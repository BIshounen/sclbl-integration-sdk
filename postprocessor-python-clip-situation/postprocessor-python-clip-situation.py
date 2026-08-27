#!/usr/bin/env python3
"""
Postprocessor Python CLIP Situation (v5 SDK)

Runs the *real* SigLIP2 image and text towers locally against every detected
object on the message, comparing each crop to operator-configured free-text
phrase(s) ("a person climbing a fence", "a package left on the ground") via
SigLIP's own sigmoid-calibrated similarity score. Above a configurable
threshold, tags the matching object with a `Situation` attribute describing
which phrase matched and at what score, and raises a real Analytics Event
(`clip_situation.match`) referencing that same object's track ID.

Attached directly to the detector (Model 1) via `ReceiveInputImage: true` --
no second model, no Feature Extraction chaining, no relay model. Earlier
iterations of this pipeline (see `../hires-crop-relay-model`,
`../clip-visual-to-onnx`, `../preprocessor-python-hires-downscale`) ran SigLIP
against a hi-res crop that a tiny dummy "relay" model chained via Feature
Extraction handed over through `OriginalObjectID` -- that indirection existed
only to work around the v4 socket transport's lack of a "give me the raw
frame directly" option cheap enough to use per-object. v5's `ImageData` root
key removes the reason for it: the whole frame arrives inline on the same
message as the detections, so this postprocessor crops each object's region
itself (with context padding -- see `crop_for_box`/`context_target_side`) and
never needs the relay model, the chaining, or `OriginalObjectID` at all. Those
directories remain in the repo for reference but are no longer part of this
processor's deployment.

CLIP runs on a background thread (`embedding_worker`), never inline on the
reply path -- see the comment above `_embed_queue` for why: a single crop is
~119ms on an RTX 3050 Laptop with fp16, several times a camera's frame
interval, so replying only after CLIP finishes would back up the metadata
stream and make bounding boxes lag or vanish from Nx.

Behavior changes from v4:
- No relay model / no `OriginalObjectID`: this postprocessor is registered
  against the detector directly, receives the whole frame via `ImageData`
  (`ReceiveInputImage: true`, replacing v4's `ReceiveInputTensor` + SHM read),
  and crops per object out of that frame using `msg.objects`' normalized
  `x`/`y`/`width`/`height` -- always a fraction of the frame in v5, so the old
  `bboxes_are_normalized()` pixel-vs-fraction sniffing is gone.
- `Situation` is written straight to `obj.attributes` (a plain dict per
  object) instead of v4's parallel `AttributeKeys`/`AttributeValues` arrays,
  and object identity is `obj.track_id`, already a UUID string -- no more
  `uuid.UUID(bytes=...)` decoding of a raw 16-byte ID.
- Operator-facing controls (`similarity_threshold`, `clip_throttle_seconds`,
  `crop_context_percent`, `negative_margin`, the phrase/negative Repeater) are
  `UserSettings` (ADR 0004), read fresh every frame with a hardcoded default
  fallback -- replacing v4's `externalprocessor.`-prefixed
  `ExternalProcessorSettings` keys and the sticky `_cached_settings` fallback
  that existed only because the old plugin populated that block on a fraction
  of messages. Deployment-level tunables (CLIP backbone/precision, crop-dump
  diagnostics, the per-frame crop cap) are launch-time `Config` values
  (`--key value` in `external_postprocessors.json`), replacing the removed
  `plugin.clip-situation.ini` file entirely -- matching
  `postprocessor-python-anpr-example`'s `ocr_worker_count`/`cache_ttl_sec`
  split in this same SDK.
- The emitted event drops v4's `Caption` field (v5 `EventMetadata` carries
  only `type_id` + `description`) and both the JSON-registered id and the
  emitted id are the same bare string, `clip_situation.match` -- since the
  AIMP-1459 fix (plugin 5.0.6+) the AI Manager relays event ids verbatim
  rather than prefixing them, unlike object classes. Verify against your
  plugin version (`nxai_plugin_start.log`) -- see `external_postprocessors.json`.

Runtime safeguards (see `run_message_loop`): this process self-exits, rather
than running forever, if (a) the IPC connection is confirmed lost (a
RuntimeError from the SDK), since there is no reconnect logic and retrying on
the same channel object can only fail the same way again, or (b)
`wait_for_message()` starts returning almost instantly instead of blocking, the
signature of a broken/non-inherited OS wait handle on the engine side (seen
with an actual plugin bug: Windows event handles created without
`bInheritHandle`), which otherwise spins one CPU core at ~100% forever with no
chance of receiving a real message.

An earlier version also self-exited when `os.getppid()` no longer matched the
PID captured at startup, as a defense against being orphaned by a Media Server
restart. Removed: on Windows, whatever process literally calls CreateProcess
for us is not a stable proxy for "the AI Manager is still running" -- it was,
in practice, a short-lived intermediary/launcher that exits on its own shortly
after spawning us, so this fired as a false positive on every real message
(the intermediary had already exited by the time real data started flowing),
killing the process each time real data arrived. If orphan cleanup is needed
again, it has to come from something that actually tracks the IPC relationship
(e.g. a Job Object on the engine side), not Windows process parentage.
"""

import os
import sys
import time
import queue
import logging
import threading

# The AI Manager launches this process with stdout/stderr redirected to a log
# file (LogFilepath), not a real console. On Windows a redirected stream falls
# back to the system's ANSI codepage (e.g. cp1252) instead of UTF-8, and any
# print()/log line containing a character outside that codepage's repertoire
# raises UnicodeEncodeError -- which, if it happens inside the SDK's own debug
# print() of the raw IPC sync arguments (nxai_postprocessor_sdk.IPCSync.initialize),
# aborts initialization before we ever find out whether the actual sync data was
# valid. Reconfigure both streams to UTF-8 with a non-crashing error handler
# before anything else runs, so a stray character degrades to a visible escape
# in the log instead of taking down startup.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="backslashreplace")

import numpy as np
from PIL import Image

# `torch`/`open_clip` are deliberately NOT imported here, at module level. The AI
# Manager gives an external postprocessor a fixed, short window (~5 one-second
# retries) to signal IPC readiness after spawning it -- and, discovered the hard
# way, that window is a one-way door: once exhausted, the channel is marked
# not-ready for the rest of the process's life with no further retry, silently
# (the "giving up" log line is verbose-only). torch's own import triggers CUDA
# initialization, which alone can burn most or all of that budget before this
# script ever reaches initialize_from_args() (which is what actually signals
# readiness) -- so importing it up front risked losing the race before doing
# anything. Both are imported instead in `__main__`, immediately after
# initialize_from_args() succeeds, so readiness is signaled in milliseconds
# regardless of how long the model load that follows takes.
torch = None
open_clip = None

if getattr(sys, "frozen", False):
    script_location = os.path.dirname(sys.executable)
else:
    script_location = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_location, "../nxai-utilities-v5"))
if os.path.join(script_location, "..") not in sys.path:
    sys.path.insert(0, os.path.join(script_location, ".."))

import nxai_postprocessor_sdk

logger = logging.getLogger(__name__)

PROCESSOR_NAME = "clip-situation"

# Must be declared in external_postprocessors.json's "Events" block (ID + Name), or
# it won't show up as a selectable Analytics Event / Camera Rule trigger in Nx Client.
# Emitted verbatim -- see the AIMP-1459 note in this file's module docstring.
EVENT_ID_SITUATION_MATCH = "clip_situation.match"

# Attribute key attached to a matching object (visible in the Nx object info panel).
SITUATION_ATTRIBUTE = "Situation"

# Number of situation phrases exposed in the UserSettings repeater.
NUM_PHRASE_SLOTS = 5

# ============================================================================
# Defaults -- UserSettings (operator-facing, read fresh every frame; the
# `defaultValue` in external_postprocessors.json is what the operator actually
# sees, these are only the safety-net fallback if a message ever arrives
# without the key).
# ============================================================================

# MEASURED on the original deployment, 744 crops of street traffic: median
# similarity 0.001, p90 0.098, max 0.317 -- so the theoretical 0.5 sigmoid
# midpoint fires on nothing. Start at 0.15 and tune using the per-crop scores
# logged at LogLevel=DEBUG. Crop content here (own bbox + context padding,
# taken from the same hi-res frame) is essentially the same as what the old
# relay-model path fed SigLIP, so the calibration still applies.
DEFAULT_SIMILARITY_THRESHOLD = 0.15
DEFAULT_CLIP_THROTTLE_SECONDS = 0.5
DEFAULT_CROP_CONTEXT_PERCENT = 15.0
DEFAULT_NEGATIVE_MARGIN = 0.5

# ============================================================================
# Defaults -- Config (launch-time, --key value in external_postprocessors.json)
# ============================================================================

# Must match the backbone used to export the model in ../clip-visual-to-onnx, if
# that ONNX export is ever needed again. hf-hub: prefix makes open_clip fetch+cache
# the weights from Hugging Face Hub automatically the first time this runs -- no
# manual download step.
#
# Overridable via Config, because the backbone is the single biggest lever on VRAM.
# Measured on an RTX 3050 Laptop (4GB, already ~95% consumed by the desktop and
# browsers): ViT-L at fp32 pushed the GPU to 100% utilisation with 3873/4096 MiB used,
# and PyTorch's allocator then thrashed instead of failing cleanly -- a single
# two-word phrase took over seven minutes to embed, wedging the whole postprocessor
# while the plugin logged 50 failed connections.
#
# If VRAM is tight, in rough order of preference:
#   1. clip_precision=fp16 below (halves weight memory, minimal accuracy cost)
#   2. hf-hub:timm/ViT-B-16-SigLIP2-384  (far smaller; weaker on nuanced phrasing)
DEFAULT_CLIP_MODEL_NAME = "hf-hub:timm/ViT-L-16-SigLIP2-384"

# fp16 halves the weight memory and is faster on any recent NVIDIA card. The accuracy
# cost for phrase matching is negligible -- scores shift in the third decimal, well
# inside the noise a threshold has to tolerate anyway. "fp32" restores full precision;
# ignored on CPU, where fp16 is generally slower rather than faster.
DEFAULT_CLIP_PRECISION = "fp16"

DEFAULT_DUMP_CROPS = False
DEFAULT_DUMP_CROPS_MAX = 40

# Hard cap on crops embedded per frame, to bound how long CLIP can fall behind by.
# A busy intersection frame carried 12 objects live. Rather than risk an unbounded
# backlog on a crowded frame, embed at most this many per frame and let the rest be
# picked up on a later frame (clip_throttle_seconds means they're due for a look soon
# anyway). Larger objects go first, since they carry the most readable detail.
#
# Sized from measurement, not guesswork. Benchmarked on the RTX 3050 Laptop with fp16:
# ViT-L takes ~119ms per crop, so a batch of 8 is ~950ms. Four crops is ~480ms, which
# leaves real headroom for other cameras sharing this same process. Coverage does not
# suffer: at 30fps, four per frame clears a 12-object scene in ~100ms, well inside the
# 0.5s default throttle window.
DEFAULT_MAX_CROPS_PER_FRAME = 4

# The text tower is small; use the GPU if one's available, but this is a minor
# optimization -- text embeddings are cached and only recomputed when the operator
# changes the configured phrases (see get_text_embeddings). Set in __main__, right
# after the deferred `import torch` (see the comment near the top of this file for
# why torch isn't imported at module level).
DEVICE = None

MAX_TRACKED_OBJECTS = 2048

# ============================================================================
# Runtime safeguards
#
# Added after a real incident, not speculatively: a plugin bug (non-inheritable
# Windows event handles) made wait_for_message() return almost instantly instead
# of blocking, spinning a CPU core at ~100% forever with no chance of ever
# receiving a real message. Looks identical from outside to a postprocessor
# running fine, and was only found by manually inspecting CPU time.
#
# A second safeguard -- self-exiting when os.getppid() no longer matched the
# PID captured at startup, to catch being orphaned by a Media Server restart --
# was tried and removed. On Windows, process parentage isn't a stable proxy for
# "the AI Manager is still running": a short-lived intermediary/launcher process
# calls CreateProcess for us and exits on its own shortly after, so the check
# fired as a false positive on every real message. See the note above
# run_message_loop for the full story.
# ============================================================================

# wait_for_message(timeout_ms=5000) legitimately returning in under this long
# means it isn't actually blocking -- a real timeout takes ~5s modulo scheduling
# jitter, not milliseconds.
FAST_TIMEOUT_THRESHOLD_SEC = 0.1

# How many consecutive suspiciously-fast timeouts to tolerate before treating the
# IPC wait as broken rather than a fluke. At the observed spin rate (thousands of
# iterations/sec), this threshold is reached in well under a second.
MAX_CONSECUTIVE_FAST_TIMEOUTS = 50

# Populated in __main__ once the model actually loads. Defaulted here (rather than
# left undefined) so the module stays safely importable for testing.
clip_model = None
tokenizer = None
preprocess = None
# Dtype the loaded model expects for image tensors. Set once the model is loaded;
# fp16 weights reject fp32 inputs. (None rather than torch.float32 as the
# placeholder, since torch isn't imported at module level -- see above.)
MODEL_DTYPE = None
LOGIT_SCALE = 1.0
LOGIT_BIAS = 0.0

# Launch-time tunables, overwritten from Config in __main__.
dump_crops = DEFAULT_DUMP_CROPS
dump_crops_max = DEFAULT_DUMP_CROPS_MAX
max_crops_per_frame = DEFAULT_MAX_CROPS_PER_FRAME
crops_dir = os.path.join(script_location, "crops")
_dumped_crop_count = 0

# Cache of the last-embedded phrase set, so the text tower only reruns when the
# operator actually changes the configured phrases.
_cached_phrases = None
_cached_text_embeddings = None

# CLIP runs on a background thread instead of inside the reply path.
#
# This is the difference between the feature working and the camera's metadata
# breaking. Measured with one phrase configured: embedding frames took 400-800ms to
# reply while video arrives every ~33ms, so the metadata stream backed up and
# bounding boxes stopped drawing on time. Even a single crop is ~119ms, still 3.6x
# the frame interval, so no per-frame cap or throttle can fix that on its own.
#
# So the main loop never waits for CLIP. It attaches whatever verdicts are already
# known for the objects in this frame, hands the message straight back (~5ms), and
# drops the crops that are due for a look into this queue. The worker embeds them and
# updates the verdict cache; the next frames pick that up. The only cost is that a
# situation is recognised a few hundred milliseconds after the object appears, which
# for this feature is irrelevant -- and far better than delaying every frame's
# metadata for every camera sharing this process.
#
# The queue is deliberately tiny: a crop that has been waiting is worthless, since the
# object has moved on and a fresher crop of it is already arriving. When full, new
# work is dropped rather than queued behind stale work.
_embed_queue = queue.Queue(maxsize=2)

# Objects with a job queued or being embedded, so the same object is not submitted
# repeatedly while its first look is still in flight.
_inflight = set()

# Guards _object_state and _inflight, both touched by the main loop and the worker.
_state_lock = threading.Lock()

# track_id -> {"t": last time CLIP actually ran on it, "phrase": ..., "score": ...}
# Lets a throttled object keep its attribute between embeddings instead of flickering.
_object_state = {}

# NOTE: no event de-duplication here on purpose. Nx itself lets the operator
# configure how often to receive the same event from the same device, and doing it
# here would silently override that choice with ours. So every match raises an event
# and the VMS decides what to show. The only throttle owned here is
# clip_throttle_seconds, because GPU time is genuinely ours to spend -- how often
# anything gets *reported* is the VMS's call.


# ============================================================================
# Business logic
# ============================================================================


def _get_float_setting(user_settings: dict, key: str, default: float) -> float:
    """Read a UserSettings value as a float, falling back to `default` on a bad value."""
    value = user_settings.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        logger.warning(f"Invalid value for setting '{key}': {value!r}, using default {default}")
        return default


def read_situation_settings(user_settings: dict):
    """Reads the configured situation phrase(s) and scoring controls from the message's
    UserSettings, applying hardcoded defaults for anything missing.

    Unlike v4's ExternalProcessorSettings (populated on only a fraction of messages,
    which needed a sticky `_cached_settings` fallback), v5 delivers UserSettings fresh
    on every message, so this reads it directly with no cross-message caching.
    """
    phrases, negatives = [], []
    for i in range(1, NUM_PHRASE_SLOTS + 1):
        phrase = str(user_settings.get(f"phrase{i}", "") or "").strip()
        if not phrase:
            continue
        negative = str(user_settings.get(f"negative{i}", "") or "").strip()
        phrases.append(phrase)
        negatives.append(negative)

    threshold = _get_float_setting(user_settings, "similarity_threshold", DEFAULT_SIMILARITY_THRESHOLD)
    throttle_seconds = max(0.0, _get_float_setting(
        user_settings, "clip_throttle_seconds", DEFAULT_CLIP_THROTTLE_SECONDS))
    context_percent = max(0.0, _get_float_setting(
        user_settings, "crop_context_percent", DEFAULT_CROP_CONTEXT_PERCENT))
    negative_margin = min(1.0, max(0.0, _get_float_setting(
        user_settings, "negative_margin", DEFAULT_NEGATIVE_MARGIN)))

    return phrases, negatives, threshold, throttle_seconds, context_percent, negative_margin


def get_text_embeddings(phrases: list):
    """Embeds the configured phrase(s) via SigLIP's text tower, caching the result so
    the (relatively expensive) text tower only reruns when the phrase set changes."""
    global _cached_phrases, _cached_text_embeddings

    phrases_key = tuple(phrases)
    if phrases_key == _cached_phrases:
        return _cached_text_embeddings

    logger.info("Situation phrases changed, re-embedding: %s", phrases)

    with torch.no_grad():
        tokens = tokenizer(phrases).to(DEVICE)
        text_features = clip_model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    embeddings = text_features.cpu().numpy().astype(np.float32)

    _cached_phrases = phrases_key
    _cached_text_embeddings = embeddings

    return embeddings


def context_target_side(box_w: float, box_h: float, context_fraction: float):
    """Returns the side length of the square crop to take around a box, and why.

    Aspect-aware, because SigLIP preprocesses with resize_mode 'squash': it forces the
    crop to 384x384 without preserving aspect. A tall narrow person box therefore reaches
    the model horizontally stretched, and padding it proportionally keeps the same aspect
    and the same distortion. Widening the short side instead removes the distortion AND
    supplies the surrounding scene that context-dependent phrases need.

    Two steps:

      1. Squarise -- grow the short side out to the long side.
      2. Top up -- if squarising grew the short side by LESS than `context_fraction` of
         the long side, spend the rest of the budget by growing the square further. If
         squarising already cost more than that, stop: it has had enough context, and
         adding more would bury the object in unrelated scene.

    So the percentage acts as a FLOOR on total expansion, not a ceiling, and elongated
    boxes get their context spent on fixing the aspect rather than on more background.

      50x150 person, 15%   -> squarise costs (150-50)/150 = 67% > 15%  -> side 150
      200x220 car,   15%   -> squarise costs (220-200)/220 = 9% < 15%  -> side 220*1.15 = 253
      300x300 square, 15%  -> squarise costs 0% < 15%                  -> side 345
    """
    long_side = max(box_w, box_h)
    short_side = min(box_w, box_h)
    if long_side <= 0:
        return 0.0, "degenerate"

    squarise_cost = (long_side - short_side) / long_side

    if squarise_cost < context_fraction:
        return long_side * (1.0 + context_fraction), "squarised + topped up"
    return long_side, "squarised (already over budget)"


def crop_for_box(frame: Image.Image, x: float, y: float, width: float, height: float,
                 context_fraction: float):
    """Crops the frame to a square region around one normalized [0,1] bounding box,
    sized by context_target_side(). Returns None for a box too small to be worth
    embedding.

    v5 always delivers normalized coordinates (`ObjectMetadata.x/y/width/height`), so
    unlike v4 this never needs to sniff whether a box is pixels or fractions.

    The region is CLAMPED to the frame on every edge: a crop rectangle outside the
    image would return blank padding. Clamping can of course make the result
    non-square again for a box against an edge; that is unavoidable and still better
    than a stretched crop. Note this only affects the pixels handed to CLIP --
    `msg.objects` in the reply is never touched, so what Nx draws stays the
    detector's own box regardless of how much context is added here.
    """
    frame_w, frame_h = frame.size

    x1, x2 = x * frame_w, (x + width) * frame_w
    y1, y2 = y * frame_h, (y + height) * frame_h
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))

    side, _reason = context_target_side(x2 - x1, y2 - y1, context_fraction)
    if side <= 0:
        return None

    # Grow symmetrically about the box centre, so the object stays centred in the crop.
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    half = side / 2.0

    left = int(max(0, cx - half))
    top = int(max(0, cy - half))
    right = int(min(frame_w, cx + half))
    bottom = int(min(frame_h, cy + half))

    # Below this there is nothing for CLIP to read, and preprocess() would just be
    # upscaling noise to 384x384.
    if right - left < 8 or bottom - top < 8:
        return None

    return frame.crop((left, top, right, bottom))


def decode_frame_image(image_data: "nxai_postprocessor_sdk.ImageData"):
    """Decodes a `ReceiveInputImage` `ImageData` block into an RGB PIL Image.

    Channel count isn't on the wire (only Height/Width/Layout/Data), so it's inferred
    from the buffer length against the declared frame size. Assumes 8-bit-per-channel
    samples, matching every raw camera frame this pipeline has been run against.
    Handles both Layout values (v4's equivalent SHM read only ever saw HWC): CHW is
    reshaped to (channels, h, w) and transposed to HWC before handing to PIL.
    """
    width, height = image_data.width, image_data.height
    data = image_data.bytes
    pixel_count = width * height
    if width <= 0 or height <= 0 or pixel_count <= 0 or len(data) % pixel_count != 0:
        logger.warning(
            "ImageData buffer (%d bytes) doesn't divide frame %dx%d; skipping",
            len(data), width, height,
        )
        return None

    channels = len(data) // pixel_count
    if channels not in (1, 3):
        logger.warning("Unsupported channel count %d in ImageData; skipping", channels)
        return None

    array = np.frombuffer(data, dtype=np.uint8)
    if image_data.layout == nxai_postprocessor_sdk.ImageOrder.CHW:
        array = array.reshape(channels, height, width).transpose(1, 2, 0)
    else:
        array = array.reshape(height, width, channels)

    if channels == 1:
        return Image.fromarray(array[:, :, 0], mode="L").convert("RGB")
    return Image.fromarray(array, mode="RGB")


def embed_images(images: list):
    """Embeds several PIL crops in ONE forward pass, returning (N, D) L2-normalized.

    Batching matters here: a busy frame carried 12 objects live, and 12 separate
    forward passes through a ViT-L is markedly slower than one batch of 12.
    """
    tensors = torch.stack([preprocess(image) for image in images]).to(DEVICE, dtype=MODEL_DTYPE)

    with torch.no_grad():
        image_features = clip_model.encode_image(tensors)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    return image_features.cpu().numpy().astype(np.float32)


def score_against_phrases(image_embeddings: np.ndarray, phrases: list, negatives: list):
    """Scores crops against each positive phrase, and against its paired negative if set.

    Returns (absolute, relative), both shaped (n_crops, n_phrases):

      absolute[i][p]  SigLIP's own calibrated score for phrase p -- "does this phrase describe
                      the crop at all". sigmoid(scale * cos + bias).
      relative[i][p]  how strongly the positive beats its negative -- 0.5 means they fit equally
                      well, above means the positive wins. 1.0 where no negative is configured.

    ON THE FORMULA. The obvious reading of "positive over negative" is the literal ratio
    positive_score / negative_score, but that behaves badly here. Both are sigmoid outputs, and
    measured on this deployment they sit near zero (median 0.001, p90 0.098) because SigLIP's
    calibration targets whole-image/caption pairs rather than small object crops. Dividing two
    near-zero numbers is numerically unstable and unbounded -- 0.02/0.0001 is 200, 0.02/0.00001 is
    2000, and neither tells you anything useful, nor can you put a fixed threshold on it.

    The principled form is to ask which of the two phrases better describes the crop, i.e. a
    softmax over the two logits:

        p = exp(logit_pos) / (exp(logit_pos) + exp(logit_neg)) = sigmoid(logit_pos - logit_neg)

    and since logit = logit_scale * cos + logit_bias, the bias cancels:

        p = sigmoid(logit_scale * (cos_pos - cos_neg))

    That is bounded in [0, 1], has a parameter-free decision point at 0.5, and is exactly the
    two-class zero-shot comparison CLIP-family models are built for. With logit_scale ~108 it also
    discriminates sharply: a cosine gap of just 0.02 gives 0.90.

    BOTH gates are kept, deliberately. Relative score alone would accept a photo of a cat for
    "a person with empty hands" vs "backpack", simply because a cat is even less backpack-like than
    it is person-like. So a match requires the positive to describe the crop in absolute terms AND
    to beat its negative.

    Only ONE image embedding is computed regardless -- the "double check" is two sets of text
    embeddings against the same image vector, and the text tower is cached. So negatives cost
    essentially no GPU time.
    """
    text_pos = get_text_embeddings(phrases)
    cos_pos = image_embeddings @ text_pos.T
    absolute = 1.0 / (1.0 + np.exp(-(LOGIT_SCALE * cos_pos + LOGIT_BIAS)))

    relative = np.ones_like(absolute)

    # Embed only the phrases that actually have a negative, then scatter back into place.
    active = [i for i, n in enumerate(negatives[:len(phrases)]) if n]
    if active:
        cos_neg = image_embeddings @ get_text_embeddings([negatives[i] for i in active]).T
        for column, i in enumerate(active):
            relative[:, i] = 1.0 / (
                1.0 + np.exp(-(LOGIT_SCALE * (cos_pos[:, i] - cos_neg[:, column]))))

    return absolute, relative


def pick_best_phrase(absolute_row, relative_row, phrases: list, threshold: float, margin: float):
    """Chooses the winning phrase for one crop, or None.

    A candidate must clear the absolute threshold AND beat its negative by `margin`. Among the
    survivors the highest absolute score wins, so behaviour for phrases without a negative is
    unchanged from before negatives existed.
    """
    best = None
    for index in range(len(phrases)):
        abs_score = float(absolute_row[index])
        rel_score = float(relative_row[index])
        if abs_score < threshold or rel_score < margin:
            continue
        if best is None or abs_score > best[1]:
            best = (index, abs_score, rel_score)
    return best


def dump_crop_image(crop: Image.Image, frame_width: int, frame_height: int, label: str):
    """Diagnostic: writes the crop CLIP actually sees to disk as a PNG, so the crop's
    provenance/quality can be judged by eye. Never fatal: a diagnostic must not be able
    to break the pipeline. Enable via the `dump_crops`/`dump_crops_max` Config keys.
    """
    global _dumped_crop_count
    try:
        if _dumped_crop_count >= dump_crops_max:
            return

        os.makedirs(crops_dir, exist_ok=True)

        _dumped_crop_count += 1
        w, h = crop.size
        name = (f"crop{_dumped_crop_count:03d}_{label}_{w}x{h}"
                f"_frame{frame_width}x{frame_height}.png")
        crop.convert("RGB").save(os.path.join(crops_dir, name.replace(":", "-")))
    except Exception as e:
        logger.warning("Crop dump failed (continuing): %s", e)


def message_time_seconds(msg: "nxai_postprocessor_sdk.ObjectsPostProcessorMessage"):
    """Best available clock for throttling. Prefers the pipeline's own frame Timestamp
    (microseconds) so behaviour is identical on live and recorded footage, and only
    falls back to wall time if it is absent."""
    if isinstance(msg.timestamp, (int, float)) and msg.timestamp > 0:
        return float(msg.timestamp) / 1_000_000.0
    return time.monotonic()


def prune_tracking(tracking: dict):
    """Objects come and go forever; keep only the most recently seen so a long-running
    process cannot grow without bound."""
    if len(tracking) > MAX_TRACKED_OBJECTS:
        overflow = len(tracking) - MAX_TRACKED_OBJECTS
        for stale_key in sorted(tracking, key=lambda k: tracking[k]["t"])[:overflow]:
            del tracking[stale_key]


def attach_situation(msg: "nxai_postprocessor_sdk.ObjectsPostProcessorMessage",
                     obj: "nxai_postprocessor_sdk.ObjectMetadata", class_name: str,
                     phrase: str, score: float):
    """Attaches the `Situation` attribute to one object and raises the Analytics Event.

    Fires on every match, including matches carried over from a throttled (cached)
    verdict -- Nx's own per-device event rate limiting is where repeat suppression
    belongs, so the operator stays in control of it.
    """
    obj.attributes[SITUATION_ATTRIBUTE] = f"{phrase} ({score:.2f})"

    event = nxai_postprocessor_sdk.EventMetadata()
    event.type_id = EVENT_ID_SITUATION_MATCH
    event.description = f"{phrase} ({score:.2f}) class={class_name} object_id={obj.track_id}"
    msg.events.append(event)

    logger.info(
        "Situation match: class=%s object_id=%s phrase=%r score=%.3f",
        class_name, obj.track_id, phrase, score,
    )


def apply_situation_matches(msg: "nxai_postprocessor_sdk.ObjectsPostProcessorMessage",
                            phrases: list, negatives: list, threshold: float,
                            throttle_seconds: float, context_percent: float,
                            negative_margin: float):
    """Attaches verdicts already known for the objects in this frame, and queues the
    ones due for a fresh look. Returns immediately -- typically in single-digit
    milliseconds. `embedding_worker` does the expensive part on its own thread.

    That split is not an optimisation, it is what makes the feature usable at all --
    see the comment above `_embed_queue`.
    """
    if not phrases or msg.image_data is None:
        return msg

    now = message_time_seconds(msg)
    due = []
    reused = 0

    with _state_lock:
        for class_name, objects in msg.objects.items():
            for obj in objects:
                track_id = obj.track_id
                if track_id is None:
                    continue

                state = _object_state.get(track_id)
                if state is not None and state["phrase"] is not None:
                    # Known match: show it now, regardless of whether a refresh is due.
                    attach_situation(msg, obj, class_name, state["phrase"], state["score"])

                fresh = (state is not None and throttle_seconds > 0
                         and (now - state["t"]) < throttle_seconds)
                if fresh or track_id in _inflight:
                    reused += 1
                    continue

                due.append((class_name, obj, track_id))

    if not due:
        logger.debug("%d objects, none due, %d reused from cache", reused, reused)
        return msg

    # Biggest boxes first: they carry the most readable detail, so if the cap trims the
    # list these are the ones worth spending a forward pass on.
    if max_crops_per_frame > 0 and len(due) > max_crops_per_frame:
        due.sort(key=lambda t: t[1].width * t[1].height, reverse=True)
        due = due[:max_crops_per_frame]

    try:
        _embed_queue.put_nowait(
            (msg.image_data, due, list(phrases), list(negatives), threshold,
             context_percent / 100.0, negative_margin, now))
    except queue.Full:
        # The worker is still busy. Dropping is correct: by the time it caught up this
        # crop would be stale, and a fresher one is already on its way.
        logger.debug("Embed queue full, skipping %d crops this frame", len(due))
        return msg

    with _state_lock:
        for _, _, track_id in due:
            _inflight.add(track_id)

    logger.debug(
        "%d objects, %d queued for embedding, %d reused from cache",
        len(due) + reused, len(due), reused,
    )
    return msg


def embedding_worker():
    """Consumes crop jobs and updates the verdict cache. Runs off the reply path.

    Never dies: an exception here would silently stop all situation detection while the
    pipeline carried on looking healthy, which is worse than a logged failure.
    """
    while True:
        try:
            job = _embed_queue.get()
            if job is None:
                return
            image_data, due, phrases, negatives, threshold, context_fraction, margin, now = job

            try:
                frame = decode_frame_image(image_data)
                if frame is None:
                    continue

                crops, kept = [], []
                for class_name, obj, track_id in due:
                    crop = crop_for_box(frame, obj.x, obj.y, obj.width, obj.height, context_fraction)
                    if crop is not None:
                        crops.append(crop)
                        kept.append((class_name, track_id))
                        if dump_crops:
                            dump_crop_image(crop, image_data.width, image_data.height, class_name)

                if not crops:
                    continue

                started = time.monotonic()
                image_embeddings = embed_images(crops)
                absolute, relative = score_against_phrases(image_embeddings, phrases, negatives)

                for row, (class_name, track_id) in enumerate(kept):
                    winner = pick_best_phrase(absolute[row], relative[row], phrases, threshold, margin)

                    logger.debug(
                        "  %s/%s: %s (threshold=%.3f, negative_margin=%.2f) -> %s",
                        class_name, track_id[:8] if track_id else "?",
                        {ph: (round(float(absolute[row][i]), 3), round(float(relative[row][i]), 3))
                         for i, ph in enumerate(phrases)},
                        threshold, margin,
                        f"{phrases[winner[0]]!r} abs={winner[1]:.3f} rel={winner[2]:.3f}"
                        if winner else "no match",
                    )

                    matched = winner is not None
                    best_phrase = phrases[winner[0]] if matched else None
                    best_score = winner[1] if matched else 0.0

                    with _state_lock:
                        _object_state[track_id] = {
                            "t": now,
                            "phrase": best_phrase if matched else None,
                            "score": best_score,
                        }
                        prune_tracking(_object_state)

                    if matched:
                        logger.info(
                            "Situation match: class=%s object_id=%s phrase=%r score=%.3f",
                            class_name, track_id, best_phrase, best_score,
                        )

                logger.debug("Embedded %d crops in %.0fms (off the reply path)",
                             len(crops), (time.monotonic() - started) * 1000.0)
            finally:
                with _state_lock:
                    for _, _, track_id in due:
                        _inflight.discard(track_id)
                _embed_queue.task_done()
        except Exception as e:
            logger.error("Embedding worker error (continuing): %s", e, exc_info=True)


def handle_message(msg: "nxai_postprocessor_sdk.ObjectsPostProcessorMessage"):
    """Reads this frame's settings and applies/queues situation matching.

    Args:
        msg: The parsed message, with `msg.objects` populated by the detector and
            `msg.image_data` populated (ReceiveInputImage is set in the JSON).

    Returns:
        The same msg, with cached `Situation` attributes applied to matching objects
        and any newly queued embedding work handed to the background worker.
    """
    phrases, negatives, threshold, throttle_seconds, context_percent, negative_margin = (
        read_situation_settings(msg.user_settings)
    )
    if msg.objects:
        # TEMPORARY diagnostic: apply_situation_matches() returns silently (no log
        # line at all) whenever phrases is empty or image_data is missing, so a
        # message that actually carries a detected object but never produces any
        # of our own processing log lines is otherwise invisible to explain. Only
        # logged when there's an object present, to avoid spamming every frame.
        logger.debug(
            "DIAG raw UserSettings=%r -> parsed phrases=%r image_data=%s",
            msg.user_settings, phrases, "present" if msg.image_data is not None else "MISSING",
        )
    return apply_situation_matches(
        msg, phrases, negatives, threshold, throttle_seconds, context_percent, negative_margin)


def run_message_loop(processor: "nxai_postprocessor_sdk.ExternalProcessor"):
    """Serve messages until the engine stops sending or the response channel fails.

    Exits (rather than looping forever) on conditions that would otherwise leave
    this process running indefinitely with no chance of doing useful work --
    see the module-level "Runtime safeguards" comment for why each one exists.

    NOTE: an earlier version of this also self-exited when `os.getppid()` no
    longer matched the PID captured at startup, as a defense against being
    orphaned by a Media Server restart. Removed: on Windows, whatever process
    literally calls CreateProcess for us is not a stable proxy for "the AI
    Manager is still running" -- it was, in practice, a short-lived
    intermediary/launcher that exits on its own shortly after spawning us,
    which made this fire as a false positive on every real message (the
    intermediary had exited by the time real data started flowing), killing
    the process each time. Process parentage on Windows just isn't a reliable
    liveness signal here; the IPC-connection-lost check below is.
    """
    consecutive_fast_timeouts = 0
    while True:
        logger.debug("Waiting for input message")
        try:
            wait_started = time.monotonic()
            message_bytes = processor.wait_for_message(timeout_ms=5000)
            wait_elapsed = time.monotonic() - wait_started

            if message_bytes is None:
                if processor.exit_requested:
                    logger.info("Received graceful exit signal, shutting down")
                    return

                if wait_elapsed < FAST_TIMEOUT_THRESHOLD_SEC:
                    consecutive_fast_timeouts += 1
                    if consecutive_fast_timeouts >= MAX_CONSECUTIVE_FAST_TIMEOUTS:
                        logger.critical(
                            "wait_for_message() returned in %.3fs (requested 5s), %d times in "
                            "a row -- the IPC wait is not actually blocking. This is the "
                            "signature of an invalid/non-inherited OS handle on the engine "
                            "side, not a real timeout. Exiting rather than spinning a CPU "
                            "core forever with no chance of ever receiving a message.",
                            wait_elapsed, consecutive_fast_timeouts,
                        )
                        return
                else:
                    consecutive_fast_timeouts = 0
                logger.debug("Timeout waiting for message")
                continue

            consecutive_fast_timeouts = 0
            started = time.monotonic()
            logger.debug(f"Received message ({len(message_bytes)} bytes)")

            msg = processor.parse_objects_message(message_bytes)
            if msg is None:
                logger.warning("Could not parse message, skipping")
                continue

            msg = handle_message(msg)

            response_bytes = processor.write_objects_message(msg)
            if response_bytes is None:
                logger.error("Failed to serialize message")
                continue
            if not processor.send_response(response_bytes):
                logger.error("Failed to send response")
                return

            elapsed_ms = (time.monotonic() - started) * 1000.0
            if elapsed_ms > 500.0:
                logger.warning(
                    "Slow frame: %.0fms to reply. Consider raising clip_throttle_seconds "
                    "or lowering max_crops_per_frame.", elapsed_ms,
                )
            else:
                logger.debug("Replied in %.0fms", elapsed_ms)
        except RuntimeError as e:
            # wait_for_message()/send_response() raise this specifically when the IPC
            # connection is confirmed lost. There is no reconnect logic in the SDK, so
            # retrying on the same channel object can only raise the same error again --
            # looping here just becomes another orphaned, useless process.
            logger.critical(f"IPC connection lost, shutting down: {e}")
            return
        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            continue


# ============================================================================
# Main entry point
# ============================================================================

if __name__ == "__main__":
    logger = nxai_postprocessor_sdk.setup_logging()
    processor_config = nxai_postprocessor_sdk.load_config(sys.argv)
    nxai_postprocessor_sdk.apply_config_logging(processor_config, logger)

    logger.info(f"Initializing '{processor_config.get('ProcessorName', PROCESSOR_NAME)}' postprocessor")
    logger.debug(f"Command-line arguments: {sys.argv}")

    clip_model_name = processor_config.get("clip_model_name", DEFAULT_CLIP_MODEL_NAME)
    clip_precision = processor_config.get("clip_precision", DEFAULT_CLIP_PRECISION)
    dump_crops = processor_config.get("dump_crops", "false").strip().lower() in ("1", "true", "yes")
    dump_crops_max = int(processor_config.get("dump_crops_max", DEFAULT_DUMP_CROPS_MAX))
    max_crops_per_frame = int(processor_config.get("max_crops_per_frame", DEFAULT_MAX_CROPS_PER_FRAME))
    crops_dir = processor_config.get("crops_dir", crops_dir)

    processor = nxai_postprocessor_sdk.ExternalPostprocessor()
    if not processor.initialize_from_args():
        logger.error("Failed to initialize postprocessor from arguments")
        sys.exit(1)
    if not processor.is_initialized:
        logger.error("Postprocessor initialization did not complete successfully")
        sys.exit(1)
    logger.info(f"Postprocessor '{processor.name}' initialized successfully")

    # Deferred from module level -- see the comment near the top of this file for
    # why. Import now, immediately after signaling readiness, so CUDA
    # initialization (triggered by `import torch` itself) and the actual model
    # load below can never cost us the engine's fixed, one-shot readiness window.
    import torch
    import open_clip
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        # fp16 only helps on CUDA; on CPU it is usually slower than fp32.
        use_half = clip_precision.strip().lower() in ("fp16", "half") and DEVICE == "cuda"
        precision = "fp16" if use_half else "fp32"

        logger.info(f"Loading {clip_model_name} on {DEVICE} (precision={precision})...")
        clip_model, _, preprocess = open_clip.create_model_and_transforms(
            clip_model_name, device=DEVICE, precision=precision
        )
        clip_model.eval()
        MODEL_DTYPE = torch.float16 if use_half else torch.float32
        tokenizer = open_clip.get_tokenizer(clip_model_name)

        if DEVICE == "cuda":
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            logger.info(
                "CUDA memory after loading model: %.0f MiB free of %.0f MiB total "
                "(precision=%s). If free memory is under ~500 MiB, expect the allocator "
                "to thrash rather than fail cleanly -- switch clip_model_name to "
                "hf-hub:timm/ViT-B-16-SigLIP2-384 in Config.",
                free_bytes / 1048576, total_bytes / 1048576, precision,
            )

        # SigLIP stores these as learned parameters; logit_scale is kept in log-space
        # (standard CLIP/SigLIP convention) so it must be exponentiated before use.
        # Plain CLIP checkpoints have no logit_bias -- default it to 0 in that case.
        LOGIT_SCALE = clip_model.logit_scale.exp().item()
        LOGIT_BIAS = clip_model.logit_bias.item() if hasattr(clip_model, "logit_bias") else 0.0
        logger.info(
            "Loaded %s on %s (logit_scale=%.3f, logit_bias=%.3f)",
            clip_model_name, DEVICE, LOGIT_SCALE, LOGIT_BIAS,
        )
    except Exception as e:
        logger.error("Failed to load CLIP/SigLIP model: %s", e, exc_info=True)
        sys.exit(1)

    # Daemon so a shutdown is never held up waiting for an in-progress embedding.
    worker = threading.Thread(target=embedding_worker, name="clip-embed", daemon=True)
    worker.start()
    logger.info("Embedding worker started (CLIP runs off the reply path)")

    logger.info("Waiting for messages from NXAI runtime...")
    try:
        run_message_loop(processor)
    except KeyboardInterrupt:
        logger.info("Exited with keyboard interrupt")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
    finally:
        logger.info("Postprocessor shutdown")
