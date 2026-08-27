Socket MessagePack Postprocessor Python CLIP Situation (v5)
=========================

This postprocessor tags detected objects with a `Situation` attribute and raises a
real Analytics Event when they match an operator-configured free-text phrase (e.g.
`"a person climbing a fence"`, `"a package left on the ground"`), scored with a real
SigLIP2 vision-language model.

It targets **AI Manager v5** and is registered directly against the detector (Model
1) via `ReceiveInputImage: true` — see [Migration notes from v4](#migration-notes-from-v4)
below for what changed and why.

# What it does

1. The detector (any model producing bounding boxes) runs as normal and its
   inference-results message reaches this postprocessor, which also receives the
   frame those detections were made on (`ReceiveInputImage: true`).
2. For every detected object, this postprocessor crops its region out of the frame
   itself — squarised and padded with a configurable amount of surrounding context
   (see `crop_context_percent` below) — and runs SigLIP2's real image tower on it
   locally (on the GPU if one's available).
3. The operator-configured free-text phrase(s) are embedded via SigLIP2's text tower
   (also loaded locally, once, at startup) and compared to the crop via SigLIP's own
   sigmoid-calibrated similarity score. Above a configurable threshold, the matching
   object is tagged with a `Situation` attribute describing which phrase matched and
   at what score, **and** a real Analytics Event (`clip_situation.match`) is raised
   referencing that same object's track ID — see [Events](#events) below.

CLIP never runs inline on the reply path — see the comment above `_embed_queue` in
`postprocessor-python-clip-situation.py` for why (a single crop is several times a
camera's frame interval; replying only after CLIP finishes would make bounding boxes
lag or vanish from Nx). A background worker thread does the actual embedding and
updates a per-object verdict cache; the *next* frame for that object picks the
verdict up. The trade-off is that a situation is recognised a few hundred
milliseconds after the object appears, not on the exact frame it first shows up in.

# Migration notes from v4

Earlier iterations of this pipeline ran SigLIP against a hi-res crop delivered by a
tiny dummy "relay" model (`../hires-crop-relay-model`) chained onto the detector via
Nx AI Manager's Feature Extraction mode, with this postprocessor attached to *that*
relay model instead of the detector, reading the crop via `OriginalObjectID` +
`ReceiveInputTensor: true` and a shared-memory read. That indirection existed only
to get a hi-res crop cheaply over the v4 socket transport — v5's `ImageData` root key
(the whole frame, inline, on the same message as the detections) removes the reason
for it, so this version is attached directly to the detector and crops each object's
region itself. `../hires-crop-relay-model`, `../clip-visual-to-onnx`, and
`../preprocessor-python-hires-downscale` remain in the repo for reference but are no
longer part of this postprocessor's deployment; **`NEXT_STEPS.md` and
`PREPROCESSOR_FIX_HANDOFF.md` at the repo root still describe that two-model relay
setup and are stale for this postprocessor as of this conversion.**

Other behavior changes, all driven by the v4→v5 wire/schema change:

- **Transport**: a pair of shared-memory IPC channels set up once at process start
  (`nxai_postprocessor_sdk.ExternalProcessor`), not a Unix-domain socket accepted
  per frame. The launch arguments are IPC sync descriptors, not a socket path — see
  `../nxai-utilities-v5/nxai_postprocessor_sdk.py`.
- **`Situation` is a plain attribute dict entry** (`obj.attributes["Situation"] = ...`)
  instead of v4's parallel `AttributeKeys`/`AttributeValues` arrays, and object
  identity is `obj.track_id`, already a UUID string — no more `uuid.UUID(bytes=...)`
  decoding of a raw 16-byte ID.
- **Bounding boxes are always normalized `[0,1]`** (`obj.x/y/width/height`), so the
  old pixel-vs-fraction sniffing (`bboxes_are_normalized()`) is gone.
- **Operator-facing controls are `UserSettings`**, read fresh every message with a
  hardcoded default fallback — replacing v4's `externalprocessor.`-prefixed
  `ExternalProcessorSettings` and the sticky settings cache that existed only because
  the v4 plugin populated that block on a fraction of messages.
- **Deployment-level tunables are launch-time `Config` values** (`--key value` in
  `external_postprocessors.json`) — replacing the removed `plugin.clip-situation.ini`
  file entirely. See [Settings](#settings) below for the exact split.
- **The emitted event drops v4's `Caption` field** and both the JSON-registered id
  and the emitted id are the same bare string, `clip_situation.match` (the AIMP-1459
  fix, plugin 5.0.6+, relays event ids verbatim rather than prefixing them, unlike
  object classes — verify against your plugin's version if the event doesn't show up).
- **Logging** goes to stdout/stderr, redirected by the engine via the JSON entry's
  `LogFilepath` — there's no more manually-managed log file path or ini-based
  `debug_level`; set the level via the JSON entry's `Config.LogLevel`.

# Settings

- **`UserSettings`** (operator-facing, configured per-camera in the Nx AI Manager
  Cloud UI, delivered fresh on every message):
  - **Situation phrase #1-5** (`phrase1`..`phrase5`): up to 5 free-text phrases
    describing what to detect. Empty slots are ignored.
  - **Exclude if it looks more like #1-5** (`negative1`..`negative5`): optional,
    paired with the phrase at the same index. A competing description that should
    *not* match — the object is only reported if its phrase describes it better than
    this does. Leave blank to match on the phrase alone.
  - **Similarity threshold** (`similarity_threshold`, default `0.15`): minimum
    SigLIP match score (0-1) required before an object is tagged.
  - **Re-check interval per object** (`clip_throttle_seconds`, default `0.5`): how
    often CLIP re-examines the same tracked object.
  - **Context around object** (`crop_context_percent`, default `15.0`): how much
    surrounding scene to include around each object before scoring it, as a
    percentage of its longest side.
  - **Negative rejection strength** (`negative_margin`, default `0.5`): how
    decisively a phrase must beat its paired negative to count as a match.
- **`Config`** (launch-time, fixed at process start, in `external_postprocessors.json`):
  - `clip_model_name` (default `hf-hub:timm/ViT-L-16-SigLIP2-384`): the SigLIP2
    backbone. The single biggest lever on VRAM — see the comment above
    `DEFAULT_CLIP_MODEL_NAME` in the source for a smaller fallback.
  - `clip_precision` (`fp16` default, or `fp32`): fp16 halves weight memory on CUDA
    with negligible accuracy cost; ignored on CPU.
  - `dump_crops` (`false` default), `dump_crops_max` (`40` default), `crops_dir`:
    diagnostic — writes every crop CLIP actually sees to disk as a PNG, so crop
    quality/provenance can be judged by eye.
  - `max_crops_per_frame` (default `4`): hard cap on crops embedded per frame, sized
    from GPU-time measurement — see the comment above `DEFAULT_MAX_CROPS_PER_FRAME`.

All `UserSettings` values are delivered as their JSON-declared type or a
JSON-encoded string depending on the runtime; this postprocessor defensively
`float()`-converts them (see `_get_float_setting`).

# Events

A match does two separate things, not just one:

1. Tags the object with a `Situation` attribute (`obj.attributes["Situation"]`) —
   passive metadata, visible when inspecting the object in Nx Witness client, but
   doesn't appear in the Event Log and can't drive Camera Rules on its own.
2. Appends a real Analytics Event to `msg.events`:
   ```json
   { "ID": "clip_situation.match", "Description": "<phrase> (<score>) class=<class_name> object_id=<uuid>" }
   ```
   `object_id` is the object's own `track_id` — the same UUID the detector assigned
   and that Nx already tracks it by, so whatever consumes the event can trace it back
   to the exact object.

For this Event to show up as a selectable trigger in Nx Client's Analytics Events /
Camera Rules, it must be declared in `external_postprocessors.json`'s `Events` block
(`ID` + `Name` — see that file). An Event `ID` used in code but not declared there
silently does nothing from the Nx Client's perspective.

If no phrases are configured, or the message carries no frame (`ImageData` absent),
messages pass through unchanged.

# MessagePack schema

Standard v5 postprocessor schema (`Schedule: OBJECTS`), with `ReceiveInputImage: true`
adding a root `ImageData` block to the *same* message — no second message, no
shared-memory read:

```json
{
    "Timestamp": 1786388747281667,
    "DeviceID": "...",
    "DeviceName": "...",
    "Objects": {
        "person": [
            { "ID": "<uuid>", "x": 0.21, "y": 0.19, "width": 0.14, "height": 0.21, "confidence": 0.91, "attributes": {} }
        ]
    },
    "Events": [],
    "UserSettings": { "phrase1": "...", "similarity_threshold": 0.15, "...": "..." },
    "ImageData": { "Height": 1280, "Width": 1280, "Layout": 1, "Data": "<binary blob>" }
}
```

See `../.claude/docs/input_from_ai_manager.md` and `../.claude/docs/output_to_ai_manager.md`
in the real SDK checkout for the authoritative field-by-field reference.

# How to use

Build and install like every other postprocessor in this repo — see the root
`CMakeLists.txt`/`README.md` for the full `cmake`/install flow. This processor's own
`CMakeLists.txt` builds it as a PyInstaller `--onefile` executable, pointed at the
vendored `../nxai-utilities-v5/nxai_postprocessor_sdk.py` (no native DLL dependency,
unlike the v4-era `nxai-utilities-v2` build).

Then define it in `external_postprocessors.json` using
[external_postprocessors.json](external_postprocessors.json) in this directory as a
starting point (bare-array root — a v4-style `{"externalPostprocessors": [...]}`
wrapper makes the AI Manager reject the whole file). In the Nx AI Manager Cloud UI,
select this postprocessor for Model 1 (the detector) directly — no second model, no
Feature Extraction chaining needed.

# Output logging

The AI Manager redirects this postprocessor's stdout/stderr to the JSON entry's
`LogFilepath`:

```shell
tail -f "C:\Windows\System32\config\systemprofile\AppData\Local\Network Optix\Network Optix MetaVMS Media Server\nx_ai_manager\nxai_manager\etc\plugin.clip-situation.log"
```

Set `Config.LogLevel` to `DEBUG` in `external_postprocessors.json` to see the full
per-crop similarity matrix logged per message.

# Licence

Copyright 2026, Network Optix, All rights reserved.
