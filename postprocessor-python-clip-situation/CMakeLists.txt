# Add target to build everything at once
#
# --onedir (not --onefile) and --noupx matter a lot here: a --onefile build has to
# unpack this whole ~2.7GB payload (torch, CUDA/cuDNN DLLs, transformers) to a temp
# directory on every single start before any Python code runs, measured at ~100s.
# The AI Manager gives an external postprocessor only ~5-10s (5 retries) to signal
# IPC readiness before it gives up and stops routing frames to it -- so a --onefile
# build silently never receives real data, even though it starts up fine a minute
# later. See preprocessor-python-hires-downscale/CMakeLists.txt for the same fix
# applied earlier to this repo's preprocessor, for the analogous reason.
#
# Because this is --onedir, the build artifact is a DIRECTORY (the exe plus its
# _internal/ folder) rather than a single file. Install/deploy the *contents* of
# that directory into the plugin's postprocessors/ folder (exe and _internal/ as
# siblings), not the directory itself nested one level deeper -- so that
# <exe dir>/../etc and <exe dir>/../nxai-utilities-v5 keep resolving the same way
# they did under --onefile.
add_custom_target(postprocessor-python-clip-situation ALL
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMAND pip install -r requirements.txt
    # --collect-data open_clip/transformers: PyInstaller's import analysis only sees
    # Python code, not the non-code data files these packages ship alongside them
    # (open_clip's BPE tokenizer vocab; transformers' per-tokenizer resources, needed
    # because this SigLIP2 checkpoint uses an HFTokenizer, not open_clip's own BPE
    # one) -- without this, the binary builds fine but crashes at startup the first
    # time it actually tries to tokenize text.
    # --paths points pyinstaller at nxai-utilities-v5, the vendored pure-Python v5 SDK
    # helper (nxai_postprocessor_sdk.py). No --add-binary for a native DLL: v5's IPC
    # transport is ctypes against the OS's own kernel32/shared-memory APIs, not a
    # custom shared library like nxai-utilities-v2's nxai-c-utilities-shared.dll.
    COMMAND pyinstaller postprocessor-python-clip-situation.py --onedir --noupx --collect-data open_clip --collect-data transformers --workpath ${CMAKE_CURRENT_BINARY_DIR}/build --distpath ${CMAKE_CURRENT_BINARY_DIR} --paths=${CMAKE_SOURCE_DIR}/nxai-utilities-v5 -y
)
