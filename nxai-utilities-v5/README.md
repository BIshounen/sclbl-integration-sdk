# nxai-utilities-v5

Vendored copy of `nxai_postprocessor_sdk.py` from the real NX AI Integration SDK
(`gitlab.nxvms.dev/mpodstrechny/nxai-integration-sdk`, `nxai-utilities` submodule,
`public` branch), which is the source of record for the **AI Manager v5** wire
protocol (shared-memory IPC channels, `Objects`/`Events`/`UserSettings` message
schema).

Not wired up as a git submodule like `nxai-utilities` and `nxai-utilities-v2` are --
it's a single self-contained pure-Python file (stdlib `ctypes`/`struct` against the
OS's own kernel32/shared-memory APIs, plus `msgpack`), so it's vendored directly
rather than adding submodule/SSH-auth machinery for one file. If more processors in
this repo migrate to v5, converting this into a proper submodule pointed at the
upstream repo (matching its own `nxai-utilities` naming) would be the natural next
step.

Used by `postprocessor-python-clip-situation` (see that directory's `README.md` for
the v4->v5 migration notes). Everything else in this repo still targets v4 or the
`nxai-utilities-v2` ctypes/DLL transport and does not depend on this directory.

Update by re-copying `nxai_postprocessor_sdk.py` from a fresh checkout of the
upstream SDK's `nxai-utilities` submodule.
