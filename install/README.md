# Install scripts

**Start here — run one of these from the repository root:**

| Script | Platform |
| --- | --- |
| `auto_install.bat` | Windows |
| `auto_install.sh` | Linux / macOS |

**Updating later:** run `update.bat` / `update.sh` — the only command you
need to update. It pulls the latest changes, upgrades `yt-dlp` to its newest
release, syncs dependencies (without rebuilding the venv), refreshes the
PO-token provider, and handles the CUDA-protected `pyproject.toml`/`uv.lock`
transparently. You never need to re-run `auto_install` — that is for the
first-time install only.

Install and update always lock the newest `yt-dlp`: video platforms change
their internals often, so the version pinned in `uv.lock` soon fails to
download. If the upgrade is not possible (e.g. offline), the pinned version is
used.

They detect your hardware (NVIDIA GPU → CUDA build, otherwise CPU), detect a
corporate proxy, download a portable Python if needed, and set everything up.
Force a specific build with `--cuda` / `--cpu`.

## Folder contents

- **`CUDA/`**, **`CPU/`** — the platform sub-scripts the entry points dispatch
  to. You can run one directly if you want to skip the auto-detection.
- **`helpers/`** — internal helper scripts, called automatically by the
  install scripts. **You never need to run these manually.**
