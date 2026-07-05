# Install scripts

**Start here — run one of these from the repository root:**

| Script | Platform |
| --- | --- |
| `auto_install.bat` | Windows |
| `auto_install.sh` | Linux / macOS |

**Updating later:** run `update.bat` / `update.sh` — it pulls the latest
changes, handles the CUDA-protected `pyproject.toml`/`uv.lock` transparently,
and syncs dependencies without rebuilding the venv.

They detect your hardware (NVIDIA GPU → CUDA build, otherwise CPU), detect a
corporate proxy, download a portable Python if needed, and set everything up.
Force a specific build with `--cuda` / `--cpu`.

## Folder contents

- **`CUDA/`**, **`CPU/`** — the platform sub-scripts the entry points dispatch
  to. You can run one directly if you want to skip the auto-detection.
- **`helpers/`** — internal helper scripts, called automatically by the
  install scripts. **You never need to run these manually.**
