# Handoff

## 2026-11-10: CLI available from any directory

- Added `install_cli.py` to install `~/.local/bin/watermarks-remover`, pointing to this checkout and the installing Python interpreter.
- Added an interactive wrapper around the existing cleaner: no arguments prompts for input and optional output; explicit arguments retain cleaner options. Supports spaces, home-directory expansion, and paths relative to the caller.
- Updated README usage and its existing Unreleased changelog; allowed installer and handoff in `.gitignore`.
- Installed locally; `~/.local/bin` is already on PATH. Run `watermarks-remover`, or `watermarks-remover INPUT -o OUTPUT`.
- Full pytest suite passed in `.venv` after installing pinned `requirements-dev.txt`; all 5 focused CLI tests and repository-wide Ruff lint/format checks passed after review.
- Keep checkout and Python interpreter available; reinstall if either moves. Launcher targets macOS/Linux.
- Work branch: `feat/standalone-clean-cli`. Upstream is read-only for the authenticated account; fork `Isaac12x/watermarks-remover` is configured as remote `fork` for PR publication.
- PR: https://github.com/guillaumemeyer/watermarks-remover/pull/441. Initial check found no comments or CI results.
