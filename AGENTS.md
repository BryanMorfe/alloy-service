# Agent Guide

This repo is a Python project with the package rooted at `alloy/`.

## Quick commands
- Install (client-only): `pip install -e .`
- Install (full): `pip install -e .[all]`
- Run server: `python -m alloy.alloy serve --host 0.0.0.0 --port 8000`
- CLI help: `alloy --help`

## Project notes
- Optional dependencies are split into extras (`server`, `models`, `gpu`, `ollama`, `all`).
- Many tests and scripts require GPUs, model weights, or external services (e.g., Ollama).
- Prefer `rg` for search; avoid destructive git commands.
- Keep edits ASCII unless a file already uses Unicode.

## Changes
- Keep diffs small and focused; update docs when behavior changes.
- Add tests when practical; otherwise describe manual verification steps.

## Client-only usage
- The client lives in `alloy/client/`; avoid importing server/model modules from the client.
