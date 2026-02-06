# Contributing to Alloy

Thanks for helping improve Alloy! Please keep PRs focused and include enough context to review quickly.

## Quick start
- Create a virtual environment and install what you need:
  - Client-only: `pip install -e .`
  - Server + models: `pip install -e .[all]`
  - Or pick subsets: `.[server]`, `.[models]`, `.[gpu]`, `.[ollama]`
- Run the tests or scripts relevant to your change.

## Development notes
- Some tests require GPUs, model weights, or external services (Ollama).
- If a change affects runtime behavior, include a minimal repro or script.
- Keep new dependencies scoped to optional extras whenever possible.

## Submitting changes
- Use clear commit messages and keep diffs small.
- Update docs/README when behavior or CLI changes.
- Add or update tests when feasible.

## Licensing
By contributing, you agree that your contributions will be licensed under the MIT License.
