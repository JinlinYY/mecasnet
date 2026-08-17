# Contributing

Contributions that improve portability, testing, documentation, or scientifically
equivalent implementations are welcome after the repository is public.

1. Create a focused branch and install `.[dev,analysis]`.
2. Preserve the `paper` profile's predictive information boundary.
3. Add or update tests for behavioral changes.
4. Run `ruff check .` and `pytest`.
5. Describe scientific implications, changed defaults, and compatibility with
   existing checkpoints in the pull request.

Do not submit confidential network data, event files, checkpoints trained on
restricted data, credentials, or private simulator code.

