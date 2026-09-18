# Changelog

## Unreleased

### Fixes

- Handle named benchmark parameters as categorical plot axes instead of applying
  numeric log-scale detection to strings. Check positivity before dividing so
  zero-valued parameters do not cause divide-by-zero warnings.
