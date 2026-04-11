# Archive Notes

This file records how the original notebook materials are treated after repository cleanup.

## Archived Notebooks

- `notebooks/archive/헬시기말고사코드최종 (1).ipynb`
  - historical main notebook
  - not the canonical pipeline
  - known to mix split logic and early AKI handling across sections

- `notebooks/archive/헬시기말고사코드최종_clean.ipynb`
  - cleaner notebook reference
  - useful for tracing old logic
  - still not treated as the source of truth

- `notebooks/archive/헬시기말고사코드최종 (11).ipynb`
  - empty notebook
  - kept only as historical residue

- `notebooks/exploration/성능비교_그래프.ipynb`
  - exploratory plotting notebook
  - not part of the script-based pipeline

## Current Principle

The repository now follows this rule:

1. preserve historical work
2. keep the reproducible implementation in `src/` and `scripts/`
3. keep old notebooks visible but clearly separated from the canonical workflow
