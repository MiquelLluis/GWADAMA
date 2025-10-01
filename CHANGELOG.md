## [0.4.0] 2025-10-01

GWADAMA 0.4.0 — seven months of features and fixes

Highlights
- Faster, safer resampling with robust heuristics for irregular grids; clearer up/down factor logic.
- Improved spectrogram tooling: tighter control of colormap limits (vmin/vmax), PSD handling, NaN/Inf robustness, and axis formatting.
- Flexible windowing: pass callable windows, precomputed windows, or specify the window explicitly.
- Padding utilities: new helpers (e.g. pad_to_length), consistent padding side-effects pushed into hooks.
- Whiten/PSD workflow: allow auto-estimation when PSD not provided; better defaults.

BREAKING CHANGES
- Parameter naming standardised to `fs` (sampling frequency) across modules.  
- Method renames for consistency:
  - `nonwhiten_strains` (Base) and `strains_clean` (BaseInjected) → **`strains_original`**.
  - A few functions renamed for uniform “return vs in-place” semantics.
- Deprecated functions removed.

New features
- Load strains as individual arrays inside dicts; accept function windows; support precomputed windows.
- New padding helpers (`pad_to_length`, log-pad option) and dictionary-wide padding coercion to NumPy arrays.
- Additional tests for `tat.py` and related modules; pickle backwards-compat for older attribute names.

Improvements
- Resampling: simpler path selection (uniform vs irregular), bounded denominator ratios, tolerance checks, and warnings when target `fs` exceeds input cadence.
- Spectrograms: fine control of vmin/vmax, better Y-axis units (Hz by default), improved X-axis major formatter.
- Type hints strengthened; argument names clarified; side-effects moved into dedicated hook methods to reduce cross-class coupling.

Bug fixes
- PSD computation error resolved.
- ChainAssignmentError (pandas) fixed; several kwarg typos and logical conditions corrected.
- Missing variable initialisations fixed; improved handling of NaNs/Infs and machine-precision comparisons in tests.
- Pickle loading fixed when dataset stored in whitened space without PSD.

Docs & tooling
- Plotting module now appears in the docs; docstrings cleaned for PEP-257 (no blank line before one-liners).
- Minor readme/doc clarifications and warnings in `resample()`.

Migration notes
- Search/replace parameters to `fs`.
- Update calls to the renamed methods listed above.
- If you relied on any deprecated functions removed here, switch to the documented replacements (see module docstrings and tests).
