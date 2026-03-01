"""Scoring and measurement for the SIT experiment.

Computes per-variant per-model metrics:
- Enum field accuracy against hand-labeled ground truth
- Jensen-Shannon divergence between variant pair distributions
- Delta H_f: entropy reduction attributable to semantic indices
- Structural validity confirmation (expected 100%)
"""
