"""
negative_set_specificity_ci.py

Computes the 95% confidence interval for the negative-only test set
specificity result (22/22 true negatives, 100% specificity, reported in
Section IV-C's False-Positive Analysis).

WHY CLOPPER-PEARSON, NOT A NORMAL APPROXIMATION:
A normal-approximation (Wald) interval is invalid here, since the observed
proportion sits exactly at the 100% boundary -- the standard formula
p_hat +/- z*sqrt(p_hat*(1-p_hat)/n) collapses to a zero-width interval
[1.0, 1.0] when p_hat = 1.0, which is statistically meaningless (it would
claim absolute certainty from only 22 observations). The Clopper-Pearson
"exact" interval is the standard, appropriate method for small-sample
binomial proportions, particularly at or near 0%/100% boundaries, and is
computed via the beta distribution rather than a normal approximation.

For comparison/completeness, the Wilson score interval (a less
conservative but still boundary-appropriate alternative) is also reported.

USAGE:
  python negative_set_specificity_ci.py
"""

import numpy as np
from scipy import stats

# ============================== RESULT BEING CHARACTERIZED ==============================
N_NEGATIVE_IMAGES = 22
N_TRUE_NEGATIVES = 22   # i.e. 0 false positives across all 22 negative-only test images
ALPHA = 0.05            # for a 95% confidence interval
# ==========================================================================================


def clopper_pearson_interval(x, n, alpha=0.05):
    """Exact binomial confidence interval via the beta distribution."""
    if x == 0:
        lo = 0.0
    else:
        lo = stats.beta.ppf(alpha / 2, x, n - x + 1)
    if x == n:
        hi = 1.0
    else:
        hi = stats.beta.ppf(1 - alpha / 2, x + 1, n - x)
    return lo, hi


def wilson_score_interval(x, n, alpha=0.05):
    """Wilson score interval, for comparison."""
    p_hat = x / n
    z = stats.norm.ppf(1 - alpha / 2)
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    half_width = (z / denom) * np.sqrt((p_hat * (1 - p_hat) / n) + (z**2 / (4 * n**2)))
    lo = max(0.0, center - half_width)
    hi = min(1.0, center + half_width)
    return lo, hi


def main():
    x, n = N_TRUE_NEGATIVES, N_NEGATIVE_IMAGES
    p_hat = x / n

    cp_lo, cp_hi = clopper_pearson_interval(x, n, ALPHA)
    wilson_lo, wilson_hi = wilson_score_interval(x, n, ALPHA)

    print("=" * 60)
    print("  NEGATIVE-ONLY TEST SET SPECIFICITY -- CONFIDENCE INTERVAL")
    print("=" * 60)
    print(f"\nObserved: {x}/{n} true negatives (specificity = {p_hat:.4f})")
    print(f"\nClopper-Pearson exact 95% CI:  [{cp_lo:.4f}, {cp_hi:.4f}]  <-- reported in manuscript")
    print(f"Wilson score 95% CI:           [{wilson_lo:.4f}, {wilson_hi:.4f}]  (for comparison)")

    print("\nSuggested manuscript sentence:")
    print(f'  "The results show {p_hat*100:.1f}% specificity ({x}/{n} true negatives, '
          f'95% CI [{cp_lo:.3f}, {cp_hi:.3f}], Clopper-Pearson exact interval) '
          f'without generating any false-positive detections."')


if __name__ == "__main__":
    main()
