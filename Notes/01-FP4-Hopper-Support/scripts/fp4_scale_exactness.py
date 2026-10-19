"""Is (E2M1 weight) x (E4M3 scale) exactly representable in FP8 E4M3?

Pure Python, no torch. Prints the share of exact products and the
rounding error when the product is rounded to E4M3 (round-to-nearest-even).
"""
from fractions import Fraction as F
import math


def e4m3_values():
    """All finite non-negative E4M3 (fn) values as exact fractions."""
    vals = set()
    for e in range(16):
        for m in range(8):
            if e == 15 and m == 7:
                continue  # NaN
            if e == 0:
                v = F(m, 8) * F(2) ** (1 - 7)  # subnormal
            else:
                v = (1 + F(m, 8)) * F(2) ** (e - 7)
            vals.add(v)
    return sorted(vals)


E2M1 = [F(0), F(1, 2), F(1), F(3, 2), F(2), F(3), F(4), F(6)]
E4M3 = e4m3_values()
E4M3_SET = set(E4M3)
E4M3_MAX = E4M3[-1]


def round_e4m3(x):
    """Round a non-negative fraction to the nearest E4M3 value (ties to even)."""
    if x > E4M3_MAX:
        return None  # overflow
    lo = max(v for v in E4M3 if v <= x)
    hi = min(v for v in E4M3 if v >= x)
    if x - lo < hi - x:
        return lo
    if hi - x < x - lo:
        return hi
    # tie: pick the one with an even mantissa code
    return lo if E4M3.index(lo) % 2 == 0 else hi


def main():
    # Only scales whose products stay in the normal range are interesting:
    # use the scales of one binade, [1, 2), and nonzero weights. A power of
    # two shifts the whole picture without changing exactness (if no
    # overflow or underflow happens), so one binade tells the whole story.
    scales = [v for v in E4M3 if 1 <= v < 2]
    weights = [w for w in E2M1 if w != 0]
    total = exact = 0
    worst = F(0)
    sq = 0.0
    for s in scales:
        for w in weights:
            p = w * s
            total += 1
            if p in E4M3_SET:
                exact += 1
                continue
            r = round_e4m3(p)
            rel = abs(r - p) / p
            worst = max(worst, rel)
            sq += float(rel) ** 2
    print(f"scale mantissas tested: {len(scales)}, nonzero E2M1 values: {len(weights)}")
    print(f"exact products: {exact}/{total} = {exact/total:.1%}")
    print(f"worst relative error after rounding to E4M3: {float(worst):.4%}")
    print(f"RMS relative error over all {total} pairs: {math.sqrt(sq/total):.4%}")
    # Range problem: the largest product overflows E4M3.
    print(f"largest product 6 * 448 = {6*448}, E4M3 max = {float(E4M3_MAX)}")
    # Which weights are always exact? Powers of two.
    for w in weights:
        ok = all(w * s in E4M3_SET for s in scales)
        print(f"  weight {float(w):>4}: always exact for every scale mantissa: {ok}")


if __name__ == "__main__":
    main()
    # Self-check: values the format must contain.
    assert F(448) in E4M3_SET and F(1, 512) in E4M3_SET and len(E4M3) == 127
