"""Print leave-one-out retrodiction accuracy for all supported systems.

Objective, generalisation-based accuracy metric (see
stellar_predictor.validation.retrodiction). Run before/after algorithm changes
to quantify whether prediction accuracy actually improved.

    python scripts/validate_accuracy.py
"""

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from stellar_predictor.validation.retrodiction import (
    evaluate_all_systems,
    overall_recall,
)


def main() -> None:
    results = evaluate_all_systems()

    print("=" * 78)
    print("  LEAVE-ONE-OUT RETRODICTION ACCURACY")
    print("=" * 78)
    print(f"  {'System':16s} {'tested':>6s} {'hits':>5s} {'recall':>7s} "
          f"{'med.err':>8s} {'mean.err':>8s} {'mean.score':>10s}")
    print("  " + "-" * 74)

    for name, acc in results.items():
        print(f"  {name:16s} {acc.n_tested:6d} {acc.n_hits:5d} "
              f"{acc.recall:6.0%} {acc.median_pos_err:8.1%} "
              f"{acc.mean_pos_err:8.1%} {acc.mean_score:10.3f}")

    print("  " + "-" * 74)
    print(f"  OVERALL micro-averaged recall: {overall_recall(results):.1%}")
    print("=" * 78)

    # Per-planet detail
    print("\n  Per-planet detail (hidden -> predicted):")
    for name, acc in results.items():
        print(f"\n  [{name}]")
        for m in acc.matches:
            mark = "HIT " if m.is_hit else "MISS"
            if m.predicted_a is None:
                print(f"    {mark}  {m.hidden_planet:16s} true={m.true_a:.4f}  "
                      f"(no gap predicted)")
            else:
                print(f"    {mark}  {m.hidden_planet:16s} true={m.true_a:.4f}  "
                      f"pred={m.predicted_a:.4f}  err={m.pos_error:.1%}  "
                      f"score={m.combined_score:.3f}")


if __name__ == "__main__":
    main()
