"""Run one or more baseline seeds and write a summary."""
from pathlib import Path
import json
import sys
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parent
CONFIG = PROJECT_DIR / "config" / "baseline.json"
sys.path.insert(0, str(HERE))
from radio_link import run_radio_link
from network_model import run_network_model


def main():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    seed_results = []
    for seed in cfg["experiments"]["seeds"]:
        print(f"\n=== Baseline seed {seed} ===")
        radio_df = run_radio_link(PROJECT_DIR, CONFIG, seed=int(seed))
        network_df = run_network_model(PROJECT_DIR, CONFIG, radio_df, seed=int(seed))
        attempts = int(network_df["handover_attempt"].sum())
        successes = int(network_df["handover_success"].sum())
        failures = int(network_df["handover_failure"].sum())

        handover_success_rate = (
            successes / attempts if attempts > 0 else float("nan")
)

        connected_fraction = float(network_df["connected"].mean())
        generated = float(network_df["traffic_generated_bytes"].sum())
        delivered = float(network_df["traffic_delivered_bytes"].sum())
        seed_results.append({
            "seed": int(seed),
            "samples": len(network_df),
            "source_satellite": network_df["source_satellite"].iloc[0],
            "target_satellite": network_df["target_satellite"].iloc[0],
            "handover_attempt_count": attempts,
            "handover_success_count": successes,
            "handover_failure_count": failures,
            "handover_success_rate": handover_success_rate,
            "connected_fraction": connected_fraction,
            "mean_source_snr_db": float(network_df["source_snr_db"].mean()),
            "mean_target_snr_db": float(network_df["target_snr_db"].mean()),
            "generated_bytes": generated,
            "delivered_bytes": delivered,
            "traffic_delivery_ratio": delivered / generated if generated else 0.0,
        })
        print(f"Source: {network_df['source_satellite'].iloc[0]}")
        print(f"Target: {network_df['target_satellite'].iloc[0]}")
        print(
            f"Handover attempts / successes / failures: "
            f"{attempts} / {successes} / {failures}"
)
        print(f"Handover success rate: {handover_success_rate:.3f}")
        print(f"Connected fraction: {connected_fraction:.3f}")
        print(f"Traffic delivery ratio: {delivered / generated if generated else 0.0:.3f}")

    summary = pd.DataFrame(seed_results)
    output = PROJECT_DIR / "data" / "baseline_seed_summary.csv"
    summary.to_csv(output, index=False)
    print(f"\nSummary saved to: {output}")
    print("\nMean summary across seeds:")
    numeric = [
    "handover_attempt_count",
    "handover_success_count",
    "handover_failure_count",
    "handover_success_rate",
    "connected_fraction",
    "mean_source_snr_db",
    "mean_target_snr_db",
    "traffic_delivery_ratio"
]
    print(summary[numeric].agg(["mean", "std"]).to_string())

if __name__ == "__main__":
    main()
