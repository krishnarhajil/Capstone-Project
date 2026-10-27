from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. Project paths and configuration
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_DIR / "config" / "relay_mitm.json"
DATA_DIR = PROJECT_DIR / "data"
PLOT_DIR = PROJECT_DIR / "plots"

PLOT_DIR.mkdir(parents=True, exist_ok=True)

if not CONFIG_PATH.exists():
    raise FileNotFoundError(
        f"Missing {CONFIG_PATH}. Create config/relay_mitm.json first."
    )

with open(CONFIG_PATH, "r", encoding="utf-8-sig") as f:
    attack_cfg = json.load(f)

SEEDS = attack_cfg["seeds"]
RELAY_DELAYS_MS = attack_cfg["relay_forwarding_delay_ms"]

SIGNATURE_PATH = DATA_DIR / "legitimate_signature.csv"

if not SIGNATURE_PATH.exists():
    raise FileNotFoundError(
        f"Missing {SIGNATURE_PATH}. Check that your baseline is complete."
    )


# --------------------------------------------------
# 2. Load the legitimate satellite-link signature
# --------------------------------------------------

signature = pd.read_csv(SIGNATURE_PATH)

required_columns = [
    "time_utc",
    "delay_s",
    "measured_delay_s",
    "doppler_hz",
    "measured_doppler_hz",
]

missing = [c for c in required_columns if c not in signature.columns]

if missing:
    raise ValueError(f"Missing signature columns: {missing}")

if signature.empty:
    raise ValueError("The legitimate signature CSV is empty.")

signature["time_utc"] = pd.to_datetime(
    signature["time_utc"], utc=True, errors="raise"
)

for col in required_columns[1:]:
    signature[col] = pd.to_numeric(signature[col], errors="raise")

if not np.isfinite(
    signature[required_columns[1:]].to_numpy(dtype=float)
).all():
    raise ValueError("The signature contains NaN or infinite measurements.")


# --------------------------------------------------
# 3. Relay attack simulation
# --------------------------------------------------

def run_relay_scenario(seed, relay_delay_ms):
    """Apply a configured additional delay to baseline measurements."""

    network_path = DATA_DIR / f"network_baseline_seed_{seed}.csv"

    if not network_path.exists():
        raise FileNotFoundError(
            f"Missing baseline network file: {network_path}"
        )

    network = pd.read_csv(network_path)

    if len(network) != len(signature):
        raise ValueError(
            f"Seed {seed}: network has {len(network)} rows, "
            f"but signature has {len(signature)} rows."
        )

    if "time_utc" not in network.columns:
        raise ValueError(f"{network_path.name} has no time_utc column.")

    network_times = pd.to_datetime(
        network["time_utc"], utc=True, errors="raise"
    )

    if not network_times.reset_index(drop=True).equals(
        signature["time_utc"].reset_index(drop=True)
    ):
        raise ValueError(
            f"Seed {seed}: timestamps do not match the signature."
        )

    # Start with the original signature and preserve baseline measurements.
    result = signature.copy()

    # Use measured baseline delay, not propagation delay alone.
    result["baseline_measured_delay_s"] = signature["measured_delay_s"]

    # Model the relay's additional forwarding delay.
    result["relay_forwarding_delay_s"] = relay_delay_ms / 1000.0

    result["attacked_delay_s"] = (
        result["baseline_measured_delay_s"]
        + result["relay_forwarding_delay_s"]
    )

    result["delay_increase_s"] = (
        result["attacked_delay_s"]
        - result["baseline_measured_delay_s"]
    )

    result["delay_increase_ms"] = result["delay_increase_s"] * 1000.0

    # This residual is defined relative to the original baseline.
    result["timing_residual_s"] = (
        result["attacked_delay_s"]
        - result["baseline_measured_delay_s"]
    )

    # Delay-only attack: no Doppler manipulation is modeled.
    result["baseline_doppler_hz"] = signature["measured_doppler_hz"]
    result["attacked_doppler_hz"] = signature["measured_doppler_hz"]

    result["doppler_residual_hz"] = (
        result["attacked_doppler_hz"]
        - result["baseline_doppler_hz"]
    )

    # Carry baseline network metrics for reference only.
    reference_columns = [
        "connected",
        "traffic_generated_bytes",
        "traffic_delivered_bytes",
        "throughput_mbps",
        "serving_satellite",
    ]

    for col in reference_columns:
        if col in network.columns:
            result[f"baseline_{col}"] = network[col].to_numpy()

    result["seed"] = seed
    result["relay_delay_ms"] = relay_delay_ms
    result["attack_name"] = "relay_mitm_delay_only"

    output_path = (
        DATA_DIR
        / f"relay_mitm_seed_{seed}_delay_{relay_delay_ms}ms.csv"
    )

    result.to_csv(output_path, index=False)

    print(f"Saved: {output_path.relative_to(PROJECT_DIR)}")

    return result


# --------------------------------------------------
# 4. Run every seed and relay-delay scenario
# --------------------------------------------------

all_results = []

for seed in SEEDS:
    for delay_ms in RELAY_DELAYS_MS:
        scenario = run_relay_scenario(seed, delay_ms)
        all_results.append(scenario)

if not all_results:
    raise RuntimeError("No attack scenarios were run.")


# --------------------------------------------------
# 5. Calculate summary measurements
# --------------------------------------------------

combined = pd.concat(all_results, ignore_index=True)

summary_rows = []

for delay_ms, group in combined.groupby("relay_delay_ms", sort=True):
    summary_rows.append({
        "relay_delay_ms": int(delay_ms),
        "runs": int(group["seed"].nunique()),
        "baseline_mean_delay_ms": (
            group["baseline_measured_delay_s"].mean() * 1000.0
        ),
        "attacked_mean_delay_ms": (
            group["attacked_delay_s"].mean() * 1000.0
        ),
        "mean_delay_increase_ms": group["delay_increase_ms"].mean(),
        "std_delay_increase_ms": group["delay_increase_ms"].std(ddof=0),
        "mean_timing_residual_ms": (
            group["timing_residual_s"].mean() * 1000.0
        ),
        "mean_absolute_doppler_residual_hz": (
            group["doppler_residual_hz"].abs().mean()
        ),
    })

summary = pd.DataFrame(summary_rows)

summary_path = DATA_DIR / "relay_mitm_summary.csv"
summary.to_csv(summary_path, index=False)

print("\n========== RELAY MITM SUMMARY ==========")
print(summary.round(4).to_string(index=False))
print(f"\nSummary saved to: {summary_path}")


# --------------------------------------------------
# 6. Plot baseline versus attacked delay
# --------------------------------------------------

plot_seed = SEEDS[0]
plot_data = combined[combined["seed"] == plot_seed].copy()

plt.figure(figsize=(12, 6))

plt.plot(
    plot_data.index % len(signature),
    plot_data["baseline_measured_delay_s"] * 1000.0,
    label="Baseline delay",
    linewidth=2,
)

for delay_ms in sorted(RELAY_DELAYS_MS):
    scenario = plot_data[
        plot_data["relay_delay_ms"] == delay_ms
    ]

    plt.plot(
        np.arange(len(scenario)),
        scenario["attacked_delay_s"] * 1000.0,
        label=f"Relay added delay: {delay_ms} ms",
        alpha=0.85,
    )

plt.xlabel("Sample index")
plt.ylabel("Satellite-path delay proxy (ms)")
plt.title(f"Relay MITM Delay Comparison — Seed {plot_seed}")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plot_path = PLOT_DIR / "relay_mitm_delay_comparison.png"
plt.savefig(plot_path, dpi=200)
plt.close()

print(f"Plot saved to: {plot_path}")
print("\nNote: traffic metrics are baseline references, not attacked outcomes.")
print("Note: Doppler is unchanged by assumption in this delay-only model.")