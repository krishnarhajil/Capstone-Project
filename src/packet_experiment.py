
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION: keep these values identical across scenarios
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PLOTS = ROOT / "plots"

TRACE_FILE = DATA / "legitimate_signature.csv"
BASELINE_NETWORK_FILE = DATA / "network_baseline_seed_0.csv"

PACKET_SIZE_BYTES = 1200
APPLICATION_RATE_BPS = 1_000_000
LINK_RATE_BPS = 10_000_000
RELAY_PROCESSING_MS = 2.0

# Explicit malicious behavior:
# Hold every fifth packet for an additional 100 ms.
ATTACK_EVERY_N_PACKETS = 20
ATTACK_DELAY_MS = 100.0

# Abstract relay location: fraction of satellite-ground range.
RELAY_FRACTION = 0.5

C_M_PER_S = 299_792_458.0
SEED = 0


def find_column(df, candidates, required=True):
    """Find a column while allowing common naming variations."""
    lookup = {str(c).strip().lower(): c for c in df.columns}

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    if required:
        raise ValueError(
            f"Could not find any of {candidates}. "
            f"Available columns: {list(df.columns)}"
        )

    return None


def load_trace():
    if not TRACE_FILE.exists():
        raise FileNotFoundError(
            f"Missing {TRACE_FILE}. Run your existing "
            "simulate_pass.py first."
        )

    df = pd.read_csv(TRACE_FILE)

    time_col = find_column(
        df, ["time_utc", "timestamp", "time", "datetime"]
    )
    range_col = find_column(
        df, ["range_m", "slant_range_m", "distance_m"]
    )
    elevation_col = find_column(
        df, ["elevation_deg", "elevation"], required=False
    )

    df["_time"] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    df["_range_m"] = pd.to_numeric(df[range_col], errors="coerce")

    if elevation_col:
        df["_elevation"] = pd.to_numeric(
            df[elevation_col], errors="coerce"
        )
    else:
        df["_elevation"] = np.nan

    df = (
        df.dropna(subset=["_time", "_range_m"])
        .sort_values("_time")
        .drop_duplicates("_time")
        .reset_index(drop=True)
    )

    if len(df) < 2:
        raise ValueError("Satellite trace has fewer than two valid rows.")

    return df


def load_connectivity(trace):
    """
    Use the existing network baseline if available.
    Otherwise, use satellite visibility (elevation >= 10 degrees)
    when elevation is present; if not, mark the trace connected.
    """
    if not BASELINE_NETWORK_FILE.exists():
        print(
            "NOTE: No per-second network baseline CSV found at "
            f"{BASELINE_NETWORK_FILE}."
        )
        print(
            "Using elevation >= 10 degrees where available; "
            "otherwise assuming connected during this trace."
        )

        if trace["_elevation"].notna().any():
            return trace["_elevation"].fillna(-90).to_numpy() >= 10.0

        return np.ones(len(trace), dtype=bool)

    net = pd.read_csv(BASELINE_NETWORK_FILE)

    time_col = find_column(
        net,
        ["time_utc", "timestamp", "time", "datetime"],
        required=False,
    )
    connected_col = find_column(
        net, ["connected", "is_connected"], required=False
    )
    serving_col = find_column(
        net, ["serving_satellite", "satellite"], required=False
    )

    if time_col is None:
        print(
            "WARNING: Baseline CSV has no recognizable time column. "
            "Using visibility fallback."
        )
        if trace["_elevation"].notna().any():
            return trace["_elevation"].fillna(-90).to_numpy() >= 10.0
        return np.ones(len(trace), dtype=bool)

    net["_time"] = pd.to_datetime(
        net[time_col], utc=True, errors="coerce"
    )

    if connected_col:
        net["_connected"] = (
            net[connected_col]
            .astype(str).str.strip().str.lower()
            .isin(["true", "1", "yes"])
        )
    elif serving_col:
        net["_connected"] = (
            net[serving_col].notna()
            & net[serving_col].astype(str).str.strip().ne("")
        )
    else:
        raise ValueError(
            "Network baseline must contain a connected column "
            "or serving_satellite column."
        )

    net = (
        net.dropna(subset=["_time"])
        .sort_values("_time")
        .drop_duplicates("_time")
        .reset_index(drop=True)
    )

    trace_ns = trace["_time"].astype("int64").to_numpy()
    net_ns = net["_time"].astype("int64").to_numpy()
    net_state = net["_connected"].to_numpy(dtype=bool)

    # Match each satellite sample to the most recent network state.
    positions = np.searchsorted(net_ns, trace_ns, side="right") - 1
    result = np.zeros(len(trace), dtype=bool)
    valid = positions >= 0
    result[valid] = net_state[positions[valid]]

    return result


def simulate(trace, connected):
    """
    Deterministic event-based packet model.

    Scenarios:
      1. direct_baseline
      2. benign_relay
      3. malicious_selective_delay

    This is a system-level model, not a radio/protocol-stack simulation.
    """
    rng = np.random.default_rng(SEED)  # Reproducibility marker
    del rng  # No randomness is required in this deterministic experiment.

    start_time = trace["_time"].iloc[0]
    duration_s = (
        trace["_time"].iloc[-1] - start_time
    ).total_seconds()

    if duration_s <= 0:
        raise ValueError("Trace duration must be positive.")

    packet_bits = PACKET_SIZE_BYTES * 8
    generation_interval_s = packet_bits / APPLICATION_RATE_BPS
    serialization_s = packet_bits / LINK_RATE_BPS

    times = trace["_time"].astype("int64").to_numpy()
    ranges = trace["_range_m"].to_numpy(dtype=float)

    # Each link is a single FIFO transmitter.
    direct_tx_free = 0.0
    relay_first_tx_free = 0.0
    benign_second_tx_free = 0.0
    malicious_second_tx_free = 0.0

    records = []
    packet_id = 0
    generation_s = 0.0

    while generation_s < duration_s:
        packet_id += 1
        generation_time = start_time + pd.to_timedelta(
            generation_s, unit="s"
        )

        timestamp_ns = generation_time.value
        trace_idx = np.searchsorted(times, timestamp_ns, side="right") - 1
        trace_idx = max(0, min(trace_idx, len(trace) - 1))

        slant_range_m = float(ranges[trace_idx])
        is_connected = bool(connected[trace_idx])

        # The abstract relay lies along the satellite-ground line.
        leg1_m = slant_range_m * RELAY_FRACTION
        leg2_m = slant_range_m * (1.0 - RELAY_FRACTION)

        direct_delay_ms = np.nan
        benign_delay_ms = np.nan
        malicious_delay_ms = np.nan

        direct_delivered = False
        benign_delivered = False
        malicious_delivered = False

        attack_applied = (
            packet_id % ATTACK_EVERY_N_PACKETS == 0
        )
        attack_action = (
            f"hold_{ATTACK_DELAY_MS:g}ms"
            if attack_applied
            else "forward_normally"
        )

        if is_connected:
            # ---------------- Direct baseline ----------------
            direct_start = max(generation_s, direct_tx_free)
            direct_arrival = (
                direct_start
                + serialization_s
                + slant_range_m / C_M_PER_S
            )
            direct_tx_free = direct_start + serialization_s

            direct_delay_ms = (direct_arrival - generation_s) * 1000
            direct_delivered = direct_arrival <= duration_s

            # ---------------- Benign store-and-forward relay ----------------
            first_start = max(generation_s, relay_first_tx_free)
            first_arrival = (
                first_start + serialization_s + leg1_m / C_M_PER_S
            )
            relay_first_tx_free = first_start + serialization_s

            second_start = max(
                first_arrival + RELAY_PROCESSING_MS / 1000,
                benign_second_tx_free,
            )
            benign_arrival = (
                second_start + serialization_s + leg2_m / C_M_PER_S
            )
            benign_second_tx_free = second_start + serialization_s

            benign_delay_ms = (benign_arrival - generation_s) * 1000
            benign_delivered = benign_arrival <= duration_s

            
            # ---------------- Malicious selective-delay relay ----------------
            # FIFO behavior:
            # - Packets become eligible after first-hop arrival + processing.
            # - The malicious hold occurs before forwarding.
            # - The hold blocks the forwarding queue, so later packets
            #   may have to wait behind the delayed packet.

            malicious_ready_time = (
                first_arrival + RELAY_PROCESSING_MS / 1000
            )

            malicious_hold_s = (
                ATTACK_DELAY_MS / 1000 if attack_applied else 0.0
            )

            # Wait for earlier packets, then apply this packet's hold.
            malicious_second_start = (
                max(malicious_ready_time, malicious_second_tx_free)
                + malicious_hold_s
            )

            malicious_arrival = (
                malicious_second_start
                + serialization_s
                + leg2_m / C_M_PER_S
            )

            # Keep the transmitter busy through this packet's serialization.
            # A deliberate hold delays later packets in this FIFO queue.
            malicious_second_tx_free = (
                malicious_second_start + serialization_s
            )

            malicious_delay_ms = (
                malicious_arrival - generation_s
            ) * 1000

            malicious_delivered = malicious_arrival <= duration_s


        records.append({
            "packet_id": packet_id,
            "generation_time_utc": generation_time.isoformat(),
            "trace_sample_time_utc": trace["_time"].iloc[trace_idx].isoformat(),
            "connected_at_generation": is_connected,
            "packet_size_bytes": PACKET_SIZE_BYTES,
            "slant_range_m": slant_range_m,
            "relay_leg1_distance_m": leg1_m,
            "relay_leg2_distance_m": leg2_m,

            "direct_baseline_delay_ms": direct_delay_ms,
            "direct_baseline_delivered": direct_delivered,

            "benign_relay_delay_ms": benign_delay_ms,
            "benign_relay_delivered": benign_delivered,

            "malicious_relay_delay_ms": malicious_delay_ms,
            "malicious_relay_delivered": malicious_delivered,

            "attack_applied": attack_applied,
            "attack_action": attack_action,
            "attack_delay_ms": (
                ATTACK_DELAY_MS if attack_applied else 0.0
            ),
        })

        generation_s += generation_interval_s

    return pd.DataFrame(records), duration_s


def scenario_metrics(df, duration_s):
    rows = []

    scenarios = [
        (
            "direct_baseline",
            "direct_baseline_delay_ms",
            "direct_baseline_delivered",
        ),
        (
            "benign_relay",
            "benign_relay_delay_ms",
            "benign_relay_delivered",
        ),
        (
            "malicious_selective_delay",
            "malicious_relay_delay_ms",
            "malicious_relay_delivered",
        ),
    ]

    total_packets = len(df)

    for name, delay_col, delivered_col in scenarios:
        delivered = df[delivered_col].astype(bool)
        delays = df.loc[delivered, delay_col].dropna()

        delivered_packets = int(delivered.sum())
        delivered_bytes = delivered_packets * PACKET_SIZE_BYTES

        rows.append({
            "scenario": name,
            "generated_packets": total_packets,
            "delivered_packets": delivered_packets,
            "packet_delivery_ratio": (
                delivered_packets / total_packets if total_packets else 0
            ),
            "mean_delay_ms_delivered_packets": (
                float(delays.mean()) if len(delays) else np.nan
            ),
            "p95_delay_ms_delivered_packets": (
                float(delays.quantile(0.95)) if len(delays) else np.nan
            ),
            "throughput_bps": (
                delivered_bytes * 8 / duration_s if duration_s > 0 else 0
            ),
            "throughput_kbps": (
                delivered_bytes * 8 / duration_s / 1000
                if duration_s > 0 else 0
            ),
        })

    return pd.DataFrame(rows)


def create_plots(df, summary):
    PLOTS.mkdir(parents=True, exist_ok=True)

    # Delay comparison: packet-level delay for delivered packets.
    delay_cols = [
        "direct_baseline_delay_ms",
        "benign_relay_delay_ms",
        "malicious_relay_delay_ms",
    ]
    labels = ["Direct baseline", "Benign relay", "Malicious relay"]

    plt.figure(figsize=(10, 5))
    for col, label in zip(delay_cols, labels):
        valid = df[col].dropna()
        if len(valid):
            plt.plot(
                np.arange(len(valid)),
                valid,
                label=label,
                linewidth=1.2,
            )
    plt.xlabel("Packet index (within each scenario)")
    plt.ylabel("End-to-end delay (ms)")
    plt.title("Packet delay: direct vs benign relay vs malicious relay")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS / "packet_delay_comparison.png", dpi=160)
    plt.close()

    # Delivery ratio.
    plt.figure(figsize=(8, 5))
    plt.bar(summary["scenario"], summary["packet_delivery_ratio"])
    plt.ylabel("Packet delivery ratio")
    plt.ylim(0, 1.05)
    plt.title("Packet delivery ratio by scenario")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(PLOTS / "packet_delivery_ratio.png", dpi=160)
    plt.close()

    # Throughput.
    plt.figure(figsize=(8, 5))
    plt.bar(summary["scenario"], summary["throughput_kbps"])
    plt.ylabel("Delivered throughput (kbps)")
    plt.title("Delivered throughput by scenario")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(PLOTS / "packet_throughput_comparison.png", dpi=160)
    plt.close()


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)

    trace = load_trace()
    connected = load_connectivity(trace)

    packets, duration_s = simulate(trace, connected)
    summary = scenario_metrics(packets, duration_s)

    packet_file = DATA / "packet_attack_log.csv"
    summary_file = DATA / "packet_scenario_summary.csv"
    assumptions_file = DATA / "packet_experiment_assumptions.json"

    packets.to_csv(packet_file, index=False)
    summary.to_csv(summary_file, index=False)

    assumptions = {
        "experiment": "Direct baseline vs benign relay vs malicious selective-delay relay",
        "satellite_trace": str(TRACE_FILE.relative_to(ROOT)),
        "network_connectivity_trace": (
            str(BASELINE_NETWORK_FILE.relative_to(ROOT))
            if BASELINE_NETWORK_FILE.exists()
            else "Not used; visibility fallback applied"
        ),
        "seed": SEED,
        "packet_size_bytes": PACKET_SIZE_BYTES,
        "application_rate_bps": APPLICATION_RATE_BPS,
        "per_hop_link_rate_bps": LINK_RATE_BPS,
        "relay_processing_ms": RELAY_PROCESSING_MS,
        "relay_fraction_of_slant_range": RELAY_FRACTION,
        "attack_every_nth_packet": ATTACK_EVERY_N_PACKETS,
        "malicious_hold_ms": ATTACK_DELAY_MS,
        "simulation_duration_s": duration_s,
        "model_limitations": [
            "System-level deterministic packet event model, not a full radio or NR protocol-stack simulation.",
            "Relay is an abstract point along the satellite-ground range, not a geolocated device.",
            "No packet corruption, cryptographic interception, or radio errors are simulated.",
            "Connectivity is checked at packet generation, not continuously throughout forwarding."
        ]
    }

    with open(assumptions_file, "w", encoding="utf-8") as f:
        json.dump(assumptions, f, indent=2)

    create_plots(packets, summary)

    print("\nPACKET EXPERIMENT COMPLETE")
    print(f"Packets generated: {len(packets)}")
    print(f"Simulation duration: {duration_s:.2f} seconds")
    print(f"Packet log: {packet_file}")
    print(f"Summary: {summary_file}")
    print(f"Assumptions: {assumptions_file}")

    print("\nSCENARIO SUMMARY")
    print(summary.to_string(index=False))

    print("\nPLOTS")
    print(PLOTS / "packet_delay_comparison.png")
    print(PLOTS / "packet_delivery_ratio.png")
    print(PLOTS / "packet_throughput_comparison.png")


if __name__ == "__main__":
    main()
