"""Simple UE/traffic/handover system-level model using actual TLE visibility.

The model scans the supplied TLE file for a second satellite that overlaps the
existing source pass near its end. It is not a full NR protocol-stack model.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skyfield.api import load, wgs84
from radio_link import calculate_link_budget

C = 299_792_458.0
KB_DBM_HZ = -174.0


def _elevation_range(satellite, times, observer):
    topocentric = satellite.at(times) - observer.at(times)
    alt, _, distance = topocentric.altaz()
    return np.asarray(alt.degrees), np.asarray(distance.m)


def _to_skyfield_times(ts, time_strings):
    dt = pd.to_datetime(pd.Series(time_strings), utc=True)
    datetimes = [v.to_pydatetime() for v in dt]
    return ts.from_datetimes(datetimes)


def _identify_source(satellites, t0, observer, target_range_m):
    best_sat, best_error = None, float("inf")
    for sat in satellites:
        try:
            topo = sat.at(t0) - observer.at(t0)
            _, _, dist = topo.altaz()
            err = abs(float(dist.m) - float(target_range_m))
            if err < best_error:
                best_sat, best_error = sat, err
        except Exception:
            continue
    if best_sat is None or best_error > 20_000:
        raise RuntimeError(
            "Could not match the first CSV range to a satellite in data/starlink.tle. "
            "Check that the TLE file is the same one used by simulate_pass.py and that the CSV is current."
        )
    return best_sat


def run_network_model(project_dir: Path, config_path: Path, radio_df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    tle_path = project_dir / "data" / "starlink.tle"
    source_csv = project_dir / "data" / "legitimate_signature.csv"
    if not tle_path.exists():
        raise FileNotFoundError(f"Cannot find {tle_path}")
    df = pd.read_csv(source_csv)
    ts = load.timescale()
    satellites = load.tle_file(str(tle_path), reload=False)
    if not satellites:
        raise RuntimeError("No satellites loaded from TLE file")
    site = cfg["site"]
    observer = wgs84.latlon(float(site["latitude_deg"]), float(site["longitude_deg"]), elevation_m=float(site["altitude_m"]))
    times = _to_skyfield_times(ts, df["time_utc"])
    source = _identify_source(satellites, times[0], observer, df["range_m"].iloc[0])
    source_el, source_range = _elevation_range(source, times, observer)

    min_el = float(cfg["radio"]["minimum_elevation_deg"])
    # Sample the whole pass more coarsely to identify candidate target satellites.
    coarse_idx = np.unique(np.linspace(0, len(df) - 1, min(30, len(df))).astype(int))
    coarse_times = _to_skyfield_times(ts, df["time_utc"].iloc[coarse_idx].tolist())
    late_start = int(0.65 * len(coarse_idx))
    candidate_sats = []
    for sat in satellites:
        if sat is source:
            continue
        try:
            el, _ = _elevation_range(sat, coarse_times, observer)
            if np.any(el >= min_el) and np.any(el[late_start:] >= min_el):
                candidate_sats.append((sat, el))
        except Exception:
            continue

    if not candidate_sats:
        raise RuntimeError(
            "No second satellite from this TLE file was visible during the late part of this pass. "
            "Refresh data/starlink.tle or update simulate_pass.py to select a longer time window; "
            "do not fake a handover and call it a physical satellite handover."
        )

    # Evaluate candidate tracks on every second; prefer long overlap near the source pass end.
    sample_index = np.arange(len(source_el))
    source_edge = (source_el <= float(cfg["handover"]["source_leave_elevation_deg"])) & (sample_index >= int(0.5 * len(source_el)))
    best = None
    best_score = (-1, -1.0)
    for sat, _ in candidate_sats:
        try:
            el, rng_m = _elevation_range(sat, times, observer)
            overlap = (source_el >= min_el) & (el >= min_el)
            late_overlap = overlap & source_edge
            score = (int(late_overlap.sum()), float(np.maximum(el[source_edge], 0).sum()))
            if score > best_score:
                best_score, best = score, (sat, el, rng_m)
        except Exception:
            continue
    if best is None or best_score[0] < int(cfg["handover"]["minimum_overlap_samples"]):
        raise RuntimeError(
            "A suitable overlapping target satellite was not found for this pass. "
            "Use a fresher/larger TLE set or choose another source pass. A clean handover must be demonstrated, not assumed."
        )
    target, target_el, target_range = best

    radio = cfg["radio"]
    traffic = cfg["traffic"]
    ho = cfg["handover"]

    # Use the same link-budget function as the source link.
    # The target has an independent, reproducible shadowing realization.
    target_rng = np.random.default_rng(seed + 10000)
    target_shadowing = target_rng.normal(
        0.0,
        float(radio["shadowing_sigma_db"]),
        len(df),
    )
    target_link = calculate_link_budget(
        range_m=target_range,
        elevation_deg=target_el,
        radio_cfg=radio,
        shadowing_db=target_shadowing,
    )
    target_snr = target_link["snr_db"].to_numpy()

    source_snr = radio_df["snr_db"].to_numpy()
    source_ok = radio_df["link_available"].to_numpy(dtype=bool)
    # Apply the handover-specific target elevation threshold as well.
    target_ok = (
        (target_el >= float(ho["target_min_elevation_deg"]))
        & (target_snr >= float(radio["minimum_snr_db"]))
    )

    # Start attached to the source. Trigger a target handover when source nears its edge
    # and target is available for the configured number of consecutive samples.
    dt_s = float(traffic["sample_interval_s"])
    ttt_n = max(1, int(np.ceil(float(ho["time_to_trigger_s"]) / dt_s)))
    interrupt_n = max(0, int(np.ceil(float(ho["handover_interruption_s"]) / dt_s)))
    serving = []
    handover_attempt = np.zeros(len(df), dtype=bool)
    handover_success = np.zeros(len(df), dtype=bool)
    handover_failure = np.zeros(len(df), dtype=bool)
    handover_interruption = np.zeros(len(df), dtype=bool)
    state = "source"
    target_ready_count = 0
    interruption_remaining = 0
    for i in range(len(df)):
        if state == "source":
            serving.append(source.name)
            if source_el[i] <= float(ho["source_leave_elevation_deg"]) and target_ok[i]:
                target_ready_count += 1
            else:
                target_ready_count = 0
            if target_ready_count >= ttt_n:
                handover_attempt[i] = True
                state = "target"
                handover_success[i] = True
                interruption_remaining = interrupt_n
                serving[-1] = target.name
                target_ready_count = 0
        else:
            serving.append(target.name)
        if state == "source" and not source_ok[i]:
            # No valid serving link before handover; mark as disconnected.
            serving[-1] = "DISCONNECTED"
        elif state == "target" and not target_ok[i]:
            serving[-1] = "DISCONNECTED"
        if interruption_remaining > 0:
            handover_interruption[i] = True
            interruption_remaining -= 1

    serving = np.asarray(serving, dtype=object)
    connected = (serving != "DISCONNECTED") & (~handover_interruption)
    generated_bytes = float(traffic["application_rate_mbps"]) * 1e6 / 8 * dt_s
    # A simple one-second service model; packet-level queues and scheduling are not modeled.
    delivered = np.where(connected, generated_bytes, 0.0)
    out = pd.DataFrame({
        "time_utc": df["time_utc"],
        "source_satellite": source.name,
        "target_satellite": target.name,
        "source_elevation_deg": source_el,
        "target_elevation_deg": target_el,
        "source_range_m": source_range,
        "target_range_m": target_range,
        "source_snr_db": source_snr,
        "target_snr_db": target_snr,
        "serving_satellite": serving,
        "connected": connected,
        "handover_attempt": handover_attempt,
        "handover_success": handover_success,
        "handover_failure": handover_failure,
        "handover_interruption": handover_interruption,
        "traffic_generated_bytes": generated_bytes,
        "traffic_delivered_bytes": delivered,
        "throughput_mbps": delivered * 8 / (dt_s * 1e6),
        "seed": seed,
    })
    out_dir = project_dir / "data"
    plots_dir = project_dir / "plots"
    out_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)
    out.to_csv(out_dir / f"network_baseline_seed_{seed}.csv", index=False)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(np.arange(len(out)), source_el, label=f"Source: {source.name}")
    ax.plot(np.arange(len(out)), target_el, label=f"Target: {target.name}")
    ax.axhline(min_el, linestyle="--", label="Minimum elevation")
    ax.set_xlabel("Time since start (s)")
    ax.set_ylabel("Elevation (degrees)")
    ax.set_title("Source and target satellite visibility")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / f"satellite_visibility_seed_{seed}.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 3.5))
    cell_ids = {source.name: 0, target.name: 1, "DISCONNECTED": -1}
    ax.step(np.arange(len(serving)), [cell_ids.get(x, -1) for x in serving], where="post")
    ax.set_yticks([-1, 0, 1], ["Disconnected", source.name, target.name])
    ax.set_xlabel("Time since start (s)")
    ax.set_title("Serving satellite over time")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / f"serving_satellite_seed_{seed}.png", dpi=160)
    plt.close(fig)
    return out
