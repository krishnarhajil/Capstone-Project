"""Radio-link budget model for the Python NTN baseline.

This is a transparent system-level link budget, not a full TR 38.811 channel
implementation. The fixed loss and shadowing parameters are explicit model
assumptions and should be refined against the exact selected TR 38.811 scenario.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

C = 299_792_458.0
K_BOLTZMANN_DBM_HZ = -174.0


def fspl_db(range_m: np.ndarray, frequency_hz: float) -> np.ndarray:
    range_m = np.maximum(np.asarray(range_m, dtype=float), 1.0)
    return 20.0 * np.log10(4.0 * np.pi * range_m * frequency_hz / C)

def calculate_link_budget(range_m, elevation_deg, radio_cfg, shadowing_db):
    fspl = fspl_db(
        np.asarray(range_m, dtype=float),
        float(radio_cfg["carrier_frequency_hz"])
    )

    shadowing = np.asarray(shadowing_db, dtype=float)

    total_loss = (
        fspl
        + float(radio_cfg["additional_fixed_losses_db"])
        + float(radio_cfg.get("shadowing_margin_db", 0.0))
        - shadowing
    )

    rx_power = (
        float(radio_cfg["satellite_eirp_dbm"])
        + float(radio_cfg["ue_antenna_gain_dbi"])
        - total_loss
    )

    noise_power = (
        K_BOLTZMANN_DBM_HZ
        + 10 * np.log10(float(radio_cfg["bandwidth_hz"]))
        + float(radio_cfg["noise_figure_db"])
    )

    snr = rx_power - noise_power

    link_available = (
        (np.asarray(elevation_deg) >= float(radio_cfg["minimum_elevation_deg"]))
        & (snr >= float(radio_cfg["minimum_snr_db"]))
    )

    return pd.DataFrame({
        "fspl_db": fspl,
        "shadowing_db": shadowing,
        "total_path_loss_db": total_loss,
        "received_power_dbm": rx_power,
        "noise_power_dbm": noise_power,
        "snr_db": snr,
        "link_available": link_available,
    })


def run_radio_link(project_dir: Path, config_path: Path, seed: int = 0) -> pd.DataFrame:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    input_csv = project_dir / "data" / "legitimate_signature.csv"
    if not input_csv.exists():
        raise FileNotFoundError(
            f"Cannot find {input_csv}. Run your existing simulate_pass.py first and preserve its output."
        )
    df = pd.read_csv(input_csv)
    needed = {"time_utc", "range_m", "elevation_deg"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"legitimate_signature.csv is missing columns: {sorted(missing)}")

        r = cfg["radio"]
    r = cfg["radio"]
    rng = np.random.default_rng(seed)

    # Generate reproducible shadowing for the source satellite
    shadowing = rng.normal(
        0.0,
        float(r["shadowing_sigma_db"]),
        len(df)
    )

    # Use the shared link-budget function
    link = calculate_link_budget(
        range_m=df["range_m"].to_numpy(),
        elevation_deg=df["elevation_deg"].to_numpy(),
        radio_cfg= r,
        shadowing_db=shadowing
    )
    out = pd.concat(
    [
        df[["time_utc", "range_m", "elevation_deg"]].reset_index(drop=True),
        link.reset_index(drop=True)
    ],
    axis=1
)

    out["seed"] = seed

    out_dir = project_dir / "data"
    plots_dir = project_dir / "plots"
    out_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)
    out.to_csv(out_dir / f"radio_link_seed_{seed}.csv", index=False)

    x = np.arange(len(out))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, out["received_power_dbm"], label="Received power")
    ax.set_xlabel("Time since start (s)")
    ax.set_ylabel("Received power (dBm)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / f"received_power_seed_{seed}.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(out["elevation_deg"], out["snr_db"], ".", markersize=3)
    ax.set_xlabel("Elevation (degrees)")
    ax.set_ylabel("SNR (dB)")
    ax.set_title("SNR versus elevation")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / f"snr_vs_elevation_seed_{seed}.png", dpi=160)
    plt.close(fig)
    return out
