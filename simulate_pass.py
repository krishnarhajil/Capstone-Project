
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from skyfield.api import load, wgs84


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_DIR = Path("data")
PLOT_DIR = Path("plots")

DATA_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)

# Chennai ground station
LATITUDE = 13.0827
LONGITUDE = 80.2707
GROUND_ALTITUDE_M = 0

# Satellite and simulation parameters
MIN_ELEVATION_DEG = 10.0
CARRIER_HZ = 2.0e9
SPEED_OF_LIGHT = 299_792_458.0
TIME_STEP_SECONDS = 1

# Illustrative Gaussian measurement noise
DELAY_NOISE_STD_S = 100e-9
DOPPLER_NOISE_STD_HZ = 5.0

RNG_SEED = 42

TLE_FILE = DATA_DIR / "starlink.tle"
CSV_FILE = DATA_DIR / "legitimate_signature.csv"


# ============================================================
# 2. DOWNLOAD AND LOAD REAL SATELLITE ORBITAL DATA
# ============================================================

print("Loading Skyfield timescale...")
ts = load.timescale()

if not TLE_FILE.exists():
    print("Downloading current Starlink TLE data...")

    tle_url = (
        "https://celestrak.org/NORAD/elements/"
        "gp.php?GROUP=starlink&FORMAT=TLE"
    )

    load.download(tle_url, filename=str(TLE_FILE))

satellites = load.tle_file(str(TLE_FILE))

if len(satellites) == 0:
    raise RuntimeError("No satellites were found in the TLE file.")

print(f"Loaded {len(satellites)} satellites.")


# ============================================================
# 3. DEFINE CHENNAI GROUND STATION
# ============================================================

ground_station = wgs84.latlon(
    latitude_degrees=LATITUDE,
    longitude_degrees=LONGITUDE,
    elevation_m=GROUND_ALTITUDE_M,
)


# ============================================================
# 4. FIND A COMPLETE PASS ABOVE 10 DEGREES
# ============================================================

start_time = ts.now()

end_time = ts.from_datetime(
    start_time.utc_datetime() + timedelta(days=3)
)

selected_satellite = None
rise_time = None
set_time = None
peak_elevation = None

print("Searching for a satellite pass over Chennai...")

for index, satellite in enumerate(satellites, start=1):

    event_times, events = satellite.find_events(
        ground_station,
        start_time,
        end_time,
        altitude_degrees=MIN_ELEVATION_DEG,
    )

    # Event codes:
    # 0 = satellite rises above elevation threshold
    # 1 = culmination / maximum elevation
    # 2 = satellite sets below elevation threshold

    for j, event in enumerate(events):

        if event != 0:
            continue

        culmination_index = next(
            (
                k for k in range(j + 1, len(events))
                if events[k] == 1
            ),
            None,
        )

        set_index = next(
            (
                k for k in range(j + 1, len(events))
                if events[k] == 2
            ),
            None,
        )

        if culmination_index is None or set_index is None:
            continue

        if culmination_index > set_index:
            continue

        selected_satellite = satellite
        rise_time = event_times[j]
        set_time = event_times[set_index]

        peak_elevation = (
            satellite - ground_station
        ).at(event_times[culmination_index]).altaz()[0].degrees

        break

    if selected_satellite is not None:
        break

    if index % 500 == 0:
        print(f"Checked {index} satellites...")

if selected_satellite is None:
    raise RuntimeError(
        "No complete pass found in the next 3 days. "
        "Check the TLE data or extend the search period."
    )


duration_seconds = (set_time - rise_time) * 86400.0

print("\nPASS FOUND")
print("Satellite:", selected_satellite.name)
print(f"Maximum elevation: {peak_elevation:.2f} degrees")
print(f"Pass duration: {duration_seconds:.1f} seconds")
print("Rise time (UTC):", rise_time.utc_iso())
print("Set time (UTC):", set_time.utc_iso())


# ============================================================
# 5. SAMPLE THE PASS AT ONE-SECOND INTERVALS
# ============================================================

offsets_seconds = np.arange(
    0,
    int(np.floor(duration_seconds)) + 1,
    TIME_STEP_SECONDS,
)

sample_times = ts.tt_jd(
    rise_time.tt + offsets_seconds / 86400.0
)

# Satellite position relative to Chennai ground station
topocentric = (
    selected_satellite - ground_station
).at(sample_times)

# Slant range, in metres
range_m = topocentric.distance().m

# Range rate, in metres per second
# Positive means increasing range (satellite receding)
relative_position = topocentric.position.m
relative_velocity = topocentric.velocity.m_per_s

range_rate_m_s = np.sum(relative_position * relative_velocity, axis=0) / np.linalg.norm(relative_position, axis=0)

# Elevation angle, in degrees
elevation_deg = topocentric.altaz()[0].degrees


# ============================================================
# 6. CALCULATE PROPAGATION DELAY AND DOPPLER
# ============================================================

# One-way propagation delay
delay_s = range_m / SPEED_OF_LIGHT

# Received-frequency Doppler convention:
# Approaching satellite -> positive Doppler
# Receding satellite -> negative Doppler
doppler_hz = -range_rate_m_s * CARRIER_HZ / SPEED_OF_LIGHT


# ============================================================
# 7. ADD GAUSSIAN MEASUREMENT NOISE
# ============================================================

rng = np.random.default_rng(RNG_SEED)

measured_delay_s = delay_s + rng.normal(
    loc=0.0,
    scale=DELAY_NOISE_STD_S,
    size=len(delay_s),
)

measured_doppler_hz = doppler_hz + rng.normal(
    loc=0.0,
    scale=DOPPLER_NOISE_STD_HZ,
    size=len(doppler_hz),
)


# ============================================================
# 8. SAVE LEGITIMATE SIGNATURE DATA TO CSV
# ============================================================

df = pd.DataFrame({
    "time_utc": sample_times.utc_iso(),
    "range_m": range_m,
    "delay_s": delay_s,
    "doppler_hz": doppler_hz,
    "measured_delay_s": measured_delay_s,
    "measured_doppler_hz": measured_doppler_hz,
    "elevation_deg": elevation_deg,
})

df.to_csv(CSV_FILE, index=False)

print("\nFIRST FIVE ROWS")
print(df.head().to_string(index=False))

print(f"\nCSV saved to: {CSV_FILE}")
print(f"Total samples: {len(df)}")


# ============================================================
# 9. PLOT PROPAGATION DELAY
# ============================================================

elapsed_minutes = offsets_seconds / 60.0

plt.figure(figsize=(10, 5))

plt.plot(
    elapsed_minutes,
    delay_s * 1000,
    label="Expected delay",
)

plt.plot(
    elapsed_minutes,
    measured_delay_s * 1000,
    alpha=0.7,
    label="Noisy measured delay",
)

plt.xlabel("Time since pass rise (minutes)")
plt.ylabel("One-way propagation delay (ms)")
plt.title("LEO Satellite Propagation Delay over Chennai")
plt.grid(True)
plt.legend()
plt.tight_layout()

delay_plot = PLOT_DIR / "delay_vs_time.png"
plt.savefig(delay_plot, dpi=200)
plt.show()

print("Saved:", delay_plot)


# ============================================================
# 10. PLOT DOPPLER SHIFT
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    elapsed_minutes,
    doppler_hz,
    label="Expected Doppler",
)

plt.plot(
    elapsed_minutes,
    measured_doppler_hz,
    alpha=0.7,
    label="Noisy measured Doppler",
)

plt.axhline(0, color="black", linewidth=0.8)

plt.xlabel("Time since pass rise (minutes)")
plt.ylabel("Doppler shift (Hz)")
plt.title("LEO Satellite Doppler Shift over Chennai")
plt.grid(True)
plt.legend()
plt.tight_layout()

doppler_plot = PLOT_DIR / "doppler_vs_time.png"
plt.savefig(doppler_plot, dpi=200)
plt.show()

print("Saved:", doppler_plot)


# ============================================================
# 11. FINAL SUMMARY
# ============================================================

print("\nSIMULATION COMPLETED")
print("Satellite:", selected_satellite.name)
print(f"Minimum slant range: {range_m.min() / 1000:.2f} km")
print(f"Minimum delay: {delay_s.min() * 1000:.3f} ms")
print(f"Maximum absolute Doppler: {np.max(np.abs(doppler_hz)):.2f} Hz")
print("CSV:", CSV_FILE)
print("Plots:", PLOT_DIR)
