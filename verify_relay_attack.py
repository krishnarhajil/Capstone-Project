
import csv
import glob
import os
import math
from collections import defaultdict

DATA_DIR = "data"
EXPECTED_DELAYS_MS = [0, 50, 100, 250]
EXPECTED_SEEDS = range(10)

required_columns = [
    "baseline_measured_delay_s",
    "relay_forwarding_delay_s",
    "attacked_delay_s",
    "delay_increase_s",
    "baseline_doppler_hz",
    "attacked_doppler_hz",
    "doppler_residual_hz",
    "seed",
    "relay_delay_ms",
]

errors = []
checked_files = 0
checked_rows = 0
delay_stats = defaultdict(list)

for seed in EXPECTED_SEEDS:
    for delay_ms in EXPECTED_DELAYS_MS:
        path = os.path.join(
            DATA_DIR,
            f"relay_mitm_seed_{seed}_delay_{delay_ms}ms.csv"
        )

        if not os.path.exists(path):
            errors.append(f"Missing file: {path}")
            continue

        with open(path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)

            missing = set(required_columns) - set(reader.fieldnames or [])
            if missing:
                errors.append(
                    f"{path}: missing columns {sorted(missing)}"
                )
                continue

            rows = list(reader)

        if not rows:
            errors.append(f"{path}: file contains no data rows")
            continue

        checked_files += 1

        for row_number, row in enumerate(rows, start=2):
            checked_rows += 1

            try:
                baseline = float(row["baseline_measured_delay_s"])
                forwarding = float(row["relay_forwarding_delay_s"])
                attacked = float(row["attacked_delay_s"])
                increase = float(row["delay_increase_s"])
                baseline_doppler = float(row["baseline_doppler_hz"])
                attacked_doppler = float(row["attacked_doppler_hz"])
                doppler_residual = float(row["doppler_residual_hz"])
                actual_seed = int(row["seed"])
                actual_delay = int(row["relay_delay_ms"])
            except (ValueError, TypeError) as e:
                errors.append(f"{path}, row {row_number}: invalid value ({e})")
                continue

            expected_delay_s = delay_ms / 1000.0
            tolerance = 1e-8

            checks = [
                (actual_seed == seed, "seed does not match filename"),
                (actual_delay == delay_ms, "delay does not match filename"),
                (math.isclose(forwarding, expected_delay_s,
                              abs_tol=tolerance),
                 "forwarding delay incorrect"),
                (math.isclose(attacked, baseline + forwarding,
                              abs_tol=tolerance),
                 "attacked delay != baseline + forwarding delay"),
                (math.isclose(increase, expected_delay_s,
                              abs_tol=tolerance),
                 "delay increase incorrect"),
                (math.isclose(attacked_doppler, baseline_doppler,
                              abs_tol=tolerance),
                 "Doppler changed unexpectedly in delay-only model"),
                (math.isclose(doppler_residual, 0.0,
                              abs_tol=tolerance),
                 "Doppler residual should be zero in delay-only model"),
            ]

            for passed, message in checks:
                if not passed:
                    errors.append(
                        f"{path}, row {row_number}: {message}"
                    )

            delay_stats[delay_ms].append(increase * 1000.0)

print("=" * 58)
print("RELAY / MITM DELAY-INJECTION VERIFICATION")
print("=" * 58)
print(f"Files checked: {checked_files}/40")
print(f"Rows checked:  {checked_rows}")

print("\nDelay-increase results:")
for delay_ms in EXPECTED_DELAYS_MS:
    values = delay_stats[delay_ms]
    if values:
        mean = sum(values) / len(values)
        max_error = max(abs(v - delay_ms) for v in values)
        print(
            f"{delay_ms:>3} ms scenario | "
            f"rows={len(values):>4} | "
            f"mean increase={mean:.6f} ms | "
            f"max deviation={max_error:.9f} ms"
        )

print("\nVERDICT:")
if not errors and checked_files == 40:
    print("PASS: All relay delay-injection checks passed.")
    print("The configured extra delay is present in the output CSVs.")
else:
    print(f"CHECK FAILED: {len(errors)} issue(s) found.")
    for error in errors[:30]:
        print("-", error)
    if len(errors) > 30:
        print(f"... and {len(errors) - 30} more issues.")
