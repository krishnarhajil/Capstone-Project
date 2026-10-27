# NTN Simulation and Selective-Delay Relay Attack

## Overview

A Python-based system-level simulation prototype for studying satellite-ground communication and the effects of relay-based packet delay in a Non-Terrestrial Network (NTN).

The project uses satellite TLE data and Skyfield to derive satellite-ground link geometry and propagation-related metrics. It includes a baseline model and a packet-level experiment comparing direct communication, benign store-and-forward relaying, and malicious selective-delay relaying.

## Scenarios

1. **Direct baseline:** Direct satellite-ground communication.
2. **Benign relay:** Two-hop forwarding with relay processing and serialization delay.
3. **Malicious selective-delay relay:** An additional 100 ms delay is applied to every twentieth packet.

## Metrics

- End-to-end packet delay
- 95th-percentile packet delay
- Packet delivery ratio
- Throughput
- Satellite-ground range, elevation, propagation delay, and Doppler-related measurements

## Current Results

| Scenario | Mean delay | 95th-percentile delay | Delivery ratio |
|---|---:|---:|---:|
| Direct baseline | 5.45 ms | 5.45 ms | 100.000% |
| Benign relay | 8.41 ms | 8.41 ms | 99.997% |
| Malicious selective delay | 39.88 ms | 99.77 ms | 99.993% |

These values are from the current experimental configuration and may change when the configuration or code changes.

## Project Scope

This is a Python-based system-level prototype with an abstract packet-level relay model. It is not a full 3GPP-compliant NR/NTN protocol-stack simulation, and the relay is not modeled as a separate physical satellite or radio repeater.

## Setup

Requires Python 3.11 or a compatible Python environment.

```bash
python -m venv .venv
```

Activate the environment, then install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Running the Experiments

Run the satellite pass and link-trace generation script:

```bash
python simulate_pass.py
```

Run the baseline experiment:

```bash
python src/run_baseline.py
```

Run the packet-level relay experiment:

```bash
python src/packet_experiment.py
```

Run commands from the repository root. Check the configuration and required input files before execution.

## Collaboration

The baseline should remain the shared reference. Attack-specific experiments should be developed separately so changes to one experiment do not unintentionally alter the baseline or another teammate's implementation.

## Limitations

- Relay behavior is abstracted at packet level.
- The current model does not simulate a complete NR/NTN protocol stack.
- Physical-layer interference, cryptographic interception, and a physically modeled radio repeater are not implemented.
- Results depend on the selected trace, configuration, and model assumptions.