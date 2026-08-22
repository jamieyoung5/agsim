# agsim

[![crates.io](https://img.shields.io/crates/v/agsim.svg)](https://crates.io/crates/agsim)
[![docs.rs](https://docs.rs/agsim/badge.svg)](https://docs.rs/agsim)
[![license](https://img.shields.io/crates/l/agsim.svg)](LICENSE)

Deterministic discrete-event simulation for synthetic time-series data.

## Install

```toml
[dependencies]
agsim = "2.1"
state_macros = "0.3"
chrono = "0.4"
rand = "0.8"
```

Features: `llm` (Anthropic, OpenAI-compatible), `local` (Candle in-process), `cuda`, `metal`.

## Usage

An `Agent` is a continuous-time Markov chain over your state type.

```rust
use agsim::agent::{Agent, StateType};
use agsim::simulation::Simulation;
use chrono::{Duration, TimeZone, Utc};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use state_macros::{State, StateDisplay};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Mode { Idle, Working }

#[derive(Debug, Clone, Default, State, StateDisplay)]
struct Device { cpu_percent: f32 }

let modes = HashMap::from([
    (Mode::Idle, StateType::new(
        |rng| Device { cpu_percent: rng.gen_range(0.1..5.0) },
        vec![(Mode::Working, 0.6), (Mode::Idle, 0.4)],
        3600.0,
    )),
    (Mode::Working, StateType::new(
        |rng| Device { cpu_percent: rng.gen_range(10.0..90.0) },
        vec![(Mode::Idle, 1.0)],
        600.0,
    )),
]);

let start = Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap();
let mut rng = StdRng::seed_from_u64(42);
let agents = (0..3)
    .map(|i| Agent::new(format!("device_{i}"), Mode::Idle, modes.clone(), &mut rng))
    .collect();

let mut sim = Simulation::new_with_seed(agents, start, 42);
let events = sim.run(Duration::days(1));
```

`run` returns a `Vec<StateChangeEvent>`, one per changed field.

```text
2026-01-01 00:01:16.561 UTC device_1 cpu_percent: 2.680131 -> 3.5044582
2026-01-01 00:28:21.644 UTC device_2 cpu_percent: 1.3188176 -> 33.948612
2026-01-01 00:48:06.625 UTC device_2 cpu_percent: 33.948612 -> 2.2853847
```

The same seed and start time replay the same log.

`examples/` covers shared world state, generative agents, and live runs.

```sh
cargo run --example device_simulator
cargo run --example generative_device
cargo run --example timeseries
cargo run --example live_simulation
cargo run --example shared_world
cargo run --features llm --example small_model
```

## License

GPL-3.0
