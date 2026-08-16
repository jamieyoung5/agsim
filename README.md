# agsim

[![crates.io](https://img.shields.io/crates/v/agsim.svg)](https://crates.io/crates/agsim)
[![docs.rs](https://docs.rs/agsim/badge.svg)](https://docs.rs/agsim)
[![license](https://img.shields.io/crates/l/agsim.svg)](LICENSE)

Discrete-event simulation for agent-based systems in Rust. Events come off a priority queue, so
state changes happen at irregular times rather than on a fixed tick. Intended for generating
synthetic time-series: device telemetry, server fleets, user sessions.

```toml
[dependencies]
agsim = "1.0"
```

## Quickstart

An `Agent` is a continuous-time Markov chain. Give it one `StateType` per mode: the metrics to
emit, the weighted transitions out, and the mean seconds spent there.

```rust
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

`run` returns a `Vec<StateChangeEvent>`, one per field that changed:

```
2026-01-01 00:01:16.561 UTC device_1 cpu_percent: 2.680131 -> 3.5044582
2026-01-01 00:28:21.644 UTC device_2 cpu_percent: 1.3188176 -> 33.948612
```

Same seed, agents, and start time replay the same log. Each agent draws from its own substream, so
inserting or reordering agents leaves the others alone. `Simulation::new` seeds from OS entropy and
reports it as `sim.seed()`.

## Generative agents

A `GenerativeAgent` follows a daily plan, records what happens in a memory stream, and reflects on
it, following [Park et al. (2023)](https://arxiv.org/abs/2304.03442). Use it when the output needs
time-of-day structure. It takes a `Mind` (planning and reflection) plus two functions: one mapping a
plan step to a mode, one mapping a mode to metrics.

```rust
let agent = GenerativeAgent::new(
    id.clone(),
    id,             // identity, passed to the planner as context
    device_state,   // fn(&Mode, &mut dyn RngCore) -> Device
    interpret_mode, // fn(&PlanStep) -> Mode
    ScriptedMind,   // impl Mind
    start,
    &mut rng,
);
```

With the `llm` feature a model does the planning, over `Anthropic`, `OpenAiCompat` (Ollama,
llama.cpp, vLLM, LM Studio), or `Candle` in-process:

```rust
let mind = agsim::llm::LlmMind::from_env("a home thermostat")?; // reads ANTHROPIC_API_KEY
```

## Live runs

`run_live` paces against the wall clock and streams events instead of returning them at the end:

```rust
let live = Live::at_speed(60.0); // one simulated minute per real second
let stop = live.stop_signal();   // clone into a Ctrl-C handler
sim.run_live(live, |event| {
    println!("{} -> {}", event.field, event.new_value);
    ControlFlow::Continue(())
});
```

Speed affects when events surface, not which ones occur.

## Feature flags

| flag | adds |
| --- | --- |
| `llm` | `LlmMind` and the HTTP backends (Anthropic, OpenAI-compatible) |
| `local` | the `Candle` backend, running a quantized model in-process. Slow to build |
| `cuda` / `metal` | GPU acceleration for `local` |

## Examples

```sh
cargo run --example device_simulator   # Markov fleet, with timeline output
cargo run --example generative_device  # planning, memory, reflection. offline, no API key
cargo run --example timeseries         # a week of CPU telemetry, sampled and sparklined
cargo run --example live_simulation    # wall-clock paced streaming
```

## License

GPL-3.0
