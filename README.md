# agsim

agsim is a discrete-event simulation framework for agent-based systems in Rust. It started out as a
way to generate realistic synthetic time-series data (device telemetry, to train a forecasting
model on) and grew into a more general agent simulation library.

Time moves forward by pulling events off a priority queue instead of stepping through fixed ticks,
so state changes happen irregularly, the way they do in real systems like server fleets, IoT
devices, or user sessions.

There are two kinds of agent, both implementing the same `SimAgent` trait:

- **CTMC agents.** The original stochastic model: a weighted transition matrix picks the next state
  and an exponential distribution decides when the transition happens. Memoryless.
- **Generative agents.** These follow a daily plan, remember what happens to them, and occasionally
  reflect on it. The design follows the Generative Agents paper (Park et al., 2023). The point is
  temporal structure: a Markov chain has no sense of time of day, so it can't produce something like
  a daily usage rhythm, whereas a planned agent can. That structure is what makes the generated data
  actually look real.

A companion proc-macro crate, `state_macros`, derives the boilerplate traits for your state structs.

## Installing

```sh
cargo build
```

The LLM integration is behind a feature flag:

```sh
cargo build --features llm
```

## Defining state

You describe the fields you want to track as a struct and let `state_macros` generate the trait
impls.

```rust
use state_macros::{State, StateDisplay};

#[derive(Debug, Clone, Default, State, StateDisplay)]
struct DeviceState {
    connected: bool,
    cpu_in_use_percent: f32,
}
```

## Stochastic agents

Each mode has a factory for its state values, a list of weighted transitions, and a mean event rate.
The engine runs the heap of scheduled events until the time limit and hands back the log of changes.
`examples/device_simulator.rs` has a full setup with timeline output.

## Generative agents

A `GenerativeAgent` is driven by a `Mind` (the planning and reflection logic) plus two closures at
the boundary: one that maps a plan activity to a state, and one that turns a state into concrete
metric values. The plan decides what the agent is doing and when; the factory fills in the numbers.
Agents can also perceive each other within a radius and move around as their plan changes location,
if you give them positions.

Two examples:

- `examples/generative_device.rs` is a device on a daily usage plan, with memory and reflection. It
  runs offline, no API key needed.
- `examples/timeseries.rs` generates a week of CPU telemetry, samples it on a fixed grid, adds a bit
  of noise, and draws it as a sparkline per day so you can see the daily shape.

The `Mind` is a trait, so you can plug in your own. With the `llm` feature and `ANTHROPIC_API_KEY`
set, `LlmMind` backs the planning and reflection with Claude. It talks to the API over blocking HTTP
and uses structured outputs, so it drops into the synchronous engine without going async.

```rust
let mind = agsim::llm::LlmMind::from_env("a home thermostat")?;
```

## Examples

```sh
cargo run --example device_simulator
cargo run --example generative_device
cargo run --example timeseries
```
