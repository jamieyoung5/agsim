# agsim

**agsim** is a discrete event simulation (DES) library for modeling agent-based systems in Rust. It focuses on tracking complex state changes over time using stochastic transitions.

## Overview and Design

This library models agents as **Continuous-Time Markov Chains (CTMC)**. Instead of fixed time steps, the simulation moves forward by processing events from a priority queue.

The aim of this framework is to faciliate realistic modeling of systems like server fleets, IoT device lifecycles, or user sessions, where state changes occur irregularly rather than in fixed ticks. To reduce boilerplate, the library includes a companion crate, `state_macros`, which automatically derives the necessary traits for your custom state structs.

## Installation

Clone the repository and build the project using Cargo:

```sh
cargo build
```

## Usage

### Defining State

You define the data you want to track in a struct. The state_macros crate provides derive macros to handle the underlying trait implementations for storage and display.

```rust
use state_macros::{State, StateDisplay};

#[derive(Debug, Clone, Default, State, StateDisplay)]
struct DeviceState {
    connected: bool,
    load: u32,
}
```

### simulation

To run a simulation, you initialize agents with a specific operational mode and a transition matrix (defining the probabilities of moving between modes). When run the engine then advances time, processing the binary heap of events until the specified duration is reached.

Refer to examples/device_simulator/device_simulator.rs for a complete implementation demonstrating agent configuration and timeline generation.
