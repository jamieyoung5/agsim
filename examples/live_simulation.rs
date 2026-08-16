use agsim::agent::{Agent, StateType};
use agsim::clock::Live;
use agsim::simulation::Simulation;
use chrono::{Duration, Utc};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use state_macros::{State, StateDisplay};
use std::collections::HashMap;
use std::ops::ControlFlow;
use std::time::Instant;

const SPEED: f64 = 1800.0; // simulated seconds per real second.
const MAX_EVENTS: usize = 25;
const WALL_LIMIT: std::time::Duration = std::time::Duration::from_secs(20);

#[derive(Debug, Clone, PartialEq, Eq, Hash, Copy)]
enum Mode {
    Idle,
    Working,
    HeavyLoad,
}

#[derive(Debug, Clone, Default, State, StateDisplay)]
struct DeviceState {
    active_sessions: u32,
    cpu_in_use_percent: f32,
}

fn transitions() -> HashMap<Mode, StateType<Mode, DeviceState>> {
    let mut transitions = HashMap::new();

    transitions.insert(
        Mode::Idle,
        StateType::new(
            |rng| DeviceState {
                active_sessions: 0,
                cpu_in_use_percent: rng.gen_range(0.1..5.0),
            },
            vec![(Mode::Working, 0.7), (Mode::Idle, 0.3)],
            45.0 * 60.0, // a transition every 45 simulated minutes on average.
        ),
    );
    transitions.insert(
        Mode::Working,
        StateType::new(
            |rng| DeviceState {
                active_sessions: rng.gen_range(1..4),
                cpu_in_use_percent: rng.gen_range(10.0..40.0),
            },
            vec![
                (Mode::Idle, 0.4),
                (Mode::HeavyLoad, 0.3),
                (Mode::Working, 0.3),
            ],
            30.0 * 60.0,
        ),
    );
    transitions.insert(
        Mode::HeavyLoad,
        StateType::new(
            |rng| DeviceState {
                active_sessions: rng.gen_range(3..10),
                cpu_in_use_percent: rng.gen_range(60.0..99.9),
            },
            vec![(Mode::Working, 0.8), (Mode::Idle, 0.2)],
            15.0 * 60.0,
        ),
    );

    transitions
}

fn main() {
    let start = Utc::now();
    let mut setup_rng = StdRng::seed_from_u64(42);

    let agents = (0..3)
        .map(|i| {
            Agent::new(
                format!("device_{i:03}"),
                Mode::Idle,
                transitions(),
                &mut setup_rng,
            )
        })
        .collect();

    let mut sim = Simulation::new_with_seed(agents, start, 2024);

    let live = Live::at_speed(SPEED);
    let stop = live.stop_signal();

    let watchdog = stop.clone();
    std::thread::spawn(move || {
        std::thread::sleep(WALL_LIMIT);
        watchdog.stop();
    });

    println!(
        "Running live at {SPEED}x ({:.0} simulated minutes per real second). \
         Stops after {MAX_EVENTS} events or {}s.\n",
        SPEED / 60.0,
        WALL_LIMIT.as_secs()
    );

    let wall_start = Instant::now();
    let mut seen = 0;

    let outcome = sim.run_live(live, |event| {
        seen += 1;
        println!(
            "[+{:>5.1}s wall] {} | sim {} | {} {} -> {}",
            wall_start.elapsed().as_secs_f64(),
            event.agent_id,
            event.time.format("%H:%M:%S"),
            event.field,
            event.old_value,
            event.new_value,
        );

        if seen >= MAX_EVENTS {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    });

    let simulated = sim.current_time() - start;
    println!(
        "\nFinished: {outcome:?} after {seen} events. \
         {} simulated minutes played in {:.1}s of real time.",
        simulated.num_minutes(),
        wall_start.elapsed().as_secs_f64()
    );

    let tail = sim.run(Duration::hours(1));
    println!(
        "Resumed offline from {}: {} more events in the next simulated hour.",
        sim.current_time().format("%H:%M:%S"),
        tail.len()
    );
}
