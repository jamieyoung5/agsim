use agsim::agent::{Agent, StateType};
use agsim::simulation::Simulation;
use agsim::state::Timeline;
use chrono::{Duration, Utc};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use state_macros::{State, StateDisplay};
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Copy)]
enum DeviceOperationalMode {
    Offline,
    Idle,
    Working,
    HeavyLoad,
}

#[derive(Debug, Clone, Default, State, StateDisplay)]
struct DeviceState {
    connected_status: bool,
    active_sessions: u32,
    memory_in_use_mb: u32,
    cpu_in_use_percent: f32,
}

fn main() {
    let mut transitions = HashMap::new();

    transitions.insert(
        DeviceOperationalMode::Offline,
        StateType::new_deterministic(
            || DeviceState {
                connected_status: false,
                active_sessions: 0,
                memory_in_use_mb: 0,
                cpu_in_use_percent: 0.0,
            },
            vec![
                (DeviceOperationalMode::Idle, 0.9),
                (DeviceOperationalMode::Working, 0.1),
            ],
            4.0 * 3600.0, // 4 hours
        ),
    );

    transitions.insert(
        DeviceOperationalMode::Idle,
        StateType::new(
            |rng| DeviceState {
                connected_status: true,
                active_sessions: 0,
                memory_in_use_mb: rng.gen_range(400..800),
                cpu_in_use_percent: rng.gen_range(0.1..5.0),
            },
            vec![
                (DeviceOperationalMode::Working, 0.4),
                (DeviceOperationalMode::Offline, 0.1),
                (DeviceOperationalMode::Idle, 0.5),
            ],
            3600.0, // 1 hour
        ),
    );

    transitions.insert(
        DeviceOperationalMode::Working,
        StateType::new(
            |rng| DeviceState {
                connected_status: true,
                active_sessions: rng.gen_range(1..4),
                memory_in_use_mb: rng.gen_range(1024..4096),
                cpu_in_use_percent: rng.gen_range(10.0..40.0),
            },
            vec![
                (DeviceOperationalMode::Idle, 0.4),
                (DeviceOperationalMode::HeavyLoad, 0.2),
                (DeviceOperationalMode::Working, 0.4),
            ],
            30.0 * 60.0, // 30 minutes
        ),
    );

    transitions.insert(
        DeviceOperationalMode::HeavyLoad,
        StateType::new(
            |rng| DeviceState {
                connected_status: true,
                active_sessions: rng.gen_range(3..10),
                memory_in_use_mb: rng.gen_range(4096..16384),
                cpu_in_use_percent: rng.gen_range(60.0..99.9),
            },
            vec![
                (DeviceOperationalMode::Working, 0.8),
                (DeviceOperationalMode::Idle, 0.2),
            ],
            10.0 * 60.0, // 10 minutes
        ),
    );

    let mut agents = Vec::new();
    let start_time = Utc::now();

    let mut rng = StdRng::seed_from_u64(42);

    for i in 0..5 {
        agents.push(Agent::new(
            format!("device_{:03}", i),
            DeviceOperationalMode::Idle,
            transitions.clone(),
            &mut rng,
        ));
    }

    let mut sim = Simulation::new(agents, start_time);
    let events = sim.run(Duration::days(7));

    println!("Generated {} events over 7 days.", events.len());

    let timelines = Timeline::generate(&events);

    let mut sorted_agents: Vec<_> = timelines.keys().collect();
    sorted_agents.sort();

    for agent_id in sorted_agents {
        println!("\n--- Timeline for {} (First 5 Entries) ---", agent_id);
        if let Some(timeline) = timelines.get(agent_id) {
            for entry in timeline.entries.iter().take(5) {
                println!("{}", entry);
            }
        }
    }
}
