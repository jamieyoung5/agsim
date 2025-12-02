use agsim::agent::{Agent, StateType};
use agsim::simulation::Simulation;
use agsim::state::{State, StateChangeEvent};
use chrono::{Duration, Utc};
use rand::Rng;
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
    // these factories below define the actual metrics that get set when a state is transitioned to.
    // the reason for using factories is to allow a user of the library to generate numbers that
    // are used per iteration more easily.

    let offline_factory = || {
        Box::new(DeviceState {
            connected_status: false,
            active_sessions: 0,
            memory_in_use_mb: 0,
            cpu_in_use_percent: 0.0,
        })
    };

    let idle_factory = || {
        let mut rng = rand::thread_rng();
        Box::new(DeviceState {
            connected_status: true,
            active_sessions: 0,
            // idle devices still use some memory in the background (e.g., 400-800MB)
            memory_in_use_mb: rng.gen_range(400..800),
            // low cpu usage (0.1% - 5.0%)
            cpu_in_use_percent: rng.gen_range(0.1..5.0),
        })
    };

    let working_factory = || {
        let mut rng = rand::thread_rng();
        Box::new(DeviceState {
            connected_status: true,
            // 1 to 3 users logged in
            active_sessions: rng.gen_range(1..4),
            // moderate memory usage
            memory_in_use_mb: rng.gen_range(1024..4096),
            // moderate cpu usage
            cpu_in_use_percent: rng.gen_range(10.0..40.0),
        })
    };

    let heavy_load_factory = || {
        let mut rng = rand::thread_rng();
        Box::new(DeviceState {
            connected_status: true,
            // high user count or batch processing
            active_sessions: rng.gen_range(3..10),
            // high memory usage
            memory_in_use_mb: rng.gen_range(4096..16384),
            // high cpu usage
            cpu_in_use_percent: rng.gen_range(60.0..99.9),
        })
    };

    // below is definitions for the state transition matrix

    let mut transitions = HashMap::new();

    // offline: (stays offline for ~4 hours avg, mostly goes to idle, rarely straight to working).
    transitions.insert(
        DeviceOperationalMode::Offline,
        StateType {
            factory: offline_factory as fn() -> Box<DeviceState>,
            transitions: vec![
                (DeviceOperationalMode::Idle, 0.9),
                (DeviceOperationalMode::Working, 0.1),
            ],
            event_rate: 4.0 * 3600.0, // 4 hours
        },
    );

    // idle: (stays idle for ~1 hour avg, can go offline, working, or stay idle)
    transitions.insert(
        DeviceOperationalMode::Idle,
        StateType {
            factory: idle_factory as fn() -> Box<DeviceState>,
            transitions: vec![
                (DeviceOperationalMode::Working, 0.4),
                (DeviceOperationalMode::Offline, 0.1),
                (DeviceOperationalMode::Idle, 0.5),
            ],
            event_rate: 3600.0, // 1 hour
        },
    );

    // working: (stays working for ~30 mins avg, can go to heavy load, idle, or offline)
    transitions.insert(
        DeviceOperationalMode::Working,
        StateType {
            factory: working_factory as fn() -> Box<DeviceState>,
            transitions: vec![
                (DeviceOperationalMode::Idle, 0.4),
                (DeviceOperationalMode::HeavyLoad, 0.2),
                (DeviceOperationalMode::Working, 0.4),
            ],
            event_rate: 30.0 * 60.0, // 30 minutes
        },
    );

    // heavy load (short bursts, ~10 mins avg, usually drops back to Working)
    transitions.insert(
        DeviceOperationalMode::HeavyLoad,
        StateType {
            factory: heavy_load_factory as fn() -> Box<DeviceState>,
            transitions: vec![
                (DeviceOperationalMode::Working, 0.8),
                (DeviceOperationalMode::Idle, 0.2),
            ],
            event_rate: 10.0 * 60.0, // 10 minutes
        },
    );

    let mut agents = Vec::new();
    let start_time = Utc::now();

    // we'll create 5 devices, starting in idle status
    for i in 0..5 {
        agents.push(Agent::new(
            format!("device_{:03}", i),
            DeviceOperationalMode::Idle,
            transitions.clone(),
        ));
    }

    // run the simulation over a 'week'
    let mut sim = Simulation::new(agents, start_time);
    let events = sim.run(Duration::days(7));

    println!("Generated {} events over 7 days.", events.len());

    if let Some(timeline) = sim.generate_master_timeline() {
        println!("\n--- Master Timeline Sample (First 20 Entries) ---");
        for entry in timeline.entries.iter().take(20) {
            println!("{}", entry);
        }
    }
}
