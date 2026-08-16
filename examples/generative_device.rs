use agsim::generative::{GenerativeAgent, Mind};
use agsim::memory::{Insight, Memory, MemoryKind, Reflector};
use agsim::planning::{PlanContext, PlanStep, Planner, Reaction};
use agsim::simulation::Simulation;
use agsim::state::{StateChangeEvent, Timeline};
use chrono::{Duration, TimeZone, Utc};
use rand::rngs::StdRng;
use rand::{Rng, RngCore, SeedableRng};
use state_macros::{State, StateDisplay};

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

// device_state generates plausible metrics for a given operational mode (the C -> S factory).
fn device_state(mode: &DeviceOperationalMode, rng: &mut dyn RngCore) -> DeviceState {
    match mode {
        DeviceOperationalMode::Offline => DeviceState {
            connected_status: false,
            active_sessions: 0,
            memory_in_use_mb: 0,
            cpu_in_use_percent: 0.0,
        },
        DeviceOperationalMode::Idle => DeviceState {
            connected_status: true,
            active_sessions: 0,
            memory_in_use_mb: rng.gen_range(400..800),
            cpu_in_use_percent: rng.gen_range(0.1..5.0),
        },
        DeviceOperationalMode::Working => DeviceState {
            connected_status: true,
            active_sessions: rng.gen_range(1..4),
            memory_in_use_mb: rng.gen_range(1024..4096),
            cpu_in_use_percent: rng.gen_range(10.0..40.0),
        },
        DeviceOperationalMode::HeavyLoad => DeviceState {
            connected_status: true,
            active_sessions: rng.gen_range(3..10),
            memory_in_use_mb: rng.gen_range(4096..16384),
            cpu_in_use_percent: rng.gen_range(60.0..99.9),
        },
    }
}

// interpret_mode maps a plan activity to the operational mode the device is in during it.
fn interpret_mode(step: &PlanStep) -> DeviceOperationalMode {
    let activity = step.description.to_lowercase();
    if activity.contains("offline") {
        DeviceOperationalMode::Offline
    } else if activity.contains("heavy") {
        DeviceOperationalMode::HeavyLoad
    } else if activity.contains("work") {
        DeviceOperationalMode::Working
    } else {
        DeviceOperationalMode::Idle
    }
}

// ScriptedMind is a deterministic stand-in for an LLM... a fixed daily routine, no reactions, and a
// simple reflection. (it lets the example run with no api key)
struct ScriptedMind;

impl Planner for ScriptedMind {
    fn daily_plan(&self, ctx: &PlanContext) -> Vec<PlanStep> {
        let block = |desc: &str, start_h: i64, dur_h: i64| {
            PlanStep::new(
                desc,
                ctx.now + Duration::hours(start_h),
                Duration::hours(dur_h),
            )
        };
        vec![
            block("overnight offline", 0, 7),
            block("morning idle", 7, 2),
            block("morning work", 9, 3),
            block("lunch idle", 12, 1),
            block("afternoon work", 13, 4),
            block("evening heavy load", 17, 2),
            block("evening work", 19, 3),
            block("night idle", 22, 2),
        ]
    }

    // the routine blocks are the granularity we want, so no further decomposition.
    fn decompose(&self, _step: &PlanStep, _ctx: &PlanContext) -> Vec<PlanStep> {
        Vec::new()
    }

    fn react(
        &self,
        _observation: &Memory,
        _current: Option<&PlanStep>,
        _ctx: &PlanContext,
    ) -> Reaction {
        Reaction::Continue
    }
}

impl Reflector for ScriptedMind {
    fn salient_questions(&self, _recent: &[&Memory]) -> Vec<String> {
        vec!["What is this device's typical usage pattern?".to_string()]
    }

    fn synthesize(&self, _question: &str, evidence: &[&Memory]) -> Vec<Insight> {
        vec![Insight {
            description: format!(
                "Recent activity produced {} notable state changes.",
                evidence.len()
            ),
            importance: 4.0,
            evidence: evidence.iter().map(|memory| memory.id).collect(),
            embedding: None,
        }]
    }
}

impl Mind for ScriptedMind {
    fn importance(&self, event: &StateChangeEvent) -> f64 {
        // CPU spikes and connectivity changes are more poignant than routine memory churn.
        match event.field.as_str() {
            "cpu_in_use_percent" => 6.0,
            "connected_status" => 5.0,
            _ => 2.0,
        }
    }
}

fn main() {
    // fixed start time + seeded agent construction + a seeded simulation: the whole run replays.
    let start = Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap();
    let mut rng = StdRng::seed_from_u64(7);

    let mut agents = Vec::new();
    for i in 0..3 {
        let id = format!("device_{i:03}");
        let mut agent = GenerativeAgent::new(
            id.clone(),
            id,
            device_state,
            interpret_mode,
            ScriptedMind,
            start,
            &mut rng,
        );
        // lower the reflection trigger so reflection is visible within a short demo run.
        agent.memory.reflection_threshold = 50.0;
        agents.push(agent);
    }

    let mut sim = Simulation::new_with_seed(agents, start, 42);
    let events = sim.run(Duration::days(3));

    println!(
        "Generated {} events over 3 days across {} devices.",
        events.len(),
        sim.agents().len()
    );

    for agent in sim.agents() {
        let reflections = agent
            .memory
            .memories()
            .iter()
            .filter(|memory| memory.kind == MemoryKind::Reflection)
            .count();
        println!(
            "\n=== {} ===\n  plan steps (after daily regeneration): {}\n  memories: {} ({} reflections)",
            agent.id,
            agent.plan.steps.len(),
            agent.memory.len(),
            reflections,
        );
    }

    let timelines = Timeline::generate(&events);
    let mut device_ids: Vec<_> = timelines.keys().collect();
    device_ids.sort();

    for id in device_ids {
        println!("\n--- Timeline for {id} (first 6 entries) ---");
        if let Some(timeline) = timelines.get(id) {
            for entry in timeline.entries.iter().take(6) {
                println!("{entry}");
            }
        }
    }
}
