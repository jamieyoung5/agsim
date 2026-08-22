use agsim::generative::GenerativeAgent;
use agsim::llm::openai::SchemaMode;
use agsim::llm::{LlmMind, OpenAiCompat};
use agsim::simulation::Simulation;
use agsim::state::Timeline;
use chrono::{Duration, TimeZone, Utc};
use rand::rngs::StdRng;
use rand::{Rng, RngCore, SeedableRng};
use state_macros::{State, StateDisplay};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Copy)]
enum Mode {
    Offline,
    Idle,
    Working,
    HeavyLoad,
}

#[derive(Debug, Clone, Default, State, StateDisplay)]
struct DeviceState {
    connected_status: bool,
    active_sessions: u32,
    cpu_in_use_percent: f32,
}

fn device_state(mode: &Mode, rng: &mut dyn RngCore) -> DeviceState {
    match mode {
        Mode::Offline => DeviceState {
            connected_status: false,
            active_sessions: 0,
            cpu_in_use_percent: 0.0,
        },
        Mode::Idle => DeviceState {
            connected_status: true,
            active_sessions: 0,
            cpu_in_use_percent: rng.gen_range(0.1..5.0),
        },
        Mode::Working => DeviceState {
            connected_status: true,
            active_sessions: rng.gen_range(1..4),
            cpu_in_use_percent: rng.gen_range(10.0..40.0),
        },
        Mode::HeavyLoad => DeviceState {
            connected_status: true,
            active_sessions: rng.gen_range(3..10),
            cpu_in_use_percent: rng.gen_range(60.0..99.9),
        },
    }
}

// match loosely, models drift
fn interpret_mode(step: &agsim::planning::PlanStep) -> Mode {
    let activity = step.description.to_lowercase();
    if activity.contains("offline") || activity.contains("asleep") || activity.contains("shut") {
        Mode::Offline
    } else if activity.contains("heavy") || activity.contains("peak") || activity.contains("load") {
        Mode::HeavyLoad
    } else if activity.contains("work") || activity.contains("active") || activity.contains("busy")
    {
        Mode::Working
    } else {
        Mode::Idle
    }
}

fn schema_mode() -> SchemaMode {
    match std::env::var("AGSIM_LLM_SCHEMA").as_deref() {
        Ok("json_schema") => SchemaMode::JsonSchema,
        Ok("json_object") => SchemaMode::JsonObject,
        _ => SchemaMode::Prompted,
    }
}

fn main() {
    let backend = match OpenAiCompat::from_env() {
        Ok(backend) => backend.with_schema_mode(schema_mode()),
        Err(_) => {
            eprintln!(
                "Set AGSIM_LLM_URL and AGSIM_LLM_MODEL to point at your local server, e.g.\n  \
                 AGSIM_LLM_URL=http://localhost:11434/v1 AGSIM_LLM_MODEL=llama3.1:8b \\\n    \
                 cargo run --example small_model --features llm"
            );
            std::process::exit(1);
        }
    };

    println!(
        "Planning with {} at {} (schema mode: {:?}).\n",
        backend.model,
        std::env::var("AGSIM_LLM_URL").unwrap_or_default(),
        backend.schema_mode,
    );

    let identity =
        "a shared office printer that is busy during working hours and idle overnight".to_string();
    let mind = LlmMind::with_backend(backend, identity);

    let start = Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap();
    let mut setup_rng = StdRng::seed_from_u64(1);

    let mut agent = GenerativeAgent::new(
        "printer_000".to_string(),
        "a shared office printer".to_string(),
        device_state,
        interpret_mode,
        mind,
        start,
        &mut setup_rng,
    );
    agent.memory.reflection_threshold = 40.0;

    if agent.plan.steps.is_empty() {
        eprintln!(
            "The model returned no usable plan. Try AGSIM_LLM_SCHEMA=json_schema if your server \
             supports it, or a larger model."
        );
        std::process::exit(1);
    }

    println!("Plan the model came back with:");
    for step in &agent.plan.steps {
        println!(
            "  {} - {}  {}",
            step.start.format("%H:%M"),
            step.end().format("%H:%M"),
            step.description,
        );
    }

    let mut sim = Simulation::new_with_seed(vec![agent], start, 42);
    let events = sim.run(Duration::days(1));

    let agent = &sim.agents()[0];
    println!(
        "\nGenerated {} events over a day. {} memories, {} of them reflections.",
        events.len(),
        agent.memory.len(),
        agent
            .memory
            .memories()
            .iter()
            .filter(|memory| memory.kind == agsim::memory::MemoryKind::Reflection)
            .count(),
    );

    if let Some(timeline) = Timeline::generate(&events).get("printer_000") {
        println!("\n--- Timeline ---");
        for entry in timeline.entries.iter().take(10) {
            println!("{entry}");
        }
    }
}
