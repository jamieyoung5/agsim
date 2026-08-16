use agsim::generative::{GenerativeAgent, Mind};
use agsim::memory::{Insight, Memory, Reflector};
use agsim::planning::{PlanContext, PlanStep, Planner, Reaction};
use agsim::simulation::Simulation;
use agsim::state::{StateChangeEvent, Timeline};
use chrono::{DateTime, Duration, TimeZone, Utc};
use rand::rngs::StdRng;
use rand::{Rng, RngCore, SeedableRng};
use rand_distr::{Distribution, Normal};
use state_macros::{State, StateDisplay};

const SAMPLE_MINUTES: i64 = 30;
const SAMPLES_PER_DAY: usize = (24 * 60 / SAMPLE_MINUTES) as usize;
const DAYS: usize = 7;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Copy)]
enum Mode {
    Offline,
    Idle,
    Working,
    HeavyLoad,
}

#[derive(Debug, Clone, Default, State, StateDisplay)]
struct DeviceState {
    cpu_in_use_percent: f32,
    memory_in_use_mb: u32,
}

fn device_state(mode: &Mode, rng: &mut dyn RngCore) -> DeviceState {
    match mode {
        Mode::Offline => DeviceState {
            cpu_in_use_percent: 0.0,
            memory_in_use_mb: 0,
        },
        Mode::Idle => DeviceState {
            cpu_in_use_percent: rng.gen_range(0.1..5.0),
            memory_in_use_mb: rng.gen_range(400..800),
        },
        Mode::Working => DeviceState {
            cpu_in_use_percent: rng.gen_range(10.0..40.0),
            memory_in_use_mb: rng.gen_range(1024..4096),
        },
        Mode::HeavyLoad => DeviceState {
            cpu_in_use_percent: rng.gen_range(60.0..99.9),
            memory_in_use_mb: rng.gen_range(4096..16384),
        },
    }
}

fn interpret_mode(step: &PlanStep) -> Mode {
    let activity = step.description.to_lowercase();
    if activity.contains("offline") {
        Mode::Offline
    } else if activity.contains("heavy") {
        Mode::HeavyLoad
    } else if activity.contains("work") {
        Mode::Working
    } else {
        Mode::Idle
    }
}

struct RoutineMind;

impl Planner for RoutineMind {
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

    fn decompose(&self, _step: &PlanStep, _ctx: &PlanContext) -> Vec<PlanStep> {
        Vec::new()
    }

    fn react(&self, _o: &Memory, _c: Option<&PlanStep>, _ctx: &PlanContext) -> Reaction {
        Reaction::Continue
    }
}

impl Reflector for RoutineMind {
    fn salient_questions(&self, _recent: &[&Memory]) -> Vec<String> {
        Vec::new()
    }
    fn synthesize(&self, _question: &str, _evidence: &[&Memory]) -> Vec<Insight> {
        Vec::new()
    }
}

impl Mind for RoutineMind {
    fn importance(&self, _event: &StateChangeEvent) -> f64 {
        1.0
    }
}

fn cpu_at(points: &[(DateTime<Utc>, f32)], t: DateTime<Utc>) -> f32 {
    let mut value = 0.0;
    for (timestamp, cpu) in points {
        if *timestamp <= t {
            value = *cpu;
        } else {
            break;
        }
    }
    value
}

fn sparkline(values: &[f32]) -> String {
    const BARS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    values
        .iter()
        .map(|&v| {
            let level = ((v / 100.0) * (BARS.len() - 1) as f32).round() as usize;
            BARS[level.min(BARS.len() - 1)]
        })
        .collect()
}

fn main() {
    // starting on a midnight boundary, with every generator seeded, so each run draws the same
    // week of telemetry, and the sparklines below line up with the day they describe.
    let start = Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap();

    let mut setup_rng = StdRng::seed_from_u64(1);
    let mut agent = GenerativeAgent::new(
        "device_000".to_string(),
        "device_000".to_string(),
        device_state,
        interpret_mode,
        RoutineMind,
        start,
        &mut setup_rng,
    );
    agent.memory.reflection_threshold = f64::INFINITY; // no reflection in this example.

    let mut sim = Simulation::new_with_seed(vec![agent], start, 42);
    let events = sim.run(Duration::days(DAYS as i64));

    // reconstruct the device's CPU timeline, sample it on a fixed grid, and add sensor noise.
    let timelines = Timeline::generate(&events);
    let timeline = timelines.get("device_000").expect("device timeline");
    let points: Vec<(DateTime<Utc>, f32)> = timeline
        .entries
        .iter()
        .map(|entry| {
            let cpu = entry
                .state
                .get("cpu_in_use_percent")
                .and_then(|value| value.parse().ok())
                .unwrap_or(0.0);
            (entry.timestamp, cpu)
        })
        .collect();

    let mut noise_rng = StdRng::seed_from_u64(7);
    let noise = Normal::new(0.0, 2.0).unwrap();

    let mut series = Vec::with_capacity(DAYS * SAMPLES_PER_DAY);
    for i in 0..(DAYS * SAMPLES_PER_DAY) {
        let t = start + Duration::minutes(i as i64 * SAMPLE_MINUTES);
        let sampled = cpu_at(&points, t) + noise.sample(&mut noise_rng) as f32;
        series.push(sampled.clamp(0.0, 100.0));
    }

    println!(
        "Generated {} CPU samples ({}-minute cadence over {} days) for device_000.\n",
        series.len(),
        SAMPLE_MINUTES,
        DAYS
    );
    println!("CPU usage % each row is one day, each cell {SAMPLE_MINUTES} minutes:\n");
    for day in 0..DAYS {
        let slice = &series[day * SAMPLES_PER_DAY..(day + 1) * SAMPLES_PER_DAY];
        let mean: f32 = slice.iter().sum::<f32>() / slice.len() as f32;
        let peak = slice.iter().cloned().fold(0.0_f32, f32::max);
        println!(
            "Day {} |{}|  mean {:>4.1}%  peak {:>4.1}%",
            day + 1,
            sparkline(slice),
            mean,
            peak
        );
    }
    println!("\n(00:00 ────────────────────────────────────────── 24:00)");
}
