// heap traffic per event

use agsim::agent::{Agent, SimAgent, StateType, Transitions};
use agsim::simulation::Simulation;
use agsim::state::{AgentId, StateChangeEvent, Value};
use chrono::{DateTime, Duration, TimeZone, Utc};
use rand::rngs::StdRng;
use rand::{Rng, RngCore, SeedableRng};
use state_macros::{State, StateDisplay};
use std::alloc::{GlobalAlloc, Layout, System};
use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

static ALLOCS: AtomicU64 = AtomicU64::new(0);
static BYTES: AtomicU64 = AtomicU64::new(0);

struct Counting;

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(new_size as u64, Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

fn snapshot() -> (u64, u64) {
    (
        ALLOCS.load(Ordering::Relaxed),
        BYTES.load(Ordering::Relaxed),
    )
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Copy)]
enum Mode {
    Idle,
    Active,
    Busy,
    Blocked,
}

#[derive(Debug, Clone, Default, State, StateDisplay)]
struct Metrics {
    level: i64,
    pending: u32,
    signal: i64,
    total: i64,
}

fn transitions() -> HashMap<Mode, StateType<Mode, Metrics>> {
    let mut matrix = HashMap::new();
    let modes = [
        (Mode::Idle, 45.0),
        (Mode::Active, 20.0),
        (Mode::Busy, 20.0),
        (Mode::Blocked, 90.0),
    ];

    for (mode, rate) in modes {
        matrix.insert(
            mode,
            StateType::new(
                |rng| Metrics {
                    level: rng.gen_range(-800..800),
                    pending: rng.gen_range(0..8),
                    signal: rng.gen_range(9_000..11_000),
                    total: rng.gen_range(0..400_000),
                },
                vec![
                    (Mode::Idle, 0.25),
                    (Mode::Active, 0.25),
                    (Mode::Busy, 0.25),
                    (Mode::Blocked, 0.25),
                ],
                rate,
            ),
        );
    }

    matrix
}

/// Emits four changes per transition.
struct SyntheticAgent {
    id: AgentId,
    legacy: bool,
    counter: i64,
}

impl SimAgent for SyntheticAgent {
    type State = ();
    type World = ();

    fn peek_next_event_delay(
        &self,
        _now: DateTime<Utc>,
        _world: &(),
        _rng: &mut dyn RngCore,
    ) -> Option<f64> {
        Some(1.0)
    }

    fn step(&self, _now: DateTime<Utc>, _world: &(), _rng: &mut dyn RngCore) -> Option<()> {
        Some(())
    }

    fn apply_transition(
        &mut self,
        _next: (),
        time: DateTime<Utc>,
        _world: &(),
        _rng: &mut dyn RngCore,
        out: &mut Vec<StateChangeEvent>,
    ) {
        self.counter += 1;
        const FIELDS: [&str; 4] = ["level", "pending", "signal", "total"];

        for field in FIELDS {
            out.push(if self.legacy {
                StateChangeEvent {
                    time,
                    agent_id: Arc::from(self.id.as_ref()),
                    field: Cow::Owned(field.to_string()),
                    old_value: Value::text(self.counter - 1),
                    new_value: Value::text(self.counter),
                }
            } else {
                StateChangeEvent {
                    time,
                    agent_id: self.id.clone(),
                    field: Cow::Borrowed(field),
                    old_value: Value::Int(self.counter - 1),
                    new_value: Value::Int(self.counter),
                }
            });
        }
    }
}

fn synthetic_throughput(agents_count: usize, legacy: bool) -> (f64, u64, u64) {
    let agents: Vec<_> = (0..agents_count)
        .map(|index| SyntheticAgent {
            id: Arc::from(format!("agent_{index:07}").as_str()),
            legacy,
            counter: 0,
        })
        .collect();

    let start = Utc.with_ymd_and_hms(2026, 1, 5, 14, 30, 0).unwrap();
    let mut sim = Simulation::new_with_seed(agents, start, 11);

    let mut events: u64 = 0;
    let before = snapshot();
    let clock = Instant::now();
    sim.run_streaming(Duration::minutes(30), |_| events += 1);
    let elapsed = clock.elapsed().as_secs_f64();
    let after = snapshot();

    (elapsed, events, after.0 - before.0)
}

fn main() {
    let agents_count: usize = args()
        .first()
        .and_then(|raw| raw.parse().ok())
        .unwrap_or(5_000);

    let shared = Arc::new(Transitions::compile(transitions()));
    let mut rng = StdRng::seed_from_u64(11);

    let before_build = snapshot();
    let agents: Vec<_> = (0..agents_count)
        .map(|index| {
            Agent::with_shared_transitions(
                format!("agent_{index:07}"),
                Mode::Idle,
                Arc::clone(&shared),
                &mut rng,
            )
        })
        .collect();
    let after_build = snapshot();

    let start = Utc.with_ymd_and_hms(2026, 1, 5, 14, 30, 0).unwrap();
    let mut sim = Simulation::new_with_seed(agents, start, 11);

    let mut events: u64 = 0;
    let before_run = snapshot();
    let run_clock = Instant::now();
    sim.run_streaming(Duration::minutes(60), |_| events += 1);
    let run_sec = run_clock.elapsed().as_secs_f64();
    let after_run = snapshot();

    let build_allocs = after_build.0 - before_build.0;
    let run_allocs = after_run.0 - before_run.0;
    let run_bytes = after_run.1 - before_run.1;

    println!("agents            {agents_count}");
    println!("events            {events}");
    println!();
    println!(
        "build allocations {build_allocs}  ({:.1} per agent)",
        build_allocs as f64 / agents_count as f64
    );
    println!();
    println!(
        "run allocations   {run_allocs}  ({:.2} per event)",
        run_allocs as f64 / events as f64
    );
    println!(
        "run bytes         {run_bytes}  ({:.1} per event)",
        run_bytes as f64 / events as f64
    );
    println!(
        "run time          {run_sec:.2}s  ({:.0} events/s, {:.0} ns/event)",
        events as f64 / run_sec,
        run_sec * 1e9 / events as f64
    );

    println!("\n--- cost of the event representation, all else equal ---");
    let (slow_sec, slow_events, slow_allocs) = synthetic_throughput(agents_count, true);
    let (fast_sec, fast_events, fast_allocs) = synthetic_throughput(agents_count, false);

    println!(
        "owned Strings     {:.2}s  {:>12.0} events/s  ({:.2} allocs/event)",
        slow_sec,
        slow_events as f64 / slow_sec,
        slow_allocs as f64 / slow_events as f64
    );
    println!(
        "current shape     {:.2}s  {:>12.0} events/s  ({:.2} allocs/event)",
        fast_sec,
        fast_events as f64 / fast_sec,
        fast_allocs as f64 / fast_events as f64
    );
    println!(
        "headroom          {:.2}x",
        (slow_events as f64 / slow_sec).recip() / (fast_events as f64 / fast_sec).recip()
    );
}

// cargo bench passes --bench
fn args() -> Vec<String> {
    std::env::args()
        .skip(1)
        .filter(|a| a != "--bench")
        .collect()
}
