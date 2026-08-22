// shared-world throughput and projection

use agsim::agent::{Agent, SimAgent, StateId, StateType, Transitions};
use agsim::simulation::Simulation;
use agsim::state::{StateChangeEvent, Value};
use agsim::world::World;
use chrono::{DateTime, Duration, TimeZone, Utc};
use rand::rngs::StdRng;
use rand::{Rng, RngCore, SeedableRng};
use state_macros::{State, StateDisplay};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

const DAYS_PER_YEAR: f64 = 365.0;
const YEARS: f64 = 10.0;

/// Minutes of simulated time per window.
const DEFAULT_WINDOW_MINUTES: i64 = 30;

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

/// Shared view every agent reads.
#[derive(Default)]
struct Shared {
    last_signal: i64,
    updates: u64,
    total: i64,
}

impl World for Shared {
    fn absorb(&mut self, event: &StateChangeEvent) {
        match event.field.as_ref() {
            "signal" => {
                if let Value::Int(signal) = event.new_value {
                    self.last_signal = signal;
                    self.updates += 1;
                }
            }
            "level" => {
                if let Value::Int(level) = event.new_value {
                    self.total += level.abs();
                }
            }
            _ => {}
        }
    }
}

/// An agent that reads the world.
struct SharedAgent {
    inner: Agent<Mode, Metrics>,
    reference: i64,
    idle: StateId,
}

impl SimAgent for SharedAgent {
    type State = StateId;
    type World = Shared;

    fn peek_next_event_delay(
        &self,
        now: DateTime<Utc>,
        _world: &Shared,
        rng: &mut dyn RngCore,
    ) -> Option<f64> {
        self.inner.peek_next_event_delay(now, &(), rng)
    }

    fn step(&self, now: DateTime<Utc>, world: &Shared, rng: &mut dyn RngCore) -> Option<StateId> {
        let intended = self.inner.step(now, &(), rng)?;

        // below reference, go idle
        if world.last_signal >= self.reference {
            Some(intended)
        } else {
            Some(self.idle)
        }
    }

    fn apply_transition(
        &mut self,
        next: StateId,
        time: DateTime<Utc>,
        world: &Shared,
        rng: &mut dyn RngCore,
        out: &mut Vec<StateChangeEvent>,
    ) {
        self.reference = world.last_signal;
        self.inner.apply_transition(next, time, &(), rng, out);
    }
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

/// One-minute min/max windows.
#[derive(Default)]
struct Window {
    current_minute: i64,
    first: i64,
    max: i64,
    min: i64,
    last: i64,
    completed: u64,
}

impl Window {
    fn fold(&mut self, event: &StateChangeEvent) {
        let Value::Int(signal) = event.new_value else {
            return;
        };
        if event.field != "signal" {
            return;
        }

        let minute = event.time.timestamp() / 60;
        if minute != self.current_minute {
            self.completed += 1;
            self.current_minute = minute;
            self.first = signal;
            self.max = signal;
            self.min = signal;
        }
        self.max = self.max.max(signal);
        self.min = self.min.min(signal);
        self.last = signal;
    }
}

/// FNV-1a over the event stream.
#[derive(Default)]
struct Digest(u64);

impl Digest {
    fn new() -> Self {
        Digest(0xcbf2_9ce4_8422_2325)
    }

    fn write(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.0 ^= *byte as u64;
            self.0 = self.0.wrapping_mul(0x1000_0000_01b3);
        }
    }

    fn absorb(&mut self, event: &StateChangeEvent) {
        self.write(&event.time.timestamp_millis().to_le_bytes());
        self.write(event.agent_id.as_bytes());
        self.write(event.field.as_bytes());
        self.write(event.new_value.to_string().as_bytes());
        self.write(event.old_value.to_string().as_bytes());
    }
}

struct Measurement {
    events: u64,
    run_sec: f64,
    peak_mb: f64,
    digest: u64,
}

fn measure(agents_count: usize, aggregate: bool, window_minutes: i64) -> Measurement {
    let shared = Arc::new(Transitions::compile(transitions()));
    let mut rng = StdRng::seed_from_u64(4);

    let agents: Vec<SharedAgent> = (0..agents_count)
        .map(|index| {
            let inner = Agent::with_shared_transitions(
                format!("agent_{index:07}"),
                Mode::Idle,
                Arc::clone(&shared),
                &mut rng,
            );
            // ids come from the table
            let idle = inner
                .transitions()
                .index(&Mode::Idle)
                .expect("Idle is a defined mode");
            SharedAgent {
                inner,
                reference: 10_000,
                idle,
            }
        })
        .collect();

    let start = Utc.with_ymd_and_hms(2026, 1, 5, 14, 30, 0).unwrap();
    let mut sim = Simulation::with_world(agents, start, 4, Shared::default());

    let mut events: u64 = 0;
    let mut window = Window::default();
    let mut digest = Digest::new();

    let clock = Instant::now();
    sim.run_streaming(Duration::minutes(window_minutes), |event| {
        events += 1;
        digest.absorb(&event);
        if aggregate {
            window.fold(&event);
        }
    });
    let run_sec = clock.elapsed().as_secs_f64();

    // defeat dead code elimination
    if window.completed == u64::MAX {
        println!("unreachable {}", window.last);
    }

    Measurement {
        events,
        run_sec,
        peak_mb: peak_rss_mb(),
        digest: digest.0,
    }
}

/// Writes the event stream as columns.
fn dump_columns(path: &str, agents_count: usize, window_minutes: i64) {
    use std::io::Write;

    let shared = Arc::new(Transitions::compile(transitions()));
    let mut rng = StdRng::seed_from_u64(4);
    let agents: Vec<SharedAgent> = (0..agents_count)
        .map(|index| {
            let inner = Agent::with_shared_transitions(
                format!("agent_{index:07}"),
                Mode::Idle,
                Arc::clone(&shared),
                &mut rng,
            );
            let idle = inner.transitions().index(&Mode::Idle).unwrap();
            SharedAgent {
                inner,
                reference: 10_000,
                idle,
            }
        })
        .collect();

    let start = Utc.with_ymd_and_hms(2026, 1, 5, 14, 30, 0).unwrap();
    let mut sim = Simulation::with_world(agents, start, 4, Shared::default());

    let mut times: Vec<i32> = Vec::new();
    let mut ids: Vec<u32> = Vec::new();
    let mut fields: Vec<u8> = Vec::new();
    let mut values: Vec<i64> = Vec::new();
    let mut previous_ms: i64 = start.timestamp_millis();

    sim.run_streaming(Duration::minutes(window_minutes), |event| {
        let ms = event.time.timestamp_millis();
        times.push((ms - previous_ms).clamp(i32::MIN as i64, i32::MAX as i64) as i32);
        previous_ms = ms;

        let id: u32 = event
            .agent_id
            .rsplit('_')
            .next()
            .and_then(|n| n.parse().ok())
            .unwrap_or(0);
        ids.push(id);
        fields.push(match event.field.as_ref() {
            "level" => 0,
            "pending" => 1,
            "signal" => 2,
            _ => 3,
        });
        values.push(event.new_value.as_i64().unwrap_or(0));
    });

    let mut file = std::io::BufWriter::new(std::fs::File::create(path).expect("dump file"));
    for value in &times {
        file.write_all(&value.to_le_bytes()).unwrap();
    }
    for value in &ids {
        file.write_all(&value.to_le_bytes()).unwrap();
    }
    file.write_all(&fields).unwrap();
    for value in &values {
        file.write_all(&value.to_le_bytes()).unwrap();
    }
    file.flush().unwrap();

    println!("{} events written to {path}", times.len());
}

fn peak_rss_mb() -> f64 {
    std::fs::read_to_string("/proc/self/status")
        .unwrap_or_default()
        .lines()
        .find(|line| line.starts_with("VmHWM:"))
        .and_then(|line| line.split_whitespace().nth(1))
        .and_then(|kb| kb.parse::<f64>().ok())
        .map(|kb| kb / 1024.0)
        .unwrap_or(0.0)
}

fn format_duration(seconds: f64) -> String {
    if seconds < 90.0 {
        format!("{seconds:.0}s")
    } else if seconds < 5400.0 {
        format!("{:.1}m", seconds / 60.0)
    } else if seconds < 172_800.0 {
        format!("{:.1}h", seconds / 3600.0)
    } else {
        format!("{:.1}d", seconds / 86_400.0)
    }
}

fn format_bytes(bytes: f64) -> String {
    if bytes < 1e9 {
        format!("{:.0} MB", bytes / 1e6)
    } else if bytes < 1e12 {
        format!("{:.1} GB", bytes / 1e9)
    } else {
        format!("{:.1} TB", bytes / 1e12)
    }
}

fn main() {
    let args = self::args();
    let after = |flag: &str| {
        args.iter()
            .position(|a| a == flag)
            .and_then(|at| args.get(at + 1))
    };

    let aggregate = !args.iter().any(|arg| arg == "--no-aggregate");
    let window_minutes: i64 = after("--window")
        .and_then(|raw| raw.parse().ok())
        .unwrap_or(DEFAULT_WINDOW_MINUTES);
    let counts: Vec<usize> = match after("--agents") {
        Some(list) => list
            .split(',')
            .map(|part| part.trim().parse().expect("agent counts are numbers"))
            .collect(),
        None => vec![1_000, 10_000, 100_000, 1_000_000],
    };

    if let Some(path) = after("--dump") {
        let count = counts.first().copied().unwrap_or(2000);
        dump_columns(path, count, window_minutes);
        return;
    }

    let horizon_seconds = YEARS * DAYS_PER_YEAR * 86_400.0;
    let window_seconds = (window_minutes * 60) as f64;
    let scale = horizon_seconds / window_seconds;

    println!(
        "{YEARS:.0} years = {:.0}M simulated seconds",
        horizon_seconds / 1e6
    );
    println!(
        "measured over {window_minutes} simulated minutes, consumer = {}\n",
        if aggregate {
            "1-minute windows"
        } else {
            "count only"
        }
    );

    println!(
        "{:>10} | {:>13} | {:>10} | {:>9} | {:>14} | {:>12} | {:>11}",
        "agents", "events/s", "ns/event", "peak MB", "horizon events", "horizon CPU", "if retained"
    );
    println!("{}", "-".repeat(100));

    let show_digest = args.iter().any(|arg| arg == "--digest");

    for count in counts {
        let m = measure(count, aggregate, window_minutes);
        if show_digest {
            println!(
                "{:>10} | events {:>12} | digest {:016x}",
                count, m.events, m.digest
            );
            continue;
        }
        let rate = m.events as f64 / m.run_sec;
        let horizon_events = m.events as f64 * scale;

        println!(
            "{:>10} | {:>13.0} | {:>10.0} | {:>9.1} | {:>14} | {:>12} | {:>11}",
            count,
            rate,
            m.run_sec * 1e9 / m.events as f64,
            m.peak_mb,
            format!("{:.2e}", horizon_events),
            format_duration(horizon_events / rate),
            format_bytes(horizon_events * 100.0),
        );
    }

    println!("\n'if retained' assumes ~100 bytes per event.");
}

// cargo bench passes --bench
fn args() -> Vec<String> {
    std::env::args()
        .skip(1)
        .filter(|a| a != "--bench")
        .collect()
}
