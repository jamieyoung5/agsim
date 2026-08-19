//! Capacity model for a market simulation: measured throughput, projected to a decade.
//!
//! Traders here use the architecture a market actually wants — they write to a shared tape and read
//! it back when they decide, rather than being individually notified of every other trader's
//! activity. That keeps shared context O(1) per event instead of O(agents).
//!
//! Each configuration runs a short window of market time and the result is extrapolated, since the
//! whole point is that the full run does not fit in a benchmark. The consumer side is measured too:
//! events are folded into one-minute bars, because at these rates whatever reads the events is a
//! real part of the cost.

use agsim::agent::{Agent, SimAgent, StateType, Transitions};
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

const TRADING_DAYS_PER_YEAR: f64 = 252.0;
const TRADING_HOURS_PER_DAY: f64 = 6.5;
const YEARS: f64 = 10.0;

/// Default minutes of market time each measured window covers.
const DEFAULT_WINDOW_MINUTES: i64 = 30;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Copy)]
enum TraderMode {
    Flat,
    Bidding,
    Offering,
    Holding,
}

#[derive(Debug, Clone, Default, State, StateDisplay)]
struct TraderState {
    position: i64,
    working_orders: u32,
    quote_price: i64,
    exposure: i64,
}

/// The shared view every trader reads: where the last print landed and how busy the tape is.
#[derive(Default)]
struct Tape {
    last_price: i64,
    prints: u64,
    volume: i64,
}

impl World for Tape {
    fn absorb(&mut self, event: &StateChangeEvent) {
        match event.field.as_ref() {
            "quote_price" => {
                if let Value::Int(price) = event.new_value {
                    self.last_price = price;
                    self.prints += 1;
                }
            }
            "position" => {
                if let Value::Int(size) = event.new_value {
                    self.volume += size.abs();
                }
            }
            _ => {}
        }
    }
}

/// A trader whose next move depends on where the tape is relative to its own last quote.
struct Trader {
    inner: Agent<TraderMode, TraderState>,
    reference: i64,
    flat: usize,
}

impl SimAgent for Trader {
    type State = usize;
    type World = Tape;

    fn peek_next_event_delay(
        &self,
        now: DateTime<Utc>,
        _world: &Tape,
        rng: &mut dyn RngCore,
    ) -> Option<f64> {
        self.inner.peek_next_event_delay(now, &(), rng)
    }

    fn step(&self, now: DateTime<Utc>, world: &Tape, rng: &mut dyn RngCore) -> Option<usize> {
        let intended = self.inner.step(now, &(), rng)?;

        // a trader that finds the tape below its own reference stands down instead
        if world.last_price >= self.reference {
            Some(intended)
        } else {
            Some(self.flat)
        }
    }

    fn apply_transition(
        &mut self,
        next: usize,
        time: DateTime<Utc>,
        world: &Tape,
        rng: &mut dyn RngCore,
        out: &mut Vec<StateChangeEvent>,
    ) {
        self.reference = world.last_price;
        self.inner.apply_transition(next, time, &(), rng, out);
    }
}

fn transitions() -> HashMap<TraderMode, StateType<TraderMode, TraderState>> {
    let mut matrix = HashMap::new();
    let modes = [
        (TraderMode::Flat, 45.0),
        (TraderMode::Bidding, 20.0),
        (TraderMode::Offering, 20.0),
        (TraderMode::Holding, 90.0),
    ];

    for (mode, rate) in modes {
        matrix.insert(
            mode,
            StateType::new(
                |rng| TraderState {
                    position: rng.gen_range(-800..800),
                    working_orders: rng.gen_range(0..8),
                    quote_price: rng.gen_range(9_000..11_000),
                    exposure: rng.gen_range(0..400_000),
                },
                vec![
                    (TraderMode::Flat, 0.25),
                    (TraderMode::Bidding, 0.25),
                    (TraderMode::Offering, 0.25),
                    (TraderMode::Holding, 0.25),
                ],
                rate,
            ),
        );
    }

    matrix
}

/// One-minute OHLC bars, standing in for whatever a real run would do with the flow.
#[derive(Default)]
struct Bars {
    current_minute: i64,
    open: i64,
    high: i64,
    low: i64,
    close: i64,
    completed: u64,
}

impl Bars {
    fn fold(&mut self, event: &StateChangeEvent) {
        let Value::Int(price) = event.new_value else {
            return;
        };
        if event.field != "quote_price" {
            return;
        }

        let minute = event.time.timestamp() / 60;
        if minute != self.current_minute {
            self.completed += 1;
            self.current_minute = minute;
            self.open = price;
            self.high = price;
            self.low = price;
        }
        self.high = self.high.max(price);
        self.low = self.low.min(price);
        self.close = price;
    }
}

/// FNV-1a over the whole event stream, for checking that a seed reproduces a run.
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

    let agents: Vec<Trader> = (0..agents_count)
        .map(|index| {
            let inner = Agent::with_shared_transitions(
                format!("trader_{index:07}"),
                TraderMode::Flat,
                Arc::clone(&shared),
                &mut rng,
            );
            // compiled positions follow the table, not the order the modes were written in
            let flat = inner
                .transitions()
                .index(&TraderMode::Flat)
                .expect("Flat is a defined mode");
            Trader {
                inner,
                reference: 10_000,
                flat,
            }
        })
        .collect();

    let start = Utc.with_ymd_and_hms(2026, 1, 5, 14, 30, 0).unwrap();
    let mut sim = Simulation::with_world(agents, start, 4, Tape::default());

    let mut events: u64 = 0;
    let mut bars = Bars::default();
    let mut digest = Digest::new();

    let clock = Instant::now();
    sim.run_streaming(Duration::minutes(window_minutes), |event| {
        events += 1;
        digest.absorb(&event);
        if aggregate {
            bars.fold(&event);
        }
    });
    let run_sec = clock.elapsed().as_secs_f64();

    // keep the aggregate observable so it cannot be optimised away
    if bars.completed == u64::MAX {
        println!("unreachable {}", bars.close);
    }

    Measurement {
        events,
        run_sec,
        peak_mb: peak_rss_mb(),
        digest: digest.0,
    }
}

/// Writes the event stream as four separate columns of fixed-width values.
///
/// Grouping like with like is what makes the stream compressible: timestamps become small deltas,
/// agent indices and field names repeat heavily, and prices move in narrow ranges. Interleaving
/// them as whole records, the way a row of CSV does, hides all of that from the compressor.
fn dump_columns(path: &str, agents_count: usize, window_minutes: i64) {
    use std::io::Write;

    let shared = Arc::new(Transitions::compile(transitions()));
    let mut rng = StdRng::seed_from_u64(4);
    let agents: Vec<Trader> = (0..agents_count)
        .map(|index| {
            let inner = Agent::with_shared_transitions(
                format!("trader_{index:07}"),
                TraderMode::Flat,
                Arc::clone(&shared),
                &mut rng,
            );
            let flat = inner.transitions().index(&TraderMode::Flat).unwrap();
            Trader {
                inner,
                reference: 10_000,
                flat,
            }
        })
        .collect();

    let start = Utc.with_ymd_and_hms(2026, 1, 5, 14, 30, 0).unwrap();
    let mut sim = Simulation::with_world(agents, start, 4, Tape::default());

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
            "position" => 0,
            "working_orders" => 1,
            "quote_price" => 2,
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
    let aggregate = !std::env::args().any(|arg| arg == "--no-aggregate");
    let window_minutes: i64 = std::env::args()
        .position(|a| a == "--window")
        .and_then(|at| std::env::args().nth(at + 1))
        .and_then(|raw| raw.parse().ok())
        .unwrap_or(DEFAULT_WINDOW_MINUTES);
    let counts: Vec<usize> = match std::env::args().position(|a| a == "--agents") {
        Some(at) => std::env::args()
            .nth(at + 1)
            .expect("--agents needs a list")
            .split(',')
            .map(|part| part.trim().parse().expect("agent counts are numbers"))
            .collect(),
        None => vec![1_000, 10_000, 100_000, 1_000_000],
    };

    if let Some(at) = std::env::args().position(|a| a == "--dump") {
        let path = std::env::args().nth(at + 1).expect("--dump needs a path");
        let count = counts.first().copied().unwrap_or(2000);
        dump_columns(&path, count, window_minutes);
        return;
    }

    // market seconds in a decade of trading sessions
    let decade_seconds = YEARS * TRADING_DAYS_PER_YEAR * TRADING_HOURS_PER_DAY * 3600.0;
    let window_seconds = (window_minutes * 60) as f64;
    let scale = decade_seconds / window_seconds;

    println!(
        "10 years = {TRADING_DAYS_PER_YEAR:.0} sessions/yr x {TRADING_HOURS_PER_DAY} h = {:.0}M market seconds",
        decade_seconds / 1e6
    );
    println!(
        "measured over {window_minutes} market minutes, consumer = {}\n",
        if aggregate {
            "1-minute OHLC bars"
        } else {
            "count only"
        }
    );

    println!(
        "{:>10} | {:>13} | {:>10} | {:>9} | {:>14} | {:>12} | {:>11}",
        "agents", "events/s", "ns/event", "peak MB", "decade events", "decade CPU", "if retained"
    );
    println!("{}", "-".repeat(100));

    let show_digest = std::env::args().any(|arg| arg == "--digest");

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
        let decade_events = m.events as f64 * scale;

        println!(
            "{:>10} | {:>13.0} | {:>10.0} | {:>9.1} | {:>14} | {:>12} | {:>11}",
            count,
            rate,
            m.run_sec * 1e9 / m.events as f64,
            m.peak_mb,
            format!("{:.2e}", decade_events),
            format_duration(decade_events / rate),
            format_bytes(decade_events * 100.0),
        );
    }

    println!("\n'if retained' assumes ~100 bytes per event, the measured cost of a kept log.");
}
