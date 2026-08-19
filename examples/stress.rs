//! Scaling harness: how many agents can a simulation carry before throughput collapses?
//!
//! Runs one configuration and prints a single result row. `--sweep` re-executes this example as a
//! fresh subprocess per agent count, which keeps the peak-RSS figure honest — a high-water mark
//! never falls, so several configurations in one process would all report the largest one.
//!
//! The agent is shaped like a market participant (quote, position, order book pressure) so the
//! numbers transfer to an order-flow simulation rather than to a toy two-state agent.

use agsim::agent::{Agent, SimAgent, StateType, Transitions};
use agsim::clock::{Live, LiveOutcome};
use agsim::simulation::{Perception, Simulation};
use agsim::space::Position;
use agsim::state::{StateChangeEvent, Value};
use chrono::{DateTime, Duration, TimeZone, Utc};
use rand::rngs::StdRng;
use rand::{Rng, RngCore, SeedableRng};
use state_macros::{State, StateDisplay};
use std::collections::HashMap;
use std::ops::ControlFlow;
use std::sync::Arc;
use std::time::Instant;

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

fn transitions() -> HashMap<TraderMode, StateType<TraderMode, TraderState>> {
    let mut matrix = HashMap::new();

    matrix.insert(
        TraderMode::Flat,
        StateType::new(
            |rng| TraderState {
                position: 0,
                working_orders: 0,
                quote_price: rng.gen_range(9_000..11_000),
                exposure: 0,
            },
            vec![
                (TraderMode::Bidding, 0.45),
                (TraderMode::Offering, 0.45),
                (TraderMode::Flat, 0.10),
            ],
            45.0,
        ),
    );

    matrix.insert(
        TraderMode::Bidding,
        StateType::new(
            |rng| TraderState {
                position: rng.gen_range(1..500),
                working_orders: rng.gen_range(1..8),
                quote_price: rng.gen_range(9_000..11_000),
                exposure: rng.gen_range(1_000..250_000),
            },
            vec![
                (TraderMode::Holding, 0.50),
                (TraderMode::Offering, 0.30),
                (TraderMode::Flat, 0.20),
            ],
            20.0,
        ),
    );

    matrix.insert(
        TraderMode::Offering,
        StateType::new(
            |rng| TraderState {
                position: rng.gen_range(-500..0),
                working_orders: rng.gen_range(1..8),
                quote_price: rng.gen_range(9_000..11_000),
                exposure: rng.gen_range(1_000..250_000),
            },
            vec![
                (TraderMode::Holding, 0.50),
                (TraderMode::Bidding, 0.30),
                (TraderMode::Flat, 0.20),
            ],
            20.0,
        ),
    );

    matrix.insert(
        TraderMode::Holding,
        StateType::new(
            |rng| TraderState {
                position: rng.gen_range(-800..800),
                working_orders: rng.gen_range(0..3),
                quote_price: rng.gen_range(9_000..11_000),
                exposure: rng.gen_range(0..400_000),
            },
            vec![
                (TraderMode::Bidding, 0.35),
                (TraderMode::Offering, 0.35),
                (TraderMode::Holding, 0.30),
            ],
            90.0,
        ),
    );

    matrix
}

/// A trader that actually consumes the tape. The plain [`Agent`] leaves `observe` at its no-op
/// default, which lets the optimiser delete the simulation's observer loop outright; anything that
/// reacts to other agents pays for that loop in full, so the harness needs both shapes to tell the
/// framework's cost from the optimiser's luck.
struct ObservingTrader {
    inner: Agent<TraderMode, TraderState>,
    seen: u64,
    last_price: i64,
    position: Position,
}

impl ObservingTrader {
    fn new(inner: Agent<TraderMode, TraderState>, position: Position) -> Self {
        ObservingTrader {
            inner,
            seen: 0,
            last_price: 0,
            position,
        }
    }
}

impl SimAgent for ObservingTrader {
    type State = usize;
    type World = ();

    fn peek_next_event_delay(
        &self,
        now: DateTime<Utc>,
        world: &(),
        rng: &mut dyn RngCore,
    ) -> Option<f64> {
        self.inner.peek_next_event_delay(now, world, rng)
    }

    fn step(&self, now: DateTime<Utc>, world: &(), rng: &mut dyn RngCore) -> Option<usize> {
        self.inner.step(now, world, rng)
    }

    fn apply_transition(
        &mut self,
        next: usize,
        time: DateTime<Utc>,
        world: &(),
        rng: &mut dyn RngCore,
        out: &mut Vec<StateChangeEvent>,
    ) {
        self.inner.apply_transition(next, time, world, rng, out)
    }

    // deliberately the cheapest useful reaction: track the last quote off the tape
    fn observe(&mut self, event: &StateChangeEvent) {
        self.seen += 1;
        if event.field == "quote_price"
            && let Value::Int(price) = event.new_value
        {
            self.last_price = price;
        }
    }

    fn location(&self) -> Option<Position> {
        Some(self.position)
    }
}

struct Config {
    agents: usize,
    minutes: i64,
    perception: Perception,
    retain_log: bool,
    collect: bool,
    observing: bool,
    shared: bool,
    budget_sec: f64,
    seed: u64,
}

impl Config {
    fn perception_label(&self) -> String {
        match self.perception {
            Perception::SelfOnly => "self".to_string(),
            Perception::Global => "global".to_string(),
            Perception::Proximity { radius } => format!("prox:{radius}"),
        }
    }
}

fn start_time() -> DateTime<Utc> {
    Utc.with_ymd_and_hms(2026, 1, 5, 14, 30, 0).unwrap()
}

// current and peak resident set, in mebibytes
fn memory_mb() -> (f64, f64) {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    let field = |key: &str| -> f64 {
        status
            .lines()
            .find(|line| line.starts_with(key))
            .and_then(|line| line.split_whitespace().nth(1))
            .and_then(|kb| kb.parse::<f64>().ok())
            .map(|kb| kb / 1024.0)
            .unwrap_or(0.0)
    };
    (field("VmRSS:"), field("VmHWM:"))
}

struct Outcome {
    build_sec: f64,
    run_sec: f64,
    events: u64,
    aborted: bool,
    rss_after_build_mb: f64,
    peak_rss_mb: f64,
}

fn run_once(config: &Config) -> Outcome {
    let matrix = transitions();
    let mut rng = StdRng::seed_from_u64(config.seed);

    let build_start = Instant::now();
    let mut base = Vec::with_capacity(config.agents);
    let shared_matrix = Arc::new(Transitions::compile(matrix));
    for index in 0..config.agents {
        let id = format!("trader_{index:07}");
        base.push(if config.shared {
            Agent::with_shared_transitions(
                id,
                TraderMode::Flat,
                Arc::clone(&shared_matrix),
                &mut rng,
            )
        } else {
            // the unshared path compiles its own copy of the table, which is the cost being measured
            Agent::new(id, TraderMode::Flat, transitions(), &mut rng)
        });
    }

    if config.observing {
        // agents are laid out on a line so a proximity radius selects a predictable neighbourhood
        let agents: Vec<_> = base
            .into_iter()
            .enumerate()
            .map(|(index, agent)| ObservingTrader::new(agent, Position::new(index as f64, 0.0)))
            .collect();
        let build_sec = build_start.elapsed().as_secs_f64();
        drive(config, agents, build_sec)
    } else {
        let build_sec = build_start.elapsed().as_secs_f64();
        drive(config, base, build_sec)
    }
}

fn drive<A: SimAgent>(config: &Config, agents: Vec<A>, build_sec: f64) -> Outcome
where
    A::World: Default,
{
    let (rss_after_build_mb, _) = memory_mb();

    let mut sim = Simulation::new_with_seed(agents, start_time(), config.seed);
    sim.set_perception(config.perception);

    let horizon = Duration::minutes(config.minutes);
    let run_start = Instant::now();

    let (events, aborted) = if config.collect {
        // the same retained log as `run`, but accumulated once by the caller, so the difference
        // against --keep-log is exactly what `run`'s clone-on-return costs
        let mut log = Vec::new();
        sim.run_streaming(horizon, |event| log.push(event));
        (log.len() as u64, false)
    } else if config.retain_log {
        // the accumulating path: every event is kept, then the whole log is cloned on return
        let log = sim.run(horizon);
        (log.len() as u64, false)
    } else {
        // unpaced live pacing is the streaming path plus an abort hatch, so a configuration that
        // is too slow reports a partial result instead of running until the user gives up
        let live = Live::unpaced().until(horizon);
        let mut events: u64 = 0;
        let outcome = sim.run_live(live, |_| {
            events += 1;
            if events.is_multiple_of(65_536)
                && run_start.elapsed().as_secs_f64() > config.budget_sec
            {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        });
        (events, outcome == LiveOutcome::Halted)
    };

    let run_sec = run_start.elapsed().as_secs_f64();
    let (_, peak_rss_mb) = memory_mb();

    Outcome {
        build_sec,
        run_sec,
        events,
        aborted,
        rss_after_build_mb,
        peak_rss_mb,
    }
}

fn report(config: &Config, outcome: &Outcome) {
    let rate = if outcome.run_sec > 0.0 {
        outcome.events as f64 / outcome.run_sec
    } else {
        0.0
    };
    let per_agent_ns = if outcome.events > 0 {
        outcome.run_sec * 1e9 / outcome.events as f64
    } else {
        0.0
    };

    println!(
        "{agents:>8} | {perception:>9} | {observe:>7} | {matrix:>6} | {log:>7} | {build:>8.2} | {run:>9.2} | {events:>12} | {rate:>13.0} | {per_event:>10.0} | {rss:>9.1} | {peak:>9.1} | {status}",
        agents = config.agents,
        perception = config.perception_label(),
        observe = if config.observing { "yes" } else { "no" },
        matrix = if config.shared { "shared" } else { "owned" },
        log = match (config.retain_log, config.collect) {
            (_, true) => "collect",
            (true, _) => "keep",
            _ => "stream",
        },
        build = outcome.build_sec,
        run = outcome.run_sec,
        events = outcome.events,
        rate = rate,
        per_event = per_agent_ns,
        rss = outcome.rss_after_build_mb,
        peak = outcome.peak_rss_mb,
        status = if outcome.aborted { "ABORTED" } else { "ok" },
    );
}

fn header() {
    println!(
        "{:>8} | {:>9} | {:>7} | {:>6} | {:>7} | {:>8} | {:>9} | {:>12} | {:>13} | {:>10} | {:>9} | {:>9} | status",
        "agents",
        "perceive",
        "observe",
        "matrix",
        "log",
        "build s",
        "run s",
        "events",
        "events/s",
        "ns/event",
        "build MB",
        "peak MB"
    );
    println!("{}", "-".repeat(140));
}

fn parse_args() -> (Config, Option<Vec<usize>>) {
    let mut agents = 1_000usize;
    let mut minutes = 60i64;
    let mut perception = Perception::SelfOnly;
    let mut retain_log = false;
    let mut collect = false;
    let mut observing = false;
    let mut shared = false;
    let mut budget_sec = 20.0;
    let mut seed = 7u64;
    let mut sweep = None;

    let args: Vec<String> = std::env::args().skip(1).collect();
    let value = |index: usize| -> String {
        args.get(index + 1)
            .unwrap_or_else(|| panic!("{} needs a value", args[index]))
            .clone()
    };

    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--agents" => {
                agents = value(index).parse().expect("--agents must be a number");
                index += 1;
            }
            "--minutes" => {
                minutes = value(index).parse().expect("--minutes must be a number");
                index += 1;
            }
            "--budget" => {
                budget_sec = value(index).parse().expect("--budget must be a number");
                index += 1;
            }
            "--seed" => {
                seed = value(index).parse().expect("--seed must be a number");
                index += 1;
            }
            "--perception" => {
                let raw = value(index);
                perception = match raw.as_str() {
                    "self" => Perception::SelfOnly,
                    "global" => Perception::Global,
                    other => match other.strip_prefix("prox:") {
                        Some(radius) => Perception::Proximity {
                            radius: radius.parse().expect("proximity radius must be a number"),
                        },
                        None => panic!("unknown perception {other}"),
                    },
                };
                index += 1;
            }
            "--keep-log" => retain_log = true,
            "--collect" => collect = true,
            "--observe" => observing = true,
            "--shared" => shared = true,
            "--sweep" => {
                sweep = Some(
                    value(index)
                        .split(',')
                        .map(|part| part.trim().parse().expect("--sweep takes agent counts"))
                        .collect(),
                );
                index += 1;
            }
            "--header-only" => {
                header();
                std::process::exit(0);
            }
            other => panic!("unknown flag {other}"),
        }
        index += 1;
    }

    (
        Config {
            agents,
            minutes,
            perception,
            retain_log,
            collect,
            observing,
            shared,
            budget_sec,
            seed,
        },
        sweep,
    )
}

// re-runs this example once per agent count so each measurement gets a clean address space
fn sweep(config: &Config, counts: &[usize]) {
    let exe = std::env::current_exe().expect("current exe");
    header();

    for &count in counts {
        let mut command = std::process::Command::new(&exe);
        command
            .arg("--agents")
            .arg(count.to_string())
            .arg("--minutes")
            .arg(config.minutes.to_string())
            .arg("--budget")
            .arg(config.budget_sec.to_string())
            .arg("--seed")
            .arg(config.seed.to_string())
            .arg("--perception")
            .arg(config.perception_label());
        if config.retain_log {
            command.arg("--keep-log");
        }
        if config.collect {
            command.arg("--collect");
        }
        if config.observing {
            command.arg("--observe");
        }
        if config.shared {
            command.arg("--shared");
        }

        let status = command.status().expect("failed to launch child run");
        if !status.success() {
            println!("{count:>8} | run failed: {status}");
            break;
        }
    }
}

fn main() {
    let (config, counts) = parse_args();

    match counts {
        Some(counts) => sweep(&config, &counts),
        None => {
            let outcome = run_once(&config);
            report(&config, &outcome);
        }
    }
}
