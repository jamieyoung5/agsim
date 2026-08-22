// agent-count scaling harness

use agsim::agent::{Agent, SimAgent, StateId, StateType, Transitions};
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

    matrix.insert(
        Mode::Idle,
        StateType::new(
            |rng| Metrics {
                level: 0,
                pending: 0,
                signal: rng.gen_range(9_000..11_000),
                total: 0,
            },
            vec![(Mode::Active, 0.45), (Mode::Busy, 0.45), (Mode::Idle, 0.10)],
            45.0,
        ),
    );

    matrix.insert(
        Mode::Active,
        StateType::new(
            |rng| Metrics {
                level: rng.gen_range(1..500),
                pending: rng.gen_range(1..8),
                signal: rng.gen_range(9_000..11_000),
                total: rng.gen_range(1_000..250_000),
            },
            vec![
                (Mode::Blocked, 0.50),
                (Mode::Busy, 0.30),
                (Mode::Idle, 0.20),
            ],
            20.0,
        ),
    );

    matrix.insert(
        Mode::Busy,
        StateType::new(
            |rng| Metrics {
                level: rng.gen_range(-500..0),
                pending: rng.gen_range(1..8),
                signal: rng.gen_range(9_000..11_000),
                total: rng.gen_range(1_000..250_000),
            },
            vec![
                (Mode::Blocked, 0.50),
                (Mode::Active, 0.30),
                (Mode::Idle, 0.20),
            ],
            20.0,
        ),
    );

    matrix.insert(
        Mode::Blocked,
        StateType::new(
            |rng| Metrics {
                level: rng.gen_range(-800..800),
                pending: rng.gen_range(0..3),
                signal: rng.gen_range(9_000..11_000),
                total: rng.gen_range(0..400_000),
            },
            vec![
                (Mode::Active, 0.35),
                (Mode::Busy, 0.35),
                (Mode::Blocked, 0.30),
            ],
            90.0,
        ),
    );

    matrix
}

/// An agent that consumes events.
struct ObservingAgent {
    inner: Agent<Mode, Metrics>,
    seen: u64,
    last_signal: i64,
    at: Position,
}

impl ObservingAgent {
    fn new(inner: Agent<Mode, Metrics>, at: Position) -> Self {
        ObservingAgent {
            inner,
            seen: 0,
            last_signal: 0,
            at,
        }
    }
}

impl SimAgent for ObservingAgent {
    type State = StateId;
    type World = ();

    fn peek_next_event_delay(
        &self,
        now: DateTime<Utc>,
        world: &(),
        rng: &mut dyn RngCore,
    ) -> Option<f64> {
        self.inner.peek_next_event_delay(now, world, rng)
    }

    fn step(&self, now: DateTime<Utc>, world: &(), rng: &mut dyn RngCore) -> Option<StateId> {
        self.inner.step(now, world, rng)
    }

    fn apply_transition(
        &mut self,
        next: StateId,
        time: DateTime<Utc>,
        world: &(),
        rng: &mut dyn RngCore,
        out: &mut Vec<StateChangeEvent>,
    ) {
        self.inner.apply_transition(next, time, world, rng, out)
    }

    fn observe(&mut self, event: &StateChangeEvent) {
        self.seen += 1;
        if event.field == "signal"
            && let Value::Int(signal) = event.new_value
        {
            self.last_signal = signal;
        }
    }

    fn location(&self) -> Option<Position> {
        Some(self.at)
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

// current and peak rss, mib
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
        let id = format!("agent_{index:07}");
        base.push(if config.shared {
            Agent::with_shared_transitions(id, Mode::Idle, Arc::clone(&shared_matrix), &mut rng)
        } else {
            Agent::new(id, Mode::Idle, transitions(), &mut rng)
        });
    }

    if config.observing {
        // laid out on a line
        let agents: Vec<_> = base
            .into_iter()
            .enumerate()
            .map(|(index, agent)| ObservingAgent::new(agent, Position::new(index as f64, 0.0)))
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
        let mut log = Vec::new();
        sim.run_streaming(horizon, |event| log.push(event));
        (log.len() as u64, false)
    } else if config.retain_log {
        let log = sim.run(horizon);
        (log.len() as u64, false)
    } else {
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

    let args = self::args();
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

// fresh process per agent count
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

// cargo bench passes --bench
fn args() -> Vec<String> {
    std::env::args()
        .skip(1)
        .filter(|a| a != "--bench")
        .collect()
}
