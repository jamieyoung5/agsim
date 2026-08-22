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

#[derive(Debug, Clone, PartialEq, Eq, Hash, Copy)]
enum Mood {
    Watching,
    Buying,
    Selling,
}

#[derive(Debug, Clone, Default, State, StateDisplay)]
struct Quote {
    price: i64,
    size: u32,
}

// traders read this instead
#[derive(Default)]
struct Tape {
    last_price: i64,
    prints: u64,
}

impl World for Tape {
    fn absorb(&mut self, event: &StateChangeEvent) {
        if event.field == "price"
            && let Value::Int(price) = event.new_value
        {
            self.last_price = price;
            self.prints += 1;
        }
    }
}

struct Trader {
    inner: Agent<Mood, Quote>,
    watching: StateId,
}

impl SimAgent for Trader {
    type State = StateId;
    type World = Tape;

    fn peek_next_event_delay(
        &self,
        now: DateTime<Utc>,
        _tape: &Tape,
        rng: &mut dyn RngCore,
    ) -> Option<f64> {
        self.inner.peek_next_event_delay(now, &(), rng)
    }

    fn step(&self, now: DateTime<Utc>, tape: &Tape, rng: &mut dyn RngCore) -> Option<StateId> {
        let intended = self.inner.step(now, &(), rng)?;

        // under our quote, sit out
        if tape.last_price < self.inner.data.price {
            Some(self.watching)
        } else {
            Some(intended)
        }
    }

    fn apply_transition(
        &mut self,
        next: StateId,
        time: DateTime<Utc>,
        _tape: &Tape,
        rng: &mut dyn RngCore,
        out: &mut Vec<StateChangeEvent>,
    ) {
        self.inner.apply_transition(next, time, &(), rng, out);
    }
}

fn transitions() -> HashMap<Mood, StateType<Mood, Quote>> {
    HashMap::from([
        (
            Mood::Watching,
            StateType::new(
                |rng| Quote {
                    price: rng.gen_range(990..1010),
                    size: 0,
                },
                vec![(Mood::Buying, 0.5), (Mood::Selling, 0.5)],
                90.0,
            ),
        ),
        (
            Mood::Buying,
            StateType::new(
                |rng| Quote {
                    price: rng.gen_range(1000..1030),
                    size: rng.gen_range(1..50),
                },
                vec![(Mood::Watching, 0.7), (Mood::Selling, 0.3)],
                45.0,
            ),
        ),
        (
            Mood::Selling,
            StateType::new(
                |rng| Quote {
                    price: rng.gen_range(970..1000),
                    size: rng.gen_range(1..50),
                },
                vec![(Mood::Watching, 0.7), (Mood::Buying, 0.3)],
                45.0,
            ),
        ),
    ])
}

fn main() {
    // compiled once, shared by all
    let table = Arc::new(Transitions::compile(transitions()));
    let watching = table.index(&Mood::Watching).expect("Watching is defined");

    let mut rng = StdRng::seed_from_u64(7);
    let traders: Vec<Trader> = (0..500)
        .map(|i| Trader {
            inner: Agent::with_shared_transitions(
                format!("trader_{i:03}"),
                Mood::Watching,
                Arc::clone(&table),
                &mut rng,
            ),
            watching,
        })
        .collect();

    let start = Utc.with_ymd_and_hms(2026, 1, 5, 9, 0, 0).unwrap();
    let mut sim = Simulation::with_world(traders, start, 7, Tape::default());
    let events = sim.run(Duration::hours(1));

    let price_changes = events.iter().filter(|e| e.field == "price").count();

    println!(
        "{} traders, 1 hour, {} events",
        sim.agents().len(),
        events.len()
    );
    println!("last price {}", sim.world().last_price);
    println!(
        "{} price changes, folded into the tape {} times",
        price_changes,
        sim.world().prints
    );
}
