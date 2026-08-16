use crate::agent::SimAgent;
use crate::clock::{Clock, Live, LiveOutcome, STOP_POLL_INTERVAL, SystemClock};
use crate::rng;
use crate::state::StateChangeEvent;
use chrono::{DateTime, Duration, Utc};
use rand::rngs::StdRng;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::ops::ControlFlow;
use std::time::{Duration as StdDuration, Instant};

/// Controls which agents are fed an event emitted by another.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum Perception {
    #[default]
    SelfOnly,
    Global,
    Proximity {
        radius: f64,
    },
}

struct ScheduledEvent<C> {
    time: DateTime<Utc>,
    agent_index: usize,
    next_state_type: Option<C>,
}

impl<C> PartialEq for ScheduledEvent<C> {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl<C> Eq for ScheduledEvent<C> {}
impl<C> PartialOrd for ScheduledEvent<C> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl<C> Ord for ScheduledEvent<C> {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .time
            .cmp(&self.time)
            .then_with(|| other.agent_index.cmp(&self.agent_index))
    }
}

pub struct Simulation<A: SimAgent> {
    agents: Vec<A>,
    current_time: DateTime<Utc>,
    event_log: Vec<StateChangeEvent>,
    seed: u64,
    streams: Vec<StdRng>,
    perception: Perception,
}

impl<A: SimAgent> Simulation<A> {
    pub fn new(agents: Vec<A>, start_time: DateTime<Utc>) -> Self {
        Self::new_with_seed(agents, start_time, rng::random_seed())
    }

    pub fn new_with_seed(agents: Vec<A>, start_time: DateTime<Utc>, seed: u64) -> Self {
        let streams = (0..agents.len())
            .map(|index| rng::substream(seed, index as u64))
            .collect();

        Simulation {
            agents,
            current_time: start_time,
            event_log: Vec::new(),
            seed,
            streams,
            perception: Perception::default(),
        }
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    pub fn set_perception(&mut self, perception: Perception) {
        self.perception = perception;
    }

    pub fn agents(&self) -> &[A] {
        &self.agents
    }

    pub fn current_time(&self) -> DateTime<Utc> {
        self.current_time
    }

    pub fn run(&mut self, duration: Duration) -> Vec<StateChangeEvent> {
        let end_time = self.current_time + duration;
        let mut queue = self.initialize_queue();

        while let Some(event) = queue.pop() {
            if event.time > end_time {
                break;
            }
            self.process_event_step(event, &mut queue, |changes, log| {
                log.extend(changes);
            });
        }

        self.event_log.clone()
    }

    /// Like [`run`](Self::run), but hands each event to `callback` instead of accumulating a log.
    pub fn run_streaming<F>(&mut self, duration: Duration, mut callback: F)
    where
        F: FnMut(StateChangeEvent),
    {
        let end_time = self.current_time + duration;
        let mut queue = self.initialize_queue();

        while let Some(event) = queue.pop() {
            if event.time > end_time {
                break;
            }

            self.process_event_step(event, &mut queue, |changes, _| {
                for change in changes {
                    callback(change);
                }
            });
        }
    }

    /// Runs against the wall clock, open-ended, rather than over a fixed stretch of simulated time.
    pub fn run_live<F>(&mut self, live: Live, callback: F) -> LiveOutcome
    where
        F: FnMut(StateChangeEvent) -> ControlFlow<()>,
    {
        self.run_live_with_clock(live, &SystemClock, callback)
    }

    pub fn run_live_with_clock<C, F>(
        &mut self,
        live: Live,
        clock: &C,
        mut callback: F,
    ) -> LiveOutcome
    where
        C: Clock + ?Sized,
        F: FnMut(StateChangeEvent) -> ControlFlow<()>,
    {
        let sim_start = self.current_time;
        let wall_start = clock.now();
        let deadline = live.horizon().map(|horizon| sim_start + horizon);
        let mut queue = self.initialize_queue();

        loop {
            if live.stop().is_stopped() {
                return LiveOutcome::Stopped;
            }

            let Some(event) = queue.pop() else {
                return LiveOutcome::Drained;
            };

            if deadline.is_some_and(|end| event.time > end) {
                return LiveOutcome::HorizonReached;
            }

            if !Self::wait_for(clock, &live, wall_start, sim_start, event.time) {
                return LiveOutcome::Stopped;
            }

            let mut flow = ControlFlow::Continue(());
            self.process_event_step(event, &mut queue, |changes, _| {
                for change in changes {
                    if flow.is_continue() {
                        flow = callback(change);
                    }
                }
            });

            if flow.is_break() {
                return LiveOutcome::Halted;
            }
        }
    }

    // holds until the wall clock reaches the point `target` maps to under the run's speed
    fn wait_for<C: Clock + ?Sized>(
        clock: &C,
        live: &Live,
        wall_start: Instant,
        sim_start: DateTime<Utc>,
        target: DateTime<Utc>,
    ) -> bool {
        let offset = (target - sim_start).num_milliseconds() as f64 / 1000.0 / live.speed();
        if !offset.is_finite() || offset <= 0.0 {
            return !live.stop().is_stopped();
        }

        let deadline = wall_start + StdDuration::from_secs_f64(offset);
        loop {
            if live.stop().is_stopped() {
                return false;
            }

            let now = clock.now();
            if now >= deadline {
                return true;
            }

            clock.sleep((deadline - now).min(STOP_POLL_INTERVAL));
        }
    }

    fn initialize_queue(&mut self) -> BinaryHeap<ScheduledEvent<A::State>> {
        let mut queue = BinaryHeap::new();
        for index in 0..self.agents.len() {
            self.schedule_next_event(index, &mut queue);
        }
        queue
    }

    fn process_event_step<F>(
        &mut self,
        event: ScheduledEvent<A::State>,
        queue: &mut BinaryHeap<ScheduledEvent<A::State>>,
        mut handler: F,
    ) where
        F: FnMut(Vec<StateChangeEvent>, &mut Vec<StateChangeEvent>),
    {
        self.current_time = event.time;

        if let Some(target_type) = event.next_state_type {
            let agent_index = event.agent_index;

            let changes = self.agents[agent_index].apply_transition(
                target_type,
                self.current_time,
                &mut self.streams[agent_index],
            );

            for change in &changes {
                for observer in 0..self.agents.len() {
                    let perceives = match self.perception {
                        Perception::SelfOnly => observer == agent_index,
                        Perception::Global => true,
                        Perception::Proximity { radius } => {
                            match (
                                self.agents[observer].location(),
                                self.agents[agent_index].location(),
                            ) {
                                (Some(observer_pos), Some(emitter_pos)) => {
                                    observer_pos.distance(&emitter_pos) <= radius
                                }
                                _ => false,
                            }
                        }
                    };
                    if perceives {
                        self.agents[observer].observe(change);
                    }
                }
            }

            handler(changes, &mut self.event_log);

            self.schedule_next_event(agent_index, queue);
        }
    }

    fn seconds_to_duration(seconds: f64) -> Duration {
        let millis = (seconds * 1000.0).round() as i64;
        Duration::milliseconds(millis)
    }

    fn schedule_next_event(
        &mut self,
        agent_index: usize,
        queue: &mut BinaryHeap<ScheduledEvent<A::State>>,
    ) {
        let Some(delay_sec) = self.agents[agent_index]
            .peek_next_event_delay(self.current_time, &mut self.streams[agent_index])
        else {
            return;
        };
        let Some(next_state) =
            self.agents[agent_index].step(self.current_time, &mut self.streams[agent_index])
        else {
            return;
        };

        let event_time = self.current_time + Self::seconds_to_duration(delay_sec);
        queue.push(ScheduledEvent {
            time: event_time,
            agent_index,
            next_state_type: Some(next_state),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::{Agent, StateType};
    use crate::state::{State, StateChangeEvent};
    use rand::{RngCore, SeedableRng};
    use std::collections::HashMap;

    #[derive(Clone, Default, Debug, PartialEq)]
    struct MockState {
        counter: usize,
    }

    impl State for MockState {
        fn diff(&self, other: &Self, time: DateTime<Utc>) -> Vec<StateChangeEvent> {
            if self.counter != other.counter {
                vec![StateChangeEvent {
                    time,
                    agent_id: String::new(),
                    field: "counter".to_string(),
                    old_value: self.counter.to_string(),
                    new_value: other.counter.to_string(),
                }]
            } else {
                vec![]
            }
        }
    }

    #[derive(Eq, Hash, PartialEq, Clone, Debug)]
    enum SimState {
        Step1,
        Step2,
    }

    // ObservingAgent fires a single transition and records every event fed back through observe.
    struct ObservingAgent {
        id: String,
        fired: bool,
        observed: Vec<String>,
        location: Option<crate::space::Position>,
    }

    impl ObservingAgent {
        fn new() -> Self {
            Self::named("obs")
        }

        fn named(id: &str) -> Self {
            ObservingAgent {
                id: id.to_string(),
                fired: false,
                observed: Vec::new(),
                location: None,
            }
        }

        fn at(position: crate::space::Position) -> Self {
            ObservingAgent {
                location: Some(position),
                ..Self::new()
            }
        }
    }

    impl SimAgent for ObservingAgent {
        type State = ();

        fn peek_next_event_delay(
            &self,
            _now: DateTime<Utc>,
            _rng: &mut dyn RngCore,
        ) -> Option<f64> {
            if self.fired { None } else { Some(1.0) }
        }

        fn step(&self, _now: DateTime<Utc>, _rng: &mut dyn RngCore) -> Option<()> {
            if self.fired { None } else { Some(()) }
        }

        fn apply_transition(
            &mut self,
            _next: (),
            time: DateTime<Utc>,
            _rng: &mut dyn RngCore,
        ) -> Vec<StateChangeEvent> {
            self.fired = true;
            vec![StateChangeEvent {
                time,
                agent_id: self.id.clone(),
                field: "state".to_string(),
                old_value: "0".to_string(),
                new_value: "1".to_string(),
            }]
        }

        fn observe(&mut self, event: &StateChangeEvent) {
            self.observed.push(event.field.clone());
        }

        fn location(&self) -> Option<crate::space::Position> {
            self.location
        }
    }

    #[test]
    fn test_simulation_queue_ordering() {
        let time = Utc::now();

        let event_early = ScheduledEvent {
            time: time,
            agent_index: 0,
            next_state_type: Some(1),
        };

        let event_late = ScheduledEvent {
            time: time + Duration::seconds(10),
            agent_index: 1,
            next_state_type: Some(1),
        };

        assert!(event_early > event_late);

        let mut heap = BinaryHeap::new();
        heap.push(event_late);
        heap.push(event_early);

        let popped = heap.pop().unwrap();
        assert_eq!(popped.agent_index, 0);
    }

    #[test]
    fn test_simulation_run_flow() {
        let start_time = Utc::now();
        let mut rng = StdRng::seed_from_u64(123);
        let mut transitions = HashMap::new();

        transitions.insert(
            SimState::Step1,
            StateType::new_deterministic(
                || MockState { counter: 1 },
                vec![(SimState::Step2, 1.0)],
                0.1,
            ),
        );
        transitions.insert(
            SimState::Step2,
            StateType::new_deterministic(
                || MockState { counter: 2 },
                vec![(SimState::Step1, 1.0)],
                0.1,
            ),
        );

        let agent = Agent::new(
            "sim_agent".to_string(),
            SimState::Step1,
            transitions,
            &mut rng,
        );

        let mut sim = Simulation::new(vec![agent], start_time);

        let events = sim.run(Duration::seconds(1));

        assert!(
            !events.is_empty(),
            "Events list should not be empty with a 0.1s mean delay over 1s duration"
        );

        let first_event = &events[0];
        assert_eq!(first_event.agent_id, "sim_agent");
        assert_eq!(first_event.field, "counter");

        assert!(sim.current_time > start_time);
    }

    #[test]
    fn test_simulation_termination() {
        let start_time = Utc::now();
        let mut rng = StdRng::seed_from_u64(123);
        let mut transitions = HashMap::new();

        transitions.insert(
            SimState::Step1,
            StateType::new_deterministic(|| MockState { counter: 1 }, vec![], 1.0),
        );

        let agent = Agent::new(
            "term_agent".to_string(),
            SimState::Step1,
            transitions,
            &mut rng,
        );

        let mut sim = Simulation::new(vec![agent], start_time);

        let events = sim.run(Duration::hours(1));
        assert!(events.is_empty());
    }

    #[test]
    fn test_observation_feed() {
        let start_time = Utc::now();
        let mut sim = Simulation::new(vec![ObservingAgent::new()], start_time);
        sim.run(Duration::seconds(10));

        assert_eq!(sim.agents()[0].observed, vec!["state".to_string()]);
    }

    #[test]
    fn test_global_perception() {
        let start_time = Utc::now();
        let agents = vec![ObservingAgent::new(), ObservingAgent::new()];

        let mut sim = Simulation::new(agents, start_time);
        sim.set_perception(Perception::Global);
        sim.run(Duration::seconds(10));

        assert_eq!(sim.agents()[0].observed.len(), 2);
        assert_eq!(sim.agents()[1].observed.len(), 2);
    }

    #[test]
    fn test_proximity_perception() {
        use crate::space::Position;

        let start_time = Utc::now();
        let agents = vec![
            ObservingAgent::at(Position::new(0.0, 0.0)),
            ObservingAgent::at(Position::new(1.0, 0.0)),
            ObservingAgent::at(Position::new(100.0, 0.0)),
        ];

        let mut sim = Simulation::new(agents, start_time);
        sim.set_perception(Perception::Proximity { radius: 5.0 });
        sim.run(Duration::seconds(10));

        assert_eq!(sim.agents()[0].observed.len(), 2);
        assert_eq!(sim.agents()[1].observed.len(), 2);
        assert_eq!(sim.agents()[2].observed.len(), 1);
    }

    fn markov_agent(id: &str) -> Agent<SimState, MockState> {
        let mut rng = StdRng::seed_from_u64(9);
        let mut transitions = HashMap::new();

        transitions.insert(
            SimState::Step1,
            StateType::new_deterministic(
                || MockState { counter: 1 },
                vec![(SimState::Step2, 0.7), (SimState::Step1, 0.3)],
                0.1,
            ),
        );
        transitions.insert(
            SimState::Step2,
            StateType::new_deterministic(
                || MockState { counter: 2 },
                vec![(SimState::Step1, 0.7), (SimState::Step2, 0.3)],
                0.1,
            ),
        );

        Agent::new(id.to_string(), SimState::Step1, transitions, &mut rng)
    }

    fn fingerprint(events: &[StateChangeEvent]) -> Vec<(DateTime<Utc>, String, String)> {
        events
            .iter()
            .map(|e| (e.time, e.agent_id.clone(), e.new_value.clone()))
            .collect()
    }

    fn base() -> DateTime<Utc> {
        chrono::TimeZone::timestamp_opt(&Utc, 1_600_000_000, 0).unwrap()
    }

    #[test]
    fn test_same_seed_replays_exactly() {
        let agents = || vec![markov_agent("a"), markov_agent("b"), markov_agent("c")];

        let mut first = Simulation::new_with_seed(agents(), base(), 4242);
        let mut second = Simulation::new_with_seed(agents(), base(), 4242);

        let left = first.run(Duration::seconds(60));
        let right = second.run(Duration::seconds(60));

        assert!(!left.is_empty());
        assert_eq!(fingerprint(&left), fingerprint(&right));
    }

    #[test]
    fn test_different_seed_diverges() {
        let agents = || vec![markov_agent("a"), markov_agent("b")];

        let mut first = Simulation::new_with_seed(agents(), base(), 1);
        let mut second = Simulation::new_with_seed(agents(), base(), 2);

        let left = first.run(Duration::seconds(60));
        let right = second.run(Duration::seconds(60));

        assert_ne!(fingerprint(&left), fingerprint(&right));
    }

    #[test]
    fn test_reported_seed_replays_an_unseeded_run() {
        let mut original = Simulation::new(vec![markov_agent("a")], base());
        let events = original.run(Duration::seconds(60));

        let mut replay =
            Simulation::new_with_seed(vec![markov_agent("a")], base(), original.seed());
        assert_eq!(
            fingerprint(&events),
            fingerprint(&replay.run(Duration::seconds(60)))
        );
    }

    #[test]
    fn test_agent_streams_are_independent() {
        let mut alone = Simulation::new_with_seed(vec![markov_agent("a")], base(), 77);
        let mut crowded = Simulation::new_with_seed(
            vec![markov_agent("a"), markov_agent("b"), markov_agent("c")],
            base(),
            77,
        );

        let solo = alone.run(Duration::seconds(60));
        let shared = crowded.run(Duration::seconds(60));

        let a_events: Vec<_> = shared.into_iter().filter(|e| e.agent_id == "a").collect();
        assert!(!solo.is_empty());
        assert_eq!(fingerprint(&solo), fingerprint(&a_events));
    }

    #[test]
    fn test_simultaneous_events_break_ties_by_agent() {
        let agents = vec![
            ObservingAgent::named("first"),
            ObservingAgent::named("second"),
            ObservingAgent::named("third"),
        ];

        let mut sim = Simulation::new_with_seed(agents, base(), 5);
        let events = sim.run(Duration::seconds(10));

        let ids: Vec<_> = events.iter().map(|e| e.agent_id.as_str()).collect();
        assert_eq!(ids, vec!["first", "second", "third"]);
    }

    struct MockClock {
        now: std::cell::Cell<Instant>,
        slept: std::cell::Cell<StdDuration>,
    }

    impl MockClock {
        fn new() -> Self {
            MockClock {
                now: std::cell::Cell::new(Instant::now()),
                slept: std::cell::Cell::new(StdDuration::ZERO),
            }
        }

        fn slept(&self) -> StdDuration {
            self.slept.get()
        }
    }

    impl Clock for MockClock {
        fn now(&self) -> Instant {
            self.now.get()
        }

        fn sleep(&self, duration: StdDuration) {
            self.now.set(self.now.get() + duration);
            self.slept.set(self.slept.get() + duration);
        }
    }

    // TickAgent transitions once a second, forever
    struct TickAgent {
        ticks: u32,
    }

    impl SimAgent for TickAgent {
        type State = ();

        fn peek_next_event_delay(
            &self,
            _now: DateTime<Utc>,
            _rng: &mut dyn RngCore,
        ) -> Option<f64> {
            Some(1.0)
        }

        fn step(&self, _now: DateTime<Utc>, _rng: &mut dyn RngCore) -> Option<()> {
            Some(())
        }

        fn apply_transition(
            &mut self,
            _next: (),
            time: DateTime<Utc>,
            _rng: &mut dyn RngCore,
        ) -> Vec<StateChangeEvent> {
            self.ticks += 1;
            vec![StateChangeEvent {
                time,
                agent_id: "tick".to_string(),
                field: "ticks".to_string(),
                old_value: (self.ticks - 1).to_string(),
                new_value: self.ticks.to_string(),
            }]
        }
    }

    #[test]
    fn test_live_paces_against_the_clock() {
        let clock = MockClock::new();
        let mut sim = Simulation::new_with_seed(vec![TickAgent { ticks: 0 }], base(), 1);

        let mut events = Vec::new();
        let outcome = sim.run_live_with_clock(
            Live::at_speed(2.0).until(Duration::seconds(5)),
            &clock,
            |event| {
                events.push(event);
                ControlFlow::Continue(())
            },
        );

        assert_eq!(outcome, LiveOutcome::HorizonReached);
        assert_eq!(events.len(), 5);
        assert_eq!(events.last().unwrap().time, base() + Duration::seconds(5));
        assert!((clock.slept().as_secs_f64() - 2.5).abs() < 0.1);
    }

    #[test]
    fn test_live_unpaced_never_waits() {
        let clock = MockClock::new();
        let mut sim = Simulation::new_with_seed(vec![TickAgent { ticks: 0 }], base(), 1);

        let outcome =
            sim.run_live_with_clock(Live::unpaced().until(Duration::hours(1)), &clock, |_| {
                ControlFlow::Continue(())
            });

        assert_eq!(outcome, LiveOutcome::HorizonReached);
        assert_eq!(clock.slept(), StdDuration::ZERO);
        assert_eq!(sim.agents()[0].ticks, 3600);
    }

    #[test]
    fn test_live_stops_on_signal() {
        let clock = MockClock::new();
        let mut sim = Simulation::new_with_seed(vec![TickAgent { ticks: 0 }], base(), 1);

        let live = Live::real_time();
        let stop = live.stop_signal();

        let mut seen = 0;
        let outcome = sim.run_live_with_clock(live, &clock, |_| {
            seen += 1;
            if seen == 3 {
                stop.stop();
            }
            ControlFlow::Continue(())
        });

        assert_eq!(outcome, LiveOutcome::Stopped);
        assert_eq!(seen, 3);
    }

    #[test]
    fn test_live_halts_on_callback() {
        let clock = MockClock::new();
        let mut sim = Simulation::new_with_seed(vec![TickAgent { ticks: 0 }], base(), 1);

        let mut seen = 0;
        let outcome = sim.run_live_with_clock(Live::at_speed(60.0), &clock, |_| {
            seen += 1;
            if seen == 2 {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        });

        assert_eq!(outcome, LiveOutcome::Halted);
        assert_eq!(seen, 2);
    }

    #[test]
    fn test_live_drains_when_agents_are_done() {
        let clock = MockClock::new();
        let mut sim = Simulation::new_with_seed(vec![ObservingAgent::new()], base(), 1);

        let mut seen = 0;
        let outcome = sim.run_live_with_clock(Live::unpaced(), &clock, |_| {
            seen += 1;
            ControlFlow::Continue(())
        });

        assert_eq!(outcome, LiveOutcome::Drained);
        assert_eq!(seen, 1);
    }

    #[test]
    fn test_live_advances_current_time() {
        let clock = MockClock::new();
        let mut sim = Simulation::new_with_seed(vec![TickAgent { ticks: 0 }], base(), 1);

        sim.run_live_with_clock(Live::unpaced().until(Duration::seconds(10)), &clock, |_| {
            ControlFlow::Continue(())
        });

        assert_eq!(sim.current_time(), base() + Duration::seconds(10));
    }

    #[test]
    fn test_live_is_deterministic_across_speeds() {
        let fast = MockClock::new();
        let slow = MockClock::new();

        let mut a = Simulation::new_with_seed(vec![markov_agent("a")], base(), 31);
        let mut b = Simulation::new_with_seed(vec![markov_agent("a")], base(), 31);

        let mut left = Vec::new();
        a.run_live_with_clock(
            Live::unpaced().until(Duration::seconds(30)),
            &fast,
            |event| {
                left.push(event);
                ControlFlow::Continue(())
            },
        );

        let mut right = Vec::new();
        b.run_live_with_clock(
            Live::at_speed(1000.0).until(Duration::seconds(30)),
            &slow,
            |event| {
                right.push(event);
                ControlFlow::Continue(())
            },
        );

        assert!(!left.is_empty());
        assert_eq!(fingerprint(&left), fingerprint(&right));
        assert!(slow.slept() > StdDuration::ZERO);
    }
}
