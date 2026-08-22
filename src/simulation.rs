use crate::agent::SimAgent;
use crate::clock::{Clock, Live, LiveOutcome, STOP_POLL_INTERVAL, SystemClock};
use crate::queue::{EventQueue, ScheduledEvent};
use crate::rng::{self, SimRng};
use crate::space::SpatialIndex;
use crate::state::StateChangeEvent;
use crate::world::World;
use chrono::{DateTime, Duration, Utc};
use std::ops::ControlFlow;
use std::time::{Duration as StdDuration, Instant};

/// Which agents see another's events.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum Perception {
    #[default]
    SelfOnly,
    Global,
    Proximity {
        radius: f64,
    },
}

pub struct Simulation<A: SimAgent> {
    agents: Vec<A>,
    world: A::World,
    origin: DateTime<Utc>,
    current_ms: i64,
    event_log: Vec<StateChangeEvent>,
    scratch: Vec<StateChangeEvent>,
    seed: u64,
    streams: Vec<SimRng>,
    perception: Perception,
    proximity: Option<SpatialIndex>,
}

impl<A: SimAgent> Simulation<A>
where
    A::World: Default,
{
    pub fn new(agents: Vec<A>, start_time: DateTime<Utc>) -> Self {
        Self::new_with_seed(agents, start_time, rng::random_seed())
    }

    pub fn new_with_seed(agents: Vec<A>, start_time: DateTime<Utc>, seed: u64) -> Self {
        Self::with_world(agents, start_time, seed, A::World::default())
    }
}

impl<A: SimAgent> Simulation<A> {
    pub fn with_world(
        agents: Vec<A>,
        start_time: DateTime<Utc>,
        seed: u64,
        world: A::World,
    ) -> Self {
        let streams = (0..agents.len())
            .map(|index| rng::substream(seed, index as u64))
            .collect();

        Simulation {
            agents,
            world,
            origin: start_time,
            current_ms: 0,
            event_log: Vec::new(),
            scratch: Vec::new(),
            seed,
            streams,
            perception: Perception::default(),
            proximity: None,
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

    pub fn world(&self) -> &A::World {
        &self.world
    }

    pub fn world_mut(&mut self) -> &mut A::World {
        &mut self.world
    }

    pub fn current_time(&self) -> DateTime<Utc> {
        self.time_at(self.current_ms)
    }

    /// Runs for `duration` and returns its events.
    pub fn run(&mut self, duration: Duration) -> Vec<StateChangeEvent> {
        let end_ms = self.current_ms.saturating_add(duration.num_milliseconds());
        let mut queue = self.initialize_queue();

        while let Some(event) = queue.pop() {
            if event.time_ms > end_ms {
                break;
            }
            self.process_event_step(event, &mut queue, |changes, log| {
                log.append(changes);
            });
        }

        std::mem::take(&mut self.event_log)
    }

    /// Like [`run`](Self::run), but streams to `callback`.
    pub fn run_streaming<F>(&mut self, duration: Duration, mut callback: F)
    where
        F: FnMut(StateChangeEvent),
    {
        let end_ms = self.current_ms.saturating_add(duration.num_milliseconds());
        let mut queue = self.initialize_queue();

        while let Some(event) = queue.pop() {
            if event.time_ms > end_ms {
                break;
            }

            self.process_event_step(event, &mut queue, |changes, _| {
                for change in changes.drain(..) {
                    callback(change);
                }
            });
        }
    }

    /// Runs open-ended against the wall clock.
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
        let start_ms = self.current_ms;
        let wall_start = clock.now();
        let deadline_ms = live
            .horizon()
            .map(|horizon| start_ms.saturating_add(horizon.num_milliseconds()));
        let mut queue = self.initialize_queue();

        loop {
            if live.stop().is_stopped() {
                return LiveOutcome::Stopped;
            }

            let Some(event) = queue.pop() else {
                return LiveOutcome::Drained;
            };

            if deadline_ms.is_some_and(|end| event.time_ms > end) {
                return LiveOutcome::HorizonReached;
            }

            if !Self::wait_for(clock, &live, wall_start, start_ms, event.time_ms) {
                return LiveOutcome::Stopped;
            }

            let mut flow = ControlFlow::Continue(());
            self.process_event_step(event, &mut queue, |changes, _| {
                for change in changes.drain(..) {
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

    // wait for the wall clock
    fn wait_for<C: Clock + ?Sized>(
        clock: &C,
        live: &Live,
        wall_start: Instant,
        start_ms: i64,
        target_ms: i64,
    ) -> bool {
        let offset = (target_ms - start_ms) as f64 / 1000.0 / live.speed();
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

    fn checked_time_at(&self, offset_ms: i64) -> Option<DateTime<Utc>> {
        self.origin
            .checked_add_signed(Duration::try_milliseconds(offset_ms)?)
    }

    fn time_at(&self, offset_ms: i64) -> DateTime<Utc> {
        self.checked_time_at(offset_ms)
            .expect("offsets are checked for representability when they are scheduled")
    }

    fn initialize_queue(&mut self) -> EventQueue<A::State> {
        self.proximity = match self.perception {
            Perception::Proximity { radius } => {
                SpatialIndex::build(radius, self.agents.iter().map(|agent| agent.location()))
            }
            _ => None,
        };

        let mut queue = EventQueue::with_capacity(self.agents.len());
        for index in 0..self.agents.len() {
            self.schedule_next_event(index, &mut queue);
        }
        queue
    }

    fn process_event_step<F>(
        &mut self,
        event: ScheduledEvent<A::State>,
        queue: &mut EventQueue<A::State>,
        mut handler: F,
    ) where
        F: FnMut(&mut Vec<StateChangeEvent>, &mut Vec<StateChangeEvent>),
    {
        self.current_ms = event.time_ms;

        let agent_index = event.agent_index as usize;
        let target_type = event.next_state_type;
        let now = self.time_at(self.current_ms);

        // avoid borrowing self twice
        let mut changes = std::mem::take(&mut self.scratch);
        changes.clear();

        self.agents[agent_index].apply_transition(
            target_type,
            now,
            &self.world,
            &mut self.streams[agent_index],
            &mut changes,
        );

        // only place a location changes
        if let Some(index) = self.proximity.as_mut() {
            let moved_to = self.agents[agent_index].location();
            index.place(agent_index, moved_to);
        }

        for change in &changes {
            self.world.absorb(change);
        }

        self.dispatch_observations(agent_index, &changes);
        handler(&mut changes, &mut self.event_log);

        self.scratch = changes;
        self.schedule_next_event(agent_index, queue);
    }

    // resolve perception once per observer
    fn dispatch_observations(&mut self, agent_index: usize, changes: &[StateChangeEvent]) {
        if changes.is_empty() {
            return;
        }

        match self.perception {
            Perception::SelfOnly => {
                let emitter = &mut self.agents[agent_index];
                for change in changes {
                    emitter.observe(change);
                }
            }
            Perception::Global => {
                for observer in &mut self.agents {
                    for change in changes {
                        observer.observe(change);
                    }
                }
            }
            Perception::Proximity { radius } => {
                let Some(emitter_pos) = self.agents[agent_index].location() else {
                    return;
                };
                let radius_squared = radius * radius;

                match self.proximity.as_ref() {
                    Some(index) => {
                        for observer in index.candidates(emitter_pos) {
                            let agent = &mut self.agents[observer];
                            let near = agent.location().is_some_and(|pos| {
                                pos.distance_squared(&emitter_pos) <= radius_squared
                            });
                            if near {
                                for change in changes {
                                    agent.observe(change);
                                }
                            }
                        }
                    }
                    // no grid, scan instead
                    None => {
                        for observer in &mut self.agents {
                            let near = observer.location().is_some_and(|pos| {
                                pos.distance_squared(&emitter_pos) <= radius_squared
                            });
                            if near {
                                for change in changes {
                                    observer.observe(change);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // unrepresentable times go unscheduled
    fn schedule_time(&self, delay_sec: f64) -> Option<i64> {
        if !delay_sec.is_finite() || delay_sec < 0.0 {
            return None;
        }

        let millis = (delay_sec * 1000.0).round();
        if millis > i64::MAX as f64 {
            return None;
        }

        let at = self.current_ms.checked_add(millis as i64)?;
        self.checked_time_at(at).is_some().then_some(at)
    }

    fn schedule_next_event(&mut self, agent_index: usize, queue: &mut EventQueue<A::State>) {
        let now = self.time_at(self.current_ms);

        let Some(delay_sec) = self.agents[agent_index].peek_next_event_delay(
            now,
            &self.world,
            &mut self.streams[agent_index],
        ) else {
            return;
        };
        let Some(next_state) =
            self.agents[agent_index].step(now, &self.world, &mut self.streams[agent_index])
        else {
            return;
        };

        let Some(time_ms) = self.schedule_time(delay_sec) else {
            return;
        };
        queue.push(ScheduledEvent {
            time_ms,
            agent_index: agent_index as u32,
            next_state_type: next_state,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::{Agent, StateType};
    use crate::space::Position;
    use crate::state::{AgentId, State, StateChangeEvent, ToValue, Value};
    use crate::world::World;
    use rand::RngCore;
    use std::borrow::Cow;
    use std::collections::HashMap;
    use std::sync::Arc;

    #[derive(Clone, Default, Debug, PartialEq)]
    struct MockState {
        counter: usize,
    }

    impl State for MockState {
        fn diff(
            &self,
            other: &Self,
            agent_id: &AgentId,
            time: DateTime<Utc>,
            out: &mut Vec<StateChangeEvent>,
        ) {
            if self.counter != other.counter {
                out.push(StateChangeEvent {
                    time,
                    agent_id: agent_id.clone(),
                    field: Cow::Borrowed("counter"),
                    old_value: self.counter.to_value(),
                    new_value: other.counter.to_value(),
                });
            }
        }
    }

    #[derive(Eq, Hash, PartialEq, Clone, Debug)]
    enum SimState {
        Step1,
        Step2,
    }

    struct ObservingAgent {
        id: AgentId,
        fired: bool,
        observed: Vec<String>,
        location: Option<Position>,
    }

    impl ObservingAgent {
        fn new() -> Self {
            Self::named("obs")
        }

        fn named(id: &str) -> Self {
            ObservingAgent {
                id: Arc::from(id),
                fired: false,
                observed: Vec::new(),
                location: None,
            }
        }

        fn at(position: Position) -> Self {
            ObservingAgent {
                location: Some(position),
                ..Self::new()
            }
        }
    }

    impl SimAgent for ObservingAgent {
        type State = ();
        type World = ();

        fn peek_next_event_delay(
            &self,
            _now: DateTime<Utc>,
            _world: &(),
            _rng: &mut dyn RngCore,
        ) -> Option<f64> {
            if self.fired { None } else { Some(1.0) }
        }

        fn step(&self, _now: DateTime<Utc>, _world: &(), _rng: &mut dyn RngCore) -> Option<()> {
            if self.fired { None } else { Some(()) }
        }

        fn apply_transition(
            &mut self,
            _next: (),
            time: DateTime<Utc>,
            _world: &(),
            _rng: &mut dyn RngCore,
            out: &mut Vec<StateChangeEvent>,
        ) {
            self.fired = true;
            out.push(StateChangeEvent {
                time,
                agent_id: self.id.clone(),
                field: Cow::Borrowed("state"),
                old_value: Value::Int(0),
                new_value: Value::Int(1),
            });
        }

        fn observe(&mut self, event: &StateChangeEvent) {
            self.observed.push(event.field.to_string());
        }

        fn location(&self) -> Option<Position> {
            self.location
        }
    }

    fn two_step_matrix(rate: f64) -> HashMap<SimState, StateType<SimState, MockState>> {
        HashMap::from([
            (
                SimState::Step1,
                StateType::new_deterministic(
                    || MockState { counter: 1 },
                    vec![(SimState::Step2, 1.0)],
                    rate,
                ),
            ),
            (
                SimState::Step2,
                StateType::new_deterministic(
                    || MockState { counter: 2 },
                    vec![(SimState::Step1, 1.0)],
                    rate,
                ),
            ),
        ])
    }

    #[test]
    fn test_simulation_run_flow() {
        let mut rng = SimRng::seed_from_u64(123);
        let agent = Agent::new("sim_agent", SimState::Step1, two_step_matrix(0.1), &mut rng);

        let mut sim = Simulation::new(vec![agent], base());
        let events = sim.run(Duration::seconds(1));

        assert!(
            !events.is_empty(),
            "Events list should not be empty with a 0.1s mean delay over 1s duration"
        );

        assert_eq!(&*events[0].agent_id, "sim_agent");
        assert_eq!(events[0].field, "counter");
        assert!(sim.current_time() > base());
    }

    #[test]
    fn test_simulation_termination() {
        let mut rng = SimRng::seed_from_u64(123);
        let transitions = HashMap::from([(
            SimState::Step1,
            StateType::new_deterministic(|| MockState { counter: 1 }, vec![], 1.0),
        )]);

        let agent = Agent::new("term_agent", SimState::Step1, transitions, &mut rng);
        let mut sim = Simulation::new(vec![agent], base());

        assert!(sim.run(Duration::hours(1)).is_empty());
    }

    #[test]
    fn test_observation_feed() {
        let mut sim = Simulation::new(vec![ObservingAgent::new()], base());
        sim.run(Duration::seconds(10));

        assert_eq!(sim.agents()[0].observed, vec!["state".to_string()]);
    }

    #[test]
    fn test_global_perception() {
        let agents = vec![ObservingAgent::new(), ObservingAgent::new()];

        let mut sim = Simulation::new(agents, base());
        sim.set_perception(Perception::Global);
        sim.run(Duration::seconds(10));

        assert_eq!(sim.agents()[0].observed.len(), 2);
        assert_eq!(sim.agents()[1].observed.len(), 2);
    }

    #[test]
    fn test_run_reports_own_events() {
        let mut sim = Simulation::new_with_seed(vec![markov_agent("a")], base(), 21);

        let first = sim.run(Duration::seconds(30));
        let second = sim.run(Duration::seconds(30));

        assert!(!first.is_empty());
        assert!(!second.is_empty());
        assert!(second.iter().all(|event| !first.contains(event)));
    }

    #[test]
    fn test_proximity_includes_radius_edge() {
        let agents = vec![
            ObservingAgent::at(Position::new(0.0, 0.0)),
            ObservingAgent::at(Position::new(3.0, 4.0)),
        ];

        let mut sim = Simulation::new_with_seed(agents, base(), 3);
        sim.set_perception(Perception::Proximity { radius: 5.0 });
        sim.run(Duration::seconds(10));

        assert_eq!(sim.agents()[0].observed.len(), 2);
        assert_eq!(sim.agents()[1].observed.len(), 2);
    }

    #[test]
    fn test_proximity_emitter_without_position() {
        let agents = vec![
            ObservingAgent::new(),
            ObservingAgent::at(Position::new(0.0, 0.0)),
        ];

        let mut sim = Simulation::new_with_seed(agents, base(), 3);
        sim.set_perception(Perception::Proximity { radius: 100.0 });
        sim.run(Duration::seconds(10));

        assert!(sim.agents()[0].observed.is_empty());
        assert_eq!(sim.agents()[1].observed.len(), 1);
    }

    #[test]
    fn test_proximity_perception() {
        let agents = vec![
            ObservingAgent::at(Position::new(0.0, 0.0)),
            ObservingAgent::at(Position::new(1.0, 0.0)),
            ObservingAgent::at(Position::new(100.0, 0.0)),
        ];

        let mut sim = Simulation::new(agents, base());
        sim.set_perception(Perception::Proximity { radius: 5.0 });
        sim.run(Duration::seconds(10));

        assert_eq!(sim.agents()[0].observed.len(), 2);
        assert_eq!(sim.agents()[1].observed.len(), 2);
        assert_eq!(sim.agents()[2].observed.len(), 1);
    }

    #[test]
    fn test_proximity_falls_back_without_index() {
        let agents = vec![
            ObservingAgent::at(Position::new(0.0, 0.0)),
            ObservingAgent::at(Position::new(0.0, 0.0)),
        ];

        let mut sim = Simulation::new_with_seed(agents, base(), 3);
        sim.set_perception(Perception::Proximity { radius: 0.0 });
        sim.run(Duration::seconds(10));

        assert!(sim.proximity.is_none());
        assert_eq!(sim.agents()[0].observed.len(), 2);
        assert_eq!(sim.agents()[1].observed.len(), 2);
    }

    struct MovingAgent {
        id: AgentId,
        x: f64,
        steps: u32,
        observed: usize,
    }

    impl SimAgent for MovingAgent {
        type State = ();
        type World = ();

        fn peek_next_event_delay(
            &self,
            _now: DateTime<Utc>,
            _world: &(),
            _rng: &mut dyn RngCore,
        ) -> Option<f64> {
            (self.steps < 4).then_some(1.0)
        }

        fn step(&self, _now: DateTime<Utc>, _world: &(), _rng: &mut dyn RngCore) -> Option<()> {
            (self.steps < 4).then_some(())
        }

        fn apply_transition(
            &mut self,
            _next: (),
            time: DateTime<Utc>,
            _world: &(),
            _rng: &mut dyn RngCore,
            out: &mut Vec<StateChangeEvent>,
        ) {
            self.steps += 1;
            self.x += 100.0;
            out.push(StateChangeEvent {
                time,
                agent_id: self.id.clone(),
                field: Cow::Borrowed("x"),
                old_value: Value::Float(self.x - 100.0),
                new_value: Value::Float(self.x),
            });
        }

        fn observe(&mut self, _event: &StateChangeEvent) {
            self.observed += 1;
        }

        fn location(&self) -> Option<Position> {
            Some(Position::new(self.x, 0.0))
        }
    }

    #[test]
    fn test_proximity_follows_movement() {
        let agents = vec![
            MovingAgent {
                id: Arc::from("walker"),
                x: 0.0,
                steps: 0,
                observed: 0,
            },
            MovingAgent {
                id: Arc::from("post_300"),
                x: 300.0,
                steps: 4,
                observed: 0,
            },
        ];

        let mut sim = Simulation::new_with_seed(agents, base(), 5);
        sim.set_perception(Perception::Proximity { radius: 10.0 });
        sim.run(Duration::seconds(30));

        assert_eq!(sim.agents()[1].observed, 1);
    }

    #[derive(Default)]
    struct Tape {
        last: i64,
        folded: usize,
    }

    impl World for Tape {
        fn absorb(&mut self, event: &StateChangeEvent) {
            self.folded += 1;
            if let Value::Int(value) = event.new_value {
                self.last = value;
            }
        }
    }

    struct TapeReader {
        id: AgentId,
        emitted: i64,
        seen_on_step: Vec<i64>,
    }

    impl SimAgent for TapeReader {
        type State = ();
        type World = Tape;

        fn peek_next_event_delay(
            &self,
            _now: DateTime<Utc>,
            _world: &Tape,
            _rng: &mut dyn RngCore,
        ) -> Option<f64> {
            (self.emitted < 3).then_some(1.0)
        }

        fn step(&self, _now: DateTime<Utc>, world: &Tape, _rng: &mut dyn RngCore) -> Option<()> {
            let _ = world.last;
            (self.emitted < 3).then_some(())
        }

        fn apply_transition(
            &mut self,
            _next: (),
            time: DateTime<Utc>,
            world: &Tape,
            _rng: &mut dyn RngCore,
            out: &mut Vec<StateChangeEvent>,
        ) {
            self.seen_on_step.push(world.last);
            self.emitted += 1;
            out.push(StateChangeEvent {
                time,
                agent_id: self.id.clone(),
                field: Cow::Borrowed("tick"),
                old_value: Value::Int(self.emitted - 1),
                new_value: Value::Int(self.emitted * 10),
            });
        }
    }

    #[test]
    fn test_world_folds_and_agents_read() {
        let agents = vec![
            TapeReader {
                id: Arc::from("a"),
                emitted: 0,
                seen_on_step: Vec::new(),
            },
            TapeReader {
                id: Arc::from("b"),
                emitted: 0,
                seen_on_step: Vec::new(),
            },
        ];

        let mut sim = Simulation::with_world(agents, base(), 7, Tape::default());
        let events = sim.run(Duration::seconds(10));

        assert_eq!(sim.world().folded, events.len());
        assert_eq!(sim.world().last, 30);

        let seen: Vec<i64> = sim
            .agents()
            .iter()
            .flat_map(|a| a.seen_on_step.clone())
            .collect();
        assert!(seen.contains(&0));
        assert!(seen.iter().any(|&v| v > 0));
    }

    fn markov_agent(id: &str) -> Agent<SimState, MockState> {
        let mut rng = SimRng::seed_from_u64(9);
        let transitions = HashMap::from([
            (
                SimState::Step1,
                StateType::new_deterministic(
                    || MockState { counter: 1 },
                    vec![(SimState::Step2, 0.7), (SimState::Step1, 0.3)],
                    0.1,
                ),
            ),
            (
                SimState::Step2,
                StateType::new_deterministic(
                    || MockState { counter: 2 },
                    vec![(SimState::Step1, 0.7), (SimState::Step2, 0.3)],
                    0.1,
                ),
            ),
        ]);

        Agent::new(id, SimState::Step1, transitions, &mut rng)
    }

    fn fingerprint(events: &[StateChangeEvent]) -> Vec<(DateTime<Utc>, AgentId, Value)> {
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

        let a_events: Vec<_> = shared.into_iter().filter(|e| &*e.agent_id == "a").collect();
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

        let ids: Vec<_> = events.iter().map(|e| e.agent_id.to_string()).collect();
        assert_eq!(ids, vec!["first", "second", "third"]);
    }

    struct WildDelayAgent {
        delay: f64,
    }

    impl SimAgent for WildDelayAgent {
        type State = ();
        type World = ();

        fn peek_next_event_delay(
            &self,
            _now: DateTime<Utc>,
            _world: &(),
            _rng: &mut dyn RngCore,
        ) -> Option<f64> {
            Some(self.delay)
        }

        fn step(&self, _now: DateTime<Utc>, _world: &(), _rng: &mut dyn RngCore) -> Option<()> {
            Some(())
        }

        fn apply_transition(
            &mut self,
            _next: (),
            _time: DateTime<Utc>,
            _world: &(),
            _rng: &mut dyn RngCore,
            _out: &mut Vec<StateChangeEvent>,
        ) {
        }
    }

    #[test]
    fn test_unrepresentable_delays_unscheduled() {
        for delay in [f64::INFINITY, f64::NAN, -1.0, 1e30, 1e18] {
            let mut sim = Simulation::new_with_seed(vec![WildDelayAgent { delay }], base(), 1);
            assert!(sim.run(Duration::hours(1)).is_empty(), "delay {delay}");
            assert_eq!(sim.current_time(), base());
        }
    }

    #[test]
    fn test_infinite_event_rate_does_not_panic() {
        let mut rng = SimRng::seed_from_u64(1);
        let transitions = HashMap::from([(
            SimState::Step1,
            StateType::new_deterministic(
                || MockState { counter: 1 },
                vec![(SimState::Step2, 1.0)],
                f64::INFINITY,
            ),
        )]);

        let agent = Agent::new("wild", SimState::Step1, transitions, &mut rng);
        let mut sim = Simulation::new_with_seed(vec![agent], base(), 1);

        assert!(sim.run(Duration::hours(1)).is_empty());
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

    struct TickAgent {
        ticks: u32,
    }

    impl SimAgent for TickAgent {
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
            self.ticks += 1;
            out.push(StateChangeEvent {
                time,
                agent_id: Arc::from("tick"),
                field: Cow::Borrowed("ticks"),
                old_value: Value::Int((self.ticks - 1) as i64),
                new_value: Value::Int(self.ticks as i64),
            });
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
