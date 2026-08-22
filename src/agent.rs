use crate::space::Position;
use crate::state::{AgentId, State, StateChangeEvent};
use crate::world::World;
use chrono::{DateTime, Utc};
use rand::RngCore;
use rand::distributions::{Distribution, WeightedIndex};
use rand_distr::Exp;
use std::collections::HashMap;
use std::hash::Hash;
use std::sync::Arc;

type Factory<S> = Arc<dyn Fn(&mut dyn RngCore) -> S + Send + Sync>;

#[derive(Clone)]
pub struct StateType<C, S: State> {
    pub factory: Factory<S>,
    pub transitions: Vec<(C, f64)>,
    pub event_rate: f64,
}

impl<C, S> StateType<C, S>
where
    S: State,
{
    pub fn new<F>(factory: F, transitions: Vec<(C, f64)>, event_rate: f64) -> Self
    where
        F: Fn(&mut dyn RngCore) -> S + Send + Sync + 'static,
    {
        StateType {
            factory: Arc::new(factory),
            transitions,
            event_rate,
        }
    }

    pub fn new_deterministic<F>(factory: F, transitions: Vec<(C, f64)>, event_rate: f64) -> Self
    where
        F: Fn() -> S + Send + Sync + 'static,
    {
        StateType {
            factory: Arc::new(move |_| factory()),
            transitions,
            event_rate,
        }
    }
}

/// The interface a [`Simulation`](crate::simulation::Simulation) drives.
pub trait SimAgent {
    type State;

    type World: World;

    fn peek_next_event_delay(
        &self,
        now: DateTime<Utc>,
        world: &Self::World,
        rng: &mut dyn RngCore,
    ) -> Option<f64>;

    fn step(
        &self,
        now: DateTime<Utc>,
        world: &Self::World,
        rng: &mut dyn RngCore,
    ) -> Option<Self::State>;

    /// Moves into `next`, appending changed fields.
    fn apply_transition(
        &mut self,
        next: Self::State,
        time: DateTime<Utc>,
        world: &Self::World,
        rng: &mut dyn RngCore,
        out: &mut Vec<StateChangeEvent>,
    );

    fn observe(&mut self, _event: &StateChangeEvent) {}

    /// Position for proximity perception.
    fn location(&self) -> Option<Position> {
        None
    }
}

/// A state's slot in a [`Transitions`] table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StateId(usize);

struct CompiledState<S> {
    factory: Option<Factory<S>>,
    targets: Vec<StateId>,
    picker: Option<WeightedIndex<f64>>,
    delay: Option<Exp<f64>>,
}

/// A transition matrix as a flat table.
pub struct Transitions<C, S: State> {
    states: Vec<CompiledState<S>>,
    keys: Vec<Option<C>>,
    index_of: HashMap<C, StateId>,
}

impl<C, S> Transitions<C, S>
where
    C: Eq + Hash + Clone,
    S: State,
{
    pub fn compile(matrix: HashMap<C, StateType<C, S>>) -> Self {
        let mut keys: Vec<Option<C>> = Vec::with_capacity(matrix.len() + 1);
        let mut index_of = HashMap::with_capacity(matrix.len());
        let mut defs = Vec::with_capacity(matrix.len());

        for (key, def) in matrix {
            index_of.insert(key.clone(), StateId(keys.len()));
            keys.push(Some(key));
            defs.push(def);
        }

        let dormant = StateId(keys.len());
        keys.push(None);

        let mut states: Vec<CompiledState<S>> = defs
            .into_iter()
            .map(|def| {
                let targets = def
                    .transitions
                    .iter()
                    .map(|(target, _)| index_of.get(target).copied().unwrap_or(dormant))
                    .collect();

                let weights: Vec<f64> = def.transitions.iter().map(|(_, weight)| *weight).collect();
                let picker = if weights.is_empty() {
                    None
                } else {
                    WeightedIndex::new(&weights).ok()
                };

                // no rate, agent goes dormant
                let delay = (def.event_rate.is_finite() && def.event_rate > 0.0)
                    .then(|| Exp::new(1.0 / def.event_rate).ok())
                    .flatten();

                CompiledState {
                    factory: Some(def.factory),
                    targets,
                    picker,
                    delay,
                }
            })
            .collect();

        states.push(CompiledState {
            factory: None,
            targets: Vec::new(),
            picker: None,
            delay: None,
        });

        Transitions {
            states,
            keys,
            index_of,
        }
    }

    pub fn index(&self, key: &C) -> Option<StateId> {
        self.index_of.get(key).copied()
    }

    pub fn state_type(&self, id: StateId) -> Option<&C> {
        self.keys.get(id.0).and_then(|key| key.as_ref())
    }

    pub fn dormant(&self) -> StateId {
        StateId(self.states.len() - 1)
    }

    pub fn len(&self) -> usize {
        self.states.len() - 1
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    // foreign ids read as dormant
    fn state(&self, id: StateId) -> &CompiledState<S> {
        self.states
            .get(id.0)
            .unwrap_or_else(|| &self.states[self.states.len() - 1])
    }
}

pub struct Agent<C, S>
where
    C: Eq + Hash + Clone,
    S: State,
{
    transitions: Arc<Transitions<C, S>>,
    current: StateId,
    pub data: S,
    pub id: AgentId,
}

impl<C, S> Agent<C, S>
where
    C: Eq + Hash + Clone,
    S: State + Clone,
{
    pub fn new(
        id: impl Into<AgentId>,
        initial_state_type: C,
        transition_matrix: HashMap<C, StateType<C, S>>,
        rng: &mut dyn RngCore,
    ) -> Self {
        Self::with_shared_transitions(
            id,
            initial_state_type,
            Arc::new(Transitions::compile(transition_matrix)),
            rng,
        )
    }

    /// Builds an agent over a shared table.
    pub fn with_shared_transitions(
        id: impl Into<AgentId>,
        initial_state_type: C,
        transitions: Arc<Transitions<C, S>>,
        rng: &mut dyn RngCore,
    ) -> Self {
        let current = transitions
            .index(&initial_state_type)
            .expect("Initial state type must exist in transition matrix");
        let factory = transitions
            .state(current)
            .factory
            .as_ref()
            .expect("A defined state always has a factory");
        let data = factory(rng);

        Agent {
            id: id.into(),
            transitions,
            current,
            data,
        }
    }

    /// Current state type, `None` if dormant.
    pub fn current_state_type(&self) -> Option<&C> {
        self.transitions.state_type(self.current)
    }

    pub fn transitions(&self) -> &Transitions<C, S> {
        &self.transitions
    }
}

impl<C, S> SimAgent for Agent<C, S>
where
    C: Eq + Hash + Clone,
    S: State + Clone,
{
    type State = StateId;
    type World = ();

    // markov timing is memoryless
    fn peek_next_event_delay(
        &self,
        _now: DateTime<Utc>,
        _world: &(),
        rng: &mut dyn RngCore,
    ) -> Option<f64> {
        Some(self.transitions.state(self.current).delay?.sample(rng))
    }

    fn step(&self, _now: DateTime<Utc>, _world: &(), rng: &mut dyn RngCore) -> Option<StateId> {
        let state = self.transitions.state(self.current);
        let picker = state.picker.as_ref()?;
        Some(state.targets[picker.sample(rng)])
    }

    fn apply_transition(
        &mut self,
        next: StateId,
        time: DateTime<Utc>,
        _world: &(),
        rng: &mut dyn RngCore,
        out: &mut Vec<StateChangeEvent>,
    ) {
        self.current = next;

        let Some(factory) = self.transitions.state(next).factory.as_ref() else {
            return;
        };

        let target_state = factory(rng);
        self.data.diff(&target_state, &self.id, time, out);
        self.data = target_state;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{State, StateChangeEvent, ToValue, Value};
    use chrono::TimeZone;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use std::borrow::Cow;

    #[derive(Clone, Default, Debug, PartialEq)]
    struct MockState {
        value: i32,
    }

    impl State for MockState {
        fn diff(
            &self,
            other: &Self,
            agent_id: &AgentId,
            time: DateTime<Utc>,
            out: &mut Vec<StateChangeEvent>,
        ) {
            if self.value != other.value {
                out.push(StateChangeEvent {
                    time,
                    agent_id: agent_id.clone(),
                    field: Cow::Borrowed("value"),
                    old_value: self.value.to_value(),
                    new_value: other.value.to_value(),
                });
            }
        }
    }

    #[derive(Eq, Hash, PartialEq, Clone, Debug)]
    enum AgentState {
        Idle,
        Active,
    }

    fn idle_and_active() -> HashMap<AgentState, StateType<AgentState, MockState>> {
        HashMap::from([
            (
                AgentState::Idle,
                StateType::new_deterministic(|| MockState { value: 0 }, vec![], 1.0),
            ),
            (
                AgentState::Active,
                StateType::new_deterministic(|| MockState { value: 10 }, vec![], 1.0),
            ),
        ])
    }

    #[test]
    fn test_agent_initialization() {
        let mut rng = StdRng::seed_from_u64(42);
        let transitions = HashMap::from([(
            AgentState::Idle,
            StateType::new_deterministic(|| MockState { value: 0 }, vec![], 1.0),
        )]);

        let agent = Agent::new("agent_1", AgentState::Idle, transitions, &mut rng);

        assert_eq!(&*agent.id, "agent_1");
        assert_eq!(agent.current_state_type(), Some(&AgentState::Idle));
        assert_eq!(agent.data.value, 0);
    }

    #[test]
    fn test_agent_step_transition_choice() {
        let mut rng = StdRng::seed_from_u64(42);
        let transitions = HashMap::from([
            (
                AgentState::Idle,
                StateType::new_deterministic(
                    || MockState { value: 0 },
                    vec![(AgentState::Active, 10.0)],
                    1.0,
                ),
            ),
            (
                AgentState::Active,
                StateType::new_deterministic(|| MockState { value: 1 }, vec![], 1.0),
            ),
        ]);

        let mut agent = Agent::new("test", AgentState::Idle, transitions, &mut rng);
        let next = agent.step(Utc::now(), &(), &mut rng).unwrap();

        let mut out = Vec::new();
        agent.apply_transition(next, Utc::now(), &(), &mut rng, &mut out);
        assert_eq!(agent.current_state_type(), Some(&AgentState::Active));
    }

    #[test]
    fn test_agent_peek_delay() {
        let mut rng = StdRng::seed_from_u64(42);
        let transitions = HashMap::from([
            (
                AgentState::Idle,
                StateType::new_deterministic(|| MockState { value: 0 }, vec![], 1.0),
            ),
            (
                AgentState::Active,
                StateType::new_deterministic(|| MockState { value: 1 }, vec![], 0.0),
            ),
        ]);

        let mut agent = Agent::new("test", AgentState::Idle, transitions, &mut rng);

        let delay = agent.peek_next_event_delay(Utc::now(), &(), &mut rng);
        assert!(delay.is_some_and(|d| d > 0.0));

        // zero rate goes dormant
        agent.current = agent.transitions.index(&AgentState::Active).unwrap();
        assert!(
            agent
                .peek_next_event_delay(Utc::now(), &(), &mut rng)
                .is_none()
        );
    }

    #[test]
    fn test_apply_transition_logic() {
        let mut rng = StdRng::seed_from_u64(42);
        let time = Utc.timestamp_opt(1000, 0).unwrap();

        let mut agent = Agent::new("agent_x", AgentState::Idle, idle_and_active(), &mut rng);
        let active = agent.transitions.index(&AgentState::Active).unwrap();

        let mut events = Vec::new();
        agent.apply_transition(active, time, &(), &mut rng, &mut events);

        assert_eq!(agent.current_state_type(), Some(&AgentState::Active));
        assert_eq!(agent.data.value, 10);

        assert_eq!(events.len(), 1);
        assert_eq!(events[0].field, "value");
        assert_eq!(events[0].old_value, Value::Int(0));
        assert_eq!(events[0].new_value, Value::Int(10));
        assert_eq!(&*events[0].agent_id, "agent_x");
    }

    #[test]
    fn test_apply_transition_appends() {
        let mut rng = StdRng::seed_from_u64(42);
        let time = Utc.timestamp_opt(1000, 0).unwrap();

        let mut agent = Agent::new("agent_x", AgentState::Idle, idle_and_active(), &mut rng);
        let active = agent.transitions.index(&AgentState::Active).unwrap();

        let existing = StateChangeEvent {
            time,
            agent_id: Arc::from("someone_else"),
            field: Cow::Borrowed("prior"),
            old_value: Value::Int(0),
            new_value: Value::Int(1),
        };
        let mut events = vec![existing.clone()];
        agent.apply_transition(active, time, &(), &mut rng, &mut events);

        assert_eq!(events.len(), 2);
        assert_eq!(events[0], existing);
    }

    #[test]
    fn test_undefined_state_goes_dormant() {
        let mut rng = StdRng::seed_from_u64(42);
        let transitions = HashMap::from([(
            AgentState::Idle,
            StateType::new_deterministic(
                || MockState { value: 0 },
                // Active is never defined
                vec![(AgentState::Active, 1.0)],
                1.0,
            ),
        )]);

        let mut agent = Agent::new("test", AgentState::Idle, transitions, &mut rng);
        let next = agent.step(Utc::now(), &(), &mut rng).unwrap();

        let mut out = Vec::new();
        agent.apply_transition(next, Utc::now(), &(), &mut rng, &mut out);

        assert!(out.is_empty());
        assert_eq!(agent.current_state_type(), None);
        assert!(
            agent
                .peek_next_event_delay(Utc::now(), &(), &mut rng)
                .is_none()
        );
    }

    #[test]
    fn test_state_ids_come_from_the_table() {
        let mut rng = StdRng::seed_from_u64(3);
        let table = Arc::new(Transitions::compile(idle_and_active()));

        let idle = table.index(&AgentState::Idle).unwrap();
        let active = table.index(&AgentState::Active).unwrap();

        assert_ne!(idle, active);
        assert_eq!(table.len(), 2);
        assert_eq!(table.state_type(idle), Some(&AgentState::Idle));
        assert_eq!(table.state_type(active), Some(&AgentState::Active));
        assert_eq!(table.state_type(table.dormant()), None);

        let agent = Agent::with_shared_transitions("a", AgentState::Idle, table, &mut rng);
        assert_eq!(agent.transitions().index(&AgentState::Active), Some(active));
    }

    #[test]
    fn test_foreign_state_id_goes_dormant() {
        let mut rng = StdRng::seed_from_u64(3);

        // wider table, foreign ids
        let wide = Transitions::compile(HashMap::from([
            (
                AgentState::Idle,
                StateType::new_deterministic(|| MockState { value: 0 }, vec![], 1.0),
            ),
            (
                AgentState::Active,
                StateType::new_deterministic(|| MockState { value: 1 }, vec![], 1.0),
            ),
        ]));
        let stranger = wide.dormant();

        let narrow = HashMap::from([(
            AgentState::Idle,
            StateType::new_deterministic(|| MockState { value: 0 }, vec![], 1.0),
        )]);
        let mut agent = Agent::new("a", AgentState::Idle, narrow, &mut rng);

        let mut out = Vec::new();
        agent.apply_transition(stranger, Utc::now(), &(), &mut rng, &mut out);

        assert!(out.is_empty());
        assert_eq!(agent.current_state_type(), None);
        assert!(
            agent
                .peek_next_event_delay(Utc::now(), &(), &mut rng)
                .is_none()
        );
    }

    #[test]
    fn test_shared_transitions_not_duplicated() {
        let shared = Arc::new(Transitions::compile(idle_and_active()));
        let mut rng = StdRng::seed_from_u64(1);

        let agents: Vec<_> = (0..8)
            .map(|index| {
                Agent::with_shared_transitions(
                    format!("agent_{index}"),
                    AgentState::Idle,
                    Arc::clone(&shared),
                    &mut rng,
                )
            })
            .collect();

        assert_eq!(agents.len(), 8);
        assert_eq!(Arc::strong_count(&shared), 9);
    }

    #[test]
    fn test_shared_matches_owned() {
        let time = Utc.timestamp_opt(1000, 0).unwrap();

        let mut owned_rng = StdRng::seed_from_u64(42);
        let mut owned = Agent::new("agent", AgentState::Idle, idle_and_active(), &mut owned_rng);

        let mut shared_rng = StdRng::seed_from_u64(42);
        let mut shared = Agent::with_shared_transitions(
            "agent",
            AgentState::Idle,
            Arc::new(Transitions::compile(idle_and_active())),
            &mut shared_rng,
        );

        let owned_target = owned.transitions.index(&AgentState::Active).unwrap();
        let shared_target = shared.transitions.index(&AgentState::Active).unwrap();

        let mut left = Vec::new();
        let mut right = Vec::new();
        owned.apply_transition(owned_target, time, &(), &mut owned_rng, &mut left);
        shared.apply_transition(shared_target, time, &(), &mut shared_rng, &mut right);

        assert_eq!(owned.data, shared.data);
        assert_eq!(left, right);
    }
}
