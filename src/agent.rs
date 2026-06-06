use crate::state::{State, StateChangeEvent};
use chrono::{DateTime, Utc};
use rand::RngCore;
use rand::seq::SliceRandom;
use rand_distr::{Distribution, Exp};
use std::collections::HashMap;
use std::hash::Hash;
use std::sync::Arc;

#[derive(Clone)]
pub struct StateType<C, S: State> {
    pub factory: Arc<dyn Fn(&mut dyn RngCore) -> S + Send + Sync>,
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

// SimAgent is the interface the Simulation drives. The Markov Agent implements it via its
// transition matrix; richer agents (e.g. memory/plan driven) can decide their own transitions and
// consume the events they emit through observe.
pub trait SimAgent {
    type State;

    fn peek_next_event_delay(&self, now: DateTime<Utc>, rng: &mut dyn RngCore) -> Option<f64>;
    fn step(&self, now: DateTime<Utc>, rng: &mut dyn RngCore) -> Option<Self::State>;
    fn apply_transition(
        &mut self,
        next: Self::State,
        time: DateTime<Utc>,
        rng: &mut dyn RngCore,
    ) -> Vec<StateChangeEvent>;
    fn observe(&mut self, _event: &StateChangeEvent) {}
}

pub struct Agent<C, S>
where
    C: Eq + Hash + Clone,
    S: State,
{
    transition_matrix: HashMap<C, StateType<C, S>>,
    current_state_type: C,
    pub data: S,
    pub id: String,
}

impl<C, S> Agent<C, S>
where
    C: Eq + Hash + Clone,
    S: State + Clone,
{
    pub fn new(
        id: String,
        initial_state_type: C,
        transition_matrix: HashMap<C, StateType<C, S>>,
        rng: &mut dyn RngCore,
    ) -> Self {
        let initial_def = transition_matrix
            .get(&initial_state_type)
            .expect("Initial state type must exist in transition matrix");
        let data = (initial_def.factory)(rng);

        Agent {
            id,
            transition_matrix,
            current_state_type: initial_state_type,
            data,
        }
    }

    fn get_target_state(&self, state_type: &C, rng: &mut dyn RngCore) -> Option<S> {
        self.transition_matrix
            .get(state_type)
            .map(|def| (def.factory)(rng))
    }
}

impl<C, S> SimAgent for Agent<C, S>
where
    C: Eq + Hash + Clone,
    S: State + Clone,
{
    type State = C;

    // peek_next_event_delay draws the time until the next event from an exponential distribution
    // keyed on the current state's event rate. Time is unused: a Markov agent's timing is memoryless.
    fn peek_next_event_delay(&self, _now: DateTime<Utc>, rng: &mut dyn RngCore) -> Option<f64> {
        let current_def = self.transition_matrix.get(&self.current_state_type)?;

        // lambda = 1 / Mean.
        // if the mean is 0, we can assume instant transition
        if current_def.event_rate <= 0.0 {
            return None;
        }

        let lambda = 1.0 / current_def.event_rate;
        let exp = Exp::new(lambda).ok()?;

        Some(exp.sample(rng))
    }

    // step picks the next state type by sampling the current state's weighted transitions.
    fn step(&self, _now: DateTime<Utc>, rng: &mut dyn RngCore) -> Option<C> {
        let current_def = self.transition_matrix.get(&self.current_state_type)?;

        if current_def.transitions.is_empty() {
            return None;
        }

        current_def
            .transitions
            .choose_weighted(rng, |item| item.1)
            .ok()
            .map(|(next_state, _)| next_state.clone())
    }

    fn apply_transition(
        &mut self,
        new_type: C,
        time: DateTime<Utc>,
        rng: &mut dyn RngCore,
    ) -> Vec<StateChangeEvent> {
        self.current_state_type = new_type.clone();

        let target_state = match self.get_target_state(&new_type, rng) {
            Some(state) => state,
            None => return Vec::new(),
        };

        let mut events = self.data.diff(&target_state, time);
        for event in &mut events {
            event.agent_id = self.id.clone();
        }

        self.data = target_state;

        events
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{State, StateChangeEvent};
    use chrono::TimeZone;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[derive(Clone, Default, Debug, PartialEq)]
    struct MockState {
        value: i32,
    }

    impl State for MockState {
        fn diff(&self, other: &Self, time: DateTime<Utc>) -> Vec<StateChangeEvent> {
            if self.value != other.value {
                vec![StateChangeEvent {
                    time,
                    agent_id: String::new(),
                    field: "value".to_string(),
                    old_value: self.value.to_string(),
                    new_value: other.value.to_string(),
                }]
            } else {
                vec![]
            }
        }
    }

    #[derive(Eq, Hash, PartialEq, Clone, Debug)]
    enum AgentState {
        Idle,
        Active,
    }

    #[test]
    fn test_agent_initialization() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut transitions = HashMap::new();

        transitions.insert(
            AgentState::Idle,
            StateType::new_deterministic(|| MockState { value: 0 }, vec![], 1.0),
        );

        let agent = Agent::new(
            "agent_1".to_string(),
            AgentState::Idle,
            transitions,
            &mut rng,
        );

        assert_eq!(agent.id, "agent_1");
        assert_eq!(agent.current_state_type, AgentState::Idle);
        assert_eq!(agent.data.value, 0);
    }

    #[test]
    fn test_agent_step_transition_choice() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut transitions = HashMap::new();

        transitions.insert(
            AgentState::Idle,
            StateType::new_deterministic(
                || MockState { value: 0 },
                vec![(AgentState::Active, 10.0)],
                1.0,
            ),
        );

        let agent = Agent::new("test".to_string(), AgentState::Idle, transitions, &mut rng);

        let next_state = agent.step(Utc::now(), &mut rng);
        assert_eq!(next_state, Some(AgentState::Active));
    }

    #[test]
    fn test_agent_peek_delay() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut transitions = HashMap::new();

        transitions.insert(
            AgentState::Idle,
            StateType::new_deterministic(|| MockState { value: 0 }, vec![], 1.0),
        );

        transitions.insert(
            AgentState::Active,
            StateType::new_deterministic(|| MockState { value: 1 }, vec![], 0.0),
        );

        let mut agent = Agent::new(
            "test".to_string(),
            AgentState::Idle,
            transitions.clone(),
            &mut rng,
        );

        let delay = agent.peek_next_event_delay(Utc::now(), &mut rng);
        assert!(delay.is_some());
        assert!(delay.unwrap() > 0.0);

        agent.current_state_type = AgentState::Active;
        let delay_none = agent.peek_next_event_delay(Utc::now(), &mut rng);
        assert!(delay_none.is_none());
    }

    #[test]
    fn test_apply_transition_logic() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut transitions = HashMap::new();
        let time = Utc.timestamp_opt(1000, 0).unwrap();

        transitions.insert(
            AgentState::Idle,
            StateType::new_deterministic(|| MockState { value: 0 }, vec![], 1.0),
        );
        transitions.insert(
            AgentState::Active,
            StateType::new_deterministic(|| MockState { value: 10 }, vec![], 1.0),
        );

        let mut agent = Agent::new(
            "agent_x".to_string(),
            AgentState::Idle,
            transitions,
            &mut rng,
        );

        let events = agent.apply_transition(AgentState::Active, time, &mut rng);

        assert_eq!(agent.current_state_type, AgentState::Active);
        assert_eq!(agent.data.value, 10);

        assert_eq!(events.len(), 1);
        assert_eq!(events[0].field, "value");
        assert_eq!(events[0].old_value, "0");
        assert_eq!(events[0].new_value, "10");
        assert_eq!(events[0].agent_id, "agent_x");
    }
}
