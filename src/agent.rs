use crate::state::{State, StateChangeEvent};
use chrono::{DateTime, Utc};
use rand::Rng;
use rand::seq::SliceRandom;
use rand_distr::{Distribution, Exp};
use std::collections::HashMap;
use std::hash::Hash;

pub struct StateType<C, S: State> {
    pub factory: fn() -> Box<S>,
    pub transitions: Vec<(C, f64)>,
    pub event_rate: f64,
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
    ) -> Self {
        let initial_def = transition_matrix
            .get(&initial_state_type)
            .expect("Initial state type must exist in transition matrix");
        let data = *(initial_def.factory)();

        Agent {
            id,
            transition_matrix,
            current_state_type: initial_state_type,
            data,
        }
    }

    // step moves to the next state change in the chain
    pub fn step(&self, rng: &mut impl Rng) -> Option<C> {
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

    // peek_next_event_delay calculates the time until the next event using an exponential distribution based on the event rate
    pub fn peek_next_event_delay(&self, rng: &mut impl Rng) -> Option<f64> {
        let current_def = self.transition_matrix.get(&self.current_state_type)?;

        // lambda = 1 / Mean.
        // if the mean is 0, we can assume instant transition
        if current_def.event_rate <= 0.0 {
            return Some(0.0);
        }

        let lambda = 1.0 / current_def.event_rate;
        let exp = Exp::new(lambda).ok()?;

        Some(exp.sample(rng))
    }

    // apply_transaction transitions the agent to a new state type
    pub fn apply_transition(&mut self, new_type: C, time: DateTime<Utc>) -> Vec<StateChangeEvent> {
        self.current_state_type = new_type.clone();

        let target_state = match self.get_target_state(&new_type) {
            Some(state) => state,
            None => return Vec::new(),
        };

        S::get_field_names()
            .into_iter()
            .filter_map(|field| self.sync_field(&field, &target_state, time))
            .collect()
    }

    fn get_target_state(&self, state_type: &C) -> Option<Box<S>> {
        self.transition_matrix
            .get(state_type)
            .map(|def| (def.factory)())
    }

    fn sync_field(
        &mut self,
        field: &str,
        target_state: &S,
        time: DateTime<Utc>,
    ) -> Option<StateChangeEvent> {
        let old_val = self.data.get_field(field);
        let new_val = target_state.get_field(field);

        if old_val == new_val {
            return None;
        }

        self.data.update_field(field, &new_val);

        Some(StateChangeEvent {
            time,
            field: field.to_string(),
            old_value: old_val,
            new_value: new_val,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[derive(Clone, Default, Debug, PartialEq)]
    struct MockState {
        status: String,
        counter: String,
    }

    impl State for MockState {
        fn update_field(&mut self, field: &str, value: &str) {
            match field {
                "status" => self.status = value.to_string(),
                "counter" => self.counter = value.to_string(),
                _ => (),
            }
        }

        fn get_field(&self, field: &str) -> String {
            match field {
                "status" => self.status.clone(),
                "counter" => self.counter.clone(),
                _ => "".to_string(),
            }
        }

        fn get_field_names() -> &'static [&'static str] {
            &["status", "counter"]
        }
    }

    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    enum AgentMode {
        Idle,
        Active,
    }

    fn idle_factory() -> Box<MockState> {
        Box::new(MockState {
            status: "idle".to_string(),
            counter: "0".to_string(),
        })
    }

    fn active_factory() -> Box<MockState> {
        Box::new(MockState {
            status: "active".to_string(),
            counter: "1".to_string(),
        })
    }

    fn setup_agent() -> Agent<AgentMode, MockState> {
        let mut transition_matrix = HashMap::new();

        transition_matrix.insert(
            AgentMode::Idle,
            StateType {
                factory: idle_factory,
                transitions: vec![(AgentMode::Active, 1.0)],
                event_rate: 1.0,
            },
        );

        transition_matrix.insert(
            AgentMode::Active,
            StateType {
                factory: active_factory,
                transitions: vec![(AgentMode::Idle, 1.0)],
                event_rate: 2.0,
            },
        );

        Agent::new("test_agent".to_string(), AgentMode::Idle, transition_matrix)
    }

    #[test]
    fn test_initialization() {
        let agent = setup_agent();
        assert_eq!(agent.id, "test_agent");
        assert_eq!(agent.current_state_type, AgentMode::Idle);
        assert_eq!(agent.data.status, "idle");
    }

    #[test]
    fn test_step_deterministic_transition() {
        let agent = setup_agent();
        let mut rng = StdRng::seed_from_u64(42);

        let next_state = agent.step(&mut rng);
        assert_eq!(next_state, Some(AgentMode::Active));
    }

    #[test]
    fn test_peek_next_event_delay() {
        let agent = setup_agent();
        let mut rng = StdRng::seed_from_u64(42);

        let delay = agent.peek_next_event_delay(&mut rng);
        assert!(delay.is_some());
        assert!(delay.unwrap() > 0.0);
    }

    #[test]
    fn test_apply_transition_updates_state_and_logs_changes() {
        let mut agent = setup_agent();
        let time = Utc::now();

        let changes = agent.apply_transition(AgentMode::Active, time);

        assert_eq!(agent.current_state_type, AgentMode::Active);
        assert_eq!(agent.data.status, "active");
        assert_eq!(agent.data.counter, "1");

        assert_eq!(changes.len(), 2);

        let status_change = changes.iter().find(|c| c.field == "status").unwrap();
        assert_eq!(status_change.old_value, "idle");
        assert_eq!(status_change.new_value, "active");
        assert_eq!(status_change.time, time);
    }

    #[test]
    fn test_apply_transition_no_redundant_logs() {
        let mut agent = setup_agent();
        let time = Utc::now();

        agent.apply_transition(AgentMode::Active, time);

        let changes = agent.apply_transition(AgentMode::Active, time);

        assert!(
            changes.is_empty(),
            "Should not generate events if values didn't change"
        );
    }
}
