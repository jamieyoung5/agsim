use crate::agent::Agent;
use crate::state::{State, StateChangeEvent};
use chrono::{DateTime, Duration, Utc};
use rand::rngs::StdRng;
use rand::{RngCore, SeedableRng};
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::hash::Hash;

struct ScheduledEvent<C> {
    time: DateTime<Utc>,
    agent_index: usize,
    next_state_type: Option<C>,
}

impl<C> PartialEq for ScheduledEvent<C> {
    fn eq(&self, other: &Self) -> bool {
        self.time == other.time
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
        other.time.cmp(&self.time)
    }
}

pub struct Simulation<C, S>
where
    C: Eq + Hash + Clone,
    S: State,
{
    agents: Vec<Agent<C, S>>,
    current_time: DateTime<Utc>,
    event_log: Vec<StateChangeEvent>,
    rng: Box<dyn RngCore>,
}

impl<C, S> Simulation<C, S>
where
    C: Eq + Hash + Clone + std::fmt::Debug,
    S: State + Clone + std::fmt::Debug,
{
    pub fn new(agents: Vec<Agent<C, S>>, start_time: DateTime<Utc>) -> Self {
        Simulation {
            agents,
            current_time: start_time,
            event_log: Vec::new(),
            rng: Box::new(StdRng::from_entropy()),
        }
    }

    pub fn new_with_seed(agents: Vec<Agent<C, S>>, start_time: DateTime<Utc>, seed: u64) -> Self {
        Simulation {
            agents,
            current_time: start_time,
            event_log: Vec::new(),
            rng: Box::new(StdRng::seed_from_u64(seed)),
        }
    }

    // run processes the simulation over a specified duration
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

    // run_streaming processes the simulation over a specified duration, providing a closure to stream the output to
    // a desired source (i.e, a file/stdout etc). This is usefull when generating a large number of events.
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

    fn initialize_queue(&mut self) -> BinaryHeap<ScheduledEvent<C>> {
        let mut queue = BinaryHeap::new();
        for index in 0..self.agents.len() {
            self.schedule_next_event(index, &mut queue);
        }
        queue
    }

    fn process_event_step<F>(
        &mut self,
        event: ScheduledEvent<C>,
        queue: &mut BinaryHeap<ScheduledEvent<C>>,
        mut handler: F,
    ) where
        F: FnMut(Vec<StateChangeEvent>, &mut Vec<StateChangeEvent>),
    {
        self.current_time = event.time;

        if let Some(target_type) = event.next_state_type {
            let agent_index = event.agent_index;

            let changes = {
                let agent = &mut self.agents[agent_index];
                agent.apply_transition(target_type, self.current_time, &mut self.rng)
            };

            handler(changes, &mut self.event_log);

            self.schedule_next_event(agent_index, queue);
        }
    }

    // seconds_to_duration converts a floating point value representing seconds to a Duration (TimeDelta) type.
    fn seconds_to_duration(seconds: f64) -> Duration {
        let millis = (seconds * 1000.0).round() as i64;
        Duration::milliseconds(millis)
    }

    /// schedule_next_for_agent attempts to schedule the next event for an agent, if possible.
    fn schedule_next_event(
        &mut self,
        agent_index: usize,
        queue: &mut BinaryHeap<ScheduledEvent<C>>,
    ) {
        if let Some(delay_sec) = self.agents[agent_index].peek_next_event_delay(&mut self.rng) {
            if let Some(next_state) = self.agents[agent_index].step(&mut self.rng) {
                let event_time = self.current_time + Self::seconds_to_duration(delay_sec);
                queue.push(ScheduledEvent {
                    time: event_time,
                    agent_index,
                    next_state_type: Some(next_state),
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::StateType;
    use crate::state::StateChangeEvent;
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
}
