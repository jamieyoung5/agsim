use crate::agent::SimAgent;
use crate::state::StateChangeEvent;
use chrono::{DateTime, Duration, Utc};
use rand::rngs::StdRng;
use rand::{RngCore, SeedableRng};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

// Perception controls which agents are fed an event emitted by another. SelfOnly (the default) keeps
// each agent observing only its own changes; Global broadcasts every event to every agent. A
// proximity-based variant belongs here once agents carry a location.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Perception {
    #[default]
    SelfOnly,
    Global,
}

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

pub struct Simulation<A: SimAgent> {
    agents: Vec<A>,
    current_time: DateTime<Utc>,
    event_log: Vec<StateChangeEvent>,
    rng: Box<dyn RngCore>,
    perception: Perception,
}

impl<A: SimAgent> Simulation<A> {
    pub fn new(agents: Vec<A>, start_time: DateTime<Utc>) -> Self {
        Simulation {
            agents,
            current_time: start_time,
            event_log: Vec::new(),
            rng: Box::new(StdRng::from_entropy()),
            perception: Perception::default(),
        }
    }

    pub fn new_with_seed(agents: Vec<A>, start_time: DateTime<Utc>, seed: u64) -> Self {
        Simulation {
            agents,
            current_time: start_time,
            event_log: Vec::new(),
            rng: Box::new(StdRng::seed_from_u64(seed)),
            perception: Perception::default(),
        }
    }

    pub fn set_perception(&mut self, perception: Perception) {
        self.perception = perception;
    }

    pub fn agents(&self) -> &[A] {
        &self.agents
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

            let changes =
                self.agents[agent_index].apply_transition(target_type, self.current_time, &mut self.rng);

            // route each change to the agents that perceive it. The Markov agent ignores what it
            // observes; memory-backed agents record it. SelfOnly keeps the agent's own changes local.
            for change in &changes {
                for observer in 0..self.agents.len() {
                    let perceives = match self.perception {
                        Perception::SelfOnly => observer == agent_index,
                        Perception::Global => true,
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

    // seconds_to_duration converts a floating point value representing seconds to a Duration (TimeDelta) type.
    fn seconds_to_duration(seconds: f64) -> Duration {
        let millis = (seconds * 1000.0).round() as i64;
        Duration::milliseconds(millis)
    }

    /// schedule_next_for_agent attempts to schedule the next event for an agent, if possible.
    fn schedule_next_event(
        &mut self,
        agent_index: usize,
        queue: &mut BinaryHeap<ScheduledEvent<A::State>>,
    ) {
        let Some(delay_sec) =
            self.agents[agent_index].peek_next_event_delay(self.current_time, &mut self.rng)
        else {
            return;
        };
        let Some(next_state) = self.agents[agent_index].step(self.current_time, &mut self.rng)
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

    // ObservingAgent fires a single transition and records every event fed back through observe,
    // exercising the observation feed independently of the Markov agent.
    struct ObservingAgent {
        fired: bool,
        observed: Vec<String>,
    }

    impl SimAgent for ObservingAgent {
        type State = ();

        fn peek_next_event_delay(&self, _now: DateTime<Utc>, _rng: &mut dyn RngCore) -> Option<f64> {
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
                agent_id: "obs".to_string(),
                field: "state".to_string(),
                old_value: "0".to_string(),
                new_value: "1".to_string(),
            }]
        }

        fn observe(&mut self, event: &StateChangeEvent) {
            self.observed.push(event.field.clone());
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
        let agent = ObservingAgent {
            fired: false,
            observed: Vec::new(),
        };

        let mut sim = Simulation::new(vec![agent], start_time);
        sim.run(Duration::seconds(10));

        assert_eq!(sim.agents()[0].observed, vec!["state".to_string()]);
    }

    #[test]
    fn test_global_perception() {
        let start_time = Utc::now();
        let agents = vec![
            ObservingAgent {
                fired: false,
                observed: Vec::new(),
            },
            ObservingAgent {
                fired: false,
                observed: Vec::new(),
            },
        ];

        let mut sim = Simulation::new(agents, start_time);
        sim.set_perception(Perception::Global);
        sim.run(Duration::seconds(10));

        // each agent fires once and both events reach both agents.
        assert_eq!(sim.agents()[0].observed.len(), 2);
        assert_eq!(sim.agents()[1].observed.len(), 2);
    }
}
