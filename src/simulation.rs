use crate::agent::Agent;
use crate::state::{State, StateChangeEvent, Timeline};
use chrono::{DateTime, Duration, Utc};
use rand::rngs::ThreadRng;
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
    rng: ThreadRng,
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
            rng: rand::thread_rng(),
        }
    }

    // run processes the simulation over a specified duration
    pub fn run(&mut self, duration: Duration) -> Vec<StateChangeEvent> {
        let end_time = self.current_time + duration;
        let mut queue = BinaryHeap::new();

        for index in 0..self.agents.len() {
            self.schedule_next_event(index, &mut queue);
        }

        // orchestrate event scheduling
        while let Some(event) = queue.pop() {
            if event.time > end_time {
                break;
            }

            self.current_time = event.time;

            if let Some(target_type) = event.next_state_type {
                let agent_index = event.agent_index;

                // apply the state transition and record state change
                let changes = {
                    let agent = &mut self.agents[agent_index];
                    agent.apply_transition(target_type, self.current_time)
                };
                self.event_log.extend(changes);

                self.schedule_next_event(agent_index, &mut queue);
            }
        }

        self.event_log.clone()
    }

    // generate_master_timeline generates a complete combined timeline over all agents.
    pub fn generate_master_timeline(&self) -> Option<Timeline<S>> {
        Timeline::generate(&self.event_log)
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
    use crate::agent::{Agent, StateType};
    use crate::state::State;
    use std::collections::HashMap;

    #[derive(Clone, Default, Debug, PartialEq)]
    struct SimTestState {
        val: String,
    }

    impl State for SimTestState {
        fn update_field(&mut self, field: &str, value: &str) {
            if field == "val" {
                self.val = value.to_string();
            }
        }
        fn get_field(&self, field: &str) -> String {
            if field == "val" {
                self.val.clone()
            } else {
                "".to_string()
            }
        }
        fn get_field_names() -> &'static [&'static str] {
            &["val"]
        }
    }

    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    enum SimStateMode {
        A,
        B,
    }

    fn factory_a() -> Box<SimTestState> {
        Box::new(SimTestState {
            val: "A".to_string(),
        })
    }
    fn factory_b() -> Box<SimTestState> {
        Box::new(SimTestState {
            val: "B".to_string(),
        })
    }

    fn create_test_agent(id: &str) -> Agent<SimStateMode, SimTestState> {
        let mut transitions = HashMap::new();

        transitions.insert(
            SimStateMode::A,
            StateType {
                factory: factory_a,
                transitions: vec![(SimStateMode::B, 1.0)],
                event_rate: 100.0,
            },
        );

        transitions.insert(
            SimStateMode::B,
            StateType {
                factory: factory_b,
                transitions: vec![(SimStateMode::A, 1.0)],
                event_rate: 100.0,
            },
        );

        Agent::new(id.to_string(), SimStateMode::A, transitions)
    }

    #[test]
    fn test_simulation_initialization() {
        let agent = create_test_agent("ag1");
        let start_time = Utc::now();
        let sim = Simulation::new(vec![agent], start_time);

        assert_eq!(sim.current_time, start_time);
        assert!(sim.event_log.is_empty());
    }

    #[test]
    fn test_simulation_run_advances_time() {
        let agent = create_test_agent("ag1");
        let start_time = Utc::now();
        let mut sim = Simulation::new(vec![agent], start_time);

        let duration = Duration::seconds(1);
        let events = sim.run(duration);

        assert!(!events.is_empty());
        assert!(!sim.event_log.is_empty());

        assert!(sim.current_time > start_time);
        assert!(sim.current_time <= start_time + duration);
    }

    #[test]
    fn test_simulation_log_integrity() {
        let agent = create_test_agent("ag1");
        let start_time = Utc::now();
        let mut sim = Simulation::new(vec![agent], start_time);

        sim.run(Duration::milliseconds(100));

        let mut prev_time = start_time;
        for event in &sim.event_log {
            assert!(
                event.time >= prev_time,
                "Events must be strictly ordered by time"
            );
            prev_time = event.time;
        }
    }

    #[test]
    fn test_master_timeline_generation() {
        let agent = create_test_agent("ag1");
        let mut sim = Simulation::new(vec![agent], Utc::now());

        sim.run(Duration::milliseconds(50));

        let timeline = sim.generate_master_timeline();
        assert!(timeline.is_some());

        let tl = timeline.unwrap();
        assert!(!tl.entries.is_empty());
    }

    #[test]
    fn test_seconds_to_duration_conversion() {
        let dur = Simulation::<SimStateMode, SimTestState>::seconds_to_duration(1.5);
        assert_eq!(dur, Duration::milliseconds(1500));

        let dur_small = Simulation::<SimStateMode, SimTestState>::seconds_to_duration(0.001);
        assert_eq!(dur_small, Duration::milliseconds(1));
    }
}
