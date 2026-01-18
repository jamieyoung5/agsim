use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct StateChangeEvent {
    #[serde(rename = "Time")]
    pub time: DateTime<Utc>,
    #[serde(rename = "AgentId")]
    pub agent_id: String,
    #[serde(rename = "Field")]
    pub field: String,
    #[serde(rename = "NewValue")]
    pub new_value: String,
    #[serde(rename = "OldValue")]
    pub old_value: String,
}

pub trait State: Sized + Clone + Default {
    fn diff(&self, other: &Self, time: DateTime<Utc>) -> Vec<StateChangeEvent>;
}

#[derive(Debug, Clone)]
pub struct TimelineEntry {
    pub timestamp: DateTime<Utc>,
    pub state: BTreeMap<String, String>,
    pub events: Vec<String>,
}

impl fmt::Display for TimelineEntry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let event_str = if self.events.is_empty() {
            "Initial State".to_string()
        } else {
            format!("Events: {}", self.events.join(", "))
        };

        let state_str = self
            .state
            .iter()
            .map(|(k, v)| format!("{}: {}", k, v))
            .collect::<Vec<_>>()
            .join(" | ");

        write!(
            f,
            "[{}] State -> [ {} ] *({})*",
            self.timestamp.format("%Y-%m-%d %H:%M:%S"),
            state_str,
            event_str
        )
    }
}

pub struct Timeline {
    pub entries: Vec<TimelineEntry>,
}

impl Timeline {
    pub fn generate(events: &[StateChangeEvent]) -> HashMap<String, Timeline> {
        let mut timelines = HashMap::new();

        if events.is_empty() {
            return timelines;
        }

        let mut events_by_agent: HashMap<String, Vec<StateChangeEvent>> = HashMap::new();
        for event in events {
            events_by_agent
                .entry(event.agent_id.clone())
                .or_default()
                .push(event.clone());
        }

        for (agent_id, agent_events) in events_by_agent {
            if let Some(timeline) = Self::generate_single_timeline(&agent_events) {
                timelines.insert(agent_id, timeline);
            }
        }

        timelines
    }

    fn generate_single_timeline(events: &[StateChangeEvent]) -> Option<Self> {
        if events.is_empty() {
            return None;
        }

        let mut sorted_events = events.to_vec();
        sorted_events.sort_by_key(|e| e.time);

        let mut current_state = BTreeMap::new();
        let mut seen_fields = HashSet::new();

        for event in &sorted_events {
            if !seen_fields.contains(&event.field) {
                current_state.insert(event.field.clone(), event.old_value.clone());
                seen_fields.insert(event.field.clone());
            }
        }

        let mut events_by_time = BTreeMap::<DateTime<Utc>, Vec<&StateChangeEvent>>::new();
        for event in &sorted_events {
            events_by_time.entry(event.time).or_default().push(event);
        }

        let mut entries = Vec::new();

        if !sorted_events.is_empty() {
            entries.push(TimelineEntry {
                timestamp: sorted_events[0].time - Duration::seconds(1),
                state: current_state.clone(),
                events: Vec::new(),
            });
        }

        for (timestamp, event_group) in events_by_time {
            let mut changed_fields = Vec::new();

            for event in event_group {
                current_state.insert(event.field.clone(), event.new_value.clone());
                changed_fields.push(event.field.clone());
            }

            entries.push(TimelineEntry {
                timestamp,
                state: current_state.clone(),
                events: changed_fields,
            });
        }

        Some(Timeline { entries })
    }
}

impl fmt::Display for Timeline {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for entry in &self.entries {
            writeln!(f, "{}", entry)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    #[test]
    fn test_timeline_generation_single_agent() {
        let base_time = Utc.timestamp_opt(1600000000, 0).unwrap();

        let events = vec![
            StateChangeEvent {
                time: base_time,
                agent_id: "agent_A".to_string(),
                field: "status".to_string(),
                old_value: "init".to_string(),
                new_value: "running".to_string(),
            },
            StateChangeEvent {
                time: base_time + Duration::seconds(10),
                agent_id: "agent_A".to_string(),
                field: "load".to_string(),
                old_value: "0".to_string(),
                new_value: "50".to_string(),
            },
            StateChangeEvent {
                time: base_time + Duration::seconds(10),
                agent_id: "agent_A".to_string(),
                field: "status".to_string(),
                old_value: "running".to_string(),
                new_value: "busy".to_string(),
            },
        ];

        let timelines = Timeline::generate(&events);

        assert!(timelines.contains_key("agent_A"));
        let timeline = timelines.get("agent_A").unwrap();

        assert_eq!(timeline.entries.len(), 3);

        let init_entry = &timeline.entries[0];
        assert_eq!(init_entry.state.get("status").unwrap(), "init");
        assert_eq!(init_entry.state.get("load").unwrap(), "0");

        let first_trans = &timeline.entries[1];
        assert_eq!(first_trans.timestamp, base_time);
        assert_eq!(first_trans.state.get("status").unwrap(), "running");
        assert_eq!(first_trans.events, vec!["status"]);

        let second_trans = &timeline.entries[2];
        assert_eq!(second_trans.timestamp, base_time + Duration::seconds(10));
        assert_eq!(second_trans.state.get("load").unwrap(), "50");
        assert_eq!(second_trans.state.get("status").unwrap(), "busy");
        assert!(second_trans.events.contains(&"load".to_string()));
        assert!(second_trans.events.contains(&"status".to_string()));
    }

    #[test]
    fn test_timeline_multi_agent_separation() {
        let time = Utc::now();
        let events = vec![
            StateChangeEvent {
                time,
                agent_id: "A".to_string(),
                field: "f".to_string(),
                old_value: "0".to_string(),
                new_value: "1".to_string(),
            },
            StateChangeEvent {
                time,
                agent_id: "B".to_string(),
                field: "f".to_string(),
                old_value: "0".to_string(),
                new_value: "2".to_string(),
            },
        ];

        let timelines = Timeline::generate(&events);
        assert_eq!(timelines.len(), 2);
        assert!(timelines.contains_key("A"));
        assert!(timelines.contains_key("B"));
    }
}
