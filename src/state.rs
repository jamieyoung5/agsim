use std::collections::{BTreeMap, HashSet};
use std::fmt;
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
#[derive(Clone)]
pub struct StateChangeEvent {
    #[serde(rename = "Time")]
    pub time: DateTime<Utc>,
    #[serde(rename = "Field")]
    pub field: String,
    #[serde(rename = "NewValue")]
    pub new_value: String,
    #[serde(rename = "OldValue")]
    pub old_value: String,
}

pub trait State: Sized + Clone + Default {
    fn update_field(&mut self, field: &str, value: &str);
    fn get_field(&self, field: &str) -> String;
    fn get_field_names() -> &'static [&'static str];
}

#[derive(Debug, Clone)]
pub struct TimelineEntry<S> {
    pub timestamp: DateTime<Utc>,
    pub state: S,
    pub events: Vec<String>,
}

impl<S: fmt::Display> fmt::Display for TimelineEntry<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let event_str = if self.events.is_empty() {
            "Initial State".to_string()
        } else {
            format!("Events: {}", self.events.join(", "))
        };
        write!(
            f,
            "[{}] State -> [ {} ] *({})*",
            self.timestamp.format("%Y-%m-%d %H:%M:%S"),
            self.state,
            event_str
        )
    }
}

pub struct Timeline<S> {
    pub entries: Vec<TimelineEntry<S>>,
}

impl<S: State> Timeline<S> {

    pub fn generate(events: &[StateChangeEvent]) -> Option<Self> {
        if events.is_empty() {
            return None;
        }

        let mut sorted_events = events.to_vec();
        sorted_events.sort_by_key(|e| e.time);

        let mut current_state = S::default();
        let mut seen_fields = HashSet::new();
        for event in &sorted_events {
            if seen_fields.insert(event.field.clone()) {
                current_state.update_field(&event.field, &event.old_value);
            }
        }

        let mut events_by_time = BTreeMap::<DateTime<Utc>, Vec<&StateChangeEvent>>::new();
        for event in &sorted_events {
            events_by_time.entry(event.time).or_default().push(event);
        }

        let mut entries = Vec::new();

        entries.push(TimelineEntry {
            timestamp: sorted_events[0].time - Duration::seconds(1),
            state: current_state.clone(),
            events: Vec::new(),
        });

        for (timestamp, event_group) in events_by_time {
            let mut state_for_this_entry = current_state.clone();
            let mut changed_fields = Vec::new();

            for event in event_group {
                state_for_this_entry.update_field(&event.field, &event.new_value);
                changed_fields.push(event.field.clone());
            }

            entries.push(TimelineEntry {
                timestamp,
                state: state_for_this_entry.clone(),
                events: changed_fields,
            });

            current_state = state_for_this_entry;
        }

        Some(Timeline { entries })
    }
}

impl<S: fmt::Display> fmt::Display for Timeline<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for entry in &self.entries {
            writeln!(f, "{}", entry)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use state_macros::{State, StateDisplay};
    use super::*;

    #[derive(Clone, Default, Debug, State, StateDisplay)]
    struct TestState {
        property1: String,
        property2: String,
    }

    #[test]
    fn test_timeline() {
        let entries: &[StateChangeEvent] = &[
            StateChangeEvent {
                time: Utc::now(),
                field: "property1".to_string(),
                new_value: "1".to_string(),
                old_value: "0".to_string(),
            },
            StateChangeEvent {
                time: Utc::now() + Duration::seconds(1),
                field: "property1".to_string(),
                new_value: "0".to_string(),
                old_value: "1".to_string(),
            },
            StateChangeEvent {
                time: Utc::now(),
                field: "property2".to_string(),
                new_value: "1".to_string(),
                old_value: "0".to_string(),
            },
        ];

        let timeline = Timeline::<TestState>::generate(&*entries).unwrap();
        println!("{}", timeline)
    }
}