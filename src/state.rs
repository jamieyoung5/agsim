use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;
use std::sync::Arc;

pub type AgentId = Arc<str>;

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(untagged)]
pub enum Value {
    Int(i64),
    Float(f64),
    Bool(bool),
    Text(Arc<str>),
}

impl Value {
    pub fn text(value: impl fmt::Display) -> Self {
        Value::Text(Arc::from(value.to_string().as_str()))
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::Int(v) => Some(*v as f64),
            Value::Float(v) => Some(*v),
            Value::Bool(_) | Value::Text(_) => None,
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Value::Int(v) => Some(*v),
            Value::Float(v) => Some(*v as i64),
            Value::Bool(_) | Value::Text(_) => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Text(v) => Some(v),
            _ => None,
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Int(v) => write!(f, "{v}"),
            Value::Float(v) => write!(f, "{v}"),
            Value::Bool(v) => write!(f, "{v}"),
            Value::Text(v) => write!(f, "{v}"),
        }
    }
}

pub trait ToValue {
    fn to_value(&self) -> Value;
}

macro_rules! impl_to_value {
    ($variant:ident, $cast:ty, $($ty:ty),+) => {
        $(
            impl ToValue for $ty {
                fn to_value(&self) -> Value {
                    Value::$variant(*self as $cast)
                }
            }
        )+
    };
}

impl_to_value!(Int, i64, i8, i16, i32, i64, isize, u8, u16, u32);
impl_to_value!(Float, f64, f32, f64);

// wider than i64, so Float
macro_rules! impl_to_value_wide {
    ($($ty:ty),+) => {
        $(
            impl ToValue for $ty {
                fn to_value(&self) -> Value {
                    match i64::try_from(*self) {
                        Ok(value) => Value::Int(value),
                        Err(_) => Value::Float(*self as f64),
                    }
                }
            }
        )+
    };
}

impl_to_value_wide!(u64, usize);

impl ToValue for bool {
    fn to_value(&self) -> Value {
        Value::Bool(*self)
    }
}

impl ToValue for String {
    fn to_value(&self) -> Value {
        Value::Text(Arc::from(self.as_str()))
    }
}

impl ToValue for &str {
    fn to_value(&self) -> Value {
        Value::Text(Arc::from(*self))
    }
}

impl ToValue for char {
    fn to_value(&self) -> Value {
        Value::Text(Arc::from(self.to_string().as_str()))
    }
}

impl<T: ToValue> ToValue for &T {
    fn to_value(&self) -> Value {
        (*self).to_value()
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct StateChangeEvent {
    #[serde(rename = "Time")]
    pub time: DateTime<Utc>,
    #[serde(rename = "AgentId")]
    pub agent_id: AgentId,
    #[serde(rename = "Field")]
    pub field: Cow<'static, str>,
    #[serde(rename = "NewValue")]
    pub new_value: Value,
    #[serde(rename = "OldValue")]
    pub old_value: Value,
}

pub trait State: Sized + Clone + Default {
    /// Appends one event per differing field.
    fn diff(
        &self,
        other: &Self,
        agent_id: &AgentId,
        time: DateTime<Utc>,
        out: &mut Vec<StateChangeEvent>,
    );
}

#[derive(Debug, Clone)]
pub struct TimelineEntry {
    pub timestamp: DateTime<Utc>,
    pub state: BTreeMap<String, Value>,
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
    pub fn generate(events: &[StateChangeEvent]) -> HashMap<AgentId, Timeline> {
        let mut timelines = HashMap::new();

        if events.is_empty() {
            return timelines;
        }

        let mut events_by_agent: HashMap<AgentId, Vec<StateChangeEvent>> = HashMap::new();
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
            if !seen_fields.contains(event.field.as_ref()) {
                current_state.insert(event.field.to_string(), event.old_value.clone());
                seen_fields.insert(event.field.to_string());
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
                current_state.insert(event.field.to_string(), event.new_value.clone());
                changed_fields.push(event.field.to_string());
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

    fn event(
        agent: &AgentId,
        field: &'static str,
        old: Value,
        new: Value,
        time: DateTime<Utc>,
    ) -> StateChangeEvent {
        StateChangeEvent {
            time,
            agent_id: agent.clone(),
            field: Cow::Borrowed(field),
            old_value: old,
            new_value: new,
        }
    }

    #[test]
    fn test_timeline_generation_single_agent() {
        let base_time = Utc.timestamp_opt(1600000000, 0).unwrap();
        let agent: AgentId = Arc::from("agent_A");

        let events = vec![
            event(
                &agent,
                "status",
                Value::text("init"),
                Value::text("running"),
                base_time,
            ),
            event(
                &agent,
                "load",
                Value::Int(0),
                Value::Int(50),
                base_time + Duration::seconds(10),
            ),
            event(
                &agent,
                "status",
                Value::text("running"),
                Value::text("busy"),
                base_time + Duration::seconds(10),
            ),
        ];

        let timelines = Timeline::generate(&events);

        assert!(timelines.contains_key("agent_A"));
        let timeline = timelines.get("agent_A").unwrap();

        assert_eq!(timeline.entries.len(), 3);

        let init_entry = &timeline.entries[0];
        assert_eq!(init_entry.state.get("status").unwrap().to_string(), "init");
        assert_eq!(init_entry.state.get("load").unwrap().to_string(), "0");

        let first_trans = &timeline.entries[1];
        assert_eq!(first_trans.timestamp, base_time);
        assert_eq!(
            first_trans.state.get("status").unwrap().to_string(),
            "running"
        );
        assert_eq!(first_trans.events, vec!["status"]);

        let second_trans = &timeline.entries[2];
        assert_eq!(second_trans.timestamp, base_time + Duration::seconds(10));
        assert_eq!(second_trans.state.get("load").unwrap().to_string(), "50");
        assert_eq!(
            second_trans.state.get("status").unwrap().to_string(),
            "busy"
        );
        assert!(second_trans.events.contains(&"load".to_string()));
        assert!(second_trans.events.contains(&"status".to_string()));
    }

    #[test]
    fn test_timeline_multi_agent_separation() {
        let time = Utc::now();
        let a: AgentId = Arc::from("A");
        let b: AgentId = Arc::from("B");
        let events = vec![
            event(&a, "f", Value::Int(0), Value::Int(1), time),
            event(&b, "f", Value::Int(0), Value::Int(2), time),
        ];

        let timelines = Timeline::generate(&events);
        assert_eq!(timelines.len(), 2);
        assert!(timelines.contains_key("A"));
        assert!(timelines.contains_key("B"));
    }

    #[test]
    fn test_primitive_fields_convert() {
        assert_eq!(7i32.to_value(), Value::Int(7));
        assert_eq!(7u8.to_value(), Value::Int(7));
        assert_eq!(1.5f32.to_value(), Value::Float(1.5));
        assert_eq!(true.to_value(), Value::Bool(true));
        assert_eq!(String::from("hi").to_value(), Value::Text(Arc::from("hi")));
    }

    #[test]
    fn test_field_names_clone_borrowed() {
        let agent: AgentId = Arc::from("a");
        let original = event(&agent, "cpu", Value::Int(1), Value::Int(2), Utc::now());
        let copy = original.clone();

        assert!(matches!(copy.field, Cow::Borrowed(_)));
        assert!(Arc::ptr_eq(&original.agent_id, &copy.agent_id));
    }

    #[test]
    fn test_event_round_trips_through_serde() {
        let agent: AgentId = Arc::from("a");
        let original = event(
            &agent,
            "cpu",
            Value::Int(1),
            Value::Float(2.5),
            Utc.timestamp_opt(1600000000, 0).unwrap(),
        );

        let json = serde_json::to_string(&original).unwrap();
        let restored: StateChangeEvent = serde_json::from_str(&json).unwrap();

        assert_eq!(original, restored);
    }

    // untagged picks the first fit
    #[test]
    fn test_value_variants_round_trip() {
        for value in [
            Value::Int(5),
            Value::Int(-5),
            Value::Int(i64::MAX),
            Value::Float(2.5),
            Value::Float(2.0),
            Value::Bool(true),
            Value::Bool(false),
            Value::text("hi"),
        ] {
            let json = serde_json::to_string(&value).unwrap();
            let restored: Value = serde_json::from_str(&json).unwrap();
            assert_eq!(value, restored, "json {json}");
        }
    }

    #[test]
    fn test_unsigned_beyond_i64() {
        assert_eq!(u64::MAX.to_value(), Value::Float(u64::MAX as f64));
        assert!(u64::MAX.to_value().as_f64().unwrap() > 0.0);
    }
}
