//! A Claude-backed [`Mind`] for [`GenerativeAgent`](crate::generative::GenerativeAgent).
//!
//! Calls the Anthropic Messages API over blocking HTTP (so it slots into the synchronous
//! simulation loop) and uses structured outputs so every response parses. Anthropic exposes no
//! embeddings endpoint, so memories stay un-embedded and retrieval falls back to recency +
//! importance. Every method degrades gracefully on transport or parse failure, since the trait
//! methods can't surface an error.
//!
//! Gated behind the `llm` feature. Requires `ANTHROPIC_API_KEY`.

use crate::generative::Mind;
use crate::memory::{Insight, Memory, MemoryId, Reflector};
use crate::planning::{PlanContext, PlanStep, Planner, Reaction};
use crate::state::StateChangeEvent;
use chrono::{DateTime, Duration, Utc};
use serde_json::{Value, json};
use std::collections::HashSet;

const API_URL: &str = "https://api.anthropic.com/v1/messages";
const DEFAULT_MODEL: &str = "claude-opus-4-8";

pub struct LlmMind {
    api_key: String,
    pub identity: String,
    pub model: String,
    pub effort: String,
    pub max_tokens: u32,
    pub default_importance: f64,
}

impl LlmMind {
    pub fn new(api_key: impl Into<String>, identity: impl Into<String>) -> Self {
        LlmMind {
            api_key: api_key.into(),
            identity: identity.into(),
            model: DEFAULT_MODEL.to_string(),
            effort: "high".to_string(),
            max_tokens: 16000,
            default_importance: 1.0,
        }
    }

    // from_env reads the API key from ANTHROPIC_API_KEY.
    pub fn from_env(identity: impl Into<String>) -> Result<Self, std::env::VarError> {
        Ok(Self::new(std::env::var("ANTHROPIC_API_KEY")?, identity))
    }

    // call sends one structured-output request and returns the parsed JSON, or None on failure.
    // `think` enables adaptive thinking for the heavier reasoning calls.
    fn call(&self, system: &str, user: &str, schema: Value, think: bool) -> Option<Value> {
        let mut output_config = json!({ "format": { "type": "json_schema", "schema": schema } });
        let mut body = json!({
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": [{ "role": "user", "content": user }],
        });
        if think {
            body["thinking"] = json!({ "type": "adaptive" });
            output_config["effort"] = json!(self.effort);
        } else {
            output_config["effort"] = json!("low");
        }
        body["output_config"] = output_config;

        let response = match ureq::post(API_URL)
            .set("content-type", "application/json")
            .set("x-api-key", &self.api_key)
            .set("anthropic-version", "2023-06-01")
            .send_json(body)
        {
            Ok(response) => response,
            Err(err) => {
                eprintln!("agsim::llm request failed: {err}");
                return None;
            }
        };

        let value: Value = response.into_json().ok()?;
        let content = value.get("content")?.as_array()?;
        let text = content
            .iter()
            .find(|block| block.get("type").and_then(Value::as_str) == Some("text"))?
            .get("text")?
            .as_str()?;
        serde_json::from_str(text).ok()
    }
}

// parse_steps turns a JSON array of {activity, start_offset_minutes, duration_minutes} into plan
// steps timed relative to `base`. Zero/negative durations are dropped.
fn parse_steps(plan: &Value, base: DateTime<Utc>) -> Vec<PlanStep> {
    let Some(items) = plan.as_array() else {
        return Vec::new();
    };
    items
        .iter()
        .filter_map(|step| {
            let activity = step.get("activity")?.as_str()?;
            let offset = step.get("start_offset_minutes")?.as_i64()?;
            let duration = step.get("duration_minutes")?.as_i64()?;
            if duration <= 0 {
                return None;
            }
            Some(PlanStep::new(
                activity,
                base + Duration::minutes(offset),
                Duration::minutes(duration),
            ))
        })
        .collect()
}

fn steps_schema() -> Value {
    json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["plan"],
        "properties": {
            "plan": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": false,
                    "required": ["activity", "start_offset_minutes", "duration_minutes"],
                    "properties": {
                        "activity": { "type": "string" },
                        "start_offset_minutes": { "type": "integer" },
                        "duration_minutes": { "type": "integer" }
                    }
                }
            }
        }
    })
}

fn format_memories(memories: &[Memory]) -> String {
    if memories.is_empty() {
        return "(no relevant memories)".to_string();
    }
    memories
        .iter()
        .map(|memory| format!("- {}", memory.description))
        .collect::<Vec<_>>()
        .join("\n")
}

impl Planner for LlmMind {
    fn daily_plan(&self, ctx: &PlanContext) -> Vec<PlanStep> {
        let system = format!(
            "You are {}. You plan your day as a sequence of activities.",
            self.identity
        );
        let user = format!(
            "The current time is {}. Relevant context:\n{}\n\nPlan the next day as 5-8 broad \
             activities. Use start_offset_minutes relative to now (the first activity starts at 0) \
             and duration_minutes. Activities must be contiguous and cover roughly the next 24 hours.",
            ctx.now.format("%Y-%m-%d %H:%M"),
            format_memories(&ctx.memories),
        );
        match self.call(&system, &user, steps_schema(), true) {
            Some(value) => parse_steps(&value["plan"], ctx.now),
            None => Vec::new(),
        }
    }

    fn decompose(&self, step: &PlanStep, _ctx: &PlanContext) -> Vec<PlanStep> {
        let minutes = step.duration.num_minutes();
        let system = format!("You are {}.", self.identity);
        let user = format!(
            "You are planning this activity: \"{}\", lasting {} minutes. Break it into finer \
             sub-activities that tile the full duration. Use start_offset_minutes relative to the \
             start of this activity (the first sub-activity at 0) and duration_minutes. If the \
             activity is already atomic or under ~30 minutes, return an empty plan.",
            step.description, minutes,
        );
        match self.call(&system, &user, steps_schema(), true) {
            Some(value) => parse_steps(&value["plan"], step.start),
            None => Vec::new(),
        }
    }

    fn react(
        &self,
        observation: &Memory,
        current_action: Option<&PlanStep>,
        ctx: &PlanContext,
    ) -> Reaction {
        let current = current_action
            .map(|step| step.description.as_str())
            .unwrap_or("nothing in particular");
        let system = format!("You are {}.", self.identity);
        let user = format!(
            "The current time is {}. You are currently doing: \"{}\". You just observed: \"{}\". \
             Relevant context:\n{}\n\nShould this observation make you change your plan? If not, set \
             react to false. If yes, set react to true and provide a new plan for the rest of your \
             day (start_offset_minutes relative to now).",
            ctx.now.format("%Y-%m-%d %H:%M"),
            current,
            observation.description,
            format_memories(&ctx.memories),
        );
        let schema = json!({
            "type": "object",
            "additionalProperties": false,
            "required": ["react", "plan"],
            "properties": {
                "react": { "type": "boolean" },
                "plan": steps_schema()["properties"]["plan"],
            }
        });
        match self.call(&system, &user, schema, true) {
            Some(value) if value["react"].as_bool() == Some(true) => {
                Reaction::Replan(parse_steps(&value["plan"], ctx.now))
            }
            _ => Reaction::Continue,
        }
    }
}

impl Reflector for LlmMind {
    fn salient_questions(&self, recent: &[&Memory]) -> Vec<String> {
        let statements = recent
            .iter()
            .enumerate()
            .map(|(i, memory)| format!("{}. {}", i + 1, memory.description))
            .collect::<Vec<_>>()
            .join("\n");
        let system = format!("You are {}.", self.identity);
        let user = format!(
            "Recent observations from your memory:\n{statements}\n\nGiven only this information, \
             what are the 3 most salient high-level questions you can ask about the subjects in \
             these statements?"
        );
        let schema = json!({
            "type": "object",
            "additionalProperties": false,
            "required": ["questions"],
            "properties": { "questions": { "type": "array", "items": { "type": "string" } } }
        });
        match self.call(&system, &user, schema, false) {
            Some(value) => value["questions"]
                .as_array()
                .map(|items| {
                    items
                        .iter()
                        .filter_map(|q| q.as_str().map(str::to_string))
                        .collect()
                })
                .unwrap_or_default(),
            None => Vec::new(),
        }
    }

    fn synthesize(&self, question: &str, evidence: &[&Memory]) -> Vec<Insight> {
        let statements = evidence
            .iter()
            .map(|memory| format!("id {}: {}", memory.id, memory.description))
            .collect::<Vec<_>>()
            .join("\n");
        let valid: HashSet<MemoryId> = evidence.iter().map(|memory| memory.id).collect();

        let system = format!("You are {}.", self.identity);
        let user = format!(
            "Statements (with ids):\n{statements}\n\nFocal question: {question}\n\nWhat high-level \
             insights can you infer that help answer the question? For each insight, cite the ids \
             of the supporting statements and rate its importance from 1 (mundane) to 10 (critical)."
        );
        let schema = json!({
            "type": "object",
            "additionalProperties": false,
            "required": ["insights"],
            "properties": {
                "insights": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": false,
                        "required": ["insight", "importance", "evidence_ids"],
                        "properties": {
                            "insight": { "type": "string" },
                            "importance": { "type": "number" },
                            "evidence_ids": { "type": "array", "items": { "type": "integer" } }
                        }
                    }
                }
            }
        });

        let Some(value) = self.call(&system, &user, schema, true) else {
            return Vec::new();
        };
        let Some(items) = value["insights"].as_array() else {
            return Vec::new();
        };
        items
            .iter()
            .filter_map(|item| {
                let description = item.get("insight")?.as_str()?.to_string();
                let importance = item.get("importance")?.as_f64()?.clamp(1.0, 10.0);
                let evidence = item
                    .get("evidence_ids")?
                    .as_array()?
                    .iter()
                    .filter_map(|id| id.as_u64().map(|id| id as MemoryId))
                    .filter(|id| valid.contains(id))
                    .collect();
                Some(Insight {
                    description,
                    importance,
                    evidence,
                    embedding: None,
                })
            })
            .collect()
    }
}

impl Mind for LlmMind {
    fn importance(&self, event: &StateChangeEvent) -> f64 {
        let system = format!("You are {}.", self.identity);
        let user = format!(
            "On a scale of 1 (mundane, e.g. brushing teeth) to 10 (extremely poignant, e.g. a \
             breakup or college acceptance), rate the likely poignancy of this observation: \
             \"{}: {} -> {}\".",
            event.field, event.old_value, event.new_value,
        );
        let schema = json!({
            "type": "object",
            "additionalProperties": false,
            "required": ["importance"],
            "properties": { "importance": { "type": "number" } }
        });
        match self.call(&system, &user, schema, false) {
            Some(value) => value["importance"]
                .as_f64()
                .map(|score| score.clamp(1.0, 10.0))
                .unwrap_or(self.default_importance),
            None => self.default_importance,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn base() -> DateTime<Utc> {
        Utc.timestamp_opt(1_600_000_000, 0).unwrap()
    }

    #[test]
    fn test_parse_steps() {
        let plan = json!([
            { "activity": "work", "start_offset_minutes": 0, "duration_minutes": 120 },
            { "activity": "rest", "start_offset_minutes": 120, "duration_minutes": 60 },
            { "activity": "skip", "start_offset_minutes": 180, "duration_minutes": 0 }
        ]);
        let steps = parse_steps(&plan, base());

        // the zero-duration step is dropped.
        assert_eq!(steps.len(), 2);
        assert_eq!(steps[0].description, "work");
        assert_eq!(steps[0].start, base());
        assert_eq!(steps[1].start, base() + Duration::minutes(120));
        assert_eq!(steps[1].duration, Duration::minutes(60));
    }

    #[test]
    fn test_parse_steps_handles_non_array() {
        assert!(parse_steps(&json!("not a list"), base()).is_empty());
    }
}
