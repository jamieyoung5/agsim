pub mod anthropic;
pub mod openai;
pub mod parse;

#[cfg(feature = "local")]
pub mod candle;

pub use anthropic::Anthropic;
pub use openai::OpenAiCompat;

#[cfg(feature = "local")]
pub use candle::Candle;

use crate::generative::Mind;
use crate::memory::{Insight, Memory, MemoryId, Reflector};
use crate::planning::{PlanContext, PlanStep, Planner, Reaction};
use crate::state::StateChangeEvent;
use chrono::{DateTime, Duration, Utc};
use serde_json::{Value, json};
use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering};

pub struct Request<'a> {
    pub system: &'a str,
    pub user: &'a str,
    pub schema: &'a Value,
    pub reasoning: bool,
}

pub trait Backend {
    fn generate(&self, request: &Request) -> Option<String>;

    fn enforces_schema(&self) -> bool {
        false
    }

    fn attempts(&self) -> u32 {
        if self.enforces_schema() { 1 } else { 3 }
    }

    fn label(&self) -> String {
        "model".to_string()
    }
}

const DEFAULT_MIN_DECOMPOSE_MINUTES: i64 = 30;
const DEFAULT_DECOMPOSE_BUDGET: usize = 12;
const DEFAULT_MAX_DEPTH: usize = 2;

pub struct LlmMind<B> {
    pub identity: String,
    pub default_importance: f64,
    pub min_decompose: Duration,
    pub decompose_budget: usize,
    pub max_depth: usize,
    remaining_decompose: AtomicUsize,
    pub backend: B,
}

pub type ServerMind = LlmMind<OpenAiCompat>;

#[cfg(feature = "local")]
pub type LocalMind = LlmMind<Candle>;

impl<B: Backend> LlmMind<B> {
    pub fn with_backend(backend: B, identity: impl Into<String>) -> Self {
        LlmMind {
            identity: identity.into(),
            default_importance: 1.0,
            min_decompose: Duration::minutes(DEFAULT_MIN_DECOMPOSE_MINUTES),
            decompose_budget: DEFAULT_DECOMPOSE_BUDGET,
            max_depth: DEFAULT_MAX_DEPTH,
            remaining_decompose: AtomicUsize::new(DEFAULT_DECOMPOSE_BUDGET),
            backend,
        }
    }

    pub fn with_default_importance(mut self, importance: f64) -> Self {
        self.default_importance = importance;
        self
    }

    pub fn with_min_decompose(mut self, minimum: Duration) -> Self {
        self.min_decompose = minimum;
        self
    }

    pub fn with_decompose_budget(mut self, budget: usize) -> Self {
        self.decompose_budget = budget;
        self.remaining_decompose = AtomicUsize::new(budget);
        self
    }

    pub fn with_max_depth(mut self, depth: usize) -> Self {
        self.max_depth = depth;
        self
    }

    fn spend_decompose(&self) -> bool {
        self.remaining_decompose
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |remaining| {
                remaining.checked_sub(1)
            })
            .is_ok()
    }

    fn ask(&self, system: &str, user: &str, schema: Value, reasoning: bool) -> Option<Value> {
        let user = if self.backend.enforces_schema() {
            user.to_string()
        } else {
            format!("{user}\n\n{}", parse::instructions(&schema))
        };

        let request = Request {
            system,
            user: &user,
            schema: &schema,
            reasoning,
        };

        for attempt in 1..=self.backend.attempts().max(1) {
            let Some(reply) = self.backend.generate(&request) else {
                continue;
            };
            if let Some(value) = parse::extract_json(&reply) {
                return Some(value);
            }
            eprintln!(
                "agsim::llm: unparseable reply from {} (attempt {attempt})",
                self.backend.label()
            );
        }

        None
    }
}

impl LlmMind<Anthropic> {
    pub fn new(api_key: impl Into<String>, identity: impl Into<String>) -> Self {
        Self::with_backend(Anthropic::new(api_key), identity)
    }

    pub fn from_env(identity: impl Into<String>) -> Result<Self, std::env::VarError> {
        Ok(Self::new(std::env::var("ANTHROPIC_API_KEY")?, identity))
    }
}
fn parse_steps(plan: &Value, base: DateTime<Utc>) -> Vec<PlanStep> {
    let mut steps = Vec::new();
    let mut cursor = 0;

    for item in parse::array(plan) {
        let Some(activity) = parse::field(item, &["activity", "action", "description", "name"])
            .and_then(parse::text)
        else {
            continue;
        };
        let Some(duration) = parse::field(
            item,
            &["duration_minutes", "duration", "minutes", "length_minutes"],
        )
        .and_then(parse::integer) else {
            continue;
        };
        if duration <= 0 {
            continue;
        }

        let offset = parse::field(
            item,
            &[
                "start_offset_minutes",
                "start_offset",
                "start_minutes",
                "start",
            ],
        )
        .and_then(parse::integer)
        .unwrap_or(cursor);

        cursor = offset + duration;
        steps.push(PlanStep::new(
            activity,
            base + Duration::minutes(offset),
            Duration::minutes(duration),
        ));
    }

    steps
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

impl<B: Backend> Planner for LlmMind<B> {
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

        self.remaining_decompose
            .store(self.decompose_budget, Ordering::Relaxed);

        match self.ask(&system, &user, steps_schema(), true) {
            Some(value) => parse_steps(&value["plan"], ctx.now),
            None => Vec::new(),
        }
    }

    fn decompose(&self, step: &PlanStep, _ctx: &PlanContext) -> Vec<PlanStep> {
        if step.duration <= self.min_decompose || !self.spend_decompose() {
            return Vec::new();
        }

        let minutes = step.duration.num_minutes();
        let floor = self.min_decompose.num_minutes();
        let system = format!("You are {}.", self.identity);
        let user = format!(
            "You are planning this activity: \"{}\", lasting {} minutes. Break it into finer \
             sub-activities that tile the full duration. Use start_offset_minutes relative to the \
             start of this activity (the first sub-activity at 0) and duration_minutes. Keep \
             sub-activities to a handful of substantial blocks, none shorter than {} minutes. If \
             the activity is already atomic, return an empty plan.",
            step.description, minutes, floor,
        );
        match self.ask(&system, &user, steps_schema(), true) {
            Some(value) => parse_steps(&value["plan"], step.start),
            None => Vec::new(),
        }
    }

    fn max_depth(&self) -> usize {
        self.max_depth
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
             react to false and leave plan empty. If yes, set react to true and provide a new plan \
             for the rest of your day (start_offset_minutes relative to now).",
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

        let Some(value) = self.ask(&system, &user, schema, true) else {
            return Reaction::Continue;
        };
        let reacting = parse::field(&value, &["react", "replan", "change_plan"])
            .and_then(parse::boolean)
            .unwrap_or(false);
        if !reacting {
            return Reaction::Continue;
        }

        let steps = parse_steps(&value["plan"], ctx.now);
        if steps.is_empty() {
            Reaction::Continue
        } else {
            Reaction::Replan(steps)
        }
    }
}

impl<B: Backend> Reflector for LlmMind<B> {
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
        match self.ask(&system, &user, schema, false) {
            Some(value) => match parse::field(&value, &["questions", "question"]) {
                Some(questions) => parse::array(questions)
                    .into_iter()
                    .filter_map(parse::text)
                    .collect(),
                None => Vec::new(),
            },
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

        let Some(value) = self.ask(&system, &user, schema, true) else {
            return Vec::new();
        };
        let Some(items) = parse::field(&value, &["insights", "insight"]) else {
            return Vec::new();
        };

        parse::array(items)
            .into_iter()
            .filter_map(|item| {
                let description = parse::field(item, &["insight", "description", "text"])
                    .and_then(parse::text)?;
                let importance = parse::field(item, &["importance", "score", "poignancy"])
                    .and_then(parse::number)
                    .unwrap_or(self.default_importance)
                    .clamp(1.0, 10.0);
                let evidence = parse::field(item, &["evidence_ids", "evidence", "ids"])
                    .map(|ids| {
                        parse::array(ids)
                            .into_iter()
                            .filter_map(parse::integer)
                            .filter(|id| *id >= 0)
                            .map(|id| id as MemoryId)
                            .filter(|id| valid.contains(id))
                            .collect()
                    })
                    .unwrap_or_default();
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

impl<B: Backend> Mind for LlmMind<B> {
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
        match self.ask(&system, &user, schema, false) {
            Some(value) => parse::field(&value, &["importance", "score", "poignancy", "rating"])
                .and_then(parse::number)
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
    use std::cell::RefCell;

    fn base() -> DateTime<Utc> {
        Utc.timestamp_opt(1_600_000_000, 0).unwrap()
    }

    fn ctx() -> PlanContext {
        PlanContext {
            identity: "a test device".to_string(),
            now: base(),
            memories: Vec::new(),
        }
    }

    struct ScriptedBackend {
        replies: RefCell<Vec<Option<String>>>,
        prompts: RefCell<Vec<String>>,
        enforces: bool,
    }

    impl ScriptedBackend {
        fn new(replies: Vec<Option<&str>>) -> Self {
            ScriptedBackend {
                replies: RefCell::new(
                    replies
                        .into_iter()
                        .map(|reply| reply.map(str::to_string))
                        .collect(),
                ),
                prompts: RefCell::new(Vec::new()),
                enforces: false,
            }
        }

        fn strict(mut self) -> Self {
            self.enforces = true;
            self
        }

        fn calls(&self) -> usize {
            self.prompts.borrow().len()
        }

        fn last_prompt(&self) -> String {
            self.prompts.borrow().last().cloned().unwrap_or_default()
        }
    }

    impl Backend for ScriptedBackend {
        fn generate(&self, request: &Request) -> Option<String> {
            self.prompts.borrow_mut().push(request.user.to_string());
            let mut replies = self.replies.borrow_mut();
            if replies.is_empty() {
                return None;
            }
            replies.remove(0)
        }

        fn enforces_schema(&self) -> bool {
            self.enforces
        }
    }

    fn scripted(replies: Vec<Option<&str>>) -> LlmMind<ScriptedBackend> {
        LlmMind::with_backend(ScriptedBackend::new(replies), "a test device")
    }

    #[test]
    fn test_parse_steps() {
        let plan = json!([
            { "activity": "work", "start_offset_minutes": 0, "duration_minutes": 120 },
            { "activity": "rest", "start_offset_minutes": 120, "duration_minutes": 60 },
            { "activity": "skip", "start_offset_minutes": 180, "duration_minutes": 0 }
        ]);
        let steps = parse_steps(&plan, base());

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

    #[test]
    fn test_parse_steps_tolerates_loose_fields() {
        let plan = json!([
            { "action": "morning work", "duration": "120" },
            { "description": "lunch", "minutes": 60 },
            { "activity": "afternoon work", "start": 180, "duration_minutes": 90 }
        ]);
        let steps = parse_steps(&plan, base());

        assert_eq!(steps.len(), 3);
        assert_eq!(steps[0].start, base());
        assert_eq!(steps[1].start, base() + Duration::minutes(120));
        assert_eq!(steps[2].start, base() + Duration::minutes(180));
        assert_eq!(steps[2].duration, Duration::minutes(90));
    }

    #[test]
    fn test_prompt_carries_an_example_when_schema_is_not_enforced() {
        let mind = scripted(vec![Some(r#"{"plan": []}"#)]);
        mind.daily_plan(&ctx());

        let prompt = mind.backend.last_prompt();
        assert!(prompt.contains("JSON only"));
        assert!(prompt.contains("\"start_offset_minutes\": 0"));
    }

    #[test]
    fn test_prompt_stays_clean_when_the_backend_enforces_the_schema() {
        let mind = LlmMind::with_backend(
            ScriptedBackend::new(vec![Some(r#"{"plan": []}"#)]).strict(),
            "a test device",
        );
        mind.daily_plan(&ctx());

        assert!(!mind.backend.last_prompt().contains("JSON only"));
    }

    #[test]
    fn test_retries_unparseable_replies() {
        let mind = scripted(vec![
            Some("I'm not sure I understand the question."),
            Some("```json\n{\"plan\": [{\"activity\": \"rest\", \"duration_minutes\": 60}]}\n```"),
        ]);

        let steps = mind.daily_plan(&ctx());
        assert_eq!(mind.backend.calls(), 2);
        assert_eq!(steps.len(), 1);
        assert_eq!(steps[0].description, "rest");
    }

    #[test]
    fn test_gives_up_after_exhausting_attempts() {
        let mind = scripted(vec![Some("no"), Some("still no"), Some("nope")]);
        assert!(mind.daily_plan(&ctx()).is_empty());
        assert_eq!(mind.backend.calls(), 3);
    }

    #[test]
    fn test_importance_falls_back_when_the_model_fails() {
        let event = StateChangeEvent {
            time: base(),
            agent_id: "device".to_string(),
            field: "cpu".to_string(),
            old_value: "1".to_string(),
            new_value: "90".to_string(),
        };

        let mind = scripted(vec![Some(r#"{"importance": "8.5"}"#)]);
        assert_eq!(mind.importance(&event), 8.5);

        let mind = scripted(vec![Some(r#"{"score": 99}"#)]);
        assert_eq!(mind.importance(&event), 10.0);

        let mind = scripted(vec![None, None, None]);
        assert_eq!(mind.importance(&event), 1.0);
    }

    #[test]
    fn test_react_needs_a_usable_plan() {
        let observation = Memory::new("cpu spiked", 9.0, base());

        let mind = scripted(vec![Some(r#"{"react": true, "plan": []}"#)]);
        assert!(matches!(
            mind.react(&observation, None, &ctx()),
            Reaction::Continue
        ));

        let mind = scripted(vec![Some(
            r#"{"react": "yes", "plan": [{"action": "shut down", "duration": 30}]}"#,
        )]);
        match mind.react(&observation, None, &ctx()) {
            Reaction::Replan(steps) => {
                assert_eq!(steps.len(), 1);
                assert_eq!(steps[0].description, "shut down");
            }
            Reaction::Continue => panic!("expected a replan"),
        }
    }

    #[test]
    fn test_synthesize_keeps_only_cited_evidence() {
        let first = Memory {
            id: 1,
            ..Memory::new("cpu spiked", 5.0, base())
        };
        let second = Memory {
            id: 2,
            ..Memory::new("went offline", 5.0, base())
        };
        let evidence = vec![&first, &second];

        let mind = scripted(vec![Some(
            r#"{"insights": [
                {"insight": "the device overheats", "importance": 7, "evidence_ids": [1, 2, 99]},
                {"importance": 3, "evidence_ids": [1]}
            ]}"#,
        )]);

        let insights = mind.synthesize("what is happening?", &evidence);

        assert_eq!(insights.len(), 1);
        assert_eq!(insights[0].description, "the device overheats");
        assert_eq!(insights[0].importance, 7.0);
        assert_eq!(insights[0].evidence, vec![1, 2]);
    }

    fn step(minutes: i64) -> PlanStep {
        PlanStep::new("a long stretch of work", base(), Duration::minutes(minutes))
    }

    #[test]
    fn test_short_steps_are_not_sent_to_the_model() {
        let mind = scripted(vec![Some(
            r#"{"plan": [{"activity": "x", "duration": 5}]}"#,
        )]);

        assert!(mind.decompose(&step(20), &ctx()).is_empty());
        assert!(mind.decompose(&step(30), &ctx()).is_empty());
        assert_eq!(mind.backend.calls(), 0);

        assert_eq!(mind.decompose(&step(60), &ctx()).len(), 1);
        assert_eq!(mind.backend.calls(), 1);
    }

    #[test]
    fn test_decompose_budget_bounds_one_plan() {
        let reply = r#"{"plan": [{"activity": "x", "duration": 45}]}"#;
        let mind = scripted(vec![Some(reply); 6]).with_decompose_budget(2);

        for _ in 0..5 {
            mind.decompose(&step(120), &ctx());
        }
        assert_eq!(mind.backend.calls(), 2);
    }

    #[test]
    fn test_generating_a_plan_refreshes_the_budget() {
        let plan = r#"{"plan": [{"activity": "morning", "duration": 240}]}"#;
        let mind = scripted(vec![Some(plan); 6]).with_decompose_budget(1);

        mind.decompose(&step(120), &ctx());
        assert!(mind.decompose(&step(120), &ctx()).is_empty());

        mind.daily_plan(&ctx());
        assert!(!mind.decompose(&step(120), &ctx()).is_empty());
    }

    #[test]
    fn test_default_depth_splits_the_day_once() {
        let mind = scripted(vec![Some("{}"); 40]);
        assert_eq!(Planner::max_depth(&mind), 2);
        assert_eq!(Planner::max_depth(&mind.with_max_depth(3)), 3);
    }

    #[test]
    fn test_salient_questions_accepts_a_bare_list() {
        let mind = scripted(vec![Some(r#"{"questions": ["why?", "when?"]}"#)]);
        assert_eq!(mind.salient_questions(&[]).len(), 2);

        let mind = scripted(vec![Some(r#"{"question": "why?"}"#)]);
        assert_eq!(mind.salient_questions(&[]), vec!["why?".to_string()]);
    }
}
