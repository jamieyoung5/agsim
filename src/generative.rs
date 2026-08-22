use crate::agent::SimAgent;
use crate::memory::{Insight, Memory, MemoryStream, Reflector};
use crate::planning::{Plan, PlanContext, PlanStep, Planner, Reaction};
use crate::space::Position;
use crate::state::{AgentId, State, StateChangeEvent};
use chrono::{DateTime, Utc};
use rand::RngCore;
use std::collections::HashMap;
use std::sync::Arc;

const REACTION_CONTEXT: usize = 10;

/// Plans, reflects, and rates observations.
pub trait Mind: Planner + Reflector {
    fn importance(&self, event: &StateChangeEvent) -> f64;
}

/// Rates an observation from 1 to 10.
pub trait Importance {
    fn importance(&self, event: &StateChangeEvent) -> f64;
}

pub const IMPORTANCE_FLOOR: f64 = 1.0;
pub const IMPORTANCE_CEILING: f64 = 10.0;

/// Rates an observation by delta size.
#[derive(Debug, Clone)]
pub struct MagnitudeImportance {
    scales: HashMap<String, f64>,
    categorical: f64,
}

impl Default for MagnitudeImportance {
    fn default() -> Self {
        MagnitudeImportance {
            scales: HashMap::new(),
            // non-numeric has no size
            categorical: (IMPORTANCE_FLOOR + IMPORTANCE_CEILING) / 2.0,
        }
    }
}

impl MagnitudeImportance {
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the delta that rates ceiling.
    pub fn with_scale(mut self, field: impl Into<String>, scale: f64) -> Self {
        self.scales.insert(field.into(), scale);
        self
    }

    /// Sets the non-numeric rating.
    pub fn with_categorical(mut self, rating: f64) -> Self {
        self.categorical = rating.clamp(IMPORTANCE_FLOOR, IMPORTANCE_CEILING);
        self
    }
}

impl Importance for MagnitudeImportance {
    fn importance(&self, event: &StateChangeEvent) -> f64 {
        let (Some(old), Some(new)) = (event.old_value.as_f64(), event.new_value.as_f64()) else {
            return self.categorical;
        };

        let delta = (new - old).abs();
        if !delta.is_finite() {
            return self.categorical;
        }

        let ratio = match self.scales.get(event.field.as_ref()) {
            Some(&scale) if scale.is_finite() && scale > 0.0 => delta / scale,
            // no scale, use old value
            _ => delta / old.abs().max(f64::EPSILON),
        };

        IMPORTANCE_FLOOR + (IMPORTANCE_CEILING - IMPORTANCE_FLOOR) * ratio.clamp(0.0, 1.0)
    }
}

/// Plans through `mind`, rates through `rating`.
pub struct WithImportance<M, I> {
    mind: M,
    rating: I,
}

impl<M, I> WithImportance<M, I> {
    pub fn new(mind: M, rating: I) -> Self {
        WithImportance { mind, rating }
    }

    pub fn inner(&self) -> &M {
        &self.mind
    }

    pub fn into_inner(self) -> M {
        self.mind
    }
}

impl<M: Planner, I> Planner for WithImportance<M, I> {
    fn daily_plan(&self, ctx: &PlanContext) -> Vec<PlanStep> {
        self.mind.daily_plan(ctx)
    }

    fn decompose(&self, step: &PlanStep, ctx: &PlanContext) -> Vec<PlanStep> {
        self.mind.decompose(step, ctx)
    }

    fn react(
        &self,
        observation: &Memory,
        current_action: Option<&PlanStep>,
        ctx: &PlanContext,
    ) -> Reaction {
        self.mind.react(observation, current_action, ctx)
    }

    fn max_depth(&self) -> usize {
        self.mind.max_depth()
    }
}

impl<M: Reflector, I> Reflector for WithImportance<M, I> {
    fn salient_questions(&self, recent: &[&Memory]) -> Vec<String> {
        self.mind.salient_questions(recent)
    }

    fn synthesize(&self, question: &str, evidence: &[&Memory]) -> Vec<Insight> {
        self.mind.synthesize(question, evidence)
    }

    fn embed_query(&self, question: &str) -> Option<Vec<f32>> {
        self.mind.embed_query(question)
    }
}

impl<M: Planner + Reflector, I: Importance> Mind for WithImportance<M, I> {
    fn importance(&self, event: &StateChangeEvent) -> f64 {
        self.rating.importance(event)
    }
}

type StateFactory<C, S> = Arc<dyn Fn(&C, &mut dyn RngCore) -> S + Send + Sync>;
type Interpreter<C> = Arc<dyn Fn(&PlanStep) -> C + Send + Sync>;
type Locator = Arc<dyn Fn(&PlanStep) -> Option<Position> + Send + Sync>;

pub struct GenerativeAgent<C, S, M> {
    pub id: AgentId,
    identity: String,
    state_factory: StateFactory<C, S>,
    interpret: Interpreter<C>,
    locate: Option<Locator>,
    mind: M,
    pub data: S,
    pub memory: MemoryStream,
    pub plan: Plan,
    /// Position for proximity perception.
    pub location: Option<Position>,
}

impl<C, S, M> GenerativeAgent<C, S, M>
where
    S: State,
    M: Mind,
{
    pub fn new<F, G>(
        id: impl Into<AgentId>,
        identity: String,
        state_factory: F,
        interpret: G,
        mind: M,
        start: DateTime<Utc>,
        rng: &mut dyn RngCore,
    ) -> Self
    where
        F: Fn(&C, &mut dyn RngCore) -> S + Send + Sync + 'static,
        G: Fn(&PlanStep) -> C + Send + Sync + 'static,
    {
        let state_factory: StateFactory<C, S> = Arc::new(state_factory);
        let interpret: Interpreter<C> = Arc::new(interpret);

        let ctx = PlanContext {
            identity: identity.clone(),
            now: start,
            memories: Vec::new(),
        };
        let plan = Plan::generate(&mind, &ctx);

        let data = match plan.current_action(start) {
            Some(action) => state_factory(&interpret(action), rng),
            None => S::default(),
        };

        GenerativeAgent {
            id: id.into(),
            identity,
            state_factory,
            interpret,
            locate: None,
            mind,
            data,
            memory: MemoryStream::new(),
            plan,
            location: None,
        }
    }

    /// Maps plan steps to world positions.
    pub fn with_locator<F>(mut self, locate: F) -> Self
    where
        F: Fn(&PlanStep) -> Option<Position> + Send + Sync + 'static,
    {
        let locate: Locator = Arc::new(locate);
        if let Some(start) = self.plan.steps.first().map(|step| step.start)
            && let Some(action) = self.plan.current_action(start)
        {
            self.location = locate(action);
        }
        self.locate = Some(locate);
        self
    }

    // append a follow-on plan
    fn extend_horizon(&mut self, now: DateTime<Utc>) {
        let plan_end = match self.plan.current_action(now) {
            Some(action) => action.end(),
            None => return,
        };
        if self.plan.current_action(plan_end).is_some() {
            return;
        }

        let memories = self
            .memory
            .retrieve(None, plan_end, REACTION_CONTEXT)
            .into_iter()
            .map(|s| s.memory)
            .collect();
        let ctx = PlanContext {
            identity: self.identity.clone(),
            now: plan_end,
            memories,
        };
        let next = Plan::generate(&self.mind, &ctx);
        self.plan.steps.extend(next.steps);
    }
}

impl<C, S, M> SimAgent for GenerativeAgent<C, S, M>
where
    S: State + Clone,
    M: Mind,
{
    type State = C;
    type World = ();

    fn peek_next_event_delay(
        &self,
        now: DateTime<Utc>,
        _world: &(),
        _rng: &mut dyn RngCore,
    ) -> Option<f64> {
        let action = self.plan.current_action(now)?;
        let secs = (action.end() - now).num_milliseconds() as f64 / 1000.0;
        (secs > 0.0).then_some(secs)
    }

    fn step(&self, now: DateTime<Utc>, _world: &(), _rng: &mut dyn RngCore) -> Option<C> {
        let action = self.plan.current_action(now)?;
        let next = self.plan.current_action(action.end())?;
        Some((self.interpret)(next))
    }

    fn apply_transition(
        &mut self,
        next: C,
        time: DateTime<Utc>,
        _world: &(),
        rng: &mut dyn RngCore,
        out: &mut Vec<StateChangeEvent>,
    ) {
        let target = (self.state_factory)(&next, rng);
        self.data.diff(&target, &self.id, time, out);
        self.data = target;

        let moved = self.locate.as_ref().and_then(|locate| {
            self.plan
                .current_action(time)
                .and_then(|action| locate(action))
        });
        if moved.is_some() {
            self.location = moved;
        }

        self.extend_horizon(time);
    }

    fn observe(&mut self, event: &StateChangeEvent) {
        let importance = self.mind.importance(event);
        let description = format!(
            "{}: {} -> {}",
            event.field, event.old_value, event.new_value
        );
        self.memory
            .observe(description.clone(), importance, event.time);

        if self.memory.should_reflect() {
            self.memory.reflect(&self.mind, event.time);
        }

        let memories = self
            .memory
            .retrieve(None, event.time, REACTION_CONTEXT)
            .into_iter()
            .map(|s| s.memory)
            .collect();
        let ctx = PlanContext {
            identity: self.identity.clone(),
            now: event.time,
            memories,
        };
        let observation = Memory::new(description, importance, event.time);
        self.plan.react(&self.mind, &observation, &ctx);
    }

    fn location(&self) -> Option<Position> {
        self.location
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::{Insight, MemoryKind};
    use crate::planning::Reaction;
    use crate::rng::SimRng;
    use crate::simulation::Simulation;
    use crate::state::{ToValue, Value};
    use chrono::Duration;
    use std::borrow::Cow;

    #[derive(Clone, Default, PartialEq)]
    struct St {
        value: i32,
    }

    impl State for St {
        fn diff(
            &self,
            other: &Self,
            agent_id: &AgentId,
            time: DateTime<Utc>,
            out: &mut Vec<StateChangeEvent>,
        ) {
            if self.value != other.value {
                out.push(StateChangeEvent {
                    time,
                    agent_id: agent_id.clone(),
                    field: Cow::Borrowed("value"),
                    old_value: self.value.to_value(),
                    new_value: other.value.to_value(),
                });
            }
        }
    }

    #[derive(Clone)]
    enum Activity {
        Rest,
        Work,
    }

    struct MockMind {
        replan: bool,
    }

    impl Planner for MockMind {
        fn daily_plan(&self, ctx: &PlanContext) -> Vec<PlanStep> {
            vec![
                PlanStep::new("rest", ctx.now, Duration::hours(2)),
                PlanStep::new("work", ctx.now + Duration::hours(2), Duration::hours(2)),
            ]
        }

        fn decompose(&self, _step: &PlanStep, _ctx: &PlanContext) -> Vec<PlanStep> {
            Vec::new()
        }

        fn react(
            &self,
            _observation: &Memory,
            _current: Option<&PlanStep>,
            ctx: &PlanContext,
        ) -> Reaction {
            if self.replan {
                Reaction::Replan(vec![PlanStep::new("REACTED", ctx.now, Duration::hours(1))])
            } else {
                Reaction::Continue
            }
        }
    }

    impl Reflector for MockMind {
        fn salient_questions(&self, _recent: &[&Memory]) -> Vec<String> {
            vec!["what is happening?".to_string()]
        }

        fn synthesize(&self, _question: &str, evidence: &[&Memory]) -> Vec<Insight> {
            vec![Insight {
                description: "an insight".to_string(),
                importance: 1.0,
                evidence: evidence.iter().map(|m| m.id).collect(),
                embedding: None,
            }]
        }
    }

    impl Mind for MockMind {
        fn importance(&self, _event: &StateChangeEvent) -> f64 {
            1.0
        }
    }

    fn change(field: &'static str, old: Value, new: Value) -> StateChangeEvent {
        StateChangeEvent {
            time: Utc::now(),
            agent_id: Arc::from("rated"),
            field: Cow::Borrowed(field),
            old_value: old,
            new_value: new,
        }
    }

    #[test]
    fn test_magnitude_scales_with_delta() {
        let rating = MagnitudeImportance::new().with_scale("price", 100.0);

        let tenth = rating.importance(&change("price", Value::Int(1000), Value::Int(1010)));
        let half = rating.importance(&change("price", Value::Int(1000), Value::Int(1050)));
        let full = rating.importance(&change("price", Value::Int(1000), Value::Int(1100)));

        assert!(tenth < half && half < full);
        assert!(
            (half - 5.5).abs() < 1e-9,
            "half a scale sits mid-range: {half}"
        );
        assert_eq!(full, IMPORTANCE_CEILING);
    }

    #[test]
    fn test_rating_ignores_direction() {
        let rating = MagnitudeImportance::new().with_scale("price", 100.0);

        let up = rating.importance(&change("price", Value::Int(1000), Value::Int(1040)));
        let down = rating.importance(&change("price", Value::Int(1000), Value::Int(960)));

        assert_eq!(up, down);
    }

    #[test]
    fn test_change_beyond_scale_clamps() {
        let rating = MagnitudeImportance::new().with_scale("price", 10.0);
        let huge = rating.importance(&change("price", Value::Int(0), Value::Int(1_000_000)));

        assert_eq!(huge, IMPORTANCE_CEILING);
    }

    #[test]
    fn test_unconfigured_field_is_relative() {
        let rating = MagnitudeImportance::new();

        // both are 10% moves
        let small = rating.importance(&change("x", Value::Float(10.0), Value::Float(11.0)));
        let large = rating.importance(&change("x", Value::Float(10_000.0), Value::Float(11_000.0)));

        assert!((small - large).abs() < 1e-9);
        assert!(small > IMPORTANCE_FLOOR && small < IMPORTANCE_CEILING);
    }

    #[test]
    fn test_moving_off_zero_hits_ceiling() {
        let rating = MagnitudeImportance::new();
        let off_zero = rating.importance(&change("x", Value::Int(0), Value::Int(1)));

        assert_eq!(off_zero, IMPORTANCE_CEILING);
    }

    #[test]
    fn test_non_numeric_takes_categorical() {
        let rating = MagnitudeImportance::new().with_categorical(4.0);

        let flag = rating.importance(&change("online", Value::Bool(false), Value::Bool(true)));
        let text = rating.importance(&change("mode", Value::text("idle"), Value::text("busy")));

        assert_eq!(flag, 4.0);
        assert_eq!(text, 4.0);
    }

    #[test]
    fn test_non_finite_values_fall_back() {
        let rating = MagnitudeImportance::new().with_categorical(3.0);

        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let rated = rating.importance(&change("x", Value::Float(1.0), Value::Float(value)));
            assert_eq!(rated, 3.0, "value {value}");
        }
    }

    #[test]
    fn test_unusable_scale_falls_back() {
        for scale in [0.0, -5.0, f64::NAN, f64::INFINITY] {
            let rating = MagnitudeImportance::new().with_scale("x", scale);
            let rated = rating.importance(&change("x", Value::Float(100.0), Value::Float(110.0)));

            let relative = MagnitudeImportance::new().importance(&change(
                "x",
                Value::Float(100.0),
                Value::Float(110.0),
            ));
            assert_eq!(rated, relative, "scale {scale}");
        }
    }

    #[test]
    fn test_ratings_stay_in_range() {
        let rating = MagnitudeImportance::new().with_scale("x", 1.0);

        for (old, new) in [(0.0, 0.0), (1.0, 1.0), (-1e18, 1e18), (1e-18, 2e-18)] {
            let rated = rating.importance(&change("x", Value::Float(old), Value::Float(new)));
            assert!(
                (IMPORTANCE_FLOOR..=IMPORTANCE_CEILING).contains(&rated),
                "{old} -> {new} rated {rated}"
            );
        }
    }

    #[test]
    fn test_wrapper_rates_locally() {
        let wrapped = WithImportance::new(
            MockMind { replan: true },
            MagnitudeImportance::new().with_scale("value", 10.0),
        );

        // heuristic, not MockMind's 1.0
        let rated = wrapped.importance(&change("value", Value::Int(0), Value::Int(5)));
        assert!(rated > 1.0, "expected the heuristic's rating, got {rated}");

        let ctx = PlanContext {
            identity: "a test persona".to_string(),
            now: base(),
            memories: Vec::new(),
        };
        assert_eq!(wrapped.daily_plan(&ctx).len(), 2);
        assert_eq!(wrapped.salient_questions(&[]), vec!["what is happening?"]);
        assert!(matches!(
            wrapped.react(&Memory::new("obs", 1.0, base()), None, &ctx),
            Reaction::Replan(_)
        ));
    }

    #[test]
    fn test_wrapper_end_to_end() {
        let start = base();
        let mut rng = SimRng::seed_from_u64(1);
        let agent = GenerativeAgent::new(
            "agent",
            "a test persona".to_string(),
            |a: &Activity, _rng: &mut dyn RngCore| match a {
                Activity::Rest => St { value: 1 },
                Activity::Work => St { value: 2 },
            },
            |step: &PlanStep| {
                if step.description.starts_with("work") {
                    Activity::Work
                } else {
                    Activity::Rest
                }
            },
            WithImportance::new(MockMind { replan: false }, MagnitudeImportance::new()),
            start,
            &mut rng,
        );

        let mut sim = Simulation::new(vec![agent], start);
        let events = sim.run(Duration::hours(7));

        assert_eq!(events.len(), 3);
        assert_eq!(sim.agents()[0].memory.len(), 3);
    }

    fn base() -> DateTime<Utc> {
        chrono::TimeZone::timestamp_opt(&Utc, 1_600_000_000, 0).unwrap()
    }

    fn build_agent(
        mind: MockMind,
        start: DateTime<Utc>,
    ) -> GenerativeAgent<Activity, St, MockMind> {
        let mut rng = SimRng::seed_from_u64(1);
        GenerativeAgent::new(
            "agent".to_string(),
            "a test persona".to_string(),
            |a: &Activity, _rng: &mut dyn RngCore| match a {
                Activity::Rest => St { value: 1 },
                Activity::Work => St { value: 2 },
            },
            |step: &PlanStep| {
                if step.description.starts_with("work") {
                    Activity::Work
                } else {
                    Activity::Rest
                }
            },
            mind,
            start,
            &mut rng,
        )
    }

    #[test]
    fn test_plan_drives_transitions() {
        let start = Utc::now();
        let agent = build_agent(MockMind { replan: false }, start);
        assert_eq!(agent.data.value, 1);

        let mut sim = Simulation::new(vec![agent], start);
        let events = sim.run(Duration::hours(7));

        // regenerates at the 4h horizon
        assert_eq!(events.len(), 3);
        assert_eq!(&*events[0].agent_id, "agent");
        assert_eq!(events[0].new_value, Value::Int(2));
        assert_eq!(events[0].time, start + Duration::hours(2));
        assert_eq!(events[1].new_value, Value::Int(1));
        assert_eq!(events[1].time, start + Duration::hours(4));
        assert_eq!(events[2].new_value, Value::Int(2));
        assert_eq!(events[2].time, start + Duration::hours(6));
    }

    #[test]
    fn test_observation_recorded() {
        let start = Utc::now();
        let agent = build_agent(MockMind { replan: false }, start);

        let mut sim = Simulation::new(vec![agent], start);
        sim.run(Duration::hours(5));

        assert_eq!(sim.agents()[0].memory.len(), 2);
    }

    #[test]
    fn test_reflection_fires() {
        let start = Utc::now();
        let mut agent = build_agent(MockMind { replan: false }, start);
        agent.memory.reflection_threshold = 0.5;

        let mut sim = Simulation::new(vec![agent], start);
        sim.run(Duration::hours(5));

        let memory = &sim.agents()[0].memory;
        assert!(
            memory
                .memories()
                .iter()
                .any(|m| m.kind == MemoryKind::Reflection)
        );
    }

    #[test]
    fn test_locator_moves_agent() {
        use crate::space::Position;

        let start = Utc::now();
        let agent =
            build_agent(MockMind { replan: false }, start).with_locator(|step: &PlanStep| {
                if step.description.starts_with("work") {
                    Some(Position::new(10.0, 0.0))
                } else {
                    Some(Position::new(0.0, 0.0))
                }
            });

        assert_eq!(agent.location, Some(Position::new(0.0, 0.0)));

        let mut sim = Simulation::new(vec![agent], start);
        sim.run(Duration::hours(3)); // crosses the 2h boundary

        assert_eq!(sim.agents()[0].location, Some(Position::new(10.0, 0.0)));
    }

    #[test]
    fn test_reaction_replans() {
        let start = Utc::now();
        let agent = build_agent(MockMind { replan: true }, start);

        let mut sim = Simulation::new(vec![agent], start);
        sim.run(Duration::hours(5));

        let plan = &sim.agents()[0].plan;
        assert_eq!(plan.steps.last().unwrap().description, "REACTED");
    }
}
