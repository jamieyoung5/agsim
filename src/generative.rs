use crate::agent::SimAgent;
use crate::memory::{Memory, MemoryStream, Reflector};
use crate::planning::{Plan, PlanContext, PlanStep, Planner};
use crate::space::Position;
use crate::state::{AgentId, State, StateChangeEvent};
use chrono::{DateTime, Utc};
use rand::RngCore;
use std::sync::Arc;

// number of memories retrieved to give the Planner context when reacting to an observation.
const REACTION_CONTEXT: usize = 10;

/// The cognition behind a [`GenerativeAgent`]: it plans, reflects, and rates how poignant an
/// observation is. Back it with a model (see [`llm`](crate::llm)) or script it, as
/// `examples/generative_device.rs` does.
pub trait Mind: Planner + Reflector {
    fn importance(&self, event: &StateChangeEvent) -> f64;
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
    /// Position in the world, for proximity-based perception. `None` unless set via
    /// [`with_locator`](Self::with_locator) or assigned directly.
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

        // opening in the plan's current activity means the first emitted event is a real transition
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

    /// Attaches a function mapping a plan step to a world position. The agent then moves to its
    /// current step's location on every transition.
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

    // appends a follow-on plan once the agent reaches its final action, so a run never outlasts
    // the plan. a no-op until then.
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

    // reflects once enough importance has built up, then lets the Mind interrupt the plan
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
        assert_eq!(agent.data.value, 1); // starts in the opening "rest" action.

        let mut sim = Simulation::new(vec![agent], start);
        let events = sim.run(Duration::hours(7));

        // the plan regenerates at its 4h horizon, so transitions keep coming on the 2h cadence:
        // rest -> work (2h), work -> rest (4h), rest -> work (6h).
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

        // two transitions inside the window (2h and 4h) recorded as observations.
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

        // seeded to the opening "rest" action's location.
        assert_eq!(agent.location, Some(Position::new(0.0, 0.0)));

        let mut sim = Simulation::new(vec![agent], start);
        sim.run(Duration::hours(3)); // crosses the 2h rest -> work boundary

        // moved to the "work" location after the transition.
        assert_eq!(sim.agents()[0].location, Some(Position::new(10.0, 0.0)));
    }

    #[test]
    fn test_reaction_replans() {
        let start = Utc::now();
        let agent = build_agent(MockMind { replan: true }, start);

        let mut sim = Simulation::new(vec![agent], start);
        sim.run(Duration::hours(5));

        // the reaction at the 2h boundary replaced the plan's tail.
        let plan = &sim.agents()[0].plan;
        assert_eq!(plan.steps.last().unwrap().description, "REACTED");
    }
}
