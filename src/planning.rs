use crate::memory::Memory;
use chrono::{DateTime, Duration, Utc};

// how far Plan::generate recurses when no planner says otherwise.
pub const MAX_PLAN_DEPTH: usize = 3;

#[derive(Debug, Clone)]
pub struct PlanStep {
    pub description: String,
    pub start: DateTime<Utc>,
    pub duration: Duration,
    pub subplan: Vec<PlanStep>,
}

impl PlanStep {
    pub fn new(description: impl Into<String>, start: DateTime<Utc>, duration: Duration) -> Self {
        PlanStep {
            description: description.into(),
            start,
            duration,
            subplan: Vec::new(),
        }
    }

    pub fn end(&self) -> DateTime<Utc> {
        self.start + self.duration
    }

    pub fn contains(&self, now: DateTime<Utc>) -> bool {
        now >= self.start && now < self.end()
    }

    pub fn is_leaf(&self) -> bool {
        self.subplan.is_empty()
    }

    // leaf_at descends to the finest sub-step active at now, or None if now lies outside this step.
    pub fn leaf_at(&self, now: DateTime<Utc>) -> Option<&PlanStep> {
        if !self.contains(now) {
            return None;
        }
        for sub in &self.subplan {
            if let Some(leaf) = sub.leaf_at(now) {
                return Some(leaf);
            }
        }
        Some(self)
    }
}

#[derive(Debug, Clone)]
pub struct PlanContext {
    pub identity: String,
    pub now: DateTime<Utc>,
    pub memories: Vec<Memory>,
}

#[derive(Debug, Clone)]
pub enum Reaction {
    Continue,
    Replan(Vec<PlanStep>),
}

// Planner is the model-backed half of planning: it decides what goes in the plan and whether to
// react. The recursion, current-action lookup, and replanning are handled by Plan.
pub trait Planner {
    fn daily_plan(&self, ctx: &PlanContext) -> Vec<PlanStep>;
    fn decompose(&self, step: &PlanStep, ctx: &PlanContext) -> Vec<PlanStep>;
    fn react(
        &self,
        observation: &Memory,
        current_action: Option<&PlanStep>,
        ctx: &PlanContext,
    ) -> Reaction;

    // max_depth caps how far generate recurses. Each level multiplies the number of decompose
    // calls, which is free for a scripted planner and expensive for a model-backed one — so the
    // planner, not the Plan, decides how deep is worth it.
    fn max_depth(&self) -> usize {
        MAX_PLAN_DEPTH
    }
}

#[derive(Debug, Clone, Default)]
pub struct Plan {
    pub steps: Vec<PlanStep>,
}

impl Plan {
    // generate lays out the day's broad strokes and recursively decomposes each one.
    pub fn generate<P: Planner>(planner: &P, ctx: &PlanContext) -> Self {
        let mut steps = planner.daily_plan(ctx);
        for step in &mut steps {
            decompose_step(planner, step, ctx, 1);
        }
        Plan { steps }
    }

    pub fn current_action(&self, now: DateTime<Utc>) -> Option<&PlanStep> {
        self.steps.iter().find_map(|s| s.leaf_at(now))
    }

    // react applies the Planner's verdict on an observation, replanning if it decides to. Returns
    // true when the plan changed.
    pub fn react<P: Planner>(
        &mut self,
        planner: &P,
        observation: &Memory,
        ctx: &PlanContext,
    ) -> bool {
        let current = self.current_action(ctx.now).cloned();
        match planner.react(observation, current.as_ref(), ctx) {
            Reaction::Continue => false,
            Reaction::Replan(new_steps) => {
                self.replan_from(ctx.now, new_steps);
                true
            }
        }
    }

    // replan_from drops everything scheduled at or after now and appends a new tail. The step
    // straddling now is trimmed to end there.
    pub fn replan_from(&mut self, now: DateTime<Utc>, new_steps: Vec<PlanStep>) {
        self.steps.retain(|s| s.start < now);
        if let Some(last) = self.steps.last_mut()
            && last.end() > now
        {
            last.duration = now - last.start;
            last.subplan.clear();
        }
        self.steps.extend(new_steps);
    }
}

fn decompose_step<P: Planner>(planner: &P, step: &mut PlanStep, ctx: &PlanContext, depth: usize) {
    if depth >= planner.max_depth() {
        return;
    }
    let subs = planner.decompose(step, ctx);
    if subs.is_empty() {
        return;
    }
    step.subplan = subs;
    for sub in &mut step.subplan {
        decompose_step(planner, sub, ctx, depth + 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::Memory;
    use chrono::TimeZone;

    fn base() -> DateTime<Utc> {
        Utc.timestamp_opt(1_600_000_000, 0).unwrap()
    }

    fn ctx_at(now: DateTime<Utc>) -> PlanContext {
        PlanContext {
            identity: "test agent".to_string(),
            now,
            memories: Vec::new(),
        }
    }

    struct MockPlanner;
    impl Planner for MockPlanner {
        fn daily_plan(&self, ctx: &PlanContext) -> Vec<PlanStep> {
            let half = Duration::hours(4);
            vec![
                PlanStep::new("morning", ctx.now, half),
                PlanStep::new("afternoon", ctx.now + half, half),
            ]
        }

        fn decompose(&self, step: &PlanStep, _ctx: &PlanContext) -> Vec<PlanStep> {
            if step.duration <= Duration::hours(1) {
                return Vec::new();
            }
            let half = step.duration / 2;
            vec![
                PlanStep::new(format!("{} (1)", step.description), step.start, half),
                PlanStep::new(
                    format!("{} (2)", step.description),
                    step.start + half,
                    step.duration - half,
                ),
            ]
        }

        fn react(
            &self,
            observation: &Memory,
            _current: Option<&PlanStep>,
            ctx: &PlanContext,
        ) -> Reaction {
            if observation.importance >= 8.0 {
                Reaction::Replan(vec![PlanStep::new("react!", ctx.now, Duration::hours(1))])
            } else {
                Reaction::Continue
            }
        }
    }

    fn tree_depth(steps: &[PlanStep]) -> usize {
        steps
            .iter()
            .map(|s| 1 + tree_depth(&s.subplan))
            .max()
            .unwrap_or(0)
    }

    #[test]
    fn test_generate_depth() {
        let plan = Plan::generate(&MockPlanner, &ctx_at(base()));
        assert_eq!(plan.steps.len(), 2);
        assert_eq!(tree_depth(&plan.steps), MAX_PLAN_DEPTH);
    }

    // ShallowPlanner is MockPlanner with the recursion capped one level earlier. A planner that
    // pays per decompose call wants this knob.
    struct ShallowPlanner;
    impl Planner for ShallowPlanner {
        fn daily_plan(&self, ctx: &PlanContext) -> Vec<PlanStep> {
            MockPlanner.daily_plan(ctx)
        }
        fn decompose(&self, step: &PlanStep, ctx: &PlanContext) -> Vec<PlanStep> {
            MockPlanner.decompose(step, ctx)
        }
        fn react(&self, o: &Memory, c: Option<&PlanStep>, ctx: &PlanContext) -> Reaction {
            MockPlanner.react(o, c, ctx)
        }
        fn max_depth(&self) -> usize {
            2
        }
    }

    #[test]
    fn test_planner_can_cap_the_depth() {
        let plan = Plan::generate(&ShallowPlanner, &ctx_at(base()));

        assert_eq!(plan.steps.len(), 2);
        // the day is split once and the sub-steps are left alone.
        assert_eq!(tree_depth(&plan.steps), 2);
        assert!(plan.steps[0].subplan.iter().all(PlanStep::is_leaf));
    }

    #[test]
    fn test_current_action() {
        let t = base();
        let plan = Plan::generate(&MockPlanner, &ctx_at(t));

        let action = plan.current_action(t + Duration::minutes(30)).unwrap();
        assert!(action.is_leaf());
        assert!(action.description.starts_with("morning"));
        assert!(action.contains(t + Duration::minutes(30)));
    }

    #[test]
    fn test_current_action_out_of_range() {
        let t = base();
        let plan = Plan::generate(&MockPlanner, &ctx_at(t));
        assert!(plan.current_action(t + Duration::hours(10)).is_none());
    }

    #[test]
    fn test_react_continue() {
        let t = base();
        let mut plan = Plan::generate(&MockPlanner, &ctx_at(t));
        let before = plan.steps.len();

        let trivial = Memory::new("a leaf falls", 2.0, t + Duration::hours(1));
        let reacted = plan.react(&MockPlanner, &trivial, &ctx_at(t + Duration::hours(1)));

        assert!(!reacted);
        assert_eq!(plan.steps.len(), before);
    }

    #[test]
    fn test_react_replan() {
        let t = base();
        let mut plan = Plan::generate(&MockPlanner, &ctx_at(t));

        let now = t + Duration::hours(3);
        let urgent = Memory::new("the house is on fire", 10.0, now);
        assert!(plan.react(&MockPlanner, &urgent, &ctx_at(now)));

        let morning = &plan.steps[0];
        assert_eq!(morning.description, "morning");
        assert_eq!(morning.end(), now);
        assert!(morning.subplan.is_empty());

        assert_eq!(plan.steps.last().unwrap().description, "react!");
        let action = plan.current_action(now + Duration::minutes(10)).unwrap();
        assert_eq!(action.description, "react!");
    }
}
