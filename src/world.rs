use crate::state::StateChangeEvent;

/// Shared state a [`Simulation`](crate::simulation::Simulation) maintains alongside its agents.
///
/// Perception pushes each change to every agent that can see it, which is what the generative
/// architecture needs but costs one call per observer per change. A world is the pull-shaped
/// alternative: the simulation folds each change in once, and agents read the result when they act.
/// That makes shared context cost O(1) per change instead of O(agents), at the price of agents
/// seeing an aggregate rather than individual events.
///
/// The two compose — an agent can observe its neighbours and read a world in the same run.
pub trait World {
    /// Folds one emitted change into the shared view.
    fn absorb(&mut self, event: &StateChangeEvent);
}

/// The absent world, for agents that rely only on perception.
impl World for () {
    fn absorb(&mut self, _event: &StateChangeEvent) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::Value;
    use chrono::Utc;
    use std::borrow::Cow;
    use std::sync::Arc;

    #[derive(Default)]
    struct LastPrice {
        price: i64,
        seen: usize,
    }

    impl World for LastPrice {
        fn absorb(&mut self, event: &StateChangeEvent) {
            self.seen += 1;
            if event.field == "price"
                && let Value::Int(price) = event.new_value
            {
                self.price = price;
            }
        }
    }

    #[test]
    fn test_world_folds_changes() {
        let mut world = LastPrice::default();
        for price in [10, 20, 30] {
            world.absorb(&StateChangeEvent {
                time: Utc::now(),
                agent_id: Arc::from("a"),
                field: Cow::Borrowed("price"),
                old_value: Value::Int(0),
                new_value: Value::Int(price),
            });
        }

        assert_eq!(world.price, 30);
        assert_eq!(world.seen, 3);
    }
}
