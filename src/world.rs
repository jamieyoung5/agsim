use crate::state::StateChangeEvent;

/// Shared state agents read.
pub trait World {
    fn absorb(&mut self, event: &StateChangeEvent);
}

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
