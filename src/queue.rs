//! The pending-event queue.
//!
//! A simulation pushes and pops this once per event, so its shape is a large part of the cost of
//! running one. It is a 4-ary min-heap rather than the binary heap in the standard library: four
//! children sit in a single cache line, and the tree is half as deep, which is what matters once a
//! population is too large for the queue to stay in cache.

use std::cmp::Ordering;

/// How many children each node has. Four 24-byte entries span one 64-byte cache line closely
/// enough that a sift step touches one line instead of two.
const ARITY: usize = 4;

/// A transition waiting to happen.
///
/// The state is stored directly rather than behind an `Option`, since nothing is ever queued
/// without one, and the agent is a `u32`, which keeps the whole entry inside three words.
pub(crate) struct ScheduledEvent<C> {
    pub time_ms: i64,
    pub agent_index: u32,
    pub next_state_type: C,
}

impl<C> ScheduledEvent<C> {
    /// Earliest first, ties broken by agent so that simultaneous events keep a stable order.
    fn before(&self, other: &Self) -> bool {
        match self.time_ms.cmp(&other.time_ms) {
            Ordering::Less => true,
            Ordering::Greater => false,
            Ordering::Equal => self.agent_index < other.agent_index,
        }
    }
}

pub(crate) struct EventQueue<C> {
    items: Vec<ScheduledEvent<C>>,
}

impl<C> EventQueue<C> {
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        EventQueue {
            items: Vec::with_capacity(capacity),
        }
    }

    pub(crate) fn push(&mut self, event: ScheduledEvent<C>) {
        self.items.push(event);
        self.sift_up(self.items.len() - 1);
    }

    pub(crate) fn pop(&mut self) -> Option<ScheduledEvent<C>> {
        let last = self.items.pop()?;
        if self.items.is_empty() {
            return Some(last);
        }

        // the tail takes the root's place and sinks to where it belongs
        let root = std::mem::replace(&mut self.items[0], last);
        self.sift_down(0);
        Some(root)
    }

    fn sift_up(&mut self, mut at: usize) {
        while at > 0 {
            let parent = (at - 1) / ARITY;
            if !self.items[at].before(&self.items[parent]) {
                break;
            }
            self.items.swap(at, parent);
            at = parent;
        }
    }

    fn sift_down(&mut self, mut at: usize) {
        loop {
            let first_child = at * ARITY + 1;
            if first_child >= self.items.len() {
                break;
            }

            let last_child = (first_child + ARITY).min(self.items.len());
            let mut smallest = first_child;
            for child in first_child + 1..last_child {
                if self.items[child].before(&self.items[smallest]) {
                    smallest = child;
                }
            }

            if !self.items[smallest].before(&self.items[at]) {
                break;
            }
            self.items.swap(at, smallest);
            at = smallest;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn event(time_ms: i64, agent_index: u32) -> ScheduledEvent<u32> {
        ScheduledEvent {
            time_ms,
            agent_index,
            next_state_type: agent_index,
        }
    }

    fn drain(queue: &mut EventQueue<u32>) -> Vec<(i64, u32)> {
        let mut out = Vec::new();
        while let Some(item) = queue.pop() {
            out.push((item.time_ms, item.agent_index));
        }
        out
    }

    #[test]
    fn test_pops_in_time_order() {
        let mut queue = EventQueue::with_capacity(0);
        for time in [50, 10, 90, 30, 70, 20] {
            queue.push(event(time, 0));
        }

        let times: Vec<i64> = drain(&mut queue).into_iter().map(|(t, _)| t).collect();
        assert_eq!(times, vec![10, 20, 30, 50, 70, 90]);
    }

    #[test]
    fn test_simultaneous_events_order_by_agent() {
        let mut queue = EventQueue::with_capacity(0);
        for agent in [4, 1, 3, 0, 2] {
            queue.push(event(100, agent));
        }

        let agents: Vec<u32> = drain(&mut queue).into_iter().map(|(_, a)| a).collect();
        assert_eq!(agents, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_empty_and_single() {
        let mut queue: EventQueue<u32> = EventQueue::with_capacity(0);
        assert!(queue.pop().is_none());

        queue.push(event(5, 7));
        assert_eq!(queue.pop().unwrap().agent_index, 7);
        assert!(queue.pop().is_none());
    }

    #[test]
    fn test_interleaved_pushes_and_pops_stay_ordered() {
        let mut queue = EventQueue::with_capacity(0);
        let mut expected = Vec::new();

        // a spread wide enough to exercise several levels of the tree in both directions
        let mut value: i64 = 1;
        for round in 0..200i64 {
            value = (value * 48271) % 65537;
            queue.push(event(value, round as u32));
            expected.push(value);

            if round % 3 == 0 {
                let popped = queue.pop().unwrap();
                expected.sort_unstable();
                assert_eq!(popped.time_ms, expected.remove(0));
            }
        }

        expected.sort_unstable();
        let remaining: Vec<i64> = drain(&mut queue).into_iter().map(|(t, _)| t).collect();
        assert_eq!(remaining, expected);
    }

    #[test]
    fn test_matches_a_sorted_reference_under_many_entries() {
        let mut queue = EventQueue::with_capacity(0);
        let mut reference = Vec::new();

        let mut value: i64 = 7;
        for agent in 0..5_000u32 {
            value = (value * 1103515245 + 12345) % 1_000_003;
            queue.push(event(value, agent));
            reference.push((value, agent));
        }

        reference.sort_unstable();
        assert_eq!(drain(&mut queue), reference);
    }
}
