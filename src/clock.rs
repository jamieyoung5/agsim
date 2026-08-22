use chrono::Duration;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration as StdDuration, Instant};

// max sleep between stop checks
pub(crate) const STOP_POLL_INTERVAL: StdDuration = StdDuration::from_millis(50);

/// Wall-clock source for a live run.
pub trait Clock {
    fn now(&self) -> Instant;
    fn sleep(&self, duration: StdDuration);
}

/// Monotonic system clock.
#[derive(Debug, Clone, Copy, Default)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn now(&self) -> Instant {
        Instant::now()
    }

    fn sleep(&self, duration: StdDuration) {
        std::thread::sleep(duration);
    }
}

/// Ends a live run.
#[derive(Debug, Clone, Default)]
pub struct StopSignal(Arc<AtomicBool>);

impl StopSignal {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn stop(&self) {
        self.0.store(true, Ordering::Relaxed);
    }

    pub fn is_stopped(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }
}

/// Pacing for [`Simulation::run_live`](crate::simulation::Simulation::run_live).
#[derive(Debug, Clone)]
pub struct Live {
    speed: f64,
    horizon: Option<Duration>,
    stop: StopSignal,
}

impl Default for Live {
    fn default() -> Self {
        Live::real_time()
    }
}

impl Live {
    pub fn real_time() -> Self {
        Live::at_speed(1.0)
    }

    pub fn at_speed(speed: f64) -> Self {
        Live {
            speed: if speed > 0.0 { speed } else { f64::INFINITY },
            horizon: None,
            stop: StopSignal::new(),
        }
    }

    pub fn unpaced() -> Self {
        Live::at_speed(f64::INFINITY)
    }

    pub fn until(mut self, horizon: Duration) -> Self {
        self.horizon = Some(horizon);
        self
    }

    pub fn with_stop(mut self, stop: StopSignal) -> Self {
        self.stop = stop;
        self
    }

    pub fn stop_signal(&self) -> StopSignal {
        self.stop.clone()
    }

    pub fn speed(&self) -> f64 {
        self.speed
    }

    pub fn horizon(&self) -> Option<Duration> {
        self.horizon
    }

    pub(crate) fn stop(&self) -> &StopSignal {
        &self.stop
    }
}

/// Why a live run ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LiveOutcome {
    /// The stop signal was raised.
    Stopped,
    /// The callback returned [`ControlFlow::Break`](std::ops::ControlFlow::Break).
    Halted,
    /// The horizon was passed.
    HorizonReached,
    /// No agent has anything left to do.
    Drained,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stop_signal_shared() {
        let signal = StopSignal::new();
        let handle = signal.clone();
        assert!(!signal.is_stopped());

        handle.stop();
        assert!(signal.is_stopped());
    }

    #[test]
    fn test_speed_normalizes_non_positive() {
        assert_eq!(Live::at_speed(0.0).speed(), f64::INFINITY);
        assert_eq!(Live::at_speed(-2.0).speed(), f64::INFINITY);
        assert_eq!(Live::real_time().speed(), 1.0);
    }

    #[test]
    fn test_horizon_builder() {
        let live = Live::at_speed(10.0).until(Duration::hours(3));
        assert_eq!(live.horizon(), Some(Duration::hours(3)));
        assert!(Live::real_time().horizon().is_none());
    }
}
