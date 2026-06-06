// Position is a point in the simulated world, used for proximity-based perception. Coordinates are
// unitless — interpret the radius in the same units.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Position {
    pub x: f64,
    pub y: f64,
}

impl Position {
    pub fn new(x: f64, y: f64) -> Self {
        Position { x, y }
    }

    pub fn distance(&self, other: &Position) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distance() {
        assert_eq!(Position::new(0.0, 0.0).distance(&Position::new(3.0, 4.0)), 5.0);
        assert_eq!(Position::new(2.0, 2.0).distance(&Position::new(2.0, 2.0)), 0.0);
    }
}
