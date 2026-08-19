use std::collections::HashMap;

/// A point in the simulated world, for proximity-based perception. Coordinates are unitless;
/// interpret [`Perception::Proximity`](crate::simulation::Perception::Proximity)'s radius in the
/// same units.
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
        self.distance_squared(other).sqrt()
    }

    /// The squared distance to `other`. Ordering by this matches ordering by
    /// [`distance`](Self::distance), so radius tests can skip the square root by comparing against
    /// a squared radius.
    pub fn distance_squared(&self, other: &Position) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        dx * dx + dy * dy
    }
}

/// A uniform grid over agent positions, for answering "who is within a radius of here".
///
/// Cells are exactly one radius across, so everything within the radius of a point lies in that
/// point's cell or one of the eight around it. That turns a proximity query from a scan of the
/// whole population into a scan of a neighbourhood, and because only the agent that just acted can
/// have moved, keeping the grid current costs one reinsertion per event rather than a rebuild.
pub(crate) struct SpatialIndex {
    cell_size: f64,
    cells: HashMap<(i64, i64), Vec<usize>>,
    placed: Vec<Option<(i64, i64)>>,
}

impl SpatialIndex {
    /// Builds an index over `positions`, or `None` if the radius cannot define a usable grid, in
    /// which case the caller should fall back to scanning.
    pub(crate) fn build(
        radius: f64,
        positions: impl ExactSizeIterator<Item = Option<Position>>,
    ) -> Option<Self> {
        if !radius.is_finite() || radius <= 0.0 {
            return None;
        }

        let mut index = SpatialIndex {
            cell_size: radius,
            cells: HashMap::new(),
            placed: vec![None; positions.len()],
        };

        for (agent, position) in positions.enumerate() {
            index.place(agent, position);
        }

        Some(index)
    }

    fn cell_of(&self, position: Position) -> Option<(i64, i64)> {
        let x = (position.x / self.cell_size).floor();
        let y = (position.y / self.cell_size).floor();

        // a coordinate that does not land on a cell cannot be indexed, so it is left out of the
        // grid and simply never matches
        (x.is_finite() && y.is_finite() && x.abs() < i64::MAX as f64 && y.abs() < i64::MAX as f64)
            .then_some((x as i64, y as i64))
    }

    /// Moves `agent` to the cell for `position`, doing nothing if it is already there.
    pub(crate) fn place(&mut self, agent: usize, position: Option<Position>) {
        let target = position.and_then(|p| self.cell_of(p));
        let current = self.placed[agent];
        if current == target {
            return;
        }

        if let Some(cell) = current
            && let Some(members) = self.cells.get_mut(&cell)
            && let Some(at) = members.iter().position(|&member| member == agent)
        {
            members.swap_remove(at);
        }

        if let Some(cell) = target {
            self.cells.entry(cell).or_default().push(agent);
        }

        self.placed[agent] = target;
    }

    /// Every agent close enough to `center` to be worth a distance test.
    pub(crate) fn candidates(&self, center: Position) -> impl Iterator<Item = usize> + '_ {
        let origin = self.cell_of(center);

        origin
            .into_iter()
            .flat_map(|(cx, cy)| {
                (-1..=1).flat_map(move |dx| (-1..=1).map(move |dy| (cx + dx, cy + dy)))
            })
            .filter_map(move |cell| self.cells.get(&cell))
            .flatten()
            .copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn test_distance() {
        assert_eq!(
            Position::new(0.0, 0.0).distance(&Position::new(3.0, 4.0)),
            5.0
        );
        assert_eq!(
            Position::new(2.0, 2.0).distance(&Position::new(2.0, 2.0)),
            0.0
        );
    }

    // the index is only allowed to narrow the search, never to hide a genuine neighbour
    fn assert_index_finds_every_neighbour(positions: &[Option<Position>], radius: f64) {
        let index = SpatialIndex::build(radius, positions.iter().copied()).unwrap();

        for origin in positions.iter().flatten() {
            let found: HashSet<usize> = index
                .candidates(*origin)
                .filter(|&agent| {
                    positions[agent].is_some_and(|p| p.distance_squared(origin) <= radius * radius)
                })
                .collect();

            let expected: HashSet<usize> = positions
                .iter()
                .enumerate()
                .filter(|(_, p)| p.is_some_and(|p| p.distance_squared(origin) <= radius * radius))
                .map(|(agent, _)| agent)
                .collect();

            assert_eq!(found, expected, "origin {origin:?} radius {radius}");
        }
    }

    #[test]
    fn test_index_matches_a_full_scan() {
        let positions: Vec<Option<Position>> = (0..60)
            .map(|i| {
                let f = i as f64;
                Some(Position::new(f * 1.7 - 40.0, (f * 0.9).sin() * 30.0))
            })
            .collect();

        for radius in [0.5, 3.0, 11.0, 50.0] {
            assert_index_finds_every_neighbour(&positions, radius);
        }
    }

    #[test]
    fn test_index_skips_agents_without_a_position() {
        let positions = vec![
            Some(Position::new(0.0, 0.0)),
            None,
            Some(Position::new(1.0, 0.0)),
        ];
        assert_index_finds_every_neighbour(&positions, 5.0);

        let index = SpatialIndex::build(5.0, positions.iter().copied()).unwrap();
        assert!(!index.candidates(Position::new(0.0, 0.0)).any(|a| a == 1));
    }

    #[test]
    fn test_placing_an_agent_moves_it_between_cells() {
        let positions = [Some(Position::new(0.0, 0.0)), Some(Position::new(1.0, 0.0))];
        let mut index = SpatialIndex::build(10.0, positions.iter().copied()).unwrap();

        assert!(index.candidates(Position::new(0.0, 0.0)).any(|a| a == 1));

        index.place(1, Some(Position::new(1000.0, 1000.0)));
        assert!(!index.candidates(Position::new(0.0, 0.0)).any(|a| a == 1));
        assert!(
            index
                .candidates(Position::new(1000.0, 1000.0))
                .any(|a| a == 1)
        );

        // dropping a position removes the agent from the grid entirely
        index.place(1, None);
        assert!(
            !index
                .candidates(Position::new(1000.0, 1000.0))
                .any(|a| a == 1)
        );
    }

    #[test]
    fn test_unusable_radius_has_no_index() {
        let positions = [Some(Position::new(0.0, 0.0))];
        assert!(SpatialIndex::build(0.0, positions.iter().copied()).is_none());
        assert!(SpatialIndex::build(-1.0, positions.iter().copied()).is_none());
        assert!(SpatialIndex::build(f64::NAN, positions.iter().copied()).is_none());
        assert!(SpatialIndex::build(f64::INFINITY, positions.iter().copied()).is_none());
    }

    #[test]
    fn test_distance_squared_orders_like_distance() {
        let origin = Position::new(0.0, 0.0);
        let near = Position::new(3.0, 4.0);
        let far = Position::new(6.0, 8.0);

        assert_eq!(origin.distance_squared(&near), 25.0);
        assert!(origin.distance_squared(&near) < origin.distance_squared(&far));
        assert_eq!(
            origin.distance_squared(&near).sqrt(),
            origin.distance(&near)
        );
    }
}
