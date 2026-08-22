use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

pub type MemoryId = usize;

const REFLECTION_RECENT_WINDOW: usize = 100;
const REFLECTION_RETRIEVE_TOP_K: usize = 10;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryKind {
    Observation,
    Reflection,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Memory {
    pub id: MemoryId,
    pub kind: MemoryKind,
    pub description: String,
    pub created: DateTime<Utc>,
    pub last_accessed: DateTime<Utc>,
    pub importance: f64,
    pub embedding: Option<Vec<f32>>,
    pub evidence: Vec<MemoryId>,
}

impl Memory {
    pub fn new(description: impl Into<String>, importance: f64, created: DateTime<Utc>) -> Self {
        Memory {
            id: 0,
            kind: MemoryKind::Observation,
            description: description.into(),
            created,
            last_accessed: created,
            importance,
            embedding: None,
            evidence: Vec::new(),
        }
    }

    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = Some(embedding);
        self
    }
}

pub struct Insight {
    pub description: String,
    pub importance: f64,
    pub evidence: Vec<MemoryId>,
    pub embedding: Option<Vec<f32>>,
}

/// Produces reflection questions and insights.
pub trait Reflector {
    fn salient_questions(&self, recent: &[&Memory]) -> Vec<String>;
    fn synthesize(&self, question: &str, evidence: &[&Memory]) -> Vec<Insight>;
    fn embed_query(&self, _question: &str) -> Option<Vec<f32>> {
        None
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RetrievalBounds {
    pub recent: usize,
    pub important: usize,
}

impl RetrievalBounds {
    pub fn new(recent: usize, important: usize) -> Self {
        RetrievalBounds { recent, important }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RetrievalWeights {
    pub recency: f64,
    pub importance: f64,
    pub relevance: f64,
}

impl Default for RetrievalWeights {
    fn default() -> Self {
        RetrievalWeights {
            recency: 1.0,
            importance: 1.0,
            relevance: 1.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ScoredMemory {
    pub memory: Memory,
    pub recency: f64,
    pub importance: f64,
    pub relevance: f64,
    pub score: f64,
}

#[derive(Debug, Clone)]
pub struct MemoryStream {
    memories: Vec<Memory>,
    next_id: MemoryId,
    pub recency_decay: f64,
    pub weights: RetrievalWeights,
    pub reflection_threshold: f64,
    importance_since_reflection: f64,
    bounds: Option<RetrievalBounds>,
    important: Vec<usize>,
}

impl Default for MemoryStream {
    fn default() -> Self {
        MemoryStream {
            memories: Vec::new(),
            next_id: 0,
            recency_decay: 0.99,
            weights: RetrievalWeights::default(),
            reflection_threshold: 150.0,
            importance_since_reflection: 0.0,
            bounds: None,
            important: Vec::new(),
        }
    }
}

impl MemoryStream {
    pub fn new() -> Self {
        Self::default()
    }

    /// Assigns an id, accumulates importance.
    pub fn add(&mut self, mut memory: Memory) -> MemoryId {
        let id = self.next_id;
        self.next_id += 1;
        memory.id = id;
        self.importance_since_reflection += memory.importance;
        self.memories.push(memory);
        self.note_importance(self.memories.len() - 1);
        id
    }

    /// Bounds how much a retrieval scores.
    pub fn set_retrieval_bounds(&mut self, bounds: Option<RetrievalBounds>) {
        self.bounds = bounds;
        self.rebuild_important();
    }

    pub fn retrieval_bounds(&self) -> Option<RetrievalBounds> {
        self.bounds
    }

    fn note_importance(&mut self, index: usize) {
        let Some(bounds) = self.bounds else {
            return;
        };
        if bounds.important == 0 {
            return;
        }

        let importance = self.memories[index].importance;
        let slot = self
            .important
            .iter()
            .position(|&held| self.memories[held].importance < importance)
            .unwrap_or(self.important.len());

        if slot < bounds.important {
            self.important.insert(slot, index);
            self.important.truncate(bounds.important);
        } else if self.important.len() < bounds.important {
            self.important.push(index);
        }
    }

    fn rebuild_important(&mut self) {
        self.important.clear();
        let Some(bounds) = self.bounds else {
            return;
        };
        if bounds.important == 0 {
            return;
        }

        let mut ranked: Vec<usize> = (0..self.memories.len()).collect();
        ranked.sort_by(|&a, &b| {
            self.memories[b]
                .importance
                .partial_cmp(&self.memories[a].importance)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });
        ranked.truncate(bounds.important);
        self.important = ranked;
    }

    fn candidate_indices(&self) -> Vec<usize> {
        let Some(bounds) = self.bounds else {
            return (0..self.memories.len()).collect();
        };

        let window_start = self.memories.len().saturating_sub(bounds.recent);
        let mut candidates: Vec<usize> = (window_start..self.memories.len()).collect();
        candidates.extend(
            self.important
                .iter()
                .copied()
                .filter(|&index| index < window_start),
        );
        candidates
    }

    pub fn observe(
        &mut self,
        description: impl Into<String>,
        importance: f64,
        time: DateTime<Utc>,
    ) -> MemoryId {
        self.add(Memory::new(description, importance, time))
    }

    pub fn add_reflection(&mut self, insight: Insight, time: DateTime<Utc>) -> MemoryId {
        self.add(Memory {
            id: 0,
            kind: MemoryKind::Reflection,
            description: insight.description,
            created: time,
            last_accessed: time,
            importance: insight.importance,
            embedding: insight.embedding,
            evidence: insight.evidence,
        })
    }

    pub fn len(&self) -> usize {
        self.memories.len()
    }

    pub fn is_empty(&self) -> bool {
        self.memories.is_empty()
    }

    pub fn memories(&self) -> &[Memory] {
        &self.memories
    }

    pub fn should_reflect(&self) -> bool {
        self.importance_since_reflection >= self.reflection_threshold
    }

    fn recent(&self, n: usize) -> Vec<&Memory> {
        let mut refs: Vec<&Memory> = self.memories.iter().collect();
        refs.sort_by(|a, b| b.created.cmp(&a.created));
        refs.truncate(n);
        refs
    }

    /// Reflects on recent memories, stores insights.
    // insights land after all retrieval
    pub fn reflect<R: Reflector>(&mut self, reflector: &R, now: DateTime<Utc>) -> Vec<MemoryId> {
        let questions = {
            let recent = self.recent(REFLECTION_RECENT_WINDOW);
            if recent.is_empty() {
                return Vec::new();
            }
            reflector.salient_questions(&recent)
        };

        let mut pending = Vec::new();
        for question in &questions {
            let query = reflector.embed_query(question);
            let retrieved = self.retrieve(query.as_deref(), now, REFLECTION_RETRIEVE_TOP_K);
            let evidence: Vec<&Memory> = retrieved.iter().map(|s| &s.memory).collect();
            pending.extend(reflector.synthesize(question, &evidence));
        }

        let new_ids = pending
            .into_iter()
            .map(|insight| self.add_reflection(insight, now))
            .collect();

        // don't re-trigger on new insights
        self.importance_since_reflection = 0.0;
        new_ids
    }

    /// Returns the `top_k` best-scoring memories.
    pub fn retrieve(
        &mut self,
        query: Option<&[f32]>,
        now: DateTime<Utc>,
        top_k: usize,
    ) -> Vec<ScoredMemory> {
        let mut scored = self.score_indices(query, now);
        let keep = top_k.min(scored.len());
        if keep == 0 {
            return Vec::new();
        }

        if keep < scored.len() {
            scored.select_nth_unstable_by(keep - 1, Scored::rank);
            scored.truncate(keep);
        }
        scored.sort_by(Scored::rank);

        let retrieved: Vec<ScoredMemory> = scored
            .iter()
            .map(|s| ScoredMemory {
                memory: self.memories[s.index].clone(),
                recency: s.recency,
                importance: s.importance,
                relevance: s.relevance,
                score: s.score,
            })
            .collect();

        for s in &scored {
            self.memories[s.index].last_accessed = now;
        }

        retrieved
    }

    pub fn score_all(&self, query: Option<&[f32]>, now: DateTime<Utc>) -> Vec<ScoredMemory> {
        self.score_indices(query, now)
            .into_iter()
            .map(|s| ScoredMemory {
                memory: self.memories[s.index].clone(),
                recency: s.recency,
                importance: s.importance,
                relevance: s.relevance,
                score: s.score,
            })
            .collect()
    }

    fn score_indices(&self, query: Option<&[f32]>, now: DateTime<Utc>) -> Vec<Scored> {
        let candidates = self.candidate_indices();
        if candidates.is_empty() {
            return Vec::new();
        }

        let recency_raw: Vec<f64> = candidates
            .iter()
            .map(|&i| {
                let hours =
                    (now - self.memories[i].last_accessed).num_milliseconds() as f64 / 3_600_000.0;
                self.recency_decay.powf(hours.max(0.0))
            })
            .collect();

        let importance_raw: Vec<f64> = candidates
            .iter()
            .map(|&i| self.memories[i].importance)
            .collect();

        let query_squared_norm = query.map(squared_norm).unwrap_or(0.0);
        let relevance_raw: Vec<f64> = candidates
            .iter()
            .map(|&i| match (query, self.memories[i].embedding.as_deref()) {
                (Some(q), Some(e)) => cosine_similarity_to(q, query_squared_norm, e),
                _ => 0.0,
            })
            .collect();

        let recency = min_max_normalize(&recency_raw);
        let importance = min_max_normalize(&importance_raw);
        let relevance = min_max_normalize(&relevance_raw);

        candidates
            .iter()
            .enumerate()
            .map(|(slot, &index)| Scored {
                index,
                recency: recency[slot],
                importance: importance[slot],
                relevance: relevance[slot],
                score: self.weights.recency * recency[slot]
                    + self.weights.importance * importance[slot]
                    + self.weights.relevance * relevance[slot],
            })
            .collect()
    }
}

struct Scored {
    index: usize,
    recency: f64,
    importance: f64,
    relevance: f64,
    score: f64,
}

impl Scored {
    // ties settled by stream order
    fn rank(a: &Scored, b: &Scored) -> std::cmp::Ordering {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.index.cmp(&b.index))
    }
}

fn min_max_normalize(values: &[f64]) -> Vec<f64> {
    let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = max - min;

    if range.abs() < f64::EPSILON {
        return vec![0.0; values.len()];
    }

    values.iter().map(|v| (v - min) / range).collect()
}

fn squared_norm(vector: &[f32]) -> f64 {
    vector.iter().map(|x| *x as f64 * *x as f64).sum()
}

// caller computes the query norm
fn cosine_similarity_to(query: &[f32], query_squared_norm: f64, other: &[f32]) -> f64 {
    if query.len() != other.len() || query_squared_norm == 0.0 {
        return 0.0;
    }

    let mut dot = 0.0;
    let mut other_squared_norm = 0.0;
    for (x, y) in query.iter().zip(other.iter()) {
        let (x, y) = (*x as f64, *y as f64);
        dot += x * y;
        other_squared_norm += y * y;
    }

    if other_squared_norm == 0.0 {
        return 0.0;
    }

    dot / (query_squared_norm.sqrt() * other_squared_norm.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Duration, TimeZone};

    fn base() -> DateTime<Utc> {
        Utc.timestamp_opt(1_600_000_000, 0).unwrap()
    }

    struct MockReflector;
    impl Reflector for MockReflector {
        fn salient_questions(&self, _recent: &[&Memory]) -> Vec<String> {
            vec!["what is happening?".to_string()]
        }

        fn synthesize(&self, _question: &str, evidence: &[&Memory]) -> Vec<Insight> {
            vec![Insight {
                description: format!("insight from {} memories", evidence.len()),
                importance: 8.0,
                evidence: evidence.iter().map(|m| m.id).collect(),
                embedding: None,
            }]
        }
    }

    #[test]
    fn test_cosine_similarity() {
        fn similarity(query: &[f32], other: &[f32]) -> f64 {
            cosine_similarity_to(query, squared_norm(query), other)
        }

        assert!((similarity(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-9);
        assert!(similarity(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-9);
        assert_eq!(similarity(&[1.0], &[1.0, 0.0]), 0.0);
        assert_eq!(similarity(&[0.0, 0.0], &[1.0, 1.0]), 0.0);
        assert_eq!(similarity(&[1.0, 1.0], &[0.0, 0.0]), 0.0);
    }

    #[test]
    fn test_min_max_normalize() {
        assert_eq!(min_max_normalize(&[0.0, 5.0, 10.0]), vec![0.0, 0.5, 1.0]);
        assert_eq!(min_max_normalize(&[3.0, 3.0]), vec![0.0, 0.0]);
    }

    #[test]
    fn test_recency_ranking() {
        let t = base();
        let mut stream = MemoryStream::new();
        let old = stream.observe("old", 5.0, t);
        stream.observe("new", 5.0, t);

        let m = stream.memories.iter_mut().find(|m| m.id == old).unwrap();
        m.last_accessed = t - Duration::hours(48);

        let results = stream.retrieve(None, t, 2);
        assert_eq!(results[0].memory.description, "new");
        assert_eq!(results[1].memory.description, "old");
    }

    #[test]
    fn test_importance_ranking() {
        let t = base();
        let mut stream = MemoryStream::new();
        stream.observe("trivial", 1.0, t);
        stream.observe("salient", 9.0, t);

        let results = stream.retrieve(None, t, 2);
        assert_eq!(results[0].memory.description, "salient");
    }

    #[test]
    fn test_relevance_ranking() {
        let t = base();
        let mut stream = MemoryStream::new();
        stream.add(Memory::new("about cats", 5.0, t).with_embedding(vec![1.0, 0.0]));
        stream.add(Memory::new("about dogs", 5.0, t).with_embedding(vec![0.0, 1.0]));

        let results = stream.retrieve(Some(&[0.9, 0.1]), t, 2);
        assert_eq!(results[0].memory.description, "about cats");
    }

    #[test]
    fn test_retrieve_updates_access_time() {
        let t = base();
        let mut stream = MemoryStream::new();
        stream.observe("seen", 5.0, t);

        let later = t + Duration::hours(5);
        stream.retrieve(None, later, 1);
        assert_eq!(stream.memories()[0].last_accessed, later);
    }

    #[test]
    fn test_top_k() {
        let t = base();
        let mut stream = MemoryStream::new();
        for i in 0..10 {
            stream.observe(format!("m{i}"), i as f64, t);
        }
        assert_eq!(stream.retrieve(None, t, 3).len(), 3);
    }

    #[test]
    fn test_top_k_beyond_stream() {
        let t = base();
        let mut stream = MemoryStream::new();
        for i in 0..3 {
            stream.observe(format!("m{i}"), i as f64, t);
        }

        assert_eq!(stream.retrieve(None, t, 99).len(), 3);
        assert!(stream.retrieve(None, t, 0).is_empty());
    }

    #[test]
    fn test_retrieve_snapshots_access_time() {
        let t = base();
        let mut stream = MemoryStream::new();
        stream.observe("only", 1.0, t);

        let later = t + Duration::hours(3);
        let results = stream.retrieve(None, later, 1);

        assert_eq!(results[0].memory.last_accessed, t);
        assert_eq!(stream.memories()[0].last_accessed, later);
    }

    #[test]
    fn test_ties_use_stream_order() {
        let t = base();
        let mut stream = MemoryStream::new();
        for i in 0..8 {
            stream.observe(format!("m{i}"), 1.0, t);
        }

        let ids: Vec<_> = stream
            .retrieve(None, t, 4)
            .iter()
            .map(|s| s.memory.id)
            .collect();

        assert_eq!(ids, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_retrieve_matches_score_all() {
        let t = base();
        let mut stream = MemoryStream::new();
        for i in 0..20 {
            stream.observe(format!("m{i}"), (i % 5) as f64, t + Duration::minutes(i));
        }

        let now = t + Duration::hours(2);
        let mut expected = stream.score_all(None, now);
        expected.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.memory.id.cmp(&b.memory.id))
        });

        let retrieved = stream.retrieve(None, now, 6);

        assert_eq!(retrieved.len(), 6);
        for (got, want) in retrieved.iter().zip(expected.iter()) {
            assert_eq!(got.memory.id, want.memory.id);
            assert!((got.score - want.score).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn test_bounds_limit_to_recent_window() {
        let t = base();
        let mut stream = MemoryStream::new();
        stream.set_retrieval_bounds(Some(RetrievalBounds::new(10, 0)));

        for i in 0..500 {
            stream.observe(format!("m{i}"), 1.0, t + Duration::seconds(i));
        }

        let ids: Vec<MemoryId> = stream
            .retrieve(None, t + Duration::seconds(500), 50)
            .iter()
            .map(|s| s.memory.id)
            .collect();

        // only ten candidates
        assert_eq!(ids.len(), 10);
        assert!(ids.iter().all(|&id| id >= 490));
    }

    #[test]
    fn test_bounds_keep_an_important_old_memory_reachable() {
        let t = base();
        let mut stream = MemoryStream::new();
        stream.set_retrieval_bounds(Some(RetrievalBounds::new(5, 2)));

        let landmark = stream.observe("landmark", 100.0, t);
        for i in 0..200 {
            stream.observe(format!("m{i}"), 0.1, t + Duration::seconds(i + 1));
        }

        let ids: Vec<MemoryId> = stream
            .retrieve(None, t + Duration::seconds(500), 10)
            .iter()
            .map(|s| s.memory.id)
            .collect();

        assert!(
            ids.contains(&landmark),
            "an old but important memory should survive the window: {ids:?}"
        );
    }

    #[test]
    fn test_bounds_applied_late() {
        let t = base();
        let mut stream = MemoryStream::new();

        let landmark = stream.observe("landmark", 100.0, t);
        for i in 0..100 {
            stream.observe(format!("m{i}"), 0.1, t + Duration::seconds(i + 1));
        }

        // rebuilt from earlier memories
        stream.set_retrieval_bounds(Some(RetrievalBounds::new(3, 1)));

        let ids: Vec<MemoryId> = stream
            .retrieve(None, t + Duration::seconds(500), 10)
            .iter()
            .map(|s| s.memory.id)
            .collect();

        assert!(ids.contains(&landmark), "{ids:?}");
        assert_eq!(ids.len(), 4);
    }

    #[test]
    fn test_removing_bounds_restores_a_full_scan() {
        let t = base();
        let mut stream = MemoryStream::new();
        stream.set_retrieval_bounds(Some(RetrievalBounds::new(2, 0)));

        for i in 0..20 {
            stream.observe(format!("m{i}"), 1.0, t + Duration::seconds(i));
        }
        assert_eq!(stream.retrieve(None, t, 50).len(), 2);

        stream.set_retrieval_bounds(None);
        assert_eq!(stream.retrieval_bounds(), None);
        assert_eq!(stream.retrieve(None, t, 50).len(), 20);
    }

    #[test]
    fn test_id_assignment() {
        let t = base();
        let mut stream = MemoryStream::new();
        assert_eq!(stream.observe("a", 1.0, t), 0);
        assert_eq!(stream.observe("b", 1.0, t), 1);
    }

    #[test]
    fn test_reflect_trigger() {
        let t = base();
        let mut stream = MemoryStream::new();
        stream.reflection_threshold = 10.0;
        stream.observe("x", 4.0, t);
        assert!(!stream.should_reflect());
        stream.observe("y", 7.0, t);
        assert!(stream.should_reflect());
    }

    #[test]
    fn test_reflect_evidence() {
        let t = base();
        let mut stream = MemoryStream::new();
        let a = stream.observe("a", 5.0, t);
        let b = stream.observe("b", 5.0, t);

        let new_ids = stream.reflect(&MockReflector, t);
        assert_eq!(new_ids.len(), 1);

        let reflection = stream
            .memories()
            .iter()
            .find(|m| m.id == new_ids[0])
            .unwrap();
        assert_eq!(reflection.kind, MemoryKind::Reflection);
        assert!(reflection.evidence.iter().all(|e| *e == a || *e == b));
    }

    #[test]
    fn test_reflect_resets_trigger() {
        let t = base();
        let mut stream = MemoryStream::new();
        stream.reflection_threshold = 5.0;
        stream.observe("x", 6.0, t);
        assert!(stream.should_reflect());

        stream.reflect(&MockReflector, t);
        assert!(!stream.should_reflect());
    }

    #[test]
    fn test_reflect_empty() {
        let t = base();
        let mut stream = MemoryStream::new();
        assert!(stream.reflect(&MockReflector, t).is_empty());
        assert!(stream.is_empty());
    }
}
