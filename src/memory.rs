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

/// Produces the focal questions for a reflection and the insights synthesized from them.
/// Triggering, retrieval, and storage live in [`MemoryStream`].
pub trait Reflector {
    fn salient_questions(&self, recent: &[&Memory]) -> Vec<String>;
    fn synthesize(&self, question: &str, evidence: &[&Memory]) -> Vec<Insight>;
    fn embed_query(&self, _question: &str) -> Option<Vec<f32>> {
        None
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
        }
    }
}

impl MemoryStream {
    pub fn new() -> Self {
        Self::default()
    }

    /// Assigns a stable id and accumulates importance toward the next reflection.
    pub fn add(&mut self, mut memory: Memory) -> MemoryId {
        let id = self.next_id;
        self.next_id += 1;
        memory.id = id;
        self.importance_since_reflection += memory.importance;
        self.memories.push(memory);
        id
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

    /// Asks the [`Reflector`] for focal questions over recent memories, retrieves evidence for
    /// each, and stores the synthesized insights.
    //
    // insights are added after all retrieval so they can't cite each other within a single pass
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

        // reset after storing so the fresh reflections don't immediately re-trigger.
        self.importance_since_reflection = 0.0;
        new_ids
    }

    /// Returns the `top_k` memories by combined recency, importance, and relevance score, and
    /// refreshes their access time.
    pub fn retrieve(
        &mut self,
        query: Option<&[f32]>,
        now: DateTime<Utc>,
        top_k: usize,
    ) -> Vec<ScoredMemory> {
        let mut scored = self.score_all(query, now);
        scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);

        for s in &scored {
            if let Some(m) = self.memories.iter_mut().find(|m| m.id == s.memory.id) {
                m.last_accessed = now;
            }
        }

        scored
    }

    pub fn score_all(&self, query: Option<&[f32]>, now: DateTime<Utc>) -> Vec<ScoredMemory> {
        if self.memories.is_empty() {
            return Vec::new();
        }

        let recency_raw: Vec<f64> = self
            .memories
            .iter()
            .map(|m| {
                let hours = (now - m.last_accessed).num_milliseconds() as f64 / 3_600_000.0;
                self.recency_decay.powf(hours.max(0.0))
            })
            .collect();

        let importance_raw: Vec<f64> = self.memories.iter().map(|m| m.importance).collect();

        let relevance_raw: Vec<f64> = self
            .memories
            .iter()
            .map(|m| match (query, m.embedding.as_deref()) {
                (Some(q), Some(e)) => cosine_similarity(q, e),
                _ => 0.0,
            })
            .collect();

        let recency = min_max_normalize(&recency_raw);
        let importance = min_max_normalize(&importance_raw);
        let relevance = min_max_normalize(&relevance_raw);

        self.memories
            .iter()
            .enumerate()
            .map(|(i, m)| {
                let score = self.weights.recency * recency[i]
                    + self.weights.importance * importance[i]
                    + self.weights.relevance * relevance[i];
                ScoredMemory {
                    memory: m.clone(),
                    recency: recency[i],
                    importance: importance[i],
                    relevance: relevance[i],
                    score,
                }
            })
            .collect()
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

fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() {
        return 0.0;
    }

    let mut dot = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        let (x, y) = (*x as f64, *y as f64);
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a.sqrt() * norm_b.sqrt())
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
        assert!((cosine_similarity(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-9);
        assert!(cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-9);
        assert_eq!(cosine_similarity(&[1.0], &[1.0, 0.0]), 0.0);
        assert_eq!(cosine_similarity(&[0.0, 0.0], &[1.0, 1.0]), 0.0);
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

        let reflection = stream.memories().iter().find(|m| m.id == new_ids[0]).unwrap();
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
