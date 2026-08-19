//! How retrieval cost grows with the size of an agent's memory stream.
//!
//! `GenerativeAgent::observe` calls `MemoryStream::retrieve` for every event it perceives, and the
//! stream grows by one memory per observation. That makes the cost of the Nth observation a
//! function of N, so this measures the shape of that curve rather than a single point.
//!
//! Pass an embedding width as the first argument to include vector relevance in the score.

use agsim::memory::{MemoryStream, RetrievalBounds};
use chrono::{Duration, TimeZone, Utc};
use std::time::Instant;

const RETRIEVE_TOP_K: usize = 10;

fn main() {
    let embedding_width: usize = std::env::args()
        .nth(1)
        .and_then(|raw| raw.parse().ok())
        .unwrap_or(0);
    // second argument caps the candidate set: `<recent>` most recent plus that many most important
    let bound: Option<usize> = std::env::args().nth(2).and_then(|raw| raw.parse().ok());

    let start = Utc.with_ymd_and_hms(2026, 1, 5, 9, 0, 0).unwrap();
    let query: Option<Vec<f32>> =
        (embedding_width > 0).then(|| (0..embedding_width).map(|i| i as f32 * 0.01).collect());

    println!(
        "bound {}",
        bound.map(|b| b.to_string()).unwrap_or("none".to_string())
    );
    println!(
        "embedding width {}\n",
        if embedding_width == 0 {
            "none".to_string()
        } else {
            embedding_width.to_string()
        }
    );
    println!(
        "{:>12} | {:>14} | {:>16}",
        "memories", "us per observe", "projected total s"
    );
    println!("{}", "-".repeat(50));

    for &target in &[1_000usize, 5_000, 10_000, 25_000, 50_000] {
        let mut stream = MemoryStream::new();
        // a threshold this high keeps reflection out of the measurement
        stream.reflection_threshold = f64::INFINITY;
        if let Some(recent) = bound {
            stream.set_retrieval_bounds(Some(RetrievalBounds::new(recent, recent / 10)));
        }

        // fill to just under the target so the timed window sits at a known stream size
        for index in 0..target {
            let time = start + Duration::seconds(index as i64);
            let memory = agsim::memory::Memory::new(
                format!("trader_{index:05}: quote_price -> {index}"),
                (index % 10) as f64 / 10.0,
                time,
            );
            let memory = match &query {
                Some(vector) => memory.with_embedding(vector.clone()),
                None => memory,
            };
            stream.add(memory);
        }

        // one observation is an add plus the retrieve that GenerativeAgent::observe performs
        let sample = 200.min(target);
        let clock = Instant::now();
        for index in 0..sample {
            let time = start + Duration::seconds((target + index) as i64);
            stream.observe(format!("observation {index}"), 0.5, time);
            let _ = stream.retrieve(query.as_deref(), time, RETRIEVE_TOP_K);
        }
        let per_observe_us = clock.elapsed().as_secs_f64() * 1e6 / sample as f64;

        // what it would cost one agent to accumulate this many observations from empty
        let projected_total_s = per_observe_us * target as f64 / 2.0 / 1e6;

        println!("{target:>12} | {per_observe_us:>14.1} | {projected_total_s:>16.1}");
    }
}
