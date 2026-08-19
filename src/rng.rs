use rand::RngCore;
use rand::SeedableRng;
use rand::rngs::StdRng;

const PHI: u64 = 0x9E37_79B9_7F4A_7C15;

/// The generator behind each agent's substream.
///
/// A simulation holds one of these per agent and touches them in event order, so the array of them
/// is walked randomly and its size shows up directly as cache misses. This is SplitMix64: a single
/// word of state against the 32 bytes of xoshiro256++ and the 136 of [`StdRng`], emitting 64 bits
/// per call. The algorithm is pinned here rather than taken from a dependency so a seed keeps
/// replaying the same run across upgrades.
///
/// Its period is 2^64 per agent. That is far short of xoshiro's 2^256 but still leaves room for
/// 10^19 draws on a single substream, which no run this is built for will approach.
#[derive(Debug, Clone)]
pub struct SimRng {
    state: u64,
}

impl SimRng {
    pub fn seed_from_u64(seed: u64) -> Self {
        SimRng { state: seed }
    }
}

impl RngCore for SimRng {
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(PHI);
        splitmix64_mix(self.state)
    }

    fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        let mut chunks = dest.chunks_exact_mut(8);
        for chunk in &mut chunks {
            chunk.copy_from_slice(&self.next_u64().to_le_bytes());
        }

        let tail = chunks.into_remainder();
        if !tail.is_empty() {
            let bytes = self.next_u64().to_le_bytes();
            tail.copy_from_slice(&bytes[..tail.len()]);
        }
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand::Error> {
        self.fill_bytes(dest);
        Ok(())
    }
}

pub fn random_seed() -> u64 {
    StdRng::from_entropy().next_u64()
}

pub fn substream(seed: u64, index: u64) -> SimRng {
    SimRng::seed_from_u64(derive(seed, index))
}

/// [`substream`]'s seed without the generator, for handing to something that seeds itself, such as
/// sampling in a local model.
pub fn derive(seed: u64, index: u64) -> u64 {
    splitmix64(seed ^ splitmix64(index))
}

fn splitmix64(x: u64) -> u64 {
    splitmix64_mix(x.wrapping_add(PHI))
}

fn splitmix64_mix(x: u64) -> u64 {
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn draw(mut rng: SimRng) -> Vec<u64> {
        (0..8).map(|_| rng.next_u64()).collect()
    }

    #[test]
    fn test_substream_is_reproducible() {
        assert_eq!(draw(substream(42, 0)), draw(substream(42, 0)));
        assert_eq!(draw(substream(42, 7)), draw(substream(42, 7)));
    }

    #[test]
    fn test_substream_differs_by_index() {
        assert_ne!(draw(substream(42, 0)), draw(substream(42, 1)));
        assert_ne!(draw(substream(42, 0)), draw(substream(42, 2)));
    }

    #[test]
    fn test_substream_differs_by_seed() {
        assert_ne!(draw(substream(1, 0)), draw(substream(2, 0)));
    }

    #[test]
    fn test_random_seed_varies() {
        assert_ne!(random_seed(), random_seed());
    }

    #[test]
    fn test_output_does_not_get_stuck() {
        let values = draw(substream(0, 0));
        assert!(values.iter().any(|&v| v != 0));
        assert_eq!(
            values
                .iter()
                .collect::<std::collections::HashSet<_>>()
                .len(),
            8
        );
    }

    #[test]
    fn test_fill_bytes_handles_a_partial_tail() {
        let mut rng = substream(9, 1);
        let mut buffer = [0u8; 13];
        rng.fill_bytes(&mut buffer);

        assert!(buffer.iter().any(|&b| b != 0));

        // the same seed fills the same bytes, tail included
        let mut again = [0u8; 13];
        substream(9, 1).fill_bytes(&mut again);
        assert_eq!(buffer, again);
    }

    #[test]
    fn test_state_is_one_word() {
        assert_eq!(std::mem::size_of::<SimRng>(), 8);
        assert!(std::mem::size_of::<SimRng>() < std::mem::size_of::<StdRng>());
    }
}
