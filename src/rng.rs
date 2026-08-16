use rand::SeedableRng;
use rand::rngs::StdRng;

const PHI: u64 = 0x9E37_79B9_7F4A_7C15;

pub fn random_seed() -> u64 {
    use rand::RngCore;
    StdRng::from_entropy().next_u64()
}

pub fn substream(seed: u64, index: u64) -> StdRng {
    StdRng::seed_from_u64(derive(seed, index))
}

// derive is substream's seed without the generator, for handing to something that seeds itself —
// sampling in a local model, for instance.
pub fn derive(seed: u64, index: u64) -> u64 {
    splitmix64(seed ^ splitmix64(index))
}

fn splitmix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(PHI);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::RngCore;

    fn draw(mut rng: StdRng) -> Vec<u64> {
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
}
