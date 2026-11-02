//! A tiny seeded RNG. Deterministic and portable, so a `(seed, spec)` pair always
//! reproduces the same trace. splitmix64 for the stream; standard transforms for
//! the exponential and Zipf draws the trace generator needs.
//!
//! Stands in for `rand_chacha` + `rand_distr`; swap those in later for
//! crypto-grade streams without touching callers.

/// splitmix64.
#[derive(Debug, Clone)]
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        // Avoid the all-zero state producing a degenerate first few outputs.
        Rng { state: seed ^ 0x9E37_79B9_7F4A_7C15 }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)`, 53-bit.
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform integer in `[0, n)`. `n` must be > 0.
    #[inline]
    pub fn below(&mut self, n: usize) -> usize {
        debug_assert!(n > 0);
        (self.next_u64() % n as u64) as usize
    }

    /// Exponential with the given rate (mean = `1/rate`). Used for Poisson
    /// inter-arrival gaps.
    #[inline]
    pub fn exp(&mut self, rate: f64) -> f64 {
        // Inverse-CDF; clamp u away from 0 so ln is finite.
        let u = (1.0 - self.next_f64()).max(1e-12);
        -u.ln() / rate
    }

    /// A Poisson inter-arrival gap in nanoseconds for a mean gap of `mean_ns`.
    #[inline]
    pub fn interarrival_ns(&mut self, mean_ns: f64) -> u64 {
        let rate = 1.0 / mean_ns.max(1.0);
        self.exp(rate).round() as u64
    }
}

/// A precomputed Zipf sampler over `1..=n`, skew `s` (0 = uniform, higher = more
/// skewed toward the first few items). Used for popular system prompts and for
/// heavy tenants.
#[derive(Debug, Clone)]
pub struct Zipf {
    cdf: Vec<f64>,
}

impl Zipf {
    pub fn new(n: usize, s: f64) -> Self {
        assert!(n > 0);
        let mut weights = Vec::with_capacity(n);
        let mut total = 0.0;
        for i in 1..=n {
            let w = 1.0 / (i as f64).powf(s);
            total += w;
            weights.push(total);
        }
        for w in &mut weights {
            *w /= total;
        }
        Zipf { cdf: weights }
    }

    /// Sample an index in `[0, n)`.
    pub fn sample(&self, rng: &mut Rng) -> usize {
        let u = rng.next_f64();
        match self.cdf.binary_search_by(|p| p.partial_cmp(&u).unwrap()) {
            Ok(i) => i,
            Err(i) => i.min(self.cdf.len() - 1),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_stream() {
        let a: Vec<u64> = {
            let mut r = Rng::new(7);
            (0..5).map(|_| r.next_u64()).collect()
        };
        let b: Vec<u64> = {
            let mut r = Rng::new(7);
            (0..5).map(|_| r.next_u64()).collect()
        };
        assert_eq!(a, b);
    }

    #[test]
    fn zipf_is_skewed() {
        let z = Zipf::new(100, 1.2);
        let mut rng = Rng::new(1);
        let mut hits0 = 0;
        for _ in 0..10_000 {
            if z.sample(&mut rng) == 0 {
                hits0 += 1;
            }
        }
        // Item 0 should dominate under s=1.2 — well above the 100-uniform rate.
        assert!(hits0 > 1500, "expected heavy skew, got {hits0}");
    }
}
