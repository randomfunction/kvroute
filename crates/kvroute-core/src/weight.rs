//! The locality-vs-load weight, factored out as a trait.
//!
//! `ScoreRouter` scores a replica as `tokens_saved − λ · load`. `λ` is not a
//! constant — it comes from a [`WeightPolicy`]. Today the only implementation is
//! [`FixedLambda`]; the deferred session-aware policy (phase 3) replaces it with
//! something adaptive. Keeping it a trait now means that policy is a drop-in and
//! the interface does not move.

use crate::types::{Request, Snapshot};

/// Supplies `λ` for one decision. Must be cheap: it is called on the hot path.
pub trait WeightPolicy: Send + Sync + std::fmt::Debug {
    /// The weight on the load term for this request against this snapshot.
    /// Larger `λ` favours spreading load; smaller `λ` favours cache locality.
    fn lambda(&self, req: &Request, snap: &Snapshot<'_>) -> f32;
}

/// A constant weight — the phase-1 baseline. `λ` is in tokens-per-load-unit, so
/// it reads as "how many prefill tokens of locality one extra queued request is
/// worth".
#[derive(Clone, Copy, Debug)]
pub struct FixedLambda(pub f32);

impl WeightPolicy for FixedLambda {
    #[inline]
    fn lambda(&self, _req: &Request, _snap: &Snapshot<'_>) -> f32 {
        self.0
    }
}

impl Default for FixedLambda {
    fn default() -> Self {
        // One queued request is worth ~200 prefill tokens of locality. Chosen so
        // a single shared 16-token block (16 tokens) never outweighs real queue
        // depth, but a long shared system prompt does.
        FixedLambda(200.0)
    }
}
