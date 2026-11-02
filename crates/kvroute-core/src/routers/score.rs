//! Score-based router — the tunable one, and the novel policy's parent.
//!
//! For each healthy replica:
//!
//! ```text
//! score = tokens_saved(prefix_match) − λ · load(waiting, running, kv_usage)
//! ```
//!
//! and the max wins. `λ` comes from a [`WeightPolicy`], not a constant, so a
//! fixed weight today becomes an adaptive (session-aware) one later without the
//! interface moving. This is the exact seam `NovelRouter` replaces.

use std::sync::Mutex;

use super::{least_loaded, Router};
use crate::ids::ReplicaId;
use crate::types::{Reason, Request, RouteDecision, Snapshot};
use crate::weight::{FixedLambda, WeightPolicy};

#[derive(Debug)]
pub struct ScoreRouter {
    block_size: u32,
    kv_weight: f32,
    policy: Box<dyn WeightPolicy>,
    scratch: Mutex<Vec<(ReplicaId, u32)>>,
    /// Keep per-replica scores on the decision for tracing/ablation.
    emit_scores: bool,
}

impl ScoreRouter {
    pub fn new(block_size: u32, kv_weight: f32, policy: Box<dyn WeightPolicy>) -> Self {
        Self {
            block_size: block_size.max(1),
            kv_weight,
            policy,
            scratch: Mutex::new(Vec::new()),
            emit_scores: false,
        }
    }

    /// Fixed-λ construction — the phase-1 default.
    pub fn fixed(block_size: u32, lambda: f32) -> Self {
        Self::new(block_size, 0.5, Box::new(FixedLambda(lambda)))
    }

    /// Attach per-replica score vectors to every decision (ablation mode).
    pub fn with_scores(mut self, on: bool) -> Self {
        self.emit_scores = on;
        self
    }
}

impl Router for ScoreRouter {
    fn name(&self) -> &'static str {
        "score"
    }

    fn select(&self, req: &Request, snap: &Snapshot<'_>) -> RouteDecision {
        let lambda = self.policy.lambda(req, snap);

        // Matched blocks per replica (0 for those not present).
        let mut matches = self.scratch.lock().unwrap();
        snap.prefixes.matched_blocks(&req.block_hashes, &mut matches);

        let mut scores: Vec<(ReplicaId, f32)> = if self.emit_scores {
            Vec::with_capacity(snap.replicas.len())
        } else {
            Vec::new()
        };

        let mut best: Option<(ReplicaId, f32, u32)> = None;
        for state in snap.replicas.iter().filter(|r| r.healthy) {
            let blocks = matches
                .iter()
                .find(|(r, _)| *r == state.replica_id)
                .map(|(_, b)| *b)
                .unwrap_or(0);
            let tokens_saved = (blocks * self.block_size) as f32;
            let load = state.load(self.kv_weight);
            let score = tokens_saved - lambda * load;

            if self.emit_scores {
                scores.push((state.replica_id, score));
            }
            best = Some(match best {
                None => (state.replica_id, score, blocks),
                Some((bid, bscore, bblocks)) => {
                    // Higher score wins; exact ties broken by lower id.
                    if score > bscore || (score == bscore && state.replica_id < bid) {
                        (state.replica_id, score, blocks)
                    } else {
                        (bid, bscore, bblocks)
                    }
                }
            });
        }

        drop(matches);

        match best {
            Some((rid, _, blocks)) => {
                let mut d = RouteDecision::new(rid, Reason::Score { matched_blocks: blocks });
                if self.emit_scores {
                    d.scores = Some(scores);
                }
                d
            }
            None => {
                let (r, _) = least_loaded(snap.replicas, self.kv_weight);
                RouteDecision::new(r.replica_id, Reason::Fallback)
            }
        }
    }
}
