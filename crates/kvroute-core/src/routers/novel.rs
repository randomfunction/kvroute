//! Novel session-aware policy — **deferred to phase 3**.
//!
//! This is the seam, not the policy. It implements [`Router`] today by behaving
//! as the score router (so the workspace builds and benches run with it present),
//! and it already carries the hooks the real policy will use:
//!
//! * `on_dispatch` / `on_complete` record per-session timing — the signal an
//!   adaptive λ or a resume predictor needs.
//! * It reads the same shared prefix store as the baselines.
//!
//! When phase 3 starts, the work is to replace [`WeightPolicy`] with an adaptive
//! one and add resume prediction here — nothing in the `Router` contract or the
//! state tracker has to move.

use super::{Router, ScoreRouter};
use crate::types::{Outcome, Request, RouteDecision, Snapshot};
use crate::weight::FixedLambda;

#[derive(Debug)]
pub struct NovelRouter {
    // TODO(phase 3): swap this for an adaptive, session-aware WeightPolicy and a
    // resume predictor keyed on (tenant_id, session_id).
    inner: ScoreRouter,
}

impl Default for NovelRouter {
    fn default() -> Self {
        Self {
            inner: ScoreRouter::new(16, 0.5, Box::new(FixedLambda::default())),
        }
    }
}

impl NovelRouter {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Router for NovelRouter {
    fn name(&self) -> &'static str {
        "novel"
    }

    fn select(&self, req: &Request, snap: &Snapshot<'_>) -> RouteDecision {
        // Placeholder behaviour: identical to score-based until the real policy
        // lands. Kept wired so ablations can include `novel` from day one.
        self.inner.select(req, snap)
    }

    fn on_dispatch(&self, _req: &Request, _decision: &RouteDecision) {
        // TODO(phase 3): stamp last-dispatch time for (tenant, session).
    }

    fn on_complete(&self, _req: &Request, _decision: &RouteDecision, _outcome: &Outcome) {
        // TODO(phase 3): record think-time / tool-call pause to predict resume.
    }
}
