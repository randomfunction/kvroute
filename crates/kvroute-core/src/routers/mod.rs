//! The [`Router`] trait and the baseline strategies.
//!
//! [`Router::select`] is synchronous and pure. All I/O lives in the state
//! tracker; all a router does is read the snapshot and return a decision. That
//! is what lets the same object run in the single-threaded simulator and in the
//! live proxy unchanged.

mod hash;
mod novel;
mod radix;
mod round_robin;
mod score;

pub use hash::HashRouter;
pub use novel::NovelRouter;
pub use radix::RadixRouter;
pub use round_robin::RoundRobin;
pub use score::ScoreRouter;

use crate::types::{Outcome, Request, ReplicaState, RouteDecision, Snapshot};

/// A routing strategy. Implementations are shared behind `&self` (an `Arc` or a
/// `Box<dyn Router>` in practice), so any mutable state must use interior
/// mutability — and must stay off the decision's critical section.
pub trait Router: Send + Sync {
    /// Short stable name, used in configs, logs and the bench table.
    fn name(&self) -> &'static str;

    /// Choose a replica for `req` given `snap`. Pure and fast.
    fn select(&self, req: &Request, snap: &Snapshot<'_>) -> RouteDecision;

    /// Dispatch hook. The *shared* prefix store is updated by the runtime (it is
    /// shared infrastructure), so routers need not touch it here. Session-aware
    /// policies use this to record per-session timing; the baselines don't.
    fn on_dispatch(&self, _req: &Request, _decision: &RouteDecision) {}

    /// Completion hook. Baselines ignore it; adaptive policies learn from it.
    fn on_complete(&self, _req: &Request, _decision: &RouteDecision, _outcome: &Outcome) {}
}

// ----- helpers shared by the strategies -----

/// The least-loaded healthy replica, ties broken by lower id for determinism.
/// Falls back to the very first replica if none are healthy, so a decision is
/// always returned rather than dropping the request.
pub(crate) fn least_loaded<'a>(replicas: &'a [ReplicaState], kv_weight: f32) -> (&'a ReplicaState, bool) {
    let mut best: Option<&ReplicaState> = None;
    for r in replicas.iter().filter(|r| r.healthy) {
        best = Some(match best {
            None => r,
            Some(b) => {
                let (lr, lb) = (r.load(kv_weight), b.load(kv_weight));
                if lr < lb || (lr == lb && r.replica_id < b.replica_id) {
                    r
                } else {
                    b
                }
            }
        });
    }
    match best {
        Some(r) => (r, true),
        None => (&replicas[0], false),
    }
}
