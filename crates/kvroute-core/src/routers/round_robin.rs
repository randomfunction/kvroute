//! Round-robin — the control arm.
//!
//! Hands each request to the next healthy replica by an atomic counter. No
//! state, no prefix awareness: perfect load balance, zero locality. Every other
//! strategy is measured against this.

use std::sync::atomic::{AtomicU64, Ordering};

use super::{least_loaded, Router};
use crate::types::{Reason, Request, RouteDecision, Snapshot};

#[derive(Debug, Default)]
pub struct RoundRobin {
    counter: AtomicU64,
}

impl RoundRobin {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Router for RoundRobin {
    fn name(&self) -> &'static str {
        "round_robin"
    }

    fn select(&self, _req: &Request, snap: &Snapshot<'_>) -> RouteDecision {
        // Count healthy replicas; walk from the next counter slot to the first
        // healthy one. Relaxed ordering is fine — we only need distinctness over
        // time, not a global sequence.
        let n = snap.replicas.len();
        let start = self.counter.fetch_add(1, Ordering::Relaxed) as usize;
        for off in 0..n {
            let idx = (start + off) % n;
            let r = &snap.replicas[idx];
            if r.healthy {
                return RouteDecision::new(r.replica_id, Reason::RoundRobin);
            }
        }
        // Nothing healthy: least_loaded returns the first replica with a flag.
        let (r, _) = least_loaded(snap.replicas, 0.0);
        RouteDecision::new(r.replica_id, Reason::Fallback)
    }
}
