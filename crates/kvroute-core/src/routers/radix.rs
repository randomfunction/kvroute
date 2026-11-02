//! Radix-tree router — locality first (the SGLang-style approach).
//!
//! Queries the shared prefix store for the longest-prefix match, picks the
//! replica holding the most leading blocks, and tie-breaks by load. With no
//! match it falls back to least-loaded. Maximises reuse, but chasing the best
//! match can overload whoever owns a popular prefix — the load tie-break is the
//! only thing holding that back (that tension is what the score router addresses).

use std::sync::Mutex;

use super::{least_loaded, Router};
use crate::ids::ReplicaId;
use crate::types::{Reason, Request, RouteDecision, Snapshot};

#[derive(Debug)]
pub struct RadixRouter {
    kv_weight: f32,
    /// Reused scratch buffer for match results, to keep `select` allocation-free
    /// on the steady-state path.
    scratch: Mutex<Vec<(ReplicaId, u32)>>,
}

impl Default for RadixRouter {
    fn default() -> Self {
        Self::new(0.5)
    }
}

impl RadixRouter {
    pub fn new(kv_weight: f32) -> Self {
        Self { kv_weight, scratch: Mutex::new(Vec::new()) }
    }
}

impl Router for RadixRouter {
    fn name(&self) -> &'static str {
        "radix"
    }

    fn select(&self, req: &Request, snap: &Snapshot<'_>) -> RouteDecision {
        let mut matches = self.scratch.lock().unwrap();
        // `&mut matches` (a MutexGuard) deref-coerces to `&mut Vec` at the call.
        snap.prefixes.matched_blocks(&req.block_hashes, &mut matches);

        // Among healthy replicas with a match, take the longest; tie-break by
        // lower load, then lower id.
        let mut best: Option<(ReplicaId, u32, f32)> = None;
        for &(rid, blocks) in matches.iter() {
            let Some(state) = snap.replica(rid) else { continue };
            if !state.healthy {
                continue;
            }
            let load = state.load(self.kv_weight);
            best = Some(match best {
                None => (rid, blocks, load),
                Some((bid, bblocks, bload)) => {
                    let better = blocks > bblocks
                        || (blocks == bblocks && load < bload)
                        || (blocks == bblocks && load == bload && rid < bid);
                    if better {
                        (rid, blocks, load)
                    } else {
                        (bid, bblocks, bload)
                    }
                }
            });
        }

        if let Some((rid, blocks, _)) = best {
            return RouteDecision::new(rid, Reason::Prefix { blocks });
        }

        drop(matches);
        let (r, healthy) = least_loaded(snap.replicas, self.kv_weight);
        RouteDecision::new(
            r.replica_id,
            if healthy { Reason::LeastLoad } else { Reason::Fallback },
        )
    }
}
