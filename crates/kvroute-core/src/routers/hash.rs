//! Consistent-hash router — cheap stickiness.
//!
//! Maps a key (the tenant-scoped session key, else the first block hash, else
//! the request id) onto a hash ring with virtual nodes. The same key lands on
//! the same replica, so a session's turns reuse cache by coincidence. No load
//! awareness: a hot key is a hotspot. Membership changes rebalance minimally —
//! only the keys between the departed replica's vnodes move.

use std::sync::Mutex;

use super::{least_loaded, Router};
use crate::ids::ReplicaId;
use crate::prefix::hashing::mix64;
use crate::types::{Reason, Request, ReplicaState, RouteDecision, Snapshot};

/// Default virtual nodes per replica. Higher ⇒ smoother balance, bigger ring.
const DEFAULT_VNODES: u32 = 160;

#[derive(Debug)]
struct Ring {
    /// Which membership this ring was built for; rebuilt when it changes.
    membership_key: u64,
    /// `(point, replica)` sorted by `point`.
    points: Vec<(u64, ReplicaId)>,
}

#[derive(Debug)]
pub struct HashRouter {
    vnodes: u32,
    ring: Mutex<Option<Ring>>,
}

impl Default for HashRouter {
    fn default() -> Self {
        Self::new(DEFAULT_VNODES)
    }
}

impl HashRouter {
    pub fn new(vnodes: u32) -> Self {
        Self { vnodes: vnodes.max(1), ring: Mutex::new(None) }
    }

    /// A membership fingerprint over the healthy replica set, order-independent.
    fn membership_key(replicas: &[ReplicaState]) -> u64 {
        let mut acc = 0u64;
        for r in replicas {
            // XOR so order does not matter; fold health in so a flap rebuilds.
            acc ^= mix64(r.replica_id.0 as u64, r.healthy as u64 + 1);
        }
        acc
    }

    fn build(&self, replicas: &[ReplicaState]) -> Ring {
        let mut points: Vec<(u64, ReplicaId)> = Vec::with_capacity(replicas.len() * self.vnodes as usize);
        for r in replicas.iter().filter(|r| r.healthy) {
            for v in 0..self.vnodes {
                let p = mix64(r.replica_id.0 as u64, 0xA5A5 ^ v as u64);
                points.push((p, r.replica_id));
            }
        }
        points.sort_by_key(|&(p, _)| p);
        Ring { membership_key: Self::membership_key(replicas), points }
    }

    /// The routing key: tenant-scoped session if present, else the first block
    /// hash (so identical prompts stick together), else the request id.
    fn key(req: &Request) -> u64 {
        if let Some(k) = req.sticky_key() {
            return k;
        }
        if let Some(&first) = req.block_hashes.first() {
            return first;
        }
        req.id.0
    }
}

impl Router for HashRouter {
    fn name(&self) -> &'static str {
        "hash"
    }

    fn select(&self, req: &Request, snap: &Snapshot<'_>) -> RouteDecision {
        let want = Self::membership_key(snap.replicas);
        let mut guard = self.ring.lock().unwrap();
        if guard.as_ref().map(|r| r.membership_key) != Some(want) {
            *guard = Some(self.build(snap.replicas));
        }
        let ring = guard.as_ref().unwrap();

        if ring.points.is_empty() {
            // `ring` still borrows `guard`; the lock simply releases when the
            // guard drops at return. `least_loaded` reads unrelated data.
            let (r, _) = least_loaded(snap.replicas, 0.0);
            return RouteDecision::new(r.replica_id, Reason::Fallback);
        }

        let key = Self::key(req);
        // First ring point at or after `key`, wrapping around.
        let start = ring.points.partition_point(|&(p, _)| p < key) % ring.points.len();
        let r = ring.points[start].1;
        RouteDecision::new(r, Reason::Sticky)
    }
}
