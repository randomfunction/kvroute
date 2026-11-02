//! The data the routing contract is expressed in.

use std::fmt;
use std::sync::Arc;

use crate::ids::{ReplicaId, RequestId, SessionId, TenantId};
use crate::prefix::store::PrefixStore;

/// An inbound request, already fingerprinted. The router never sees raw text:
/// prefix matching happens on `block_hashes`, a salted, chained hash per
/// 16-token block, so the router's notion of a shared prefix lines up with what
/// the engine can actually reuse.
#[derive(Clone, Debug)]
pub struct Request {
    pub id: RequestId,
    pub model: Arc<str>,
    /// Owner — isolation / fairness. Optional: anonymous traffic still routes.
    pub tenant_id: Option<TenantId>,
    /// Conversation / agent run. Optional.
    pub session_id: Option<SessionId>,
    /// Mirrors vLLM's `cache_salt`; folded into the hash chain so the router
    /// never predicts a cross-tenant hit the engine won't give.
    pub cache_salt: Option<u64>,
    /// One hash per full block of the prompt prefix, chained front-to-back.
    pub block_hashes: Arc<[u64]>,
    /// Prompt length in tokens (whole prompt, not just full blocks).
    pub prompt_tokens: u32,
    /// Arrival time, nanoseconds on the run's clock.
    pub arrival_ns: u64,
}

impl Request {
    /// The tenant-scoped stickiness key, when the request carries a session.
    /// Clients choose session ids and they collide across tenants, so the key
    /// folds in the tenant.
    pub fn sticky_key(&self) -> Option<u64> {
        self.session_id.map(|s| {
            let t = self.tenant_id.map(|t| t.0).unwrap_or(0);
            crate::prefix::hashing::mix64(t, s.0)
        })
    }
}

/// A point-in-time view of one replica. Produced off the hot path by the state
/// tracker (or the simulator) and read lock-free by routers.
#[derive(Clone, Debug)]
pub struct ReplicaState {
    pub replica_id: ReplicaId,
    pub endpoint: Arc<str>,
    /// Requests queued, not yet decoding.
    pub waiting: u32,
    /// Requests in flight (decoding).
    pub running: u32,
    /// Fraction of KV blocks in use, `0.0..=1.0`.
    pub kv_usage: f32,
    pub healthy: bool,
    /// When this row was last refreshed (run clock, ns) — its staleness.
    pub updated_ns: u64,
}

impl ReplicaState {
    /// A scalar load estimate. `waiting + running`, plus a weighted KV-pressure
    /// term so a replica that is not queue-deep but is about to thrash its cache
    /// still looks loaded.
    #[inline]
    pub fn load(&self, kv_weight: f32) -> f32 {
        self.waiting as f32 + self.running as f32 + kv_weight * self.kv_usage
    }
}

/// Why a replica was chosen — human-readable, and useful in the trace log.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Reason {
    /// Round-robin counter, no state consulted.
    RoundRobin,
    /// Consistent-hash ring landed here (stickiness).
    Sticky,
    /// Longest prefix match; `blocks` leading blocks believed resident.
    Prefix { blocks: u32 },
    /// Highest score; `matched_blocks` leading blocks credited to the winner.
    Score { matched_blocks: u32 },
    /// No cache signal; fell back to the least-loaded replica.
    LeastLoad,
    /// Degenerate path (e.g. nothing healthy) — picked something to avoid a drop.
    Fallback,
}

impl fmt::Display for Reason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Reason::RoundRobin => write!(f, "round-robin"),
            Reason::Sticky => write!(f, "sticky"),
            Reason::Prefix { blocks } => write!(f, "prefix:{blocks}blk"),
            Reason::Score { matched_blocks } => write!(f, "score:{matched_blocks}blk"),
            Reason::LeastLoad => write!(f, "least-load"),
            Reason::Fallback => write!(f, "fallback"),
        }
    }
}

/// The router's answer.
#[derive(Clone, Debug)]
pub struct RouteDecision {
    pub replica_id: ReplicaId,
    pub reason: Reason,
    /// Per-replica scores, kept only when tracing/ablation asks for them.
    pub scores: Option<Vec<(ReplicaId, f32)>>,
}

impl RouteDecision {
    #[inline]
    pub fn new(replica_id: ReplicaId, reason: Reason) -> Self {
        Self { replica_id, reason, scores: None }
    }
}

/// What actually happened, fed back after the reply completes. Drives metrics
/// and any adaptive policy.
#[derive(Clone, Copy, Debug)]
pub struct Outcome {
    /// Time to first token, ns (queue wait + prefill).
    pub ttft_ns: u64,
    /// Arrival to completion, ns.
    pub total_ns: u64,
    pub output_tokens: u32,
    /// Leading prompt blocks that were a real cache hit (prefill skipped).
    pub cached_blocks: u32,
    /// Prompt tokens that had to be prefilled (uncached).
    pub prefill_tokens: u32,
}

/// Everything a router may read for one decision: the replica rows plus the
/// shared prefix store, borrowed. The live runtime holds these behind
/// `Arc`/`ArcSwap`; here they are borrowed from whoever owns the tick, which
/// keeps the simulator allocation-free on the hot path.
#[derive(Debug)]
pub struct Snapshot<'a> {
    pub replicas: &'a [ReplicaState],
    pub prefixes: &'a PrefixStore,
    /// Bumped every time the tracker publishes.
    pub version: u64,
    /// The run clock at decision time, ns.
    pub now_ns: u64,
}

impl<'a> Snapshot<'a> {
    /// Look up a replica row by id. `O(n)`; `n` is small.
    pub fn replica(&self, id: ReplicaId) -> Option<&ReplicaState> {
        self.replicas.iter().find(|r| r.replica_id == id)
    }
}
