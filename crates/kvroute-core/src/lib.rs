//! kvroute-core — the small set of abstractions every routing strategy is
//! written against, plus the four baseline strategies.
//!
//! The whole design hangs off one narrow contract: a [`Router`] sees a
//! [`Request`] and a borrowed [`Snapshot`] of replica state, and returns a
//! [`RouteDecision`]. [`Router::select`] is **synchronous and pure** — no
//! `.await`, no network — which is exactly why the identical object runs in the
//! discrete-event simulator and, later, in the live axum proxy.
//!
//! This crate has no async runtime and no I/O. The prefix store ([`PrefixStore`])
//! is a plain data structure here; the live runtime wraps it in an
//! `Arc<RwLock<_>>` and publishes snapshots via `ArcSwap` (crate `kvroute-state`,
//! not yet built).

pub mod ids;
pub mod prefix;
pub mod routers;
pub mod types;
pub mod weight;

pub use ids::{ReplicaId, RequestId, SessionId, TenantId};
pub use prefix::hashing::{block_hashes, fnv1a, mix64};
pub use prefix::store::PrefixStore;
pub use routers::{HashRouter, NovelRouter, RadixRouter, RoundRobin, Router, ScoreRouter};
pub use types::{Outcome, Reason, Request, ReplicaState, RouteDecision, Snapshot};
pub use weight::{FixedLambda, WeightPolicy};
