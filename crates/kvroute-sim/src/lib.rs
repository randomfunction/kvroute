//! kvroute-sim — a deterministic discrete-event simulator for evaluating routers
//! without a GPU in sight.
//!
//! The point of the simulator is iteration speed and reproducibility: given a
//! seed and a [`trace::TraceSpec`], every router sees the exact same request
//! stream, and a run is a pure function of its inputs. The same [`Router`] object
//! later drives the live proxy unchanged, because [`kvroute_core::Router::select`]
//! is sync and pure.
//!
//! Pieces:
//! * [`rng`] — seeded splitmix64 RNG with the distributions the trace needs.
//! * [`cache`] — a per-replica LRU KV-cache model, so a "hit" is *earned*.
//! * [`trace`] — skewed prefixes, bursts, tenants and multi-turn agent sessions.
//! * [`engine`] — the event loop that ties a router to the replica models.
//! * [`metrics`] — TTFT percentiles, hit rate, load imbalance, throughput.

pub mod cache;
pub mod engine;
pub mod metrics;
pub mod rng;
pub mod trace;

pub use engine::{Engine, SimConfig};
pub use metrics::RunReport;
pub use trace::{SimRequest, TraceSpec};
