//! The discrete-event engine.
//!
//! Requests are replayed in arrival order. For each one the engine builds a
//! [`Snapshot`] from the current replica models, asks the router to choose, then
//! services the request on the chosen replica and feeds the outcome back.
//!
//! # Service model
//!
//! Each replica is a single FIFO server whose **busy time is the prefill cost**
//! of the uncached tokens. Decode time is added to a request's end-to-end latency
//! but does *not* occupy the server — a first-order stand-in for continuous
//! batching, where decode of many sequences overlaps. This keeps queueing
//! governed by prefill throughput, which is exactly the quantity cache locality
//! reduces, so TTFT differences between routers are attributable to routing.
//!
//! TTFT = queue wait + prefill. The cache-hit prefix skips prefill, so a warm
//! route both lowers this request's prefill *and* frees the server sooner for the
//! next one.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::sync::Arc;
use std::time::Instant;

use kvroute_core::{Outcome, PrefixStore, ReplicaId, ReplicaState, Router, Snapshot};

use crate::cache::KvCache;
use crate::metrics::{Collector, RunReport};
use crate::trace::SimRequest;

/// Engine configuration — the replica fleet and its cost model.
#[derive(Clone, Debug)]
pub struct SimConfig {
    pub replicas: usize,
    pub block_size: u32,
    /// Resident blocks per replica before LRU eviction kicks in.
    pub cache_capacity_blocks: usize,
    /// Nanoseconds of server-busy time per *uncached* prompt token.
    pub prefill_ns_per_token: f64,
    /// Nanoseconds added to end-to-end latency per output token (does not occupy
    /// the server — see the module docs).
    pub decode_ns_per_token: f64,
    /// Prune the prefix-store belief older than this (ns); `0` disables pruning,
    /// leaving the router's belief to drift from the real cache (stale-view study).
    pub prune_ttl_ns: u64,
}

impl SimConfig {
    pub fn new(replicas: usize) -> Self {
        SimConfig {
            replicas: replicas.max(1),
            block_size: 16,
            cache_capacity_blocks: 20_000,
            prefill_ns_per_token: 2_000.0,     // ~500k tok/s batched prefill
            decode_ns_per_token: 5_000_000.0,  // ~200 tok/s decode
            prune_ttl_ns: 0,
        }
    }
}

struct ReplicaSim {
    cache: KvCache,
    /// When the server is next free (end of its current queue), run clock ns.
    free_at_ns: u64,
    /// Completion times of requests still in the system, for queue-depth counts.
    inflight: BinaryHeap<Reverse<u64>>,
}

/// The simulator. One [`Engine`] per `(router, trace)` run; `run` consumes the
/// router by reference so the same object can be reused across engines.
pub struct Engine {
    cfg: SimConfig,
    endpoints: Vec<Arc<str>>,
}

impl std::fmt::Debug for Engine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Engine").field("cfg", &self.cfg).finish()
    }
}

impl Engine {
    pub fn new(cfg: SimConfig) -> Self {
        let endpoints = (0..cfg.replicas)
            .map(|i| Arc::from(format!("http://replica-{i}")))
            .collect();
        Engine { cfg, endpoints }
    }

    /// Replay `trace` through `router`, returning the run report. `trace` is
    /// expected to be sorted by arrival time (as [`crate::TraceSpec::generate`]
    /// returns it).
    pub fn run(&self, router: &dyn Router, trace: &[SimRequest]) -> RunReport {
        let n = self.cfg.replicas;
        let mut reps: Vec<ReplicaSim> = (0..n)
            .map(|_| ReplicaSim {
                cache: KvCache::new(self.cfg.cache_capacity_blocks),
                free_at_ns: 0,
                inflight: BinaryHeap::new(),
            })
            .collect();

        let mut store = PrefixStore::new();
        let mut collector = Collector::new(router.name(), n);

        // Reused per-tick snapshot rows.
        let mut states: Vec<ReplicaState> = (0..n)
            .map(|i| ReplicaState {
                replica_id: ReplicaId(i as u32),
                endpoint: self.endpoints[i].clone(),
                waiting: 0,
                running: 0,
                kv_usage: 0.0,
                healthy: true,
                updated_ns: 0,
            })
            .collect();

        let bs = self.cfg.block_size;

        for (tick, sr) in trace.iter().enumerate() {
            let t = sr.req.arrival_ns;

            // Advance each replica to time `t`: drop completed requests and
            // refresh its snapshot row.
            for (i, rep) in reps.iter_mut().enumerate() {
                while let Some(&Reverse(c)) = rep.inflight.peek() {
                    if c <= t {
                        rep.inflight.pop();
                    } else {
                        break;
                    }
                }
                let inflight = rep.inflight.len() as u32;
                let running = if inflight > 0 { 1 } else { 0 };
                let st = &mut states[i];
                st.running = running;
                st.waiting = inflight - running;
                st.kv_usage = rep.cache.usage();
                st.updated_ns = t;
                st.healthy = true;
            }

            // Decide. The snapshot borrows `states` and `store` immutably; the
            // borrow ends with this block so we can mutate both afterwards.
            let (decision, dec_ns) = {
                let snap = Snapshot {
                    replicas: &states,
                    prefixes: &store,
                    version: tick as u64,
                    now_ns: t,
                };
                let t0 = Instant::now();
                let d = router.select(&sr.req, &snap);
                (d, t0.elapsed().as_nanos() as u64)
            };

            let idx = decision.replica_id.index().min(n - 1);

            // Optimistic belief update (represents the shared store write that
            // the runtime performs on dispatch), then the router hook.
            store.insert(decision.replica_id, &sr.req.block_hashes, t);
            router.on_dispatch(&sr.req, &decision);

            // Service on the chosen replica.
            let rep = &mut reps[idx];
            let start_service = t.max(rep.free_at_ns);
            let (hit_blocks, evicted) = rep.cache.admit(&sr.req.block_hashes);

            let cached_tokens = hit_blocks * bs;
            let uncached_tokens = sr.req.prompt_tokens.saturating_sub(cached_tokens);
            let prefill_ns = (uncached_tokens as f64 * self.cfg.prefill_ns_per_token).round() as u64;
            let decode_ns = (sr.output_tokens as f64 * self.cfg.decode_ns_per_token).round() as u64;

            let queue_wait = start_service - t;
            let ttft_ns = queue_wait + prefill_ns;
            // Server is busy for the prefill; decode overlaps (see module docs).
            rep.free_at_ns = start_service + prefill_ns;
            let completion_ns = start_service + prefill_ns + decode_ns;
            rep.inflight.push(Reverse(completion_ns));

            // Reconcile the belief with what the engine actually evicted.
            for h in evicted {
                store.forget(decision.replica_id, h);
            }

            let outcome = Outcome {
                ttft_ns,
                total_ns: completion_ns - t,
                output_tokens: sr.output_tokens,
                cached_blocks: hit_blocks,
                prefill_tokens: uncached_tokens,
            };
            router.on_complete(&sr.req, &decision, &outcome);

            collector.record(
                decision.replica_id,
                t,
                completion_ns,
                ttft_ns,
                dec_ns,
                cached_tokens,
                sr.req.prompt_tokens,
            );

            if self.cfg.prune_ttl_ns > 0 && tick % 2048 == 0 {
                store.prune(self.cfg.prune_ttl_ns, t);
            }
        }

        collector.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace::TraceSpec;
    use kvroute_core::{RadixRouter, RoundRobin};

    fn small_trace() -> Vec<SimRequest> {
        TraceSpec { seed: 7, sessions: 300, ..TraceSpec::default() }.generate()
    }

    #[test]
    fn run_is_deterministic() {
        let trace = small_trace();
        let engine = Engine::new(SimConfig::new(4));
        let a = engine.run(&RoundRobin::new(), &trace);
        let b = engine.run(&RoundRobin::new(), &trace);
        assert_eq!(a.ttft_p50_ns, b.ttft_p50_ns);
        assert_eq!(a.ttft_p99_ns, b.ttft_p99_ns);
        assert_eq!(a.requests, b.requests);
    }

    #[test]
    fn cache_aware_beats_round_robin_on_hit_rate() {
        let trace = small_trace();
        let engine = Engine::new(SimConfig::new(4));
        let rr = engine.run(&RoundRobin::new(), &trace);
        let radix = engine.run(&RadixRouter::default(), &trace);
        assert!(
            radix.cache_hit_rate > rr.cache_hit_rate,
            "radix hit {:.3} should beat round-robin hit {:.3}",
            radix.cache_hit_rate,
            rr.cache_hit_rate
        );
    }
}
