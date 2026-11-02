//! Behavioural tests for the four baseline routers against hand-built snapshots.

use std::sync::Arc;

use kvroute_core::{
    HashRouter, PrefixStore, RadixRouter, Reason, ReplicaId, ReplicaState, Request, RequestId,
    RoundRobin, Router, ScoreRouter, SessionId, Snapshot, TenantId,
};

fn replica(id: u32, waiting: u32, running: u32) -> ReplicaState {
    ReplicaState {
        replica_id: ReplicaId(id),
        endpoint: Arc::from(format!("http://r{id}")),
        waiting,
        running,
        kv_usage: 0.0,
        healthy: true,
        updated_ns: 0,
    }
}

fn request(id: u64, blocks: &[u64]) -> Request {
    Request {
        id: RequestId(id),
        model: Arc::from("test"),
        tenant_id: Some(TenantId(1)),
        session_id: Some(SessionId(id)),
        cache_salt: None,
        block_hashes: Arc::from(blocks.to_vec()),
        prompt_tokens: (blocks.len() * 16) as u32,
        arrival_ns: 0,
    }
}

fn snap<'a>(replicas: &'a [ReplicaState], prefixes: &'a PrefixStore) -> Snapshot<'a> {
    Snapshot { replicas, prefixes, version: 0, now_ns: 0 }
}

#[test]
fn round_robin_spreads_evenly() {
    let replicas = vec![replica(0, 0, 0), replica(1, 0, 0), replica(2, 0, 0)];
    let store = PrefixStore::new();
    let rr = RoundRobin::new();
    let mut counts = [0usize; 3];
    for i in 0..300 {
        let req = request(i, &[i]);
        let d = rr.select(&req, &snap(&replicas, &store));
        counts[d.replica_id.index()] += 1;
    }
    for c in counts {
        assert_eq!(c, 100, "round-robin should be exactly even");
    }
}

#[test]
fn hash_router_is_sticky_per_session() {
    let replicas = vec![replica(0, 0, 0), replica(1, 0, 0), replica(2, 0, 0)];
    let store = PrefixStore::new();
    let hr = HashRouter::default();
    // Same session id ⇒ same replica across turns, even with different prompts.
    let r1 = hr.select(&request(42, &[100]), &snap(&replicas, &store));
    let r2 = hr.select(&request(42, &[100, 200]), &snap(&replicas, &store));
    let r3 = hr.select(&request(42, &[100, 200, 300]), &snap(&replicas, &store));
    assert_eq!(r1.replica_id, r2.replica_id);
    assert_eq!(r2.replica_id, r3.replica_id);
    assert_eq!(r1.reason, Reason::Sticky);
}

#[test]
fn radix_router_follows_the_longest_prefix() {
    let replicas = vec![replica(0, 0, 0), replica(1, 0, 0)];
    let mut store = PrefixStore::new();
    // Replica 1 holds a long prefix; replica 0 holds nothing relevant.
    store.insert(ReplicaId(1), &[10, 20, 30, 40], 0);
    let d = RadixRouter::default().select(&request(1, &[10, 20, 30, 40, 50]), &snap(&replicas, &store));
    assert_eq!(d.replica_id, ReplicaId(1));
    assert_eq!(d.reason, Reason::Prefix { blocks: 4 });
}

#[test]
fn radix_router_falls_back_to_least_load_without_a_match() {
    let replicas = vec![replica(0, 5, 1), replica(1, 0, 0)];
    let store = PrefixStore::new();
    let d = RadixRouter::default().select(&request(1, &[7, 8, 9]), &snap(&replicas, &store));
    assert_eq!(d.replica_id, ReplicaId(1)); // the idle one
    assert_eq!(d.reason, Reason::LeastLoad);
}

#[test]
fn score_router_trades_locality_against_load() {
    let mut store = PrefixStore::new();
    // Replica 0 holds a 3-block prefix (= 48 tokens saved) but is busy.
    store.insert(ReplicaId(0), &[1, 2, 3], 0);
    let req = request(1, &[1, 2, 3]);

    // Small λ: locality dominates ⇒ stick with the warm-but-busy replica 0.
    {
        let replicas = vec![replica(0, 2, 1), replica(1, 0, 0)];
        let d = ScoreRouter::fixed(16, 1.0).select(&req, &snap(&replicas, &store));
        assert_eq!(d.replica_id, ReplicaId(0));
    }
    // Large λ: load dominates ⇒ shed to the idle replica 1 despite the cache.
    {
        let replicas = vec![replica(0, 20, 1), replica(1, 0, 0)];
        let d = ScoreRouter::fixed(16, 50.0).select(&req, &snap(&replicas, &store));
        assert_eq!(d.replica_id, ReplicaId(1));
    }
}
