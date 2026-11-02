# kvroute

Cache-aware, session-aware routing across LLM inference replicas.

kvroute sits in front of a fleet of vLLM/SGLang replicas and decides **which
replica serves each request**, balancing KV-cache locality (route to the replica
that already holds the prompt's prefix, skip the prefill, cut TTFT) against load
(don't pile every popular prefix onto one replica).

This repository is the **phase-1 implementation**: the core routing abstractions,
the four baseline strategies, and a discrete-event simulator + benchmark harness
to evaluate them. It mirrors the design doc (sections 01–09). The novel
session-aware policy is designed-for but deliberately deferred — the seam it
plugs into (`NovelRouter`, `WeightPolicy`, and the session hooks) is already here.

## Status

| Piece | Crate | State |
|---|---|---|
| Core types + `Router` trait | `kvroute-core` | ✅ implemented |
| Chained block hashing | `kvroute-core::prefix::hashing` | ✅ |
| Arena radix prefix store | `kvroute-core::prefix::store` | ✅ |
| Round-robin / hash / radix / score routers | `kvroute-core::routers` | ✅ |
| Novel session-aware policy | `kvroute-core::routers::novel` | 🔸 stub (phase 3 seam) |
| Simulator + LRU KV-cache model | `kvroute-sim` | ✅ |
| Trace generator | `kvroute-sim::trace` | ✅ |
| Metrics + bench harness | `kvroute-sim`, `kvroute-cli` | ✅ |
| Real-cluster proxy + state tracker | `kvroute-state`, `kvroute-proxy` | ⬜ phase 2 |
| PyO3 bindings | `kvroute-py` | ⬜ phase 3 |

> **Note.** This phase is deliberately **dependency-free** so it builds offline.
> The design names xxh3 (hashing), `rand_chacha`/`rand_distr` (traces),
> `tokenizers`/`minijinja` (real tokenization), `tokio`/`axum` (the proxy) and
> `arc-swap` (snapshots). Each is a drop-in behind the interfaces already here —
> the hash mixer and the RNG, for instance, have the same signatures as their
> eventual replacements.

## Build & test

Requires a Rust toolchain (1.74+). Install from <https://rustup.rs> if needed.

```bash
cargo build --workspace
cargo test  --workspace     # unit + integration tests for core and sim
```

## Run the benchmark

```bash
# All policies on the default trace, 4 replicas:
cargo run -p kvroute-cli --release -- bench

# Pick policies, scale the fleet and the workload, fix the seed:
cargo run -p kvroute-cli --release -- bench \
    --policy round_robin,radix,score --replicas 8 --sessions 4000 --seed 3

# Tune the score router's load weight λ, and isolate tenant caches:
cargo run -p kvroute-cli --release -- bench --lambda 120 --isolate
```

Example shape of the output (numbers depend on the trace):

```
kvroute bench  —  5241 requests over 1500 sessions, 4 replicas, seed 1, skew 1.1, shared caches
(ttft in ms; hit = prefill tokens served from cache; imbal = max/mean replica load)

policy        ttft_p50  ttft_p99      hit    imbal  thru_rps    dec_ns
----------------------------------------------------------------------
round_robin       ...       ...     ...%     ...      ...        ...
hash              ...       ...     ...%     ...      ...        ...
radix             ...       ...     ...%     ...      ...        ...
score             ...       ...     ...%     ...      ...        ...
novel             ...       ...     ...%     ...      ...        ...

  radix        p99 TTFT -XX.X% vs round-robin, cache hit YY.Y%
  ...
```

## Layout

```
kv-route/
├─ crates/
│  ├─ kvroute-core/     # no async, no I/O — types, Router trait, prefix, routers
│  │  ├─ src/types.rs           Request, ReplicaState, RouteDecision, Snapshot
│  │  ├─ src/prefix/hashing.rs  chained 16-token-block hashing (+ salt)
│  │  ├─ src/prefix/store.rs    arena radix tree, the shared prefix store
│  │  └─ src/routers/           round_robin, hash, radix, score, novel(stub)
│  ├─ kvroute-sim/      # discrete-event engine, LRU cache model, traces, metrics
│  └─ kvroute-cli/      # the `kvroute` binary (bench today; serve in phase 2)
├─ configs/            # example run configs
└─ analysis/           # Python plotting of harness output (phase 1c)
```

## How the pieces fit

1. A request is fingerprinted into **chained block hashes** (`prefix::hashing`).
   Shared leading content ⇒ identical leading hashes, so a longest-common-prefix
   is an integer-sequence compare. A per-tenant salt isolates caches.
2. The **prefix store** (`prefix::store`) is a radix tree tagging each prefix with
   the replicas believed to hold it — shared infrastructure the cache-aware
   routers read, not a field of any one router.
3. A **router** (`routers`) sees `(Request, Snapshot)` and returns a
   `RouteDecision`. `select` is synchronous and pure, which is exactly why the
   same object runs in the simulator and (later) the live proxy.
4. The **simulator** (`kvroute-sim`) replays a generated trace, models each
   replica's prefill cost and LRU KV-cache, and reports TTFT percentiles, cache
   hit rate, load imbalance and throughput.

See the design doc for the full rationale, the phase plan, and the phase-1
decisions (exact tokenization, one shared prefix store, the sync/async split,
block size, and the deferred learned-policy runtime).
