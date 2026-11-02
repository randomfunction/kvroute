//! Workload generation.
//!
//! Produces a reproducible request stream with the structure that makes routing
//! interesting:
//!
//! * **Skewed prefixes** — a Zipfian choice of shared system prompt, so a few
//!   prompts dominate (shared docs / system messages).
//! * **Tenants** — a Zipfian choice of tenant (a few heavy, a long tail), so a
//!   cache-blind sticky router produces visible hotspots. Optional per-tenant
//!   cache salt models isolation.
//! * **Multi-turn agent sessions** — each turn's prompt contains all prior turns,
//!   so the prefix grows and locality compounds. Think-time and occasional
//!   tool-call pauses space the turns out.
//!
//! Block hashes are fabricated (the simulator has no real tokens) but chained
//! through the same [`kvroute_core::mix64`] as the real hasher, so shared leading
//! content yields identical leading hashes exactly as it would in production.

use std::sync::Arc;

use kvroute_core::{mix64, Request, RequestId, SessionId, TenantId};

use crate::rng::{Rng, Zipf};

const SYS_SEED: u64 = 0x5359_5300; // "SYS"

/// A generated request plus the output length the engine needs (output length is
/// a workload property, not part of the routing contract, so it rides alongside
/// [`Request`] rather than inside it).
#[derive(Clone, Debug)]
pub struct SimRequest {
    pub req: Request,
    pub output_tokens: u32,
}

/// Knobs for the generator. Defaults produce a skewed, multi-turn, multi-tenant
/// workload where cache-aware routing clearly helps.
#[derive(Clone, Debug)]
pub struct TraceSpec {
    pub seed: u64,
    pub model: &'static str,
    pub block_size: u32,

    /// Number of sessions (conversations / agent runs) to generate.
    pub sessions: usize,

    /// Distinct shared system prompts, and how sharply their popularity is
    /// skewed (higher = a few dominate).
    pub system_prompts: usize,
    pub prompt_skew: f64,
    /// System-prompt length in blocks, sampled uniformly in this inclusive range.
    pub sys_blocks: (u32, u32),

    /// Tenants and their popularity skew; isolate caches per tenant via salt.
    pub tenants: usize,
    pub tenant_skew: f64,
    pub isolate_tenants: bool,

    /// Turns per session, sampled uniformly in this inclusive range.
    pub turns: (u32, u32),
    /// New user blocks added to the prompt each turn.
    pub user_blocks: (u32, u32),
    /// Assistant-reply blocks appended to the context after each turn.
    pub assistant_blocks: (u32, u32),
    /// Output tokens per turn, mean (exponential spread around it).
    pub output_tokens_mean: u32,

    /// Mean gap between session starts, nanoseconds (Poisson arrivals).
    pub session_gap_ns: f64,
    /// Probability a new session clusters onto the previous start (bursts).
    pub burstiness: f64,
    /// Mean think-time between a session's turns, nanoseconds.
    pub think_ns: f64,
    /// Probability a between-turn gap is a long tool-call pause instead.
    pub tool_pause_prob: f64,
    /// Multiplier applied to think-time when a tool-call pause fires.
    pub tool_pause_mult: f64,
}

impl Default for TraceSpec {
    fn default() -> Self {
        TraceSpec {
            seed: 1,
            model: "sim-model",
            block_size: 16,
            sessions: 1_500,
            system_prompts: 40,
            prompt_skew: 1.1,
            sys_blocks: (8, 64), // 128–1024 token system prompts
            tenants: 16,
            tenant_skew: 1.3,
            isolate_tenants: false,
            turns: (1, 6),
            user_blocks: (1, 4),
            assistant_blocks: (2, 12),
            output_tokens_mean: 256,
            session_gap_ns: 8_000_000.0, // 8 ms mean between session starts
            burstiness: 0.15,
            think_ns: 2_000_000_000.0, // 2 s mean think-time
            tool_pause_prob: 0.25,
            tool_pause_mult: 6.0,
        }
    }
}

impl TraceSpec {
    fn range(&self, rng: &mut Rng, lo_hi: (u32, u32)) -> u32 {
        let (lo, hi) = lo_hi;
        if hi <= lo {
            lo
        } else {
            lo + rng.below((hi - lo + 1) as usize) as u32
        }
    }

    /// Build the sorted request stream for this spec.
    pub fn generate(&self) -> Vec<SimRequest> {
        let mut rng = Rng::new(self.seed);
        let prompt_zipf = Zipf::new(self.system_prompts.max(1), self.prompt_skew);
        let tenant_zipf = Zipf::new(self.tenants.max(1), self.tenant_skew);

        let mut out: Vec<SimRequest> = Vec::new();
        let mut req_id: u64 = 0;
        let mut session_clock: u64 = 0;
        let mut prev_session_start: u64 = 0;

        for s in 0..self.sessions {
            // Session start: Poisson gap, occasionally clustered into a burst.
            let start = if s > 0 && rng.next_f64() < self.burstiness {
                prev_session_start
            } else {
                session_clock += rng.interarrival_ns(self.session_gap_ns);
                session_clock
            };
            prev_session_start = start;

            let tenant = tenant_zipf.sample(&mut rng);
            let prompt = prompt_zipf.sample(&mut rng);
            let session_id = s as u64;
            let salt = if self.isolate_tenants {
                mix64(tenant as u64, 0x5445_4E41_4E54) // "TENANT"
            } else {
                0
            };

            // Build the system-prompt prefix (shared across sessions with the
            // same prompt + salt).
            let mut seq: Vec<u64> = Vec::new();
            let mut prev = mix64(salt, SYS_SEED);
            let sys_n = self.range(&mut rng, self.sys_blocks);
            for i in 0..sys_n {
                let cid = mix64(prompt as u64, i as u64);
                prev = mix64(prev, cid);
                seq.push(prev);
            }

            let turns = self.range(&mut rng, self.turns);
            let mut arrival = start;

            for t in 0..turns {
                // This turn's user message extends the prompt.
                let un = self.range(&mut rng, self.user_blocks);
                for i in 0..un {
                    let cid = mix64(mix64(session_id, 0x5553_0000 ^ t as u64), i as u64); // "US"
                    prev = mix64(prev, cid);
                    seq.push(prev);
                }

                let block_hashes: Arc<[u64]> = Arc::from(seq.clone());
                let prompt_tokens = (seq.len() as u32) * self.block_size;
                let output_tokens = self.sample_output(&mut rng);

                req_id += 1;
                out.push(SimRequest {
                    req: Request {
                        id: RequestId(req_id),
                        model: Arc::from(self.model),
                        tenant_id: Some(TenantId(tenant as u64)),
                        session_id: Some(SessionId(session_id)),
                        cache_salt: if self.isolate_tenants { Some(salt) } else { None },
                        block_hashes,
                        prompt_tokens,
                        arrival_ns: arrival,
                    },
                    output_tokens,
                });

                // Assistant reply becomes context for the next turn.
                let an = self.range(&mut rng, self.assistant_blocks);
                for i in 0..an {
                    let cid = mix64(mix64(session_id, 0x4153_0000 ^ t as u64), i as u64); // "AS"
                    prev = mix64(prev, cid);
                    seq.push(prev);
                }

                // Advance to the next turn: think-time, sometimes a tool pause.
                let mut gap = rng.exp(1.0 / self.think_ns.max(1.0));
                if rng.next_f64() < self.tool_pause_prob {
                    gap *= self.tool_pause_mult;
                }
                arrival += gap.round() as u64;
            }
        }

        out.sort_by_key(|r| r.req.arrival_ns);
        out
    }

    fn sample_output(&self, rng: &mut Rng) -> u32 {
        // Exponential around the mean, clamped to a sane floor.
        let mean = self.output_tokens_mean.max(1) as f64;
        (rng.exp(1.0 / mean).round() as u32).max(16)
    }
}
