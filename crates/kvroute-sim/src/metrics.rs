//! Metric collection and the per-run report.

use kvroute_core::ReplicaId;

/// Accumulates per-request samples during a run; finalised into a [`RunReport`].
#[derive(Debug)]
pub struct Collector {
    policy: String,
    replicas: usize,
    ttft_ns: Vec<u64>,
    decision_ns: Vec<u64>,
    cached_tokens: u64,
    prompt_tokens: u64,
    per_replica: Vec<u64>,
    first_arrival_ns: u64,
    last_completion_ns: u64,
    count: u64,
}

impl Collector {
    pub fn new(policy: impl Into<String>, replicas: usize) -> Self {
        Collector {
            policy: policy.into(),
            replicas,
            ttft_ns: Vec::new(),
            decision_ns: Vec::new(),
            cached_tokens: 0,
            prompt_tokens: 0,
            per_replica: vec![0; replicas],
            first_arrival_ns: u64::MAX,
            last_completion_ns: 0,
            count: 0,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn record(
        &mut self,
        replica: ReplicaId,
        arrival_ns: u64,
        completion_ns: u64,
        ttft_ns: u64,
        decision_ns: u64,
        cached_tokens: u32,
        prompt_tokens: u32,
    ) {
        self.ttft_ns.push(ttft_ns);
        self.decision_ns.push(decision_ns);
        self.cached_tokens += cached_tokens as u64;
        self.prompt_tokens += prompt_tokens as u64;
        let idx = replica.index();
        if idx < self.per_replica.len() {
            self.per_replica[idx] += 1;
        }
        self.first_arrival_ns = self.first_arrival_ns.min(arrival_ns);
        self.last_completion_ns = self.last_completion_ns.max(completion_ns);
        self.count += 1;
    }

    pub fn finish(mut self) -> RunReport {
        self.ttft_ns.sort_unstable();
        self.decision_ns.sort_unstable();

        let makespan_ns = self.last_completion_ns.saturating_sub(self.first_arrival_ns);
        let throughput_rps = if makespan_ns > 0 {
            self.count as f64 / (makespan_ns as f64 / 1e9)
        } else {
            0.0
        };

        let mean = self.per_replica.iter().sum::<u64>() as f64 / self.replicas.max(1) as f64;
        let max = *self.per_replica.iter().max().unwrap_or(&0) as f64;
        let load_imbalance = if mean > 0.0 { max / mean } else { 0.0 };

        RunReport {
            policy: self.policy,
            requests: self.count,
            ttft_p50_ns: percentile(&self.ttft_ns, 0.50),
            ttft_p99_ns: percentile(&self.ttft_ns, 0.99),
            ttft_mean_ns: mean_u64(&self.ttft_ns),
            cache_hit_rate: if self.prompt_tokens > 0 {
                self.cached_tokens as f64 / self.prompt_tokens as f64
            } else {
                0.0
            },
            load_imbalance,
            throughput_rps,
            decision_ns_mean: mean_u64(&self.decision_ns),
            decision_ns_p99: percentile(&self.decision_ns, 0.99),
            per_replica: self.per_replica,
        }
    }
}

fn percentile(sorted: &[u64], q: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let rank = (q * (sorted.len() as f64 - 1.0)).round() as usize;
    sorted[rank.min(sorted.len() - 1)]
}

fn mean_u64(xs: &[u64]) -> u64 {
    if xs.is_empty() {
        0
    } else {
        (xs.iter().sum::<u64>() as f64 / xs.len() as f64).round() as u64
    }
}

/// The result of one `(router, trace)` run.
#[derive(Clone, Debug)]
pub struct RunReport {
    pub policy: String,
    pub requests: u64,
    pub ttft_p50_ns: u64,
    pub ttft_p99_ns: u64,
    pub ttft_mean_ns: u64,
    pub cache_hit_rate: f64,
    pub load_imbalance: f64,
    pub throughput_rps: f64,
    pub decision_ns_mean: u64,
    pub decision_ns_p99: u64,
    pub per_replica: Vec<u64>,
}

impl RunReport {
    /// One aligned table row. Pair with [`RunReport::header`].
    pub fn row(&self) -> String {
        format!(
            "{:<12} {:>9.2} {:>9.2} {:>7.1}% {:>8.2} {:>9.0} {:>9}",
            self.policy,
            ms(self.ttft_p50_ns),
            ms(self.ttft_p99_ns),
            self.cache_hit_rate * 100.0,
            self.load_imbalance,
            self.throughput_rps,
            self.decision_ns_mean,
        )
    }

    pub fn header() -> String {
        format!(
            "{:<12} {:>9} {:>9} {:>8} {:>8} {:>9} {:>9}",
            "policy", "ttft_p50", "ttft_p99", "hit", "imbal", "thru_rps", "dec_ns",
        )
    }
}

#[inline]
fn ms(ns: u64) -> f64 {
    ns as f64 / 1e6
}
