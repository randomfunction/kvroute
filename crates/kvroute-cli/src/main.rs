//! `kvroute` — the command-line tool.
//!
//! Phase 1 ships one subcommand, `bench`, which replays one reproducible trace
//! through each routing policy and prints a comparison table. `serve` (the axum
//! proxy) arrives with phase 2.
//!
//! Examples:
//! ```text
//! kvroute bench                        # all policies, defaults
//! kvroute bench --policy radix,score   # just these two
//! kvroute bench --replicas 8 --sessions 4000 --skew 1.4 --seed 3
//! kvroute bench --lambda 120 --isolate # tune the score router; isolate tenants
//! ```

use std::process::ExitCode;

use kvroute_core::{HashRouter, NovelRouter, RadixRouter, RoundRobin, Router, ScoreRouter};
use kvroute_sim::{Engine, RunReport, SimConfig, TraceSpec};

const ALL: &[&str] = &["round_robin", "hash", "radix", "score", "novel"];

struct Args {
    policies: Vec<String>,
    replicas: usize,
    sessions: usize,
    seed: u64,
    lambda: f32,
    skew: f64,
    isolate: bool,
}

impl Default for Args {
    fn default() -> Self {
        Args {
            policies: ALL.iter().map(|s| s.to_string()).collect(),
            replicas: 4,
            sessions: TraceSpec::default().sessions,
            seed: TraceSpec::default().seed,
            lambda: 200.0,
            skew: TraceSpec::default().prompt_skew,
            isolate: false,
        }
    }
}

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    match argv.first().map(String::as_str) {
        Some("bench") => match parse_bench(&argv[1..]) {
            Ok(args) => {
                run_bench(args);
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("error: {e}\n");
                usage();
                ExitCode::FAILURE
            }
        },
        Some("serve") => {
            eprintln!("`serve` (the OpenAI-compatible proxy) lands in phase 2; not built yet.");
            ExitCode::FAILURE
        }
        Some("-h") | Some("--help") | Some("help") | None => {
            usage();
            ExitCode::SUCCESS
        }
        Some(other) => {
            eprintln!("unknown command: {other}\n");
            usage();
            ExitCode::FAILURE
        }
    }
}

/// Consume the value that follows the flag at `*i`, advancing `*i` past it.
fn take<'a>(rest: &'a [String], i: &mut usize, flag: &str) -> Result<&'a str, String> {
    *i += 1;
    rest.get(*i)
        .map(String::as_str)
        .ok_or_else(|| format!("{flag} needs a value"))
}

fn parse_bench(rest: &[String]) -> Result<Args, String> {
    let mut args = Args::default();
    let mut i = 0;
    while i < rest.len() {
        let flag = rest[i].as_str();
        match flag {
            "--policy" | "-p" => {
                let v = take(rest, &mut i, flag)?;
                let picks: Vec<String> = v.split(',').map(|s| s.trim().to_string()).collect();
                for p in &picks {
                    if p != "all" && !ALL.contains(&p.as_str()) {
                        return Err(format!("unknown policy `{p}` (have: {}, all)", ALL.join(", ")));
                    }
                }
                args.policies = if picks.iter().any(|p| p == "all") {
                    ALL.iter().map(|s| s.to_string()).collect()
                } else {
                    picks
                };
            }
            "--replicas" | "-r" => args.replicas = parse(flag, take(rest, &mut i, flag)?)?,
            "--sessions" => args.sessions = parse(flag, take(rest, &mut i, flag)?)?,
            "--seed" => args.seed = parse(flag, take(rest, &mut i, flag)?)?,
            "--lambda" => args.lambda = parse(flag, take(rest, &mut i, flag)?)?,
            "--skew" => args.skew = parse(flag, take(rest, &mut i, flag)?)?,
            "--isolate" => args.isolate = true,
            "-h" | "--help" => {
                usage();
                std::process::exit(0);
            }
            other => return Err(format!("unknown flag: {other}")),
        }
        i += 1;
    }
    if args.replicas == 0 {
        return Err("--replicas must be >= 1".into());
    }
    Ok(args)
}

fn parse<T: std::str::FromStr>(flag: &str, v: &str) -> Result<T, String> {
    v.parse::<T>().map_err(|_| format!("{flag}: could not parse `{v}`"))
}

fn build(name: &str, block_size: u32, lambda: f32) -> Box<dyn Router> {
    match name {
        "round_robin" => Box::new(RoundRobin::new()),
        "hash" => Box::new(HashRouter::default()),
        "radix" => Box::new(RadixRouter::default()),
        "score" => Box::new(ScoreRouter::fixed(block_size, lambda)),
        "novel" => Box::new(NovelRouter::new()),
        _ => unreachable!("validated in parse_bench"),
    }
}

fn run_bench(args: Args) {
    let spec = TraceSpec {
        seed: args.seed,
        sessions: args.sessions,
        prompt_skew: args.skew,
        isolate_tenants: args.isolate,
        ..TraceSpec::default()
    };
    let trace = spec.generate();
    let engine = Engine::new(SimConfig::new(args.replicas));

    println!(
        "kvroute bench  —  {} requests over {} sessions, {} replicas, seed {}, skew {}, {}",
        trace.len(),
        args.sessions,
        args.replicas,
        args.seed,
        args.skew,
        if args.isolate { "tenant-isolated caches" } else { "shared caches" },
    );
    println!("(ttft in ms; hit = prefill tokens served from cache; imbal = max/mean replica load)\n");
    println!("{}", RunReport::header());
    println!("{}", "-".repeat(70));

    let mut reports: Vec<RunReport> = Vec::new();
    for name in &args.policies {
        let router = build(name, spec.block_size, args.lambda);
        let report = engine.run(router.as_ref(), &trace);
        println!("{}", report.row());
        reports.push(report);
    }

    // A short takeaway versus the round-robin control, when it was run.
    if let Some(base) = reports.iter().find(|r| r.policy == "round_robin") {
        println!();
        for r in &reports {
            if r.policy == "round_robin" {
                continue;
            }
            let p99_delta = pct_change(base.ttft_p99_ns as f64, r.ttft_p99_ns as f64);
            println!(
                "  {:<12} p99 TTFT {:+.1}% vs round-robin, cache hit {:.1}%",
                r.policy,
                p99_delta,
                r.cache_hit_rate * 100.0,
            );
        }
    }
}

/// Percent change from `base` to `now` (negative = improvement for latency).
fn pct_change(base: f64, now: f64) -> f64 {
    if base <= 0.0 {
        0.0
    } else {
        (now - base) / base * 100.0
    }
}

fn usage() {
    eprintln!(
        "kvroute — cache-aware routing for LLM inference (phase-1 simulator)\n\n\
         USAGE:\n  \
         kvroute bench [OPTIONS]\n  \
         kvroute serve                 (phase 2 — not built)\n\n\
         BENCH OPTIONS:\n  \
         -p, --policy <list>   comma-separated: round_robin,hash,radix,score,novel,all  [default: all]\n  \
         -r, --replicas <n>    replica count                                            [default: 4]\n      \
         --sessions <n>    conversations to generate                              [default: 1500]\n      \
         --seed <n>        trace seed (reproducible)                              [default: 1]\n      \
         --lambda <f>      score router's load weight λ                           [default: 200]\n      \
         --skew <f>        Zipf skew of shared system prompts                     [default: 1.1]\n      \
         --isolate         give each tenant an isolated cache (salt)\n"
    );
}
