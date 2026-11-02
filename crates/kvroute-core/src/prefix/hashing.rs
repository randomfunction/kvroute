//! Chained block hashing.
//!
//! Prompts are hashed one 16-token block at a time, and each block hash folds in
//! the previous one. That chaining is what makes a longest-common-prefix a cheap
//! integer-sequence compare instead of a string diff: two prompts that share a
//! leading span of tokens produce *identical* leading block hashes, and diverge
//! exactly where the tokens diverge.
//!
//! The salt (vLLM's `cache_salt`) seeds the chain, so a per-tenant salt makes two
//! tenants' identical prompts hash differently — the router then never predicts a
//! shared cache hit the engine would refuse for isolation reasons.
//!
//! The mixer here is a self-contained splitmix64 finalizer, not xxh3. It is a
//! stand-in with the same signature; swapping in `xxhash-rust` later changes the
//! numbers but not the logic (hashes are router-private keys either way).

/// splitmix64 finalizer. Good avalanche, no state, `const`-friendly.
#[inline]
pub fn mix64(a: u64, b: u64) -> u64 {
    let mut z = a ^ b.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// FNV-1a over bytes — used to turn opaque strings (tenant names, session ids
/// coming off the wire) into the `u64` the typed ids wrap.
#[inline]
pub fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xCBF2_9CE4_8422_2325;
    for &byte in bytes {
        h ^= byte as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01B3);
    }
    h
}

/// Hash the block-aligned prefix of a token sequence.
///
/// Only **full** blocks are hashed — a trailing partial block is not cacheable as
/// a prefix and is skipped, matching engine behaviour. Returns one `u64` per full
/// block, chained, seeded by `salt`.
///
/// This is the entry point the live proxy uses after rendering the chat template
/// and tokenizing. The simulator fabricates block hashes directly (it has no real
/// tokens) but through the same [`mix64`] chain, so both paths share prefixes the
/// same way.
pub fn block_hashes(token_ids: &[u32], block_size: usize, salt: u64) -> Vec<u64> {
    debug_assert!(block_size > 0, "block_size must be > 0");
    let full_blocks = token_ids.len() / block_size;
    let mut out = Vec::with_capacity(full_blocks);
    let mut prev = mix64(salt, 0xB10C_C0DE); // seed the chain with the salt
    for b in 0..full_blocks {
        let start = b * block_size;
        // Fold the block's tokens into a single value, then into the chain.
        let mut acc = prev;
        for &tok in &token_ids[start..start + block_size] {
            acc = mix64(acc, tok as u64);
        }
        prev = acc;
        out.push(prev);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_prefix_shares_leading_hashes() {
        // Two token streams agreeing on the first 32 tokens (two blocks) must
        // agree on the first two block hashes and then diverge.
        let mut a: Vec<u32> = (0..48).collect();
        let mut b: Vec<u32> = (0..48).collect();
        for x in b.iter_mut().skip(32) {
            *x += 1000;
        }
        let ha = block_hashes(&a, 16, 0);
        let hb = block_hashes(&b, 16, 0);
        assert_eq!(ha.len(), 3);
        assert_eq!(hb.len(), 3);
        assert_eq!(ha[0], hb[0]);
        assert_eq!(ha[1], hb[1]);
        assert_ne!(ha[2], hb[2]);
        // Touching a later block must not touch an earlier hash.
        a[40] = 7;
        let ha2 = block_hashes(&a, 16, 0);
        assert_eq!(ha2[0], ha[0]);
        assert_eq!(ha2[1], ha[1]);
    }

    #[test]
    fn salt_isolates() {
        let toks: Vec<u32> = (0..32).collect();
        let unsalted = block_hashes(&toks, 16, 0);
        let salted = block_hashes(&toks, 16, 42);
        assert_ne!(unsalted[0], salted[0]);
        assert_ne!(unsalted[1], salted[1]);
    }

    #[test]
    fn partial_trailing_block_dropped() {
        let toks: Vec<u32> = (0..40).collect(); // 2 full blocks + 8 leftover
        assert_eq!(block_hashes(&toks, 16, 0).len(), 2);
    }
}
