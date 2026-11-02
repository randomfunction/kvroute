//! A per-replica LRU KV-cache model.
//!
//! This is what makes a "cache hit" in the simulator *earned* rather than
//! assumed. Each replica has a bounded set of resident blocks. A request reuses
//! only the leading contiguous run of its blocks that are resident (prefix
//! semantics: the first miss stops reuse). After the request, all its blocks
//! become resident (prefill filled them), evicting the least-recently-used
//! blocks past capacity.

use std::collections::{BTreeMap, HashMap};

/// LRU over block hashes with a capacity in blocks.
#[derive(Debug)]
pub struct KvCache {
    capacity: usize,
    /// `hash -> clock` (its most recent use).
    at: HashMap<u64, u64>,
    /// `clock -> hash`, ordered, so eviction pops the smallest clock.
    order: BTreeMap<u64, u64>,
    clock: u64,
}

impl KvCache {
    pub fn new(capacity_blocks: usize) -> Self {
        KvCache {
            capacity: capacity_blocks.max(1),
            at: HashMap::new(),
            order: BTreeMap::new(),
            clock: 0,
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.at.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.at.is_empty()
    }

    /// Fraction of capacity in use, `0.0..=1.0`.
    #[inline]
    pub fn usage(&self) -> f32 {
        self.at.len() as f32 / self.capacity as f32
    }

    fn touch(&mut self, hash: u64) {
        self.clock += 1;
        if let Some(old) = self.at.insert(hash, self.clock) {
            self.order.remove(&old);
        }
        self.order.insert(self.clock, hash);
    }

    fn evict_to_capacity(&mut self, evicted: &mut Vec<u64>) {
        while self.at.len() > self.capacity {
            // Smallest clock = least recently used.
            if let Some((&clk, &hash)) = self.order.iter().next() {
                self.order.remove(&clk);
                self.at.remove(&hash);
                evicted.push(hash);
            } else {
                break;
            }
        }
    }

    /// Admit a request's blocks.
    ///
    /// Returns `(hit_blocks, evicted)`: `hit_blocks` is the number of leading
    /// blocks already resident (contiguous from the front), and `evicted` lists
    /// block hashes pushed out while admitting. After the call every block in
    /// `blocks` is resident and most-recently-used.
    pub fn admit(&mut self, blocks: &[u64]) -> (u32, Vec<u64>) {
        // Count the leading resident run *before* touching anything.
        let mut hit = 0u32;
        for &h in blocks {
            if self.at.contains_key(&h) {
                hit += 1;
            } else {
                break;
            }
        }
        // Now admit/touch all blocks (prefill makes the whole prompt resident).
        for &h in blocks {
            self.touch(h);
        }
        let mut evicted = Vec::new();
        self.evict_to_capacity(&mut evicted);
        (hit, evicted)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn leading_run_is_the_hit() {
        let mut c = KvCache::new(100);
        c.admit(&[1, 2, 3]);
        // Shares first two, diverges at the third.
        let (hit, _) = c.admit(&[1, 2, 9]);
        assert_eq!(hit, 2);
        // A hole in the middle still only counts the leading run.
        let mut c2 = KvCache::new(100);
        c2.admit(&[1, 2, 3, 4]);
        let (hit2, _) = c2.admit(&[1, 9, 3, 4]);
        assert_eq!(hit2, 1);
    }

    #[test]
    fn lru_evicts_and_reports() {
        let mut c = KvCache::new(2);
        c.admit(&[1]); // resident: {1}
        c.admit(&[2]); // resident: {1,2}
        let (_, ev) = c.admit(&[3]); // evicts LRU = 1
        assert_eq!(c.len(), 2);
        assert_eq!(ev, vec![1]);
    }

    #[test]
    fn usage_tracks_fill() {
        let mut c = KvCache::new(4);
        assert_eq!(c.usage(), 0.0);
        c.admit(&[1, 2]);
        assert_eq!(c.usage(), 0.5);
    }
}
