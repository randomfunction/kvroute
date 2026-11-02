//! The shared prefix store: an arena-backed radix tree over chained block-hash
//! sequences, each node tagged with the replicas believed to hold that prefix.
//!
//! This is the one source of truth the cache-aware routers read. It is *shared
//! infrastructure*, not a field of any router — radix, score, and the future
//! novel policy all query the same tree, so a comparison between them reflects
//! the policy and not a difference in what they believe.
//!
//! # Semantics
//!
//! A replica that has cached the blocks of some request holds every prefix of
//! that request. So on insert we tag the replica at **every** node along its
//! path (each depth). A query then walks its own block-hash sequence down the
//! tree, and the deepest node on that walk where a replica is tagged is exactly
//! the number of leading blocks that replica shares — the prefix match length.
//!
//! The view is explicitly *a hint, never a guarantee*: replicas evict
//! independently, so a predicted hit can miss. That only costs the prefill it
//! hoped to save; metrics are always measured from the real engine/cache, never
//! from this belief.
//!
//! # Representation
//!
//! `Vec<Node>` arena with `u32` ids and a free list, plus a global
//! `hash -> node` index. Chained block hashes are unique per prefix, so a hash
//! identifies exactly one node, which makes KV-event removal (`forget`) an
//! `O(1)` lookup followed by a subtree prune.

use std::collections::HashMap;

use crate::ids::ReplicaId;

/// Arena index. `0` is always the root.
type NodeId = u32;
const ROOT: NodeId = 0;

#[derive(Debug)]
struct Node {
    /// Edge label from parent is a block hash; children keyed by the child's
    /// block hash.
    children: HashMap<u64, NodeId>,
    /// Replicas that hold the prefix ending at this node. Small, kept sorted for
    /// cheap dedup; replica counts are tiny.
    replicas: Vec<ReplicaId>,
    parent: NodeId,
    /// The block hash on the edge from `parent` (meaningless for the root).
    edge: u64,
    last_access_ns: u64,
}

impl Node {
    fn new(parent: NodeId, edge: u64, now_ns: u64) -> Self {
        Node {
            children: HashMap::new(),
            replicas: Vec::new(),
            parent,
            edge,
            last_access_ns: now_ns,
        }
    }

    fn reset_as_root(&mut self) {
        self.children.clear();
        self.replicas.clear();
        self.parent = ROOT;
        self.edge = 0;
        self.last_access_ns = 0;
    }

    fn add_replica(&mut self, r: ReplicaId) {
        if let Err(pos) = self.replicas.binary_search(&r) {
            self.replicas.insert(pos, r);
        }
    }

    fn remove_replica(&mut self, r: ReplicaId) {
        if let Ok(pos) = self.replicas.binary_search(&r) {
            self.replicas.remove(pos);
        }
    }
}

/// The radix prefix store. See the module docs.
#[derive(Debug)]
pub struct PrefixStore {
    nodes: Vec<Node>,
    free: Vec<NodeId>,
    /// `block hash -> node`. Valid because chained hashes are prefix-unique.
    index: HashMap<u64, NodeId>,
    live: usize,
}

impl Default for PrefixStore {
    fn default() -> Self {
        Self::new()
    }
}

impl PrefixStore {
    pub fn new() -> Self {
        PrefixStore {
            nodes: vec![Node::new(ROOT, 0, 0)],
            free: Vec::new(),
            index: HashMap::new(),
            live: 1,
        }
    }

    /// Number of live nodes (including the root).
    #[inline]
    pub fn len(&self) -> usize {
        self.live
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.live <= 1
    }

    fn alloc(&mut self, parent: NodeId, edge: u64, now_ns: u64) -> NodeId {
        self.live += 1;
        if let Some(id) = self.free.pop() {
            self.nodes[id as usize] = Node::new(parent, edge, now_ns);
            id
        } else {
            let id = self.nodes.len() as NodeId;
            self.nodes.push(Node::new(parent, edge, now_ns));
            id
        }
    }

    /// Record that `replica` holds the prefix described by `blocks` (the chained
    /// block hashes of a request). Tags the replica at every depth and refreshes
    /// access times. Idempotent for a replica already on the path.
    ///
    /// Used both for authoritative KV `BlockStored` events and for the optimistic
    /// "this replica now likely holds the prefix" update at dispatch time.
    pub fn insert(&mut self, replica: ReplicaId, blocks: &[u64], now_ns: u64) {
        let mut cur = ROOT;
        self.nodes[cur as usize].last_access_ns = now_ns;
        for &h in blocks {
            cur = match self.nodes[cur as usize].children.get(&h).copied() {
                Some(child) => child,
                None => {
                    let child = self.alloc(cur, h, now_ns);
                    self.nodes[cur as usize].children.insert(h, child);
                    self.index.insert(h, child);
                    child
                }
            };
            let node = &mut self.nodes[cur as usize];
            node.add_replica(replica);
            node.last_access_ns = now_ns;
        }
    }

    /// For the query `blocks`, append `(replica, matched_block_count)` for every
    /// replica that shares at least one leading block. `out` is cleared first.
    /// The count is the length of the shared prefix in blocks.
    pub fn matched_blocks(&self, blocks: &[u64], out: &mut Vec<(ReplicaId, u32)>) {
        out.clear();
        let mut cur = ROOT;
        let mut depth: u32 = 0;
        for &h in blocks {
            match self.nodes[cur as usize].children.get(&h).copied() {
                Some(child) => {
                    cur = child;
                    depth += 1;
                    for &r in &self.nodes[cur as usize].replicas {
                        match out.iter_mut().find(|(rr, _)| *rr == r) {
                            Some((_, d)) => *d = depth, // deeper on the path ⇒ longer prefix
                            None => out.push((r, depth)),
                        }
                    }
                }
                None => break,
            }
        }
    }

    /// Convenience: the single best `(replica, blocks)` match, ties broken by
    /// smaller replica id. `None` if nothing matches.
    pub fn best_match(&self, blocks: &[u64], scratch: &mut Vec<(ReplicaId, u32)>) -> Option<(ReplicaId, u32)> {
        self.matched_blocks(blocks, scratch);
        scratch
            .iter()
            .copied()
            .max_by(|a, b| a.1.cmp(&b.1).then(b.0.cmp(&a.0)))
    }

    /// Drop a replica from the entire store (e.g. it went unhealthy or left the
    /// fleet). `O(nodes)`; called rarely.
    pub fn forget_replica(&mut self, replica: ReplicaId) {
        for node in &mut self.nodes {
            node.remove_replica(replica);
        }
    }

    /// Apply a KV `BlockRemoved` event: `replica` evicted the block whose hash is
    /// `block`. Because a block is only reusable when its whole prefix is
    /// resident, evicting it also makes every deeper block unreachable as a
    /// prefix hit, so the replica is cleared from that node and its subtree.
    pub fn forget(&mut self, replica: ReplicaId, block: u64) {
        let Some(&start) = self.index.get(&block) else { return };
        // Clear `replica` from `start` and all descendants.
        let mut stack = vec![start];
        while let Some(id) = stack.pop() {
            self.nodes[id as usize].remove_replica(replica);
            for &child in self.nodes[id as usize].children.values() {
                stack.push(child);
            }
        }
    }

    /// Prune stale **leaf** nodes (no children) whose last access is older than
    /// `ttl_ns`. Leaf-only keeps the tree well-formed; repeated passes trim deep
    /// stale paths from the bottom up. Returns the number of nodes freed.
    ///
    /// Off by default in the simulator: the gap between this belief and the real
    /// LRU cache *is* the stale-view phenomenon worth studying. Call it to bound
    /// memory on long runs or to model a fresher belief.
    pub fn prune(&mut self, ttl_ns: u64, now_ns: u64) -> usize {
        let cutoff = now_ns.saturating_sub(ttl_ns);
        let mut freed = 0;
        loop {
            let mut victims: Vec<NodeId> = Vec::new();
            for (i, node) in self.nodes.iter().enumerate() {
                let id = i as NodeId;
                if id == ROOT {
                    continue;
                }
                // A freed slot has an empty child map and parent ROOT; skip ones
                // already on the free list.
                if self.free.contains(&id) {
                    continue;
                }
                if node.children.is_empty() && node.last_access_ns < cutoff {
                    victims.push(id);
                }
            }
            if victims.is_empty() {
                break;
            }
            for id in victims {
                let (parent, edge) = {
                    let n = &self.nodes[id as usize];
                    (n.parent, n.edge)
                };
                self.nodes[parent as usize].children.remove(&edge);
                self.index.remove(&edge);
                self.nodes[id as usize].reset_as_root(); // scrub, then recycle
                self.free.push(id);
                self.live -= 1;
                freed += 1;
            }
        }
        freed
    }

    /// Reset to an empty tree, keeping allocated capacity.
    pub fn clear(&mut self) {
        self.nodes.truncate(1);
        self.nodes[ROOT as usize].reset_as_root();
        self.free.clear();
        self.index.clear();
        self.live = 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(n: u32) -> ReplicaId {
        ReplicaId(n)
    }

    #[test]
    fn matches_shared_prefix_length_per_replica() {
        let mut s = PrefixStore::new();
        // replica 0 holds [10,20], replica 1 holds [10,20,30,40]
        s.insert(r(0), &[10, 20], 1);
        s.insert(r(1), &[10, 20, 30, 40], 1);

        let mut out = Vec::new();
        s.matched_blocks(&[10, 20, 30], &mut out);
        out.sort();
        assert_eq!(out, vec![(r(0), 2), (r(1), 3)]);

        // A query sharing only the first block.
        s.matched_blocks(&[10, 99], &mut out);
        out.sort();
        assert_eq!(out, vec![(r(0), 1), (r(1), 1)]);

        // No shared block at all.
        s.matched_blocks(&[7, 8, 9], &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn best_match_prefers_longer_then_lower_id() {
        let mut s = PrefixStore::new();
        s.insert(r(2), &[1, 2, 3], 0);
        s.insert(r(5), &[1, 2], 0);
        let mut scratch = Vec::new();
        assert_eq!(s.best_match(&[1, 2, 3, 4], &mut scratch), Some((r(2), 3)));
        // Equal length ⇒ lower id wins.
        s.insert(r(1), &[1, 2, 3], 0);
        assert_eq!(s.best_match(&[1, 2, 3, 4], &mut scratch), Some((r(1), 3)));
    }

    #[test]
    fn forget_clears_node_and_subtree() {
        let mut s = PrefixStore::new();
        s.insert(r(0), &[10, 20, 30], 0);
        let mut out = Vec::new();
        s.matched_blocks(&[10, 20, 30], &mut out);
        assert_eq!(out, vec![(r(0), 3)]);
        // Evicting block 20 removes it and its descendant 30 for replica 0.
        s.forget(r(0), 20);
        s.matched_blocks(&[10, 20, 30], &mut out);
        assert_eq!(out, vec![(r(0), 1)]); // only block 10 remains
    }

    #[test]
    fn prune_frees_stale_leaves_only() {
        let mut s = PrefixStore::new();
        s.insert(r(0), &[1, 2, 3], 1_000);
        let before = s.len();
        assert!(before > 1);
        // ttl so that everything older than now-ttl = 10_000-5_000 = 5_000 goes.
        let freed = s.prune(5_000, 10_000);
        assert!(freed >= 1);
        assert!(s.len() < before);
        // Re-inserting still works after recycling slots.
        s.insert(r(1), &[1, 2], 20_000);
        let mut out = Vec::new();
        s.matched_blocks(&[1, 2], &mut out);
        assert_eq!(out, vec![(r(1), 2)]);
    }

    #[test]
    fn insert_is_idempotent_for_same_replica() {
        let mut s = PrefixStore::new();
        s.insert(r(0), &[1, 2], 0);
        let n1 = s.len();
        s.insert(r(0), &[1, 2], 1);
        assert_eq!(s.len(), n1);
        let mut out = Vec::new();
        s.matched_blocks(&[1, 2], &mut out);
        assert_eq!(out, vec![(r(0), 2)]);
    }
}
