//! Typed identifiers. Thin newtypes so a `ReplicaId` can never be passed where a
//! `SessionId` is meant, with no runtime cost.

/// Identifies one backend replica. In the simulator the inner `u32` is also the
/// index of the replica in [`crate::Snapshot::replicas`]; the engine relies on
/// that, so keep them aligned when constructing snapshots.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct ReplicaId(pub u32);

/// Monotonic per-run request id.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct RequestId(pub u64);

/// The owner of a request: used for isolation, fairness and quotas. It is **not**
/// a locality key — cross-session sharing inside a tenant is already found by the
/// block hashes.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct TenantId(pub u64);

/// A conversation / agent run. The locality key: turn N+1 contains turns `1..N`,
/// so stickiness on this id yields the biggest prefix hit. Client-chosen, so it
/// is always scoped by [`TenantId`] downstream.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct SessionId(pub u64);

impl ReplicaId {
    #[inline]
    pub fn index(self) -> usize {
        self.0 as usize
    }
}
