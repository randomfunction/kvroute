//! Prefix fingerprinting and the shared prefix store.
//!
//! [`hashing`] turns block-aligned token ids into a chained hash sequence;
//! [`store`] is the arena radix tree the cache-aware routers query.

pub mod hashing;
pub mod store;
