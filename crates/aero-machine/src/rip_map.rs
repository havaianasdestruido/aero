//! Maps and sets keyed by a guest instruction pointer.
//!
//! Split out of `jit` so the compiled tier can use them in the browser, where that
//! module does not build.

use std::collections::{HashMap, HashSet};
/// Hasher for RIP-keyed maps.
///
/// Every map here is keyed by a guest address and looked up once per block the
/// chain runs. The default hasher is SipHash, built to resist collision attacks on
/// untrusted keys — irrelevant for a RIP, and it showed up as ~3% of the IR path's
/// profile. A multiply and a shift spread instruction addresses well enough: they
/// are dense and differ mostly in the low bits, which is what the high half of a
/// 64-bit multiply mixes down.
#[derive(Default, Clone, Copy)]
pub struct RipHasher(u64);

impl std::hash::Hasher for RipHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        // Only the u64 path is used; keep this correct rather than fast.
        for &b in bytes {
            self.write_u64(u64::from(b));
        }
    }

    #[inline]
    fn write_u64(&mut self, n: u64) {
        self.0 = (self.0 ^ n).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        self.0 ^= self.0 >> 29;
    }

    #[inline]
    fn write_usize(&mut self, n: usize) {
        self.write_u64(n as u64);
    }
}

pub type RipBuildHasher = std::hash::BuildHasherDefault<RipHasher>;
pub type RipMap<V> = HashMap<u64, V, RipBuildHasher>;
pub type RipSet = HashSet<u64, RipBuildHasher>;
