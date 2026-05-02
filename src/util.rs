//! Crate-internal shared helpers. Kept small on purpose; anything that grows
//! beyond a few related primitives belongs in its own module.

/// Mirror an out-of-range index back into `[0, n)` for Neumann boundary handling.
///
/// Reflection rule: `i < 0` reflects to `-i`; `i >= n` reflects to `2n − i − 2`.
/// This convention reproduces a Neumann (zero-flux) boundary on the first
/// derivative and is consistent with the boundary treatment used throughout
/// the paper.
#[inline]
pub fn mirror(i: i32, n: i32) -> usize {
    debug_assert!(n > 0);
    let mut i = if i < 0 { -i } else { i };
    if i >= n {
        i = 2 * n - i - 2;
    }
    i.clamp(0, n - 1) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mirror_at_boundaries() {
        // n = 5, valid indices [0, 4].
        assert_eq!(mirror(-1, 5), 1);
        assert_eq!(mirror(-2, 5), 2);
        assert_eq!(mirror(0, 5), 0);
        assert_eq!(mirror(4, 5), 4);
        assert_eq!(mirror(5, 5), 3);
        assert_eq!(mirror(6, 5), 2);
    }

    #[test]
    fn mirror_idempotent_on_interior() {
        for i in 0..10 {
            assert_eq!(mirror(i, 10), i as usize);
        }
    }

    #[test]
    fn mirror_large_offsets_clamp() {
        // Far-out-of-range indices clamp into [0, n) without overflowing.
        assert!(mirror(-1000, 8) < 8);
        assert!(mirror(1000, 8) < 8);
    }
}
