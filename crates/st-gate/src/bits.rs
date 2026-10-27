//! Bit sets for change maps and window selections.
//!
//! Plain `Vec<u64>` words, the same layout `tracker::associate::Bitmask`
//! uses (bit `i` is `words[i / 64] >> (i % 64) & 1`). This is the only module
//! that knows how the bits are stored: the owner's sparsl bitpacked-spike
//! library is planned to replace it, and then [`BitSet`] is the one type to
//! swap. Callers use the methods, never `words()` directly, except to hash or
//! serialize.

/// A fixed-length set of bits. Bits at or past `len` in the last word are
/// always zero, so `==` and [`BitSet::words`] are canonical.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BitSet {
    words: Vec<u64>,
    len: usize,
}

impl BitSet {
    /// `len` bits, all clear.
    pub fn new(len: usize) -> Self {
        Self { words: vec![0; len.div_ceil(64)], len }
    }

    /// `len` bits, all set.
    pub fn full(len: usize) -> Self {
        let mut s = Self { words: vec![!0; len.div_ceil(64)], len };
        s.mask_tail();
        s
    }

    fn mask_tail(&mut self) {
        let used = self.len % 64;
        if used != 0 {
            if let Some(last) = self.words.last_mut() {
                *last &= (1u64 << used) - 1;
            }
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    /// Panics when `i >= len()`: an index out of range is a caller bug, and a
    /// silently ignored bit would be a skipped recompute.
    pub fn get(&self, i: usize) -> bool {
        assert!(i < self.len, "bit {i} out of range for a set of {}", self.len);
        self.words[i / 64] >> (i % 64) & 1 == 1
    }

    /// Panics when `i >= len()`.
    pub fn set(&mut self, i: usize, on: bool) {
        assert!(i < self.len, "bit {i} out of range for a set of {}", self.len);
        let (w, b) = (i / 64, i % 64);
        if on {
            self.words[w] |= 1 << b;
        } else {
            self.words[w] &= !(1 << b);
        }
    }

    /// Number of set bits.
    pub fn count(&self) -> usize {
        self.words.iter().map(|w| w.count_ones() as usize).sum()
    }

    pub fn none(&self) -> bool {
        self.words.iter().all(|&w| w == 0)
    }

    pub fn all(&self) -> bool {
        self.count() == self.len
    }

    pub fn clear(&mut self) {
        self.words.iter_mut().for_each(|w| *w = 0);
    }

    /// `self |= other`; the sets must have the same length.
    pub fn or_assign(&mut self, other: &Self) {
        assert_eq!(self.len, other.len, "or of sets of different lengths");
        self.words.iter_mut().zip(&other.words).for_each(|(a, b)| *a |= b);
    }

    /// The indices of the set bits, ascending.
    pub fn iter_ones(&self) -> impl Iterator<Item = usize> + '_ {
        self.words.iter().enumerate().flat_map(|(w, &word)| {
            let mut rest = word;
            std::iter::from_fn(move || {
                if rest == 0 {
                    return None;
                }
                let b = rest.trailing_zeros() as usize;
                rest &= rest - 1;
                Some(w * 64 + b)
            })
        })
    }

    /// The packed words, least significant bit first.
    pub fn words(&self) -> &[u64] {
        &self.words
    }
}

/// Which attention windows of the token grid are selected. Window `w` is
/// row `w / side`, column `w % side`, row-major: the order
/// `ojas_core::window_partition` produces, so the `j`-th selected window is
/// row `j` of a gathered batch.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct WindowSet {
    side: usize,
    bits: BitSet,
}

impl WindowSet {
    /// No windows selected, on a `side x side` grid of windows.
    pub fn new(side: usize) -> Self {
        Self { side, bits: BitSet::new(side * side) }
    }

    /// Every window selected.
    pub fn full(side: usize) -> Self {
        Self { side, bits: BitSet::full(side * side) }
    }

    /// The listed windows; panics on an index past `side * side`.
    pub fn from_indices(side: usize, windows: &[usize]) -> Self {
        let mut s = Self::new(side);
        windows.iter().for_each(|&w| s.insert(w));
        s
    }

    /// Windows per grid row.
    pub fn side(&self) -> usize {
        self.side
    }

    /// Windows in the grid, `side * side`.
    pub fn windows(&self) -> usize {
        self.side * self.side
    }

    pub fn contains(&self, w: usize) -> bool {
        self.bits.get(w)
    }

    pub fn insert(&mut self, w: usize) {
        self.bits.set(w, true);
    }

    pub fn remove(&mut self, w: usize) {
        self.bits.set(w, false);
    }

    /// Number of selected windows.
    pub fn count(&self) -> usize {
        self.bits.count()
    }

    pub fn is_empty(&self) -> bool {
        self.bits.none()
    }

    pub fn is_full(&self) -> bool {
        self.bits.all()
    }

    /// Selected windows, ascending.
    pub fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        self.bits.iter_ones()
    }

    pub fn bits(&self) -> &BitSet {
        &self.bits
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_get_count_across_word_boundaries() {
        let mut s = BitSet::new(130);
        for i in [0, 1, 63, 64, 65, 127, 128, 129] {
            assert!(!s.get(i));
            s.set(i, true);
            assert!(s.get(i));
        }
        assert_eq!(s.count(), 8);
        assert_eq!(s.iter_ones().collect::<Vec<_>>(), vec![0, 1, 63, 64, 65, 127, 128, 129]);
        s.set(64, false);
        assert_eq!(s.count(), 7);
        assert!(!s.get(64));
    }

    #[test]
    fn full_masks_the_tail_so_equality_is_canonical() {
        for len in [0, 1, 63, 64, 65, 81, 128, 5184] {
            let f = BitSet::full(len);
            assert_eq!(f.count(), len, "len {len}");
            assert!(f.all());
            let mut built = BitSet::new(len);
            (0..len).for_each(|i| built.set(i, true));
            assert_eq!(f, built, "len {len}");
        }
    }

    #[test]
    fn or_and_clear() {
        let mut a = BitSet::new(100);
        let mut b = BitSet::new(100);
        a.set(3, true);
        b.set(3, true);
        b.set(99, true);
        a.or_assign(&b);
        assert_eq!(a.iter_ones().collect::<Vec<_>>(), vec![3, 99]);
        a.clear();
        assert!(a.none());
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn an_index_past_the_end_panics_instead_of_vanishing() {
        BitSet::new(9).set(9, true);
    }

    #[test]
    fn window_set_is_row_major_and_ordered() {
        let s = WindowSet::from_indices(3, &[8, 0, 4]);
        assert_eq!(s.iter().collect::<Vec<_>>(), vec![0, 4, 8]);
        assert_eq!((s.count(), s.windows(), s.side()), (3, 9, 3));
        assert!(!s.is_empty() && !s.is_full());
        assert!(WindowSet::full(3).is_full());
        assert!(WindowSet::new(3).is_empty());
        let mut t = WindowSet::full(3);
        t.remove(4);
        assert_eq!(t.count(), 8);
        assert!(!t.contains(4));
    }
}
