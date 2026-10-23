//! CPython 3.11 `random.Random` integer path, used to make seeded boards exact.

#[derive(Clone)]
pub struct PythonRandom {
    mt: [u32; 624],
    index: usize,
}

impl PythonRandom {
    pub fn from_u64(seed: u64) -> Self {
        let key = [seed as u32, (seed >> 32) as u32];
        let key = if key[1] == 0 { &key[..1] } else { &key[..] };
        let mut rng = Self {
            mt: [0; 624],
            index: 624,
        };
        rng.init_by_array(key);
        rng
    }

    fn init_genrand(&mut self, seed: u32) {
        self.mt[0] = seed;
        for i in 1..624 {
            self.mt[i] = 1812433253u32
                .wrapping_mul(self.mt[i - 1] ^ (self.mt[i - 1] >> 30))
                .wrapping_add(i as u32);
        }
        self.index = 624;
    }

    fn init_by_array(&mut self, key: &[u32]) {
        self.init_genrand(19650218);
        let mut i = 1usize;
        let mut j = 0usize;
        for _ in 0..624.max(key.len()) {
            self.mt[i] = (self.mt[i]
                ^ (self.mt[i - 1] ^ (self.mt[i - 1] >> 30)).wrapping_mul(1664525))
            .wrapping_add(key[j])
            .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= 624 {
                self.mt[0] = self.mt[623];
                i = 1;
            }
            if j >= key.len() {
                j = 0;
            }
        }
        for _ in 0..623 {
            self.mt[i] = (self.mt[i]
                ^ (self.mt[i - 1] ^ (self.mt[i - 1] >> 30)).wrapping_mul(1566083941))
            .wrapping_sub(i as u32);
            i += 1;
            if i >= 624 {
                self.mt[0] = self.mt[623];
                i = 1;
            }
        }
        self.mt[0] = 0x8000_0000;
        self.index = 624;
    }

    pub fn next_u32(&mut self) -> u32 {
        if self.index >= 624 {
            const UPPER: u32 = 0x8000_0000;
            const LOWER: u32 = 0x7fff_ffff;
            for i in 0..624 {
                let y = (self.mt[i] & UPPER) | (self.mt[(i + 1) % 624] & LOWER);
                self.mt[i] =
                    self.mt[(i + 397) % 624] ^ (y >> 1) ^ if y & 1 != 0 { 0x9908_b0df } else { 0 };
            }
            self.index = 0;
        }
        let mut y = self.mt[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    fn getrandbits(&mut self, bits: u32) -> u32 {
        debug_assert!((1..=32).contains(&bits));
        self.next_u32() >> (32 - bits)
    }

    pub fn randbelow(&mut self, n: usize) -> usize {
        assert!(n > 0);
        let bits = usize::BITS - n.leading_zeros();
        loop {
            let value = self.getrandbits(bits) as usize;
            if value < n {
                return value;
            }
        }
    }

    pub fn sample_all<const N: usize, T: Copy>(&mut self, values: [T; N]) -> [T; N] {
        let mut pool = values;
        let mut out = values;
        for (i, slot) in out.iter_mut().enumerate() {
            let j = self.randbelow(N - i);
            *slot = pool[j];
            pool[j] = pool[N - i - 1];
        }
        out
    }

    pub fn shuffle<T>(&mut self, values: &mut [T]) {
        for i in (1..values.len()).rev() {
            let j = self.randbelow(i + 1);
            values.swap(i, j);
        }
    }

    pub fn random(&mut self) -> f64 {
        let a = (self.next_u32() >> 5) as u64;
        let b = (self.next_u32() >> 6) as u64;
        ((a << 26) + b) as f64 * (1.0 / 9_007_199_254_740_992.0)
    }
}

#[cfg(test)]
mod tests {
    use super::PythonRandom;

    #[test]
    fn matches_python_seed_123() {
        let mut rng = PythonRandom::from_u64(123);
        assert_eq!(rng.next_u32(), 224899942);
        assert_eq!(rng.next_u32(), 1149664691);
        assert_eq!(rng.randbelow(6), 0);
    }
}
