// Deterministic terrain generator. Pure function of (seed, cx, cz): worker
// threads need zero shared state. Trees cross chunk borders; solved by
// re-deriving neighbor trees from world coordinates (a 3-cell evaluation
// ring) and writing only the parts landing inside this chunk. No locks.

use super::*;
use noise::{NoiseFn, Perlin, SuperSimplex};

pub struct Generator {
    pub seed: u32,
    cont: FbmP,
    detail: FbmP,
    ridge: SuperSimplex,
    mount_mask: Perlin,
    cave: Perlin,
}

// Fbm<Perlin> alias kept concrete; no dyn, no boxes in the hot path.
type FbmP = noise::Fbm<Perlin>;

impl Generator {
    pub fn new(seed: u32) -> Self {
        let mut cont = FbmP::new(seed);
        cont.octaves = 5;
        cont.frequency = 1.0;
        cont.persistence = 0.5;
        let mut detail = FbmP::new(seed ^ 0x9E37_79B9);
        detail.octaves = 3;
        detail.frequency = 4.0;
        detail.persistence = 0.45;
        Self {
            seed,
            cont,
            detail,
            ridge: SuperSimplex::new(seed ^ 0x5105_5F0E),
            mount_mask: Perlin::new(seed ^ 0xDEAD_BEEF),
            cave: Perlin::new(seed ^ 0x1234_5678),
        }
    }

    /// Surface height (top solid block + 1) at world column (x, z). Deterministic.
    pub fn height(&self, x: i32, z: i32) -> i32 {
        let fx = x as f64 * 0.0075;
        let fz = z as f64 * 0.0075;
        let base = self.cont.get([fx, fz]); // ~[-1,1]
        let d = self.detail.get([fx, fz]) * 3.0;

        // Mountain mask: where positive, ridge noise shapes tall peaks.
        let m = self.mount_mask.get([x as f64 * 0.0032, z as f64 * 0.0032]);
        let r = self.ridge.get([x as f64 * 0.012, z as f64 * 0.012]).abs();
        let peak = (r * r) * (m.max(0.0) * 2.2).min(1.0) * 52.0;

        let h = 36.0 + base * 14.0 + d + peak;
        (h.round() as i32).clamp(4, (CH - 8) as i32)
    }

    /// Cheap integer hash for tree placement: splitmix64(finalize).
    #[inline]
    fn hash2(&self, x: i32, z: i32) -> u64 {
        let mut h = (x as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ (z as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F)
            ^ (self.seed as u64).wrapping_mul(0x1656_67B1_9E37_79F9);
        h ^= h >> 30;
        h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        h ^= h >> 27;
        h = h.wrapping_mul(0x94D0_49BB_1331_11EB);
        h ^ (h >> 31)
    }

    /// True if a tree occupies column (x, z); density and terrain gated.
    fn tree_here(&self, x: i32, z: i32) -> bool {
        let h = self.height(x, z);
        if h <= WATER_Y + 1 || h > 84 {
            return false;
        }
        // gentle slope only: max height delta over the 4-neighborhood
        let dh = [
            (self.height(x + 1, z) - h).abs(),
            (self.height(x - 1, z) - h).abs(),
            (self.height(x, z + 1) - h).abs(),
            (self.height(x, z - 1) - h).abs(),
        ]
        .into_iter()
        .max()
        .unwrap_or(0);
        if dh > 2 {
            return false;
        }
        self.hash2(x, z) % 96 == 0 // ~1.04% of eligible columns
    }

    /// Full chunk generation: columns, caves, water, then decoration ring.
    pub fn generate(&self, cx: i32, cz: i32) -> ChunkData {
        let mut d = ChunkData::new();
        let ox = cx * CW as i32;
        let oz = cz * CW as i32;

        for lx in 0..CW {
            for lz in 0..CW {
                let wx = ox + lx as i32;
                let wz = oz + lz as i32;
                let h = self.height(wx, wz) as usize;
                let beach = h <= (WATER_Y + 2) as usize;
                let snowy = h > 84;

                for y in 0..=h.min(CH - 1) {
                    let b = if y == 0 {
                        block::BEDROCK
                    } else if y + 4 < h {
                        block::STONE
                    } else if y < h {
                        if beach { block::SAND } else { block::DIRT }
                    } else {
                        // top block
                        if beach {
                            block::SAND
                        } else if snowy {
                            block::SNOW
                        } else {
                            block::GRASS
                        }
                    };
                    d.blocks[idx(lx, lz, y)] = b;
                }
                // caves: carve 3D noise below a 4-block crust so the surface
                // never breaches into shafts/sinkholes
                if h > 5 {
                    for y in 1..(h.saturating_sub(4)).max(1).min(CH - 1) {
                        let c = self.cave.get([
                            wx as f64 * 0.055,
                            y as f64 * 0.09,
                            wz as f64 * 0.055,
                        ]);
                        if c > 0.62 {
                            d.blocks[idx(lx, lz, y)] = block::AIR;
                        }
                    }
                }
                // water fills air up to sea level
                for y in h..=(WATER_Y.max(0) as usize).min(CH - 1) {
                    if d.blocks[idx(lx, lz, y)] == block::AIR {
                        d.blocks[idx(lx, lz, y)] = block::WATER;
                    }
                }
            }
        }

        // Trees: evaluate a 3-cell ring so canopies from neighbors land correctly.
        for tx in ox - 3..ox + CW as i32 + 3 {
            for tz in oz - 3..oz + CW as i32 + 3 {
                if !self.tree_here(tx, tz) {
                    continue;
                }
                self.stamp_tree(&mut d, ox, oz, tx, tz);
            }
        }

        d
    }

    fn stamp_tree(&self, d: &mut ChunkData, ox: i32, oz: i32, tx: i32, tz: i32) {
        let h = self.height(tx, tz);
        let trunk = 4 + ((self.hash2(tx, tz) >> 40) % 3) as i32; // 4..6
        // trunk
        for dy in 0..trunk {
            self.put(d, ox, oz, tx, h + 1 + dy, tz, block::LOG);
        }
        // canopy: 5x5 two layers, corner-trimmed, top cap
        let cy = h + trunk;
        for dx in -2..=2 {
            for dz in -2..=2 {
                let trim = (dx * dx + dz * dz) > 5; // round the corners
                if !trim {
                    self.put(d, ox, oz, tx + dx, cy, tz + dz, block::LEAVES);
                    self.put(d, ox, oz, tx + dx, cy + 1, tz + dz, block::LEAVES);
                }
            }
        }
        for dx in -1..=1 {
            for dz in -1..=1 {
                if (dx * dx + dz * dz) <= 1 {
                    self.put(d, ox, oz, tx + dx, cy + 2, tz + dz, block::LEAVES);
                }
            }
        }
    }

    /// World-space stamp into this chunk: bounds-checked, air/water only, so
    /// decorations never punch through terrain or each other's trunks.
    #[inline]
    fn put(&self, d: &mut ChunkData, ox: i32, oz: i32, x: i32, y: i32, z: i32, b: u8) {
        let lx = x - ox;
        let lz = z - oz;
        if lx < 0 || lx >= CW as i32 || lz < 0 || lz >= CW as i32 {
            return;
        }
        if y < 0 || y >= CH as i32 {
            return;
        }
        let i = idx(lx as usize, lz as usize, y as usize);
        if d.blocks[i] == block::AIR || d.blocks[i] == block::WATER {
            d.blocks[i] = b;
        }
    }
}

/// Spawn N generation workers. Jobs: chunk coords in; finished ChunkData out.
/// One Generator per thread; the job receiver is shared behind a Mutex, the
/// result sender is cloned. No world state is ever shared.
pub fn spawn_workers(
    seed: u32,
    n: usize,
) -> (
    std::sync::mpsc::Sender<(i32, i32)>,
    mpsc::Receiver<(i32, i32, ChunkData)>,
    Vec<std::thread::JoinHandle<()>>,
) {
    let (job_tx, job_rx) = mpsc::channel::<(i32, i32)>();
    let (res_tx, res_rx) = mpsc::channel::<(i32, i32, ChunkData)>();
    let job_rx = std::sync::Arc::new(std::sync::Mutex::new(job_rx));
    let mut handles = Vec::with_capacity(n);
    for w in 0..n {
        let job_rx = job_rx.clone();
        let res_tx = res_tx.clone();
        handles.push(
            std::thread::Builder::new()
                .name(format!("genw{w}"))
                .spawn(move || {
                    let g = Generator::new(seed);
                    while let Ok((cx, cz)) = job_rx.lock().unwrap().recv() {
                        let d = g.generate(cx, cz);
                        if res_tx.send((cx, cz, d)).is_err() {
                            break; // main thread gone
                        }
                    }
                })
                .unwrap(),
        );
    }
    drop(res_tx); // our copy unused; threads closing ends the channel
    (job_tx, res_rx, handles)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mk(seed: u32) -> Generator {
        Generator::new(seed)
    }

    #[test]
    fn height_is_deterministic_and_bounded() {
        let g = mk(7);
        for i in 0..200 {
            let h1 = g.height(i * 13, i * -7);
            let h2 = g.height(i * 13, i * -7);
            assert_eq!(h1, h2);
            assert!((4..CH as i32 - 7).contains(&h1));
        }
    }

    #[test]
    fn generate_pure_and_structural_invariants() {
        let g = mk(42);
        let a = g.generate(0, 0);
        let b = g.generate(0, 0);
        assert_eq!(&a.blocks[..], &b.blocks[..]); // pure
        // bedrock floor everywhere
        for lx in 0..CW {
            for lz in 0..CW {
                assert_eq!(a.get(lx, lz, 0), block::BEDROCK);
            }
        }
        // below surface (minus caves) mostly solid, ocean columns filled to sea level
        let mut solid_cols = 0;
        for lx in 0..CW {
            for lz in 0..CW {
                let col = &a.blocks[(lx * CW + lz) * CH..(lx * CW + lz) * CH + CH];
                let top = col.iter().rposition(|b| *b != 0).map_or(0, |y| y + 1);
                if top > WATER_Y as usize {
                    solid_cols += 1;
                }
            }
        }
        assert!(solid_cols > 0, "chunk must not be all ocean");
    }

    #[test]
    fn workers_produce_requested_chunks() {
        let (tx, rx, hs) = spawn_workers(3, 2);
        for i in 0..6i32 {
            tx.send((i, 0)).unwrap();
        }
        let mut got = 0;
        while let Ok(_) = rx.recv() {
            got += 1;
            if got == 6 {
                break;
            }
        }
        assert_eq!(got, 6);
        drop(tx);
        for h in hs {
            h.join().unwrap();
        }
    }
}

