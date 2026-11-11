// Voxel world core: block taxonomy, chunk storage, world container + streaming bookkeeping.
//
// Coordinate system: block coords (i32, i32, i32). Chunk (cx, cz) covers
// x in [cx*16, cx*16+16), z in [cz*16, cz*16+16), y in [0, 128).

pub mod terrain;

use std::collections::HashMap;
use std::sync::mpsc;

pub const CW: usize = 16; // chunk width (x and z)
pub const CH: usize = 128; // world height
pub const WATER_Y: i32 = 32;
pub const CHUNK_BLOCKS: usize = CW * CW * CH;

// ---- Block taxonomy (u8 ids, self-describing table) ----

pub mod block {
    pub const AIR: u8 = 0;
    pub const GRASS: u8 = 1;
    pub const DIRT: u8 = 2;
    pub const STONE: u8 = 3;
    pub const COBBLE: u8 = 4;
    pub const SAND: u8 = 5;
    pub const WATER: u8 = 6;
    pub const LOG: u8 = 7;
    pub const LEAVES: u8 = 8;
    pub const GLASS: u8 = 9;
    pub const SNOW: u8 = 10;
    pub const PLANKS: u8 = 11;
    pub const BEDROCK: u8 = 12;
    pub const GRAVEL: u8 = 13;
    pub const COUNT: usize = 14;

    /// Blocks that stop the player. Table-driven, branchless.
    const SOLID: [bool; COUNT] = [
        false, true, true, true, true, true, false, true, true, true, true, true, true, true,
    ];
    /// Blocks that fully occlude a neighbor face (mesh culling + skylight).
    const OPAQUE: [bool; COUNT] = [
        false, true, true, true, true, true, false, true, false, false, true, true, true, true,
    ];
    /// Tile index in the 4x4 atlas for each (block, face). Face order: -Y,+Y,-X,+X,-Z,+Z.
    /// 255 = render nothing (air).
    #[rustfmt::skip]
    const TILES: [[u8; 6]; COUNT] = [
        [255, 255, 255, 255, 255, 255], // AIR
        [  2,   0,   1,   1,   1,   1], // GRASS: dirt bottom, grass top, grass side
        [  2,   2,   2,   2,   2,   2], // DIRT
        [  3,   3,   3,   3,   3,   3], // STONE
        [  4,   4,   4,   4,   4,   4], // COBBLE
        [  5,   5,   5,   5,   5,   5], // SAND
        [  6,   6,   6,   6,   6,   6], // WATER
        [  8,   8,   7,   7,   7,   7], // LOG: top rings, side bark
        [  9,   9,   9,   9,   9,   9], // LEAVES
        [ 10,  10,  10,  10,  10,  10], // GLASS
        [ 11,  11,  12,  12,  12,  12], // SNOW
        [ 13,  13,  13,  13,  13,  13], // PLANKS
        [ 14,  14,  14,  14,  14,  14], // BEDROCK
        [ 15,  15,  15,  15,  15,  15], // GRAVEL
    ];

    #[inline(always)]
    pub fn solid(b: u8) -> bool {
        SOLID[b as usize]
    }
    #[inline(always)]
    pub fn opaque(b: u8) -> bool {
        OPAQUE[b as usize]
    }
    #[inline(always)]
    pub fn tile(b: u8, face: usize) -> u8 {
        TILES[b as usize][face]
    }
    #[inline(always)]
    pub fn is_water(b: u8) -> bool {
        b == WATER
    }
    pub fn name(b: u8) -> &'static str {
        [
            "Air", "Grass", "Dirt", "Stone", "Cobblestone", "Sand", "Water", "Log", "Leaves",
            "Glass", "Snow", "Planks", "Bedrock", "Gravel",
        ][b as usize]
    }
}

/// Hotbar contents (keys 1..9).
pub const HOTBAR: [u8; 9] = [
    block::GRASS,
    block::DIRT,
    block::STONE,
    block::COBBLE,
    block::SAND,
    block::LOG,
    block::PLANKS,
    block::GLASS,
    block::LEAVES,
];

#[inline(always)]
pub fn chunk_of(x: i32, z: i32) -> (i32, i32) {
    (x.div_euclid(CW as i32), z.div_euclid(CW as i32))
}
#[inline(always)]
pub fn local_of(x: i32, z: i32) -> (usize, usize) {
    (x.rem_euclid(CW as i32) as usize, z.rem_euclid(CW as i32) as usize)
}
#[inline(always)]
pub fn idx(lx: usize, lz: usize, ly: usize) -> usize {
    (lx * CW + lz) * CH + ly
}

pub struct ChunkData {
    pub blocks: Box<[u8; CHUNK_BLOCKS]>,
    /// Sky light 0..15 per voxel, same `idx` layout as blocks. Filled by
    /// `World::relight_region` (vertical sky seed + BFS decay-1 flood).
    pub light: Box<[u8; CHUNK_BLOCKS]>,
}

impl ChunkData {
    pub fn new() -> Self {
        Self {
            blocks: vec![0; CHUNK_BLOCKS].into_boxed_slice().try_into().unwrap(),
            light: vec![0; CHUNK_BLOCKS].into_boxed_slice().try_into().unwrap(),
        }
    }
    #[inline]
    pub fn get(&self, lx: usize, lz: usize, ly: usize) -> u8 {
        self.blocks[idx(lx, lz, ly)]
    }
}

pub struct Chunk {
    pub data: ChunkData,
    pub meshed: bool,   // mesh exists / is current
    pub mesh_dirty: bool,
}

pub struct World {
    pub chunks: HashMap<(i32, i32), Chunk>,
    pub radius: i32,
    /// Chunks awaiting region relight (loads + edits), deduped.
    pub relight_pending: std::collections::VecDeque<(i32, i32)>,
    pub relight_set: std::collections::HashSet<(i32, i32)>,
}

impl World {
    pub fn new(radius: i32) -> Self {
        Self {
            chunks: HashMap::new(),
            radius,
            relight_pending: std::collections::VecDeque::new(),
            relight_set: std::collections::HashSet::new(),
        }
    }

    pub fn queue_relight(&mut self, cx: i32, cz: i32) {
        if self.relight_set.insert((cx, cz)) {
            self.relight_pending.push_back((cx, cz));
        }
    }

    #[inline]
    pub fn get(&self, x: i32, y: i32, z: i32) -> u8 {
        if y < 0 || y >= CH as i32 {
            return 0;
        }
        let (cx, cz) = chunk_of(x, z);
        match self.chunks.get(&(cx, cz)) {
            Some(c) => {
                let (lx, lz) = local_of(x, z);
                c.data.get(lx, lz, y as usize)
            }
            None => {
                // Unloaded: treat as opaque stone above water, air below, to stop
                // mesh seams from drawing into the void.
                if y < WATER_Y { 0 } else { block::STONE }
            }
        }
    }

    /// Sky light 0..15 at a world cell. Unloaded columns read as open sky
    /// (matches the stone fallback in `get`: the surface is lit, and mesh
    /// seams redraw once the chunk arrives and relights).
    #[inline]
    pub fn light_of(&self, x: i32, y: i32, z: i32) -> u8 {
        if y < 0 || y >= CH as i32 {
            return 15;
        }
        let (cx, cz) = chunk_of(x, z);
        match self.chunks.get(&(cx, cz)) {
            Some(c) => {
                let (lx, lz) = local_of(x, z);
                c.data.light[idx(lx, lz, y as usize)]
            }
            None => 15,
        }
    }

    /// Rebuild sky light over the 3x3 chunk region around (ccx, ccz):
    /// vertical sky seed per column (15 down to the first opaque block),
    /// boundary seeds from loaded neighbors, then BFS with decay 1 into
    /// every non-opaque cell. Unloaded chunks act as opaque barriers so no
    /// phantom light leaks through seams; stale light beyond the region is
    /// bounded by the 15-block decay reach and refreshes as neighbor
    /// regions relight. O(9 chunks), run per edit/load.
    pub fn relight_region(&mut self, ccx: i32, ccz: i32) {
        const RW: usize = 3 * CW; // 48
        const RH: usize = CH;
        // region-local idx: (lx * RW + lz) * RH + ly
        let ridx = |lx: usize, lz: usize, ly: usize| (lx * RW + lz) * RH + ly;
        let ox = (ccx - 1) * CW as i32;
        let oz = (ccz - 1) * CW as i32;
        // owned region copies: block read + light write without map hops
        let mut blocks = vec![0u8; RW * RW * RH];
        let mut light = vec![0u8; RW * RW * RH];
        let mut loaded = [[false; 3]; 3];
        for ix in 0..3usize {
            for iz in 0..3usize {
                let key = (ccx - 1 + ix as i32, ccz - 1 + iz as i32);
                let Some(c) = self.chunks.get(&key) else { continue };
                loaded[ix][iz] = true;
                for lx in 0..CW {
                    for lz in 0..CW {
                        let rx = ix * CW + lx;
                        let rz = iz * CW + lz;
                        for ly in 0..CH {
                            blocks[ridx(rx, rz, ly)] = c.data.blocks[idx(lx, lz, ly)];
                        }
                    }
                }
            }
        }
        let is_loaded = |rx: usize, rz: usize| loaded[rx / CW][rz / CW];

        let mut queue: std::collections::VecDeque<u32> = std::collections::VecDeque::new();
        let push = |q: &mut std::collections::VecDeque<u32>, rx: usize, rz: usize, ly: usize| {
            q.push_back((rx as u32) | ((rz as u32) << 6) | ((ly as u32) << 12));
        };
        // vertical seed; unloaded columns are solid barriers (light 0)
        for rx in 0..RW {
            for rz in 0..RW {
                let mut lit = if is_loaded(rx, rz) { 15u8 } else { 0 };
                for ly in (0..RH).rev() {
                    let i = ridx(rx, rz, ly);
                    if block::opaque(blocks[i]) {
                        lit = 0;
                    }
                    light[i] = lit;
                    if lit > 1 {
                        push(&mut queue, rx, rz, ly);
                    }
                }
            }
        }
        // boundary seeds: loaded world chunks hugging the region walls shine in
        {
            let mut seed_b = |rx: usize, rz: usize, wx: i32, wz: i32| {
                if !is_loaded(rx, rz) {
                    return;
                }
                let (ocx, ocz) = chunk_of(wx, wz);
                if !self.chunks.contains_key(&(ocx, ocz)) {
                    return;
                }
                for ly in 0..RH {
                    let src = self.light_of(wx, ly as i32, wz);
                    let i = ridx(rx, rz, ly);
                    if src > 1 && src - 1 > light[i] && !block::opaque(blocks[i]) {
                        light[i] = src - 1;
                        push(&mut queue, rx, rz, ly);
                    }
                }
            };
            for rz in 0..RW {
                seed_b(0, rz, ox - 1, oz + rz as i32);
                seed_b(RW - 1, rz, ox + RW as i32, oz + rz as i32);
            }
            for rx in 0..RW {
                seed_b(rx, 0, ox + rx as i32, oz - 1);
                seed_b(rx, RW - 1, ox + rx as i32, oz + RW as i32);
            }
        }
        // BFS relax: want = light-1 spreads into non-opaque in-region cells
        const NB: [(i32, i32, i32); 6] =
            [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)];
        while let Some(p) = queue.pop_front() {
            let (rx, rz, ry) = ((p & 63) as i32, ((p >> 6) & 63) as i32, (p >> 12) as i32);
            let l = light[ridx(rx as usize, rz as usize, ry as usize)];
            let want = l - 1;
            for (dx, dy, dz) in NB {
                let (nx, nz, ny) = (rx + dx, rz + dz, ry + dy);
                if nx < 0 || nz < 0 || ny < 0 {
                    continue;
                }
                let (nx, nz, ny) = (nx as usize, nz as usize, ny as usize);
                if nx >= RW || nz >= RW || ny >= RH || !is_loaded(nx, nz) {
                    continue;
                }
                let i = ridx(nx, nz, ny);
                if want > light[i] && !block::opaque(blocks[i]) {
                    light[i] = want;
                    if want > 1 {
                        push(&mut queue, nx, nz, ny);
                    }
                }
            }
        }
        // write back + dirty meshes that sample this region (region + ring)
        for ix in 0..3usize {
            for iz in 0..3usize {
                if !loaded[ix][iz] {
                    continue;
                }
                let key = (ccx - 1 + ix as i32, ccz - 1 + iz as i32);
                if let Some(c) = self.chunks.get_mut(&key) {
                    for lx in 0..CW {
                        for lz in 0..CW {
                            let rx = ix * CW + lx;
                            let rz = iz * CW + lz;
                            for ly in 0..CH {
                                c.data.light[idx(lx, lz, ly)] = light[ridx(rx, rz, ly)];
                            }
                        }
                    }
                }
            }
        }
        for ix in -1..=3i32 {
            for iz in -1..=3i32 {
                if let Some(c) = self.chunks.get_mut(&(ccx - 2 + ix, ccz - 2 + iz)) {
                    c.mesh_dirty = true;
                }
            }
        }
    }

    pub fn set(&mut self, x: i32, y: i32, z: i32, b: u8) {
        if y < 0 || y >= CH as i32 {
            return;
        }
        let (cx, cz) = chunk_of(x, z);
        let (lx, lz) = local_of(x, z);
        let ly = y as usize;
        if let Some(c) = self.chunks.get_mut(&(cx, cz)) {
            c.data.blocks[idx(lx, lz, ly)] = b;
            c.mesh_dirty = true;
            self.queue_relight(cx, cz);
            // Border edits dirty the neighbor chunks whose faces face this block.
            let nb: [(i32, i32); 4] = [(cx - 1, cz), (cx + 1, cz), (cx, cz - 1), (cx, cz + 1)];
            let edge = (lx == 0, lx == CW - 1, lz == 0, lz == CW - 1);
            if edge.0 { if let Some(n) = self.chunks.get_mut(&nb[0]) { n.mesh_dirty = true; } }
            if edge.1 { if let Some(n) = self.chunks.get_mut(&nb[1]) { n.mesh_dirty = true; } }
            if edge.2 { if let Some(n) = self.chunks.get_mut(&nb[2]) { n.mesh_dirty = true; } }
            if edge.3 { if let Some(n) = self.chunks.get_mut(&nb[3]) { n.mesh_dirty = true; } }
        }
    }

    /// All 8 surrounding chunks loaded: the padded mesher samples diagonals
    /// for AO, and unloaded columns fall back to solid stone above sea
    /// level, so diagonal seams must wait for the corner chunks.
    pub fn has_8_neighbors(&self, cx: i32, cz: i32) -> bool {
        [-1i32, 0, 1].iter().all(|dx| [-1i32, 0, 1].iter().all(|dz| {
            (*dx == 0 && *dz == 0) || self.chunks.contains_key(&(cx + dx, cz + dz))
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat_world_chunk(fill: u8) -> crate::world::Chunk {
        let mut d = ChunkData::new();
        if fill != 0 {
            for b in d.blocks.iter_mut() {
                *b = fill;
            }
        }
        Chunk { data: d, meshed: true, mesh_dirty: false }
    }

    #[test]
    fn sky_fills_air_zeroes_opaque() {
        let mut w = World::new(2);
        let mut c = flat_world_chunk(0);
        for y in 0..10 {
            for x in 0..CW {
                for z in 0..CW {
                    c.data.blocks[idx(x, z, y)] = block::STONE;
                }
            }
        }
        w.chunks.insert((0, 0), c);
        w.relight_region(0, 0);
        let c = &w.chunks[&(0, 0)];
        assert_eq!(c.data.light[idx(4, 4, 10)], 15, "air above surface full sky");
        assert_eq!(c.data.light[idx(4, 4, 5)], 0, "interior stone unlit");
    }

    #[test]
    fn shaft_floods_cave_with_decay() {
        let mut w = World::new(2);
        let mut c = flat_world_chunk(block::STONE);
        // shaft down to y=64 at (8,8), cave corridor along -x at y=64
        for y in 64..CH {
            c.data.blocks[idx(8, 8, y)] = block::AIR;
        }
        for x in 0..8 {
            c.data.blocks[idx(x, 8, 64)] = block::AIR;
        }
        w.chunks.insert((0, 0), c);
        w.relight_region(0, 0);
        let c = &w.chunks[&(0, 0)];
        assert_eq!(c.data.light[idx(8, 8, 64)], 15, "shaft mouth full");
        assert_eq!(c.data.light[idx(7, 8, 64)], 14, "first cave cell decays 1");
        assert_eq!(c.data.light[idx(0, 8, 64)], 7, "8 blocks in: 15-8");
        // solid stone never carries light
        assert_eq!(c.data.light[idx(4, 4, 40)], 0, "stone interior unlit");
    }

    #[test]
    fn unloaded_neighbor_is_a_light_barrier() {
        // sealed 5x5x5 room, no sky path: must stay dark while the
        // neighbor chunk is unloaded (no phantom 15 leaking through seams)
        let mut w = World::new(2);
        let mut c = flat_world_chunk(block::STONE);
        for x in 4..9 {
            for z in 4..9 {
                for y in 40..45 {
                    c.data.blocks[idx(x, z, y)] = block::AIR;
                }
            }
        }
        w.chunks.insert((0, 0), c);
        w.relight_region(0, 0);
        assert_eq!(w.chunks[&(0, 0)].data.light[idx(6, 6, 42)], 0, "sealed room dark");
        // load an open-sky neighbor at +x and carve a tunnel from its edge
        // into the room: boundary seed + BFS must shine in with decay 1
        let n = flat_world_chunk(0); // all air = fully lit columns
        w.chunks.insert((1, 0), n);
        if let Some(c0) = w.chunks.get_mut(&(0, 0)) {
            for x in 9..16 {
                c0.data.blocks[idx(x, 6, 42)] = block::AIR;
            }
        }
        w.relight_region(0, 0);
        // world x=16 (chunk 1, open column) is 15; tunnel x=15 -> 14, ...
        assert_eq!(w.chunks[&(0, 0)].data.light[idx(15, 6, 42)], 14, "tunnel mouth");
        assert_eq!(w.chunks[&(0, 0)].data.light[idx(8, 6, 42)], 7, "room entry, 8 in");
        assert_eq!(w.chunks[&(0, 0)].data.light[idx(6, 6, 42)], 5, "room center, 10 in");
    }

    #[test]
    fn coord_roundtrip_negative() {
        for x in [-131i32, -17, -16, -1, 0, 1, 15, 16, 127] {
            for z in [-33i32, -1, 0, 7, 64] {
                let (cx, cz) = chunk_of(x, z);
                let (lx, lz) = local_of(x, z);
                assert!(lx < CW && lz < CW);
                assert_eq!(cx * CW as i32 + lx as i32, x);
                assert_eq!(cz * CW as i32 + lz as i32, z);
            }
        }
    }

    #[test]
    fn set_get_and_dirty_neighbors() {
        let mut w = World::new(1);
        for c in [(0i32, 0i32), (1, 0), (-1, 0), (0, 1), (0, -1)] {
            w.chunks.insert(c, Chunk { data: ChunkData::new(), meshed: false, mesh_dirty: false });
        }
        w.set(0, 5, 0, block::STONE);
        assert_eq!(w.get(0, 5, 0), block::STONE);
        assert!(w.chunks[&(0, 0)].mesh_dirty);
        // edit at local x=0 must dirty chunk (-1, 0)
        w.set(0, 6, 3, block::DIRT);
        assert!(w.chunks[&(-1, 0)].mesh_dirty);
    }

    #[test]
    fn block_table_consistency() {
        assert!(!block::solid(block::AIR));
        assert!(!block::solid(block::WATER));
        assert!(block::solid(block::GLASS));
        assert!(!block::opaque(block::LEAVES));
        assert_eq!(block::tile(block::AIR, 0), 255);
        for b in 0..block::COUNT as u8 {
            for f in 0..6 {
                let t = block::tile(b, f);
                assert!(t == 255 || (t as usize) < 16);
            }
        }
    }
}
