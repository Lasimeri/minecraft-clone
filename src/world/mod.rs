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
    /// Per-column y of the highest non-air block (255 sentinel unused; height <= CH).
    pub heights: Box<[u8; CW * CW]>,
}

impl ChunkData {
    pub fn new() -> Self {
        Self {
            blocks: vec![0; CHUNK_BLOCKS].into_boxed_slice().try_into().unwrap(),
            heights: vec![0; CW * CW].into_boxed_slice().try_into().unwrap(),
        }
    }
    #[inline]
    pub fn get(&self, lx: usize, lz: usize, ly: usize) -> u8 {
        self.blocks[idx(lx, lz, ly)]
    }
    #[inline]
    pub fn column_height(&self, lx: usize, lz: usize) -> usize {
        self.heights[lx * CW + lz] as usize
    }
    pub fn recalc_column(&mut self, lx: usize, lz: usize) {
        let base = lx * CW + lz;
        let col = &self.blocks[base * CH..base * CH + CH];
        // highest non-air + 1; scan from the top, early exit
        let mut top = 0usize;
        for y in (0..CH).rev() {
            if col[y] != 0 {
                top = y + 1;
                break;
            }
        }
        self.heights[base] = top as u8;
    }
}

pub struct Chunk {
    pub data: ChunkData,
    pub meshed: bool,   // mesh exists / is current
    pub mesh_dirty: bool,
}

pub struct Stream {
    pub gen_tx: mpsc::Sender<(i32, i32)>,
    pub gen_rx: mpsc::Receiver<(i32, i32, ChunkData)>,
    pub requested: std::collections::HashSet<(i32, i32)>,
    pub workers: Vec<std::thread::JoinHandle<()>>,
}

pub struct World {
    pub chunks: HashMap<(i32, i32), Chunk>,
    pub radius: i32,
    pub stream: Option<Stream>,
}

impl World {
    pub fn new(radius: i32) -> Self {
        Self {
            chunks: HashMap::new(),
            radius,
            stream: None,
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

    /// Block + its column height (for skylight). Falls back gracefully.
    #[inline]
    pub fn sky_of_neighbor(&self, x: i32, y: i32, z: i32) -> f32 {
        let (cx, cz) = chunk_of(x, z);
        let h = match self.chunks.get(&(cx, cz)) {
            Some(c) => {
                let (lx, lz) = local_of(x, z);
                c.data.column_height(lx, lz)
            }
            None => 0,
        };
        // Above surface = full sky; below = attenuation by depth, floor 0.42.
        let depth = h.saturating_sub(y as usize) as f32;
        (1.0 - 0.035 * depth).clamp(0.42, 1.0)
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
            c.data.recalc_column(lx, lz);
            c.mesh_dirty = true;
            // Border edits dirty the neighbor chunks whose faces face this block.
            let nb: [(i32, i32); 4] = [(cx - 1, cz), (cx + 1, cz), (cx, cz - 1), (cx, cz + 1)];
            let edge = (lx == 0, lx == CW - 1, lz == 0, lz == CW - 1);
            if edge.0 { if let Some(n) = self.chunks.get_mut(&nb[0]) { n.mesh_dirty = true; } }
            if edge.1 { if let Some(n) = self.chunks.get_mut(&nb[1]) { n.mesh_dirty = true; } }
            if edge.2 { if let Some(n) = self.chunks.get_mut(&nb[2]) { n.mesh_dirty = true; } }
            if edge.3 { if let Some(n) = self.chunks.get_mut(&nb[3]) { n.mesh_dirty = true; } }
        }
    }

    pub fn has_4_neighbors(&self, cx: i32, cz: i32) -> bool {
        [(cx - 1, cz), (cx + 1, cz), (cx, cz - 1), (cx, cz + 1)]
            .iter()
            .all(|c| self.chunks.contains_key(c))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
