// Chunk mesher + procedural texture atlas.
//
// Vertices carry world coordinates (baked), so drawing needs only the global
// view-proj uniform: zero per-chunk CPU cost beyond the draw call itself.
// Two vertex streams: opaque and water (separate pipelines/pass).

use crate::world::{block, ChunkData, World, CW, CH};
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Vertex {
    pub pos: [f32; 4],   // xyz world, w unused (vec4 alignment)
    pub uv: [f32; 2],    // atlas coord, already tile-offset
    pub light: [f32; 2], // x: face_brightness * ao, y: sky
}

/// Face geometry. Order: 0=-Y 1=+Y 2=-X 3=+X 4=-Z 5=+Z (matches block::TILES).
/// Each face: normal, 4 corners in CCW order (winding = cross(e1,e2) = normal),
/// and per-corner (a,b) grid coords driving UV.
pub struct Face {
    pub n: [i32; 3],
    pub c: [[i32; 3]; 4],
    pub ab: [[u8; 2]; 4],
    pub light: f32,
}

pub const FACES: [Face; 6] = [
    // -Y
    Face { n: [0, -1, 0], c: [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]], ab: [[0, 0], [1, 0], [1, 1], [0, 1]], light: 0.55 },
    // +Y
    Face { n: [0, 1, 0], c: [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]], ab: [[0, 0], [0, 1], [1, 1], [1, 0]], light: 1.00 },
    // -X
    Face { n: [-1, 0, 0], c: [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]], ab: [[0, 0], [1, 0], [1, 1], [0, 1]], light: 0.82 },
    // +X
    Face { n: [1, 0, 0], c: [[1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 0, 1]], ab: [[0, 0], [1, 0], [1, 1], [0, 1]], light: 0.82 },
    // -Z
    Face { n: [0, 0, -1], c: [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]], ab: [[0, 0], [1, 0], [1, 1], [0, 1]], light: 0.70 },
    // +Z
    Face { n: [0, 0, 1], c: [[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], ab: [[0, 0], [1, 0], [1, 1], [0, 1]], light: 0.70 },
];

/// AO brightness by level 0..3 (occluded..open).
const AO_LUT: [f32; 4] = [0.33, 0.52, 0.74, 1.0];

// ---- Procedural texture atlas: 4x4 tiles of 16px, RGBA8, 64x64 total ----

pub const ATLAS_PX: usize = 64;
pub const TILE_PX: usize = 16;

#[inline(always)]
fn tile_rand(t: usize, x: usize, y: usize, salt: u64) -> f32 {
    // xorshift-ish hash per texel: deterministic speckle
    let mut h = (t as u64) | ((x as u64) << 8) | ((y as u64) << 16) | (salt << 32);
    h ^= h << 13;
    h ^= h >> 7;
    h ^= h << 17;
    (h as f32 / u64::MAX as f32).abs()
}

/// Build the RGBA8 atlas. Tiles indexed as block::TILES:
/// 0 grass_top 1 grass_side 2 dirt 3 stone 4 cobble 5 sand 6 water
/// 7 log_side 8 log_top 9 leaves 10 glass 11 snow_top 12 snow_side
/// 13 planks 14 bedrock 15 gravel
pub fn build_atlas() -> Vec<u8> {
    let mut px = vec![0u8; ATLAS_PX * ATLAS_PX * 4];
    for t in 0..16usize {
        let tx = (t % 4) * TILE_PX;
        let ty = (t / 4) * TILE_PX;
        for y in 0..TILE_PX {
            for x in 0..TILE_PX {
                let (r, g, b, a) = tile_color(t, x, y);
                let i = ((ty + y) * ATLAS_PX + tx + x) * 4;
                px[i] = r;
                px[i + 1] = g;
                px[i + 2] = b;
                px[i + 3] = a;
            }
        }
    }
    px
}

fn tile_color(t: usize, x: usize, y: usize) -> (u8, u8, u8, u8) {
    let n = tile_rand(t, x, y, 0); // 0..1 speckle
    let s = |base: [u8; 3], amp: u8| {
        (
            (base[0] as i16 - amp as i16 + (n * 2.0 * amp as f32) as i16).clamp(0, 255) as u8,
            (base[1] as i16 - amp as i16 + (n * 2.0 * amp as f32) as i16).clamp(0, 255) as u8,
            (base[2] as i16 - amp as i16 + (n * 2.0 * amp as f32) as i16).clamp(0, 255) as u8,
            255,
        )
    };
    match t {
        0 => s([96, 168, 64], 18),                          // grass top
        1 => {                                              // grass side: dirt w/ green lip
            if y < 3 + (n * 2.0) as usize {
                s([96, 168, 64], 18)
            } else {
                s([134, 96, 67], 14)
            }
        }
        2 => s([134, 96, 67], 14),                          // dirt
        3 => s([128, 128, 128], 16),                        // stone
        4 => {                                              // cobble: blobby cells
            let cell = tile_rand(t, x / 4, y / 4, 7);
            s([118, 118, 118], (20.0 * cell) as u8)
        }
        5 => s([219, 207, 160], 12),                        // sand
        6 => (48, 106, 190, 190),                           // water (translucent)
        7 => {                                              // log side: vertical bark streaks
            let streak = tile_rand(t, x, 0, 3);
            s([104, 82, 50], (30.0 * streak) as u8)
        }
        8 => {                                              // log top: rings
            let dx = x as f32 - 7.5;
            let dy = y as f32 - 7.5;
            let r = (dx * dx + dy * dy).sqrt();
            let ring = ((r * 1.8) as usize % 2) as f32;
            let base = if ring > 0.5 { [178, 152, 98] } else { [120, 94, 58] };
            s(base, 10)
        }
        9 => {                                              // leaves: holes via alpha
            if n < 0.18 {
                (0, 0, 0, 0)
            } else {
                s([54, 128, 44], 26)
            }
        }
        10 => {                                             // glass: frame + clear pane
            let edge = x == 0 || y == 0 || x == 15 || y == 15;
            if edge {
                (200, 226, 232, 235)
            } else {
                (176, 216, 228, 46)
            }
        }
        11 => s([240, 246, 250], 8),                        // snow top
        12 => {                                             // snow side: white over dirt
            if y < 4 {
                s([240, 246, 250], 8)
            } else {
                s([134, 96, 67], 14)
            }
        }
        13 => {                                             // planks: horizontal boards
            let board = (y % 4 == 0) as u8;
            if board == 1 {
                s([116, 90, 54], 8)
            } else {
                s([168, 132, 82], 12)
            }
        }
        14 => {                                             // bedrock: dark chaos
            let c = tile_rand(t, x, y, 11);
            s([60, 60, 66], (46.0 * c) as u8)
        }
        _ => {                                              // gravel
            let c = tile_rand(t, x / 2, y / 2, 5);
            s([130, 126, 124], (36.0 * c) as u8)
        }
    }
}

// ---- Meshing kernels ----
//
// Production kernel: the chunk is copied into a 1-cell padded volume so the
// inner kernel reads neighbors by pure array indexing (no hash lookups, no
// bounds branches), and AO is branchless integer arithmetic:
//     occ = s1 + s2 + c + (m & (1 - c)),  m = s1 & s2
// which equals the textbook `if s1 && s2 { 3 } else { s1 + s2 + c }`.
// mesh_chunk_plain is the honest baseline: hash lookups + textbook branches.

// AO samples reach 2 cells outside the block (face cell + side/diagonal), so
// the pad is 2 cells on every side: block (lx,lz,ly) lives at (lx+2, lz+2, ly+2).
const PW: usize = CW + 4; // padded width, 20
const PH: usize = CH + 4; // padded height, y = -2..=129

#[inline(always)]
fn pad_i(px: usize, pz: usize, py: usize) -> usize {
    (px * PW + pz) * PH + py
}

pub struct Padded {
    pub v: Vec<u8>,
    pub heights: [u8; PW * PW],
}

/// Copy chunk + 2-cell world border into a contiguous padded volume.
pub fn pad_chunk(world: &World, cx: i32, cz: i32, data: &ChunkData) -> Padded {
    let ox = cx * CW as i32;
    let oz = cz * CW as i32;
    let mut v = vec![0u8; PW * PW * PH];
    let mut heights = [0u8; PW * PW];
    for px in 0..PW {
        for pz in 0..PW {
            let wx = ox + px as i32 - 2;
            let wz = oz + pz as i32 - 2;
            let interior = px >= 2 && px < 2 + CW && pz >= 2 && pz < 2 + CW;
            let mut h = 0u8;
            for py in 0..PH {
                let y = py as i32 - 2;
                let b = if interior && y >= 0 && y < CH as i32 {
                    data.get(px - 2, pz - 2, y as usize)
                } else {
                    world.get(wx, y, wz)
                };
                v[pad_i(px, pz, py)] = b;
                h = if b != 0 { (py).min(255) as u8 } else { h };
            }
            heights[px * PW + pz] = h;
        }
    }
    Padded { v, heights }
}

/// Sky brightness from padded column height at (px,pz) for cell row py.
#[inline(always)]
fn sky_at(pad: &Padded, px: usize, pz: usize, py: usize) -> f32 {
    let h = pad.heights[px * PW + pz] as i32;
    let depth = (h - py as i32).max(0) as f32;
    (1.0 - 0.035 * depth).clamp(0.42, 1.0)
}

/// Mesh one chunk. Appends opaque + water vertices. Assumes 4 horizontal
/// neighbors are loaded (world.get's unloaded fallback keeps seams sealed
/// either way; streaming just delays correct seams).
pub fn mesh_chunk(
    world: &World,
    cx: i32,
    cz: i32,
    data: &ChunkData,
    opaque: &mut Vec<Vertex>,
    water: &mut Vec<Vertex>,
) {
    let pad = pad_chunk(world, cx, cz, data);
    let ox = cx * CW as i32;
    let oz = cz * CW as i32;

    for lx in 0..CW {
        for lz in 0..CW {
            for ly in 0..CH {
                // padded coords of the block
                let (px, pz, py) = (lx + 2, lz + 2, ly + 2);
                let b = pad.v[pad_i(px, pz, py)];
                if b == block::AIR {
                    continue;
                }
                let isw = block::is_water(b);
                for (fi, f) in FACES.iter().enumerate() {
                    let (qx, qz, qy) =
                        (px as i32 + f.n[0], pz as i32 + f.n[1], py as i32 + f.n[2]);
                    // padded volume always contains the neighbor cell (py in 0..PH)
                    let nb = pad.v[pad_i(qx as usize, qz as usize, qy as usize)];
                    let emit = if isw {
                        nb != block::WATER && !block::opaque(nb)
                    } else {
                        !block::opaque(nb)
                    };
                    if !emit {
                        continue;
                    }
                    let tile = block::tile(b, fi);
                    let sky = sky_at(&pad, qx as usize, qz as usize, qy as usize);
                    // per-corner AO in the neighbor plane
                    let mut ao = [0u8; 4];
                    let mut any_ao = false;
                    if !isw {
                        // side axes e1,e2 derived from the corner table
                        let e1 = [
                            f.c[1][0] - f.c[0][0],
                            f.c[1][1] - f.c[0][1],
                            f.c[1][2] - f.c[0][2],
                        ];
                        let e2 = [
                            f.c[2][0] - f.c[1][0],
                            f.c[2][1] - f.c[1][1],
                            f.c[2][2] - f.c[1][2],
                        ];
                        for i in 0..4 {
                            let (a, bb) = (f.ab[i][0] as i32 * 2 - 1, f.ab[i][1] as i32 * 2 - 1);
                            let s1o = (qx as i32 + e1[0] * a, qz as i32 + e1[1] * a, qy as i32 + e1[2] * a);
                            let s2o = (qx as i32 + e2[0] * bb, qz as i32 + e2[1] * bb, qy as i32 + e2[2] * bb);
                            let coo = (
                                s1o.0 + e2[0] * bb,
                                s1o.1 + e2[1] * bb,
                                s1o.2 + e2[2] * bb,
                            );
                            let s1 = u8::from(block::opaque(pad.v[pad_i(s1o.0 as usize, s1o.1 as usize, s1o.2 as usize)]));
                            let s2 = u8::from(block::opaque(pad.v[pad_i(s2o.0 as usize, s2o.1 as usize, s2o.2 as usize)]));
                            let c = u8::from(block::opaque(pad.v[pad_i(coo.0 as usize, coo.1 as usize, coo.2 as usize)]));
                            let m = s1 & s2;
                            let occ = s1 + s2 + c + (m & (1 - c)); // 0..3, branchless
                            let lvl = 3 - occ;
                            ao[i] = lvl;
                            any_ao |= lvl < 3;
                        }
                    }
                    let _ = any_ao;
                    // quad flip to avoid AO diagonal seams
                    let flip = ((ao[0] + ao[2]) < (ao[1] + ao[3])) as usize;
                    let tri: [usize; 6] = if flip == 0 {
                        [0, 1, 2, 0, 2, 3]
                    } else {
                        [1, 2, 3, 1, 3, 0]
                    };
                    let tx = (tile % 4) as f32 * TILE_PX as f32;
                    let ty = (tile / 4) as f32 * TILE_PX as f32;
                    let dst: &mut Vec<Vertex> = if isw { &mut *water } else { &mut *opaque };
                    for k in 0..6 {
                        let ci = tri[k];
                        let (a, bb) = (f.ab[ci][0], f.ab[ci][1]);
                        let lvl = if isw { 3 } else { ao[ci] } as usize;
                        let u = (tx + if a == 1 { 15.5 } else { 0.5 }) / ATLAS_PX as f32;
                        let v = (ty + if bb == 1 { 15.5 } else { 0.5 }) / ATLAS_PX as f32;
                        dst.push(Vertex {
                            pos: [
                                (ox + f.c[ci][0]) as f32,
                                (ly as i32 + f.c[ci][1]) as f32,
                                (oz + f.c[ci][2]) as f32,
                                1.0,
                            ],
                            uv: [u, v],
                            light: [f.light * AO_LUT[lvl], sky],
                        });
                    }
                }
            }
        }
    }
}

/// Baseline mesher: textbook branches, direct world lookups. Bench target.
pub fn mesh_chunk_plain(
    world: &World,
    cx: i32,
    cz: i32,
    data: &ChunkData,
    opaque: &mut Vec<Vertex>,
    water: &mut Vec<Vertex>,
) {
    let ox = cx * CW as i32;
    let oz = cz * CW as i32;
    for lx in 0..CW as i32 {
        for lz in 0..CW as i32 {
            for ly in 0..CH as i32 {
                let (wx, wy, wz) = (ox + lx, ly, oz + lz);
                let b = data.get(lx as usize, lz as usize, ly as usize);
                if b == block::AIR {
                    continue;
                }
                let isw = block::is_water(b);
                for (fi, f) in FACES.iter().enumerate() {
                    let nb = world.get(wx + f.n[0], wy + f.n[1], wz + f.n[2]);
                    let emit = if isw {
                        if nb == block::WATER {
                            false
                        } else {
                            !block::opaque(nb)
                        }
                    } else if block::opaque(nb) {
                        false
                    } else {
                        true
                    };
                    if !emit {
                        continue;
                    }
                    let tile = block::tile(b, fi);
                    let sky = world.sky_of_neighbor(wx + f.n[0], wy + f.n[1], wz + f.n[2]);
                    let mut ao = [3u8; 4];
                    if !isw {
                        let e1 = [
                            f.c[1][0] - f.c[0][0],
                            f.c[1][1] - f.c[0][1],
                            f.c[1][2] - f.c[0][2],
                        ];
                        let e2 = [
                            f.c[2][0] - f.c[1][0],
                            f.c[2][1] - f.c[1][1],
                            f.c[2][2] - f.c[1][2],
                        ];
                        for i in 0..4 {
                            let a = f.ab[i][0] as i32 * 2 - 1;
                            let bb = f.ab[i][1] as i32 * 2 - 1;
                            let q = (wx + f.n[0], wy + f.n[1], wz + f.n[2]);
                            let s1 = block::opaque(world.get(q.0 + e1[0] * a, q.1 + e1[1] * a, q.2 + e1[2] * a));
                            let s2 = block::opaque(world.get(q.0 + e2[0] * bb, q.1 + e2[1] * bb, q.2 + e2[2] * bb));
                            let cc = block::opaque(world.get(
                                q.0 + e1[0] * a + e2[0] * bb,
                                q.1 + e1[1] * a + e2[1] * bb,
                                q.2 + e1[2] * a + e2[2] * bb,
                            ));
                            ao[i] = if s1 && s2 {
                                0
                            } else {
                                3 - (u8::from(s1) + u8::from(s2) + u8::from(cc))
                            };
                        }
                    }
                    let flip = if (ao[0] + ao[2]) < (ao[1] + ao[3]) { 1 } else { 0 };
                    let tri: [usize; 6] = if flip == 0 {
                        [0, 1, 2, 0, 2, 3]
                    } else {
                        [1, 2, 3, 1, 3, 0]
                    };
                    let tx = (tile % 4) as f32 * TILE_PX as f32;
                    let ty = (tile / 4) as f32 * TILE_PX as f32;
                    let dst: &mut Vec<Vertex> = if isw { &mut *water } else { &mut *opaque };
                    for k in 0..6 {
                        let ci = tri[k];
                        let (a, bb) = (f.ab[ci][0], f.ab[ci][1]);
                        let lvl = if isw { 3 } else { ao[ci] as usize };
                        dst.push(Vertex {
                            pos: [
                                (ox + f.c[ci][0]) as f32,
                                (ly + f.c[ci][1]) as f32,
                                (oz + f.c[ci][2]) as f32,
                                1.0,
                            ],
                            uv: [
                                (tx + if a == 1 { 15.5 } else { 0.5 }) / ATLAS_PX as f32,
                                (ty + if bb == 1 { 15.5 } else { 0.5 }) / ATLAS_PX as f32,
                            ],
                            light: [f.light * AO_LUT[lvl], sky],
                        });
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::{idx, Chunk, ChunkData, World};

    fn world_with(mut d: ChunkData) -> World {
        let mut w = World::new(1);
        for c in [(0i32, 0i32), (1, 0), (-1, 0), (0, 1), (0, -1)] {
            w.chunks
                .insert(c, Chunk { data: ChunkData::new(), meshed: false, mesh_dirty: false });
        }
        d.recalc_column(0, 0);
        w.chunks.get_mut(&(0, 0)).unwrap().data = d;
        w
    }

    #[test]
    fn single_block_makes_exactly_6_faces() {
        let mut d = ChunkData::new();
        d.blocks[idx(4, 4, 40)] = block::STONE;
        let w = world_with(d);
        let (mut o, mut wt) = (Vec::new(), Vec::new());
        mesh_chunk(&w, 0, 0, &w.chunks[&(0, 0)].data, &mut o, &mut wt);
        assert_eq!(o.len(), 36, "6 faces x 6 verts");
        assert_eq!(wt.len(), 0);
    }

    #[test]
    fn solid_2x2x2_hides_all_internal_faces() {
        let mut d = ChunkData::new();
        for x in 4..6 {
            for z in 4..6 {
                for y in 40..42 {
                    d.blocks[idx(x, z, y)] = block::STONE;
                }
            }
        }
        let w = world_with(d);
        let (mut o, _): (Vec<Vertex>, Vec<Vertex>) = (Vec::new(), Vec::new());
        mesh_chunk(&w, 0, 0, &w.chunks[&(0, 0)].data, &mut o, &mut Vec::new());
        assert_eq!(o.len(), 24 * 6, "cube surface = 24 faces");
    }

    #[test]
    fn atlas_has_opaque_texels() {
        let a = build_atlas();
        let opaque = a.chunks_exact(4).filter(|p| p[3] > 12).count();
        assert!(opaque > 60 * 64 / 2, "opaque texels: {opaque}");
    }

    /// Winding law for this stack (glam look_to_rh + perspective_rh, wgpu
    /// FrontFace::Ccw evaluated in y-up clip space, Vulkan y-down framebuffer
    /// flip absorbed by naga): a camera-facing face must have POSITIVE signed
    /// shoelace area in NDC. Verified empirically 2026-11-10: the opposite
    /// table was fully culled by cull_mode::Back with zero validation errors.
    #[test]
    fn faces_wind_ccw_for_wgpu() {
        use glam::{Mat4, Vec3};
        let eye = Vec3::new(8.5, 39.0, 8.5);
        let (yaw, pitch) = (2.35f32, -0.28f32);
        let dir = Vec3::new(-yaw.sin() * pitch.cos(), pitch.sin(), -yaw.cos() * pitch.cos());
        let vp = Mat4::perspective_rh(std::f32::consts::FRAC_PI_2 * 0.78, 16.0 / 9.0, 0.08, 1200.0)
            * Mat4::look_to_rh(eye, dir, Vec3::Y);
        let b = [1.0f32, 35.0, 16.0];
        for f in FACES.iter() {
            let p: Vec<Vec3> = f
                .c
                .iter()
                .map(|o| {
                    vp.project_point3(Vec3::new(
                        b[0] + o[0] as f32,
                        b[1] + o[1] as f32,
                        b[2] + o[2] as f32,
                    ))
                })
                .collect();
            let area = (p[1].x - p[0].x) * (p[2].y - p[0].y) - (p[1].y - p[0].y) * (p[2].x - p[0].x);
            let center = Vec3::new(b[0] + 0.5, b[1] + 0.5, b[2] + 0.5);
            let to_cam = eye - center;
            let facing = f.n[0] as f32 * to_cam.x + f.n[1] as f32 * to_cam.y + f.n[2] as f32 * to_cam.z;
            assert!(
                area.signum() == facing.signum() && area != 0.0,
                "face winding wrong: n {:?} area {area} facing {facing}",
                f.n
            );
        }
    }

    #[test]
    fn optimized_and_plain_agree_on_face_count() {
        let mut d = ChunkData::new();
        for x in 2..10 {
            for z in 2..7 {
                let h = 30 + (x % 3);
                for y in 25..h {
                    d.blocks[idx(x, z, y as usize)] = block::STONE;
                }
            }
        }
        let w = world_with(d);
        let (a, b) = (&mut Vec::new(), &mut Vec::new());
        mesh_chunk(&w, 0, 0, &w.chunks[&(0, 0)].data, a, &mut Vec::new());
        mesh_chunk_plain(&w, 0, 0, &w.chunks[&(0, 0)].data, b, &mut Vec::new());
        assert_eq!(a.len(), b.len(), "face culling must agree");
    }

    #[test]
    fn branchless_ao_matches_textbook() {
        // exhaustive over the 8 (s1,s2,c) states
        for s1 in 0..2u8 {
            for s2 in 0..2u8 {
                for c in 0..2u8 {
                    let m = s1 & s2;
                    let fast = s1 + s2 + c + (m & (1 - c));
                    let slow = if s1 == 1 && s2 == 1 { 3 } else { s1 + s2 + c };
                    assert_eq!(fast, slow, "s1={s1} s2={s2} c={c}");
                }
            }
        }
    }
}

