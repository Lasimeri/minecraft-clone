// Amanatides & Woo voxel traversal. No per-step float rounding of the ray:
// tMax advances by exact tDelta, so long rays stay on-grid.

use crate::world::block;
use glam::Vec3;

pub struct RayHit {
    pub block: [i32; 3],
    /// Normal of the entered face (points back toward the ray origin cell).
    pub normal: [i32; 3],
    /// Traversal distance to entry in ray-parameter units. Read by tests;
    /// gameplay uses `block`/`normal` only.
    #[allow(dead_code)]
    pub dist: f32,
}

pub fn raycast(
    get: impl Fn(i32, i32, i32) -> u8,
    origin: Vec3,
    dir: Vec3,
    max_dist: f32,
) -> Option<RayHit> {
    let dir = dir.normalize();
    let mut p = [origin.x.floor() as i32, origin.y.floor() as i32, origin.z.floor() as i32];
    let step = [
        (dir.x > 0.0) as i32 - (dir.x < 0.0) as i32,
        (dir.y > 0.0) as i32 - (dir.y < 0.0) as i32,
        (dir.z > 0.0) as i32 - (dir.z < 0.0) as i32,
    ];
    // tDelta = full cell / |dir component|; guard exact zeros
    let inv = [dir.x.abs(), dir.y.abs(), dir.z.abs()];
    let t_delta = [
        if inv[0] > 1e-9 { 1.0 / inv[0] } else { f32::INFINITY },
        if inv[1] > 1e-9 { 1.0 / inv[1] } else { f32::INFINITY },
        if inv[2] > 1e-9 { 1.0 / inv[2] } else { f32::INFINITY },
    ];
    // tMax = distance along the ray to the next cell boundary on each axis
    let mut t_max = [
        if dir.x > 1e-9 {
            ((p[0] + 1) as f32 - origin.x) / dir.x
        } else if dir.x < -1e-9 {
            (origin.x - p[0] as f32) / -dir.x
        } else {
            f32::INFINITY
        },
        if dir.y > 1e-9 {
            ((p[1] + 1) as f32 - origin.y) / dir.y
        } else if dir.y < -1e-9 {
            (origin.y - p[1] as f32) / -dir.y
        } else {
            f32::INFINITY
        },
        if dir.z > 1e-9 {
            ((p[2] + 1) as f32 - origin.z) / dir.z
        } else if dir.z < -1e-9 {
            (origin.z - p[2] as f32) / -dir.z
        } else {
            f32::INFINITY
        },
    ];

    let mut axis;
    let mut dist;
    loop {
        // step the axis whose boundary is nearest
        if t_max[0] <= t_max[1] && t_max[0] <= t_max[2] {
            axis = 0;
        } else if t_max[1] <= t_max[2] {
            axis = 1;
        } else {
            axis = 2;
        }
        dist = t_max[axis];
        if dist > max_dist {
            return None;
        }
        p[axis] += step[axis];
        t_max[axis] += t_delta[axis];

        let b = get(p[0], p[1], p[2]);
        if b != block::AIR && b != block::WATER {
            let mut normal = [0i32; 3];
            normal[axis] = -step[axis];
            return Some(RayHit { block: p, normal, dist });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::{ChunkData, idx};

    fn flat_world() -> ChunkData {
        // solid stone plane at y=0..=9 everywhere in test range
        let mut d = ChunkData::new();
        for x in 0..16 {
            for z in 0..16 {
                for y in 0..10 {
                    d.blocks[idx(x, z, y)] = block::STONE;
                }
            }
        }
        d
    }

    #[test]
    fn straight_down_hits_floor_top() {
        let d = flat_world();
        let get = |x: i32, y: i32, z: i32| {
            if x >= 0 && x < 16 && z >= 0 && z < 16 && y >= 0 && y < 128 {
                d.get(x as usize, z as usize, y as usize)
            } else {
                0
            }
        };
        let h = raycast(&get, Vec3::new(3.5, 20.0, 4.5), Vec3::NEG_Y, 50.0).unwrap();
        assert_eq!(h.block, [3, 9, 4]);
        assert_eq!(h.normal, [0, 1, 0]);
        assert!((h.dist - 10.0).abs() < 1e-4);
    }

    #[test]
    fn horizontal_hits_wall_with_correct_normal() {
        let d = flat_world();
        let get = |x: i32, y: i32, z: i32| {
            if x >= 0 && x < 16 && z >= 0 && z < 16 && y >= 0 && y < 128 {
                d.get(x as usize, z as usize, y as usize)
            } else {
                0
            }
        };
        // travel +x at y=5 (inside the solid plane is wrong; raise plane test):
        // cast from below the plane upward at an angle
        let h = raycast(&get, Vec3::new(3.5, 2.0, 3.5), Vec3::X, 50.0).unwrap();
        // origin is inside solid stone; first step enters another solid cell
        assert_eq!(h.normal, [-1, 0, 0]);
        assert_eq!(h.block, [4, 2, 3]);
    }

    #[test]
    fn misses_when_out_of_range() {
        let d = flat_world();
        let get = |x: i32, y: i32, z: i32| {
            if x >= 0 && x < 16 && z >= 0 && z < 16 && y >= 0 && y < 128 {
                d.get(x as usize, z as usize, y as usize)
            } else {
                0
            }
        };
        assert!(raycast(&get, Vec3::new(3.5, 30.0, 3.5), Vec3::Y, 8.0).is_none());
    }
}
