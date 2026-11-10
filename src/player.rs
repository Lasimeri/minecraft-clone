// Player physics: swept-per-axis AABB against the voxel grid. Axis order is
// chosen by velocity sign so a corner clip never tunnels a block.

use glam::Vec3;

pub const HALF_W: f32 = 0.3;
pub const HEIGHT: f32 = 1.8;
pub const EYE: f32 = 1.62;

pub struct Player {
    pub pos: Vec3, // feet center
    pub vel: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub on_ground: bool,
    pub fly: bool,
}

pub struct Input {
    pub fwd: f32,   // -1..1
    pub strafe: f32,
    pub jump: bool,
    pub sneak: bool,
    pub sprint: bool,
}

impl Player {
    pub fn new(pos: Vec3) -> Self {
        Self { pos, vel: Vec3::ZERO, yaw: 0.0, pitch: -0.2, on_ground: false, fly: false }
    }

    pub fn eye(&self) -> Vec3 {
        self.pos + Vec3::new(0.0, EYE, 0.0)
    }

    pub fn look_dir(&self) -> Vec3 {
        let (cp, sp) = (self.pitch.cos(), self.pitch.sin());
        let (cy, sy) = (self.yaw.cos(), self.yaw.sin());
        Vec3::new(-sy * cp, sp, -cy * cp)
    }

    #[inline]
    fn collides(&self, p: Vec3, solid: impl Fn(i32, i32, i32) -> bool) -> bool {
        let min = [p.x - HALF_W, p.y, p.z - HALF_W];
        let max = [p.x + HALF_W, p.y + HEIGHT, p.z + HALF_W];
        for bx in min[0].floor() as i32..=max[0].floor() as i32 {
            for by in min[1].floor() as i32..=max[1].floor() as i32 {
                for bz in min[2].floor() as i32..=max[2].floor() as i32 {
                    if solid(bx, by, bz) {
                        return true;
                    }
                }
            }
        }
        false
    }

    /// Integrate one tick. `dt` clamped by caller.
    pub fn tick(&mut self, input: &Input, dt: f32, is_solid: impl Fn(i32, i32, i32) -> bool) {
        // movement basis in the horizontal plane
        let (cy, sy) = (self.yaw.cos(), self.yaw.sin());
        let fwd_v = Vec3::new(-sy, 0.0, -cy);
        let right_v = Vec3::new(cy, 0.0, -sy);

        let speed = if self.fly {
            if input.sprint { 22.0 } else { 11.0 }
        } else if input.sprint {
            8.0
        } else if input.sneak {
            1.6
        } else {
            4.3
        };

        let wish = (fwd_v * input.fwd + right_v * input.strafe).try_normalize().unwrap_or(Vec3::ZERO);

        if self.fly {
            let vy = (f32::from(u8::from(input.jump)) - f32::from(u8::from(input.sneak))) * speed;
            self.vel = wish * speed + Vec3::new(0.0, vy, 0.0);
        } else {
            // accelerate toward wish direction; friction on the ground
            let accel = if self.on_ground { 60.0 } else { 12.0 };
            let target = wish * speed;
            let mut hv = Vec3::new(self.vel.x, 0.0, self.vel.z);
            hv += (target - hv).clamp_length_max(accel * dt);
            // gravity + jump
            let mut vy = self.vel.y - 28.0 * dt;
            if input.jump && self.on_ground {
                vy = 8.4;
            }
            if input.sneak && self.sneak_anchor(&is_solid) {
                // walking off an edge while sneaking: cancel fall
                if vy < 0.0 && self.on_ground {
                    vy = 0.0;
                }
            }
            self.vel = hv + Vec3::new(0.0, vy, 0.0);
        }

        // sweep each axis independently
        self.move_axis(0, self.vel.x * dt, &is_solid);
        self.move_axis(2, self.vel.z * dt, &is_solid);
        self.on_ground = false;
        self.move_axis(1, self.vel.y * dt, &is_solid);
    }

    fn sneak_anchor(&self, is_solid: &(impl Fn(i32, i32, i32) -> bool)) -> bool {
        // is there ground directly under the feet box?
        let p = self.pos;
        let min = [ (p.x - HALF_W).floor() as i32, (p.y - 0.05).floor() as i32, (p.z - HALF_W).floor() as i32 ];
        let max = [ (p.x + HALF_W).floor() as i32, (p.y - 0.05).floor() as i32, (p.z + HALF_W).floor() as i32 ];
        for bx in min[0]..=max[0] {
            for bz in min[2]..=max[2] {
                if is_solid(bx, min[1], bz) {
                    return true;
                }
            }
        }
        false
    }

    fn move_axis(&mut self, axis: usize, d: f32, is_solid: &impl Fn(i32, i32, i32) -> bool) {
        if d == 0.0 {
            if axis == 1 {
                // at rest, gravity was canceled this tick (sneak anchor): probe contact
                let mut q = self.pos;
                add_axis(&mut q, 1, -0.001);
                self.on_ground = self.collides(q, is_solid);
            }
            return;
        }
        let mut p = self.pos;
        add_axis(&mut p, axis, d);
        if !self.collides(p, is_solid) {
            self.pos = p;
            return;
        }
        // step back to contact: binary search along the axis
        let (mut lo, mut hi) = (0.0f32, d);
        for _ in 0..20 {
            let mid = (lo + hi) * 0.5;
            let mut q = self.pos;
            add_axis(&mut q, axis, mid);
            if self.collides(q, is_solid) {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        let mut q = self.pos;
        add_axis(&mut q, axis, lo);
        self.pos = q;
        if axis == 1 {
            if d < 0.0 {
                self.on_ground = true;
            }
        }
        set_axis(&mut self.vel, axis, 0.0);
    }
}

#[inline]
fn add_axis(v: &mut Vec3, i: usize, d: f32) {
    match i {
        0 => v.x += d,
        1 => v.y += d,
        _ => v.z += d,
    }
}
#[inline]
fn set_axis(v: &mut Vec3, i: usize, x: f32) {
    match i {
        0 => v.x = x,
        1 => v.y = x,
        _ => v.z = x,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ground_at(y_max: i32) -> impl Fn(i32, i32, i32) -> bool {
        move |x: i32, y: i32, z: i32| {
            let _ = (x, z);
            y >= 0 && y <= y_max
        }
    }

    #[test]
    fn falls_and_rests_on_floor() {
        let mut p = Player::new(Vec3::new(0.5, 4.0, 0.5));
        let inp = Input { fwd: 0.0, strafe: 0.0, jump: false, sneak: false, sprint: false };
        for _ in 0..300 {
            p.tick(&inp, 1.0 / 60.0, ground_at(0));
        }
        assert!(p.on_ground);
        assert!((p.pos.y - 1.0).abs() < 0.02, "feet rest on top of y=0 block: {}", p.pos.y);
        assert!(p.vel.y.abs() < 1e-3);
    }

    #[test]
    fn jump_gains_then_returns_height() {
        let mut p = Player::new(Vec3::new(0.5, 1.0, 0.5));
        let stand = Input { fwd: 0.0, strafe: 0.0, jump: false, sneak: false, sprint: false };
        let jump = Input { fwd: 0.0, strafe: 0.0, jump: true, sneak: false, sprint: false };
        for _ in 0..30 {
            p.tick(&stand, 1.0 / 60.0, ground_at(0));
        }
        p.tick(&jump, 1.0 / 60.0, ground_at(0));
        assert!(p.vel.y > 5.0);
        let mut peak = p.pos.y;
        for _ in 0..120 {
            p.tick(&stand, 1.0 / 60.0, ground_at(0));
            peak = peak.max(p.pos.y);
        }
        assert!(peak > 1.3, "jumped to {peak}");
        assert!(p.on_ground);
    }

    #[test]
    fn horizontal_motion_blocked_by_wall() {
        // floor y=0, wall column at x>=3
        let solid = |x: i32, y: i32, z: i32| (y >= 0 && y <= 4 && x >= 3) || (y == 0);
        let mut p = Player::new(Vec3::new(2.0, 1.0, 0.5));
        let inp = Input { fwd: 1.0, strafe: 0.0, jump: false, sneak: false, sprint: false };
        // face +x: yaw such that fwd = +x  => look_dir -sy=1 => yaw=-pi/2
        p.yaw = -std::f32::consts::FRAC_PI_2;
        for _ in 0..120 {
            p.tick(&inp, 1.0 / 60.0, &solid);
        }
        assert!(p.pos.x < 3.0 - HALF_W + 0.01, "stopped by wall at {}", p.pos.x);
    }
}
