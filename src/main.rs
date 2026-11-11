// Minecraft-clone: wgpu voxel renderer, streaming world, FPS controls.
// Modes: default = game; --bench-mesh N = headless mesher benchmark.

mod gpu;
mod hud;
mod mesh;
mod player;
mod raycast;
mod world;

use glam::Vec3;
use gpu::{ChunkGpuMap, Gpu};
use player::{Input, Player};
use std::collections::HashSet;
use std::time::{Duration, Instant};
use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, DeviceId, ElementState, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, Window, WindowId};

const MESH_BUDGET: usize = 8; // chunk remeshes per frame (~0.25 ms each)

struct Game {
    window: Option<std::sync::Arc<Window>>,
    gpu: Option<Gpu>,
    meshes: ChunkGpuMap,
    wld: Option<world::World>,
    player: Player,
    keys: HashSet<KeyCode>,
    locked: bool,
    hotbar: usize,
    last: Instant,
    next_frame: Instant,
    fps: f32,
    acc: f32,
    seed: u32,
    gen_tx: Option<std::sync::mpsc::Sender<(i32, i32)>>,
    gen_rx: Option<std::sync::mpsc::Receiver<(i32, i32, world::ChunkData)>>,
    requested: HashSet<(i32, i32)>,
    workers: Vec<std::thread::JoinHandle<()>>,
    stats: bool,
    stat_last: Instant,
    frames: u64,
}

impl Game {
    fn new(seed: u32, radius: i32) -> Self {
        Self {
            window: None,
            gpu: None,
            meshes: ChunkGpuMap::new(),
            wld: Some(world::World::new(radius)),
            player: Player::new(Vec3::new(8.5, 90.0, 8.5)),
            keys: HashSet::new(),
            locked: false,
            hotbar: 0,
            last: Instant::now(),
            next_frame: Instant::now(),
            fps: 0.0,
            acc: 0.0,
            seed,
            gen_tx: None,
            gen_rx: None,
            requested: HashSet::new(),
            workers: Vec::new(),
            stats: false,
            stat_last: Instant::now(),
            frames: 0,
        }
    }

    fn pcx(&self) -> i32 {
        (self.player.pos.x.floor() as i32).div_euclid(world::CW as i32)
    }
    fn pcz(&self) -> i32 {
        (self.player.pos.z.floor() as i32).div_euclid(world::CW as i32)
    }

    /// Queue chunk generation for the ring around the player.
    fn request_chunks(&mut self) {
        let (Some(w), Some(tx)) = (&self.wld, &self.gen_tx) else { return };
        let r = w.radius;
        let (pcx, pcz) = (self.pcx(), self.pcz());
        for dx in -r..=r {
            for dz in -r..=r {
                if dx * dx + dz * dz > r * r + r {
                    continue; // round-ish streaming disk
                }
                let c = (pcx + dx, pcz + dz);
                if w.chunks.contains_key(&c) || self.requested.contains(&c) {
                    continue;
                }
                if tx.send(c).is_ok() {
                    self.requested.insert(c);
                }
            }
        }
    }

    /// Drain finished generation into the world.
    fn poll_generated(&mut self) {
        let Some(rx) = &self.gen_rx else { return };
        let mut n = 0;
        while let Ok((cx, cz, data)) = rx.try_recv() {
            self.requested.remove(&(cx, cz));
            if let Some(w) = &mut self.wld {
                w.chunks.insert(
                    (cx, cz),
                    world::Chunk { data, meshed: false, mesh_dirty: false },
                );
                w.queue_relight(cx, cz);
            }
            n += 1;
            if n >= 16 {
                break; // spread insertion cost across frames
            }
        }
    }

    /// Drain the relight queue: full region flood per entry, budgeted so a
    /// load burst never stalls a frame (each region ~2-4 ms worst case).
    /// Light is a vertex attribute, so every already-meshed chunk whose
    /// light plane the flood rewrote must be re-meshed.
    fn relight_pass(&mut self) {
        const BUDGET: usize = 3;
        let Some(w) = &mut self.wld else { return };
        let mut n = 0;
        while n < BUDGET {
            let Some(c) = w.relight_pending.pop_front() else { break };
            w.relight_set.remove(&c);
            w.relight_region(c.0, c.1);
            for dx in -1..=1i32 {
                for dz in -1..=1i32 {
                    if let Some(ch) = w.chunks.get_mut(&(c.0 + dx, c.1 + dz)) {
                        if ch.meshed {
                            ch.mesh_dirty = true;
                        }
                    }
                }
            }
            n += 1;
        }
    }

    /// Mesh dirty/ready chunks within budget, nearest first.
    fn mesh_pass(&mut self) {
        let (pcx, pcz) = (self.pcx(), self.pcz());
        let (Some(w), Some(g)) = (&mut self.wld, &mut self.gpu) else { return };
        let mut todo: Vec<(i64, (i32, i32))> = w
            .chunks
            .iter()
            .filter(|(c, ch)| (!ch.meshed || ch.mesh_dirty) && w.has_8_neighbors(c.0, c.1) && !w.relight_set.contains(c))
            .map(|(c, _)| {
                let dx = (c.0 - pcx) as i64;
                let dz = (c.1 - pcz) as i64;
                (dx * dx + dz * dz, *c)
            })
            .collect();
        todo.sort_unstable();
        let mut opaque = Vec::new();
        let mut water = Vec::new();
        for (_, c) in todo.into_iter().take(MESH_BUDGET) {
            opaque.clear();
            water.clear();
            let data = &w.chunks[&c].data;
            mesh::mesh_chunk(w, c.0, c.1, data, &mut opaque, &mut water);
            self.meshes.upsert(&g.device, c, false, &opaque);
            self.meshes.upsert(&g.device, c, true, &water);
            let ch = w.chunks.get_mut(&c).unwrap();
            ch.meshed = true;
            ch.mesh_dirty = false;
        }
    }

    /// Drop far chunks (bounded memory while roaming).
    fn unload_far(&mut self) {
        let (pcx, pcz) = (self.pcx(), self.pcz());
        let Some(w) = &mut self.wld else { return };
        let cutoff = w.radius + 4;
        let far: Vec<(i32, i32)> = w
            .chunks
            .keys()
            .filter(|c| (c.0 - pcx).abs().max((c.1 - pcz).abs()) > cutoff)
            .copied()
            .collect();
        for c in far {
            w.chunks.remove(&c);
            self.meshes.drop_chunk(c);
        }
    }

    fn current_input(&self) -> Input {
        let k = &self.keys;
        let b = |c: KeyCode| k.contains(&c);
        Input {
            fwd: f32::from(u8::from(b(KeyCode::KeyW))) - f32::from(u8::from(b(KeyCode::KeyS))),
            strafe: f32::from(u8::from(b(KeyCode::KeyD))) - f32::from(u8::from(b(KeyCode::KeyA))),
            jump: b(KeyCode::Space),
            sneak: b(KeyCode::ShiftLeft),
            sprint: b(KeyCode::ControlLeft),
        }
    }

    fn targeted(&self) -> Option<raycast::RayHit> {
        let w = self.wld.as_ref()?;
        raycast::raycast(
            |x, y, z| w.get(x, y, z),
            self.player.eye(),
            self.player.look_dir(),
            7.0,
        )
    }

    fn break_block(&mut self) {
        if let Some(h) = self.targeted() {
            let [x, y, z] = h.block;
            if let Some(w) = &mut self.wld {
                if w.get(x, y, z) != world::block::BEDROCK {
                    w.set(x, y, z, world::block::AIR);
                }
            }
        }
    }

    fn place_block(&mut self) {
        if let Some(h) = self.targeted() {
            let (x, y, z) = (
                h.block[0] + h.normal[0],
                h.block[1] + h.normal[1],
                h.block[2] + h.normal[2],
            );
            let Some(w) = &mut self.wld else { return };
            let cur = w.get(x, y, z);
            if cur != world::block::AIR && cur != world::block::WATER {
                return;
            }
            // refuse placement inside the player's AABB
            let p = self.player.pos;
            let overlap = (x as f32) < p.x + player::HALF_W
                && ((x + 1) as f32) > p.x - player::HALF_W
                && (y as f32) < p.y + player::HEIGHT
                && ((y + 1) as f32) > p.y
                && (z as f32) < p.z + player::HALF_W
                && ((z + 1) as f32) > p.z - player::HALF_W;
            if overlap {
                return;
            }
            w.set(x, y, z, world::HOTBAR[self.hotbar]);
        }
    }

    fn set_locked(&mut self, locked: bool) {
        let Some(win) = &self.window else { return };
        let mode = if locked { CursorGrabMode::Locked } else { CursorGrabMode::None };
        // Locked first; some Wayland compositors only honor Confined
        if win.set_cursor_grab(mode).is_err() && locked {
            let _ = win.set_cursor_grab(CursorGrabMode::Confined);
        }
        win.set_cursor_visible(!locked);
        self.locked = locked;
    }
}

impl ApplicationHandler for Game {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.gpu.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title("Minecraft-Clone // wgpu")
            .with_inner_size(winit::dpi::PhysicalSize::new(1440, 810));
        let window = std::sync::Arc::new(event_loop.create_window(attrs).unwrap());
        self.gpu = Some(Gpu::new(window.clone()));
        self.window = Some(window);

        let (tx, rx, workers) = world::terrain::spawn_workers(self.seed, 4);
        self.gen_tx = Some(tx);
        self.gen_rx = Some(rx);
        self.workers = workers;

        // drop the player onto the surface at spawn
        let g = world::terrain::Generator::new(self.seed);
        let h = g.height(8, 8);
        self.player.pos = Vec3::new(8.5, (h + 2) as f32, 8.5);

        self.request_chunks();
        event_loop.set_control_flow(ControlFlow::Poll);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match &event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(s) => {
                if let Some(g) = &mut self.gpu {
                    g.resize(s.width, s.height);
                }
            }
            WindowEvent::CursorMoved { .. } => {
                // Look rotation comes from DeviceEvent::MouseMotion (raw
                // relative deltas via zwp_relative_pointer); CursorMoved
                // carries absolute surface coords that are meaningless
                // under a locked pointer. Forward to egui only.
                if !self.locked {
                    if let Some(g) = &mut self.gpu {
                        let _ = g.egui_state.on_window_event(&self.window.clone().unwrap(), &event);
                    }
                }
            }
            WindowEvent::MouseInput { state: ElementState::Pressed, button, .. } => {
                if !self.locked {
                    self.set_locked(true);
                } else {
                    match button {
                        winit::event::MouseButton::Left => self.break_block(),
                        winit::event::MouseButton::Right => self.place_block(),
                        _ => {}
                    }
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let d = match delta {
                    MouseScrollDelta::LineDelta(_, y) => *y,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32,
                };
                if d > 0.0 {
                    self.hotbar = (self.hotbar + 8) % 9;
                } else if d < 0.0 {
                    self.hotbar = (self.hotbar + 1) % 9;
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                let PhysicalKey::Code(code) = event.physical_key else { return };
                let pressed = event.state == ElementState::Pressed;
                if pressed && !event.repeat {
                    match code {
                        KeyCode::KeyF => self.player.fly = !self.player.fly,
                        KeyCode::Escape => self.set_locked(false),
                        KeyCode::Space | KeyCode::Digit1 | KeyCode::Digit2 | KeyCode::Digit3
                        | KeyCode::Digit4 | KeyCode::Digit5 | KeyCode::Digit6 | KeyCode::Digit7
                        | KeyCode::Digit8 | KeyCode::Digit9 => {}
                        _ => {}
                    }
                    if let Some(d) = digit_of(code) {
                        self.hotbar = d;
                    }
                }
                if pressed {
                    self.keys.insert(code);
                } else {
                    self.keys.remove(&code);
                }
            }
            WindowEvent::RedrawRequested => {
                self.frame();
            }
            _ => {
                if !self.locked {
                    if let Some(g) = &mut self.gpu {
                        let win = self.window.clone().unwrap();
                        let _ = g.egui_state.on_window_event(&win, &event);
                    }
                }
            }
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: DeviceId,
        event: DeviceEvent,
    ) {
        // Raw unaccelerated deltas from zwp_relative_pointer; only arrive
        // while the pointer is grabbed, but gate on locked regardless.
        if !self.locked {
            return;
        }
        if let DeviceEvent::MouseMotion { delta: (dx, dy) } = event {
            self.player.yaw -= dx as f32 * 0.0022;
            self.player.pitch -= dy as f32 * 0.0022;
            self.player.pitch = self.player.pitch.clamp(-1.5533, 1.5533);
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(w) = &self.window {
            w.request_redraw();
        }
    }

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        // closing the job sender ends the workers
        self.gen_tx = None;
        for h in self.workers.drain(..) {
            let _ = h.join();
        }
    }
}

fn digit_of(code: KeyCode) -> Option<usize> {
    Some(match code {
        KeyCode::Digit1 => 0,
        KeyCode::Digit2 => 1,
        KeyCode::Digit3 => 2,
        KeyCode::Digit4 => 3,
        KeyCode::Digit5 => 4,
        KeyCode::Digit6 => 5,
        KeyCode::Digit7 => 6,
        KeyCode::Digit8 => 7,
        KeyCode::Digit9 => 8,
        _ => return None,
    })
}

impl Game {
    /// One rendered frame: stream, simulate, raycast, draw.
    fn frame(&mut self) {
        let now = Instant::now();
        let dt = now.duration_since(self.last).as_secs_f32().min(0.25);
        self.last = now;
        if dt > 0.0 {
            let inst = 1.0 / dt;
            self.fps = if self.fps == 0.0 { inst } else { self.fps * 0.95 + inst * 0.05 };
        }

        self.poll_generated();
        self.request_chunks();
        self.relight_pass();
        self.mesh_pass();
        self.unload_far();
        self.frames += 1;
        if self.stats && self.stat_last.elapsed().as_secs_f32() >= 2.0 {
            let chunks = self.wld.as_ref().map(|w| w.chunks.len()).unwrap_or(0);
            println!(
                "stats: fps {:.1} chunks {} queue {} pos {:.1} {:.1} {:.1}",
                self.fps, chunks, self.requested.len(),
                self.player.pos.x, self.player.pos.y, self.player.pos.z
            );
            self.stat_last = Instant::now();
        }

        // fixed 60 Hz physics steps
        self.acc += dt;
        let input = self.current_input();
        let mut steps = 0;
        while self.acc >= 1.0 / 60.0 && steps < 4 {
            if let Some(w) = &self.wld {
                self.player.tick(&input, 1.0 / 60.0, |x, y, z| {
                    world::block::solid(w.get(x, y, z))
                });
            }
            self.acc -= 1.0 / 60.0;
            steps += 1;
        }
        // safety: fell out of the world → respawn
        if self.player.pos.y < -8.0 {
            let g = world::terrain::Generator::new(self.seed);
            let h = g.height(8, 8);
            self.player.pos = Vec3::new(8.5, (h + 2) as f32, 8.5);
            self.player.vel = Vec3::ZERO;
        }

        let hit = self.targeted().map(|h| h.block);
        let (Some(g), Some(w)) = (&mut self.gpu, &self.wld) else { return };
        let eye = self.player.eye();
        let dir = self.player.look_dir();
        let payload = hud::Payload {
            fps: self.fps,
            pos: self.player.pos.to_array(),
            chunks: w.chunks.len(),
            queue: self.requested.len(),
            hotbar_sel: self.hotbar,
            fly: self.player.fly,
            locked: self.locked,
        };
        match g.render_frame(w, &self.meshes, eye, dir, hit, &payload) {
            Ok(()) => {}
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::OutOfMemory) => {
                if let Some(g) = self.gpu.as_mut() {
                    let (ww, hh) = (g.config.width, g.config.height);
                    g.resize(ww, hh);
                }
            }
            Err(e) => eprintln!("surface error: {e:?}"),
        }

        // Deadline pacing: measured 2000 Hz loop under Fifo on this KDE
        // Wayland stack (present applies no backpressure), i.e. ~1940
        // rendered frames/s discarded by the compositor. Sleep to a 60 Hz
        // accumulator; drift-recover when a frame overshoots.
        const FRAME: f32 = 1.0 / 60.0;
        self.next_frame += Duration::from_secs_f32(FRAME);
        let now2 = Instant::now();
        if self.next_frame < now2 {
            self.next_frame = now2 + Duration::from_secs_f32(FRAME);
        } else {
            std::thread::sleep(self.next_frame.saturating_duration_since(Instant::now()));
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut seed: u32 = 1337;
    let mut radius: i32 = 8;
    let mut bench: Option<usize> = None;
    let mut bench_stats = false;
    let mut shot_path: Option<String> = None;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--seed" => {
                seed = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(seed);
                i += 2;
            }
            "--radius" => {
                radius = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(radius);
                i += 2;
            }
            "--bench-mesh" => {
                let n: usize = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(64);
                bench = Some(n.max(1));
                i += 2;
            }
            "--stats" => {
                bench_stats = true;
                i += 1;
            }
            "--shot" => {
                shot_path = args.get(i + 1).cloned();
                if shot_path.is_none() {
                    eprintln!("--shot requires a file path");
                    std::process::exit(2);
                }
                i += 2;
            }
            "--xsec" => {
                let (x, z): (i32, i32) = (
                    args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(8),
                    args.get(i + 2).and_then(|s| s.parse().ok()).unwrap_or(8),
                );
                xsec_dump(seed, x, z);
                return;
            }
            _ => {
                eprintln!("usage: minecraft-clone [--seed N] [--radius R] [--bench-mesh N] [--stats] [--xsec X Z]");
                std::process::exit(2);
            }
        }
    }

    if let Some(n) = bench {
        bench_mesh(seed, n);
        return;
    }

    if let Some(path) = shot_path {
        shot(&path, seed);
        return;
    }

    let event_loop: EventLoop<()> = EventLoop::new().unwrap();
    let mut game = Game::new(seed, radius);
    game.stats = bench_stats;
    event_loop.run_app(&mut game).unwrap();
}

/// Offscreen one-shot renderer: hidden 1280x720 window (adapter/surface
/// compatibility), synchronous world around spawn, one GPU render, PPM out.
struct ShotApp {
    path: String,
    seed: u32,
    done: bool,
}

impl ApplicationHandler for ShotApp {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        if self.done {
            return;
        }
        self.done = true;
        let window = std::sync::Arc::new(
            el.create_window(
                winit::window::Window::default_attributes()
                    .with_visible(false)
                    .with_inner_size(winit::dpi::PhysicalSize::new(1280, 720)),
            )
            .unwrap(),
        );
        shot_render(&self.path, self.seed, window);
        el.exit();
    }

    fn window_event(&mut self, _el: &ActiveEventLoop, _id: winit::window::WindowId, _e: winit::event::WindowEvent) {}
}

fn shot(path: &str, seed: u32) {
    let event_loop: EventLoop<()> = EventLoop::new().unwrap();
    let mut app = ShotApp { path: path.to_string(), seed, done: false };
    event_loop.run_app(&mut app).expect("shot event loop");
}

/// Debug: print a vertical column cross-section + porosity + face audit for
/// the chunk containing (x,z), from the same generator the game uses.
fn xsec_dump(seed: u32, x: i32, z: i32) {
    let g = world::terrain::Generator::new(seed);
    let mut wld = world::World::new(2);
    for cx in -2..=2 {
        for cz in -2..=2 {
            wld.chunks.insert(
                (cx, cz),
                world::Chunk { data: g.generate(cx, cz), meshed: false, mesh_dirty: false },
            );
        }
    }
    let sym = |b: u8| match b {
        0 => '.',
        1 => 'G',
        2 => 'd',
        3 => '#',
        4 => 'c',
        5 => 's',
        6 => '~',
        7 => '|',
        8 => '*',
        9 => 'o',
        10 => 'S',
        11 => 'p',
        12 => 'B',
        13 => 'g',
        _ => '?',
    };
    // column at (x,z), y 0..70
    for y in (0..70).rev() {
        let b = wld.get(x, y, z);
        print!("{}", sym(b));
    }
    println!("  <- column x={x} z={z} y69..0");
    // porosity: air fraction below surface in the center chunk
    let (ccx, ccz) = (x.div_euclid(16), z.div_euclid(16));
    let cd = &wld.chunks[&(ccx, ccz)].data;
    let mut air_below = 0usize;
    let mut solid_below = 0usize;
    let mut surf_sum = 0usize;
    for lx in 0..world::CW {
        for lz in 0..world::CW {
            let col = &cd.blocks[(lx * world::CW + lz) * world::CH..(lx * world::CW + lz) * world::CH + world::CH];
            let top = col.iter().rposition(|b| *b != 0 && *b != 6).unwrap_or(0);
            surf_sum += top;
            for y in 0..top {
                if col[y] == 0 {
                    air_below += 1;
                } else {
                    solid_below += 1;
                }
            }
        }
    }
    println!(
        "chunk ({ccx},{ccz}): mean_surface_y {} air_below {air_below} solid_below {solid_below} porosity {:.3}",
        surf_sum / (world::CW * world::CW),
        air_below as f64 / (air_below + solid_below) as f64
    );
    // surface breaches: columns where air sits within 3 of the terrain top
    let mut breach = 0usize;
    for lx in 0..world::CW {
        for lz in 0..world::CW {
            let col = &cd.blocks[(lx * world::CW + lz) * world::CH..(lx * world::CW + lz) * world::CH + world::CH];
            let top = col.iter().rposition(|b| *b != 0 && *b != 6).unwrap_or(0);
            if (1..=3).any(|d| top > d && col[top - d] == 0) {
                breach += 1;
            }
        }
    }
    println!("breach_columns {breach}/256");
    // face audit: recount expected faces from blocks vs emitted vertices
    let (mut o, mut wt) = (Vec::new(), Vec::new());
    mesh::mesh_chunk(&wld, ccx, ccz, cd, &mut o, &mut wt);
    let mut exp = 0usize;
    for lx in 0..world::CW {
        for lz in 0..world::CW {
            for ly in 0..world::CH {
                let b = cd.get(lx, lz, ly);
                if b == 0 {
                    continue;
                }
                for (dx, dz, dy) in [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)] {
                    let nb = wld.get(
                        ccx * 16 + lx as i32 + dx,
                        ly as i32 + dy,
                        ccz * 16 + lz as i32 + dz,
                    );
                    let e = if b == 6 { nb != 6 && !world::block::opaque(nb) } else { !world::block::opaque(nb) };
                    exp += usize::from(e);
                }
            }
        }
    }
    println!(
        "emitted opaque_faces {} water_faces {} expected(non-water) {}",
        o.len() / 6,
        wt.len() / 6,
        exp
    );
    // per-direction area histogram from emitted vertex winding
    let mut dirs = [0u32; 6];
    for tri in o.chunks_exact(3) {
        let (a, b, c) = (&tri[0].pos, &tri[1].pos, &tri[2].pos);
        let e1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let e2 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
        let n = [
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0],
        ];
        let k = if n[0] > 0.0 {
            1
        } else if n[0] < 0.0 {
            0
        } else if n[1] > 0.0 {
            3
        } else if n[1] < 0.0 {
            2
        } else if n[2] > 0.0 {
            5
        } else {
            4
        };
        dirs[k] += 1;
    }
    println!("opaque tri dirs -X,+X,-Y,+Y,-Z,+Z: {dirs:?}");
}

fn shot_render(path: &str, seed: u32, window: std::sync::Arc<winit::window::Window>) {
    let mut gpu = Gpu::new(window);

    // synchronous world: 9x9 chunks around origin
    let g = world::terrain::Generator::new(seed);
    let mut wld = world::World::new(4);
    for cx in -4..=4 {
        for cz in -4..=4 {
            let data = g.generate(cx, cz);
            wld.chunks.insert(
                (cx, cz),
                world::Chunk { data, meshed: true, mesh_dirty: false },
            );
        }
    }
    let mut meshes = ChunkGpuMap::new();
    // flood sky light to convergence (two sweeps: boundary light flows in)
    for _ in 0..2 {
        for cx in -4..=4 {
            for cz in -4..=4 {
                wld.relight_region(cx, cz);
            }
        }
    }
    let (mut o, mut wt) = (Vec::new(), Vec::new());
    for c in wld.chunks.keys().copied().collect::<Vec<_>>() {
        o.clear();
        wt.clear();
        mesh::mesh_chunk(&wld, c.0, c.1, &wld.chunks[&c].data, &mut o, &mut wt);
        meshes.upsert(&gpu.device, c, false, &o);
        meshes.upsert(&gpu.device, c, true, &wt);
    }

    let h = g.height(8, 8);
    let eye = Vec3::new(8.5, (h + 3) as f32, 8.5);
    let (yaw, pitch) = (2.35f32, -0.28f32);
    let dir = Vec3::new(
        -yaw.sin() * pitch.cos(),
        pitch.sin(),
        -yaw.cos() * pitch.cos(),
    );
    let rgba = gpu.capture(&wld, &meshes, eye, dir, 1280, 720);

    // PPM (P6): drop alpha, RGB rows
    let mut ppm = format!("P6\n1280 720\n255\n").into_bytes();
    ppm.reserve(1280 * 720 * 3);
    // readback bytes are already R,G,B,A on this stack (verified via clear-color decode)
    for px in rgba.chunks_exact(4) {
        ppm.extend_from_slice(&px[..3]);
    }
    std::fs::write(path, &ppm).expect("write shot");
    let (mut opaque_v, mut water_v) = (0usize, 0usize);
    for m in &meshes.opaque {
        opaque_v += m.count as usize;
    }
    for m in &meshes.water {
        water_v += m.count as usize;
    }
    println!(
        "shot: {path} seed {seed} chunks {} opaque_faces {opaque_v} water_faces {water_v} surface {:?} spawn_h {h}",
        wld.chunks.len(),
        wld.get(8, h, 8)
    );
}

/// Headless mesher benchmark: padded+branchless kernel vs plain baseline.
/// Interleaved rounds on identical terrain; reports median ms per chunk.
fn bench_mesh(seed: u32, n: usize) {
    eprintln!("bench-mesh: seed={seed} chunks={n} rounds=5 host={} threads", std::thread::available_parallelism().map(|v| v.get()).unwrap_or(0));
    let g = world::terrain::Generator::new(seed);

    // generate the corpus once
    let mut world = world::World::new(2);
    let coords: Vec<(i32, i32)> = (0..n as i32).map(|i| (i % 8, i / 8)).collect();
    // neighbors for meshing (plain kernel reads through World::get)
    for cx in -2..=9 {
        for cz in -2..=(n as i32 / 8 + 2) {
            let data = g.generate(cx, cz);
            world.chunks.insert(
                (cx, cz),
                world::Chunk { data, meshed: false, mesh_dirty: false },
            );
        }
    }

    let mut fast_times: Vec<f64> = Vec::new();
    let mut plain_times: Vec<f64> = Vec::new();
    let (mut vf, mut wf, mut vp, mut wp) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());

    for round in 0..5 {
        for &c in &coords {
            let t = Instant::now();
            mesh::mesh_chunk(&world, c.0, c.1, &world.chunks[&c].data, &mut vf, &mut wf);
            let el = t.elapsed().as_secs_f64() * 1e3;
            fast_times.push(el);

            let t = Instant::now();
            mesh::mesh_chunk_plain(&world, c.0, c.1, &world.chunks[&c].data, &mut vp, &mut wp);
            let el = t.elapsed().as_secs_f64() * 1e3;
            plain_times.push(el);

            assert_eq!(vf.len(), vp.len(), "round {round}: opaque vertex mismatch");
            assert_eq!(wf.len(), wp.len(), "round {round}: water vertex mismatch");
            vf.clear(); wf.clear(); vp.clear(); wp.clear();
        }
    }

    let med = |v: &mut Vec<f64>| {
        v.sort_by(|a, b| a.total_cmp(b));
        v[v.len() / 2]
    };
    let fm = med(&mut fast_times.clone());
    let pm = med(&mut plain_times.clone());
    println!("mesher bench: {} chunks, seed {}", n, seed);
    println!("  padded+branchless: median {fm:.3} ms/chunk");
    println!("  plain+branchy    : median {pm:.3} ms/chunk");
    println!("  speedup          : {:.2}x", pm / fm);
}

