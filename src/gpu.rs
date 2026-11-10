// wgpu plumbing: device, surface, terrain pipelines, atlas, uniforms, egui.
// API shapes verified against vendored sources: wgpu 25.0.2, egui-wgpu 0.33.3,
// egui-winit 0.33.3.

use crate::mesh::{Vertex, ATLAS_PX};
use crate::world::{World, CW, CH};
use glam::{Mat4, Vec3};
use std::sync::Arc;
use wgpu::util::DeviceExt;
use winit::window::Window;

pub struct Gpu {
    pub window: Arc<Window>,
    pub surface: wgpu::Surface<'static>,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub config: wgpu::SurfaceConfiguration,
    pub format: wgpu::TextureFormat,
    depth: wgpu::Texture,
    depth_view: wgpu::TextureView,
    terrain_pipeline: wgpu::RenderPipeline,
    hl_pipeline: wgpu::RenderPipeline,
    hl_vb: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    uniform: wgpu::Buffer,
    hl_uniform: wgpu::Buffer,
    pub egui_renderer: egui_wgpu::Renderer,
    pub egui_state: egui_winit::State,
    pub ppp: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uni {
    view_proj: [[f32; 4]; 4],
    cam_pos: [f32; 4],
    fog: [f32; 4],
    sky: [f32; 4],
}

pub const SKY: Vec3 = Vec3::new(0.53, 0.71, 0.94);

impl Gpu {
    pub fn new(window: Arc<Window>) -> Self {
        let size = window.inner_size();
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..Default::default()
        });
        let surface = instance
            .create_surface(window.clone())
            .expect("create surface");
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
            power_preference: wgpu::PowerPreference::HighPerformance,
        }))
        .expect("no Vulkan adapter");
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("mc"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                ..Default::default()
            },
        ))
        .expect("request device");

        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);
        let mut config = surface
            .get_default_config(&adapter, size.width.max(1), size.height.max(1))
            .expect("surface config");
        config.format = format;
        config.present_mode = wgpu::PresentMode::Fifo;
        surface.configure(&device, &config);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("terrain"),
            source: wgpu::ShaderSource::Wgsl(include_str!("terrain.wgsl").into()),
        });

        // ---- atlas ----
        let atlas = crate::mesh::build_atlas();
        let atlas_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("atlas"),
            size: wgpu::Extent3d { width: ATLAS_PX as u32, height: ATLAS_PX as u32, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            atlas_tex.as_image_copy(),
            &atlas,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * ATLAS_PX as u32),
                rows_per_image: Some(ATLAS_PX as u32),
            },
            wgpu::Extent3d { width: ATLAS_PX as u32, height: ATLAS_PX as u32, depth_or_array_layers: 1 },
        );
        let atlas_view = atlas_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("atlas-nearest"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // ---- uniforms ----
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("uni"),
            size: std::mem::size_of::<Uni>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let hl_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hl-uni"),
            size: 64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let hl_vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("hl-vb"),
            contents: bytemuck::cast_slice(&highlight_vertices()),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let uniform_ty = wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("terrain-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::FRAGMENT, ty: uniform_ty, count: None },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::VERTEX, ty: uniform_ty, count: None },
            ],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("terrain-bg"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&atlas_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&sampler) },
                wgpu::BindGroupEntry { binding: 3, resource: hl_uniform.as_entire_binding() },
            ],
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("terrain-layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let vs_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 0, shader_location: 0 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x2, offset: 16, shader_location: 1 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x2, offset: 24, shader_location: 2 },
            ],
        };

        let target = wgpu::ColorTargetState {
            format,
            blend: Some(wgpu::BlendState::ALPHA_BLENDING),
            write_mask: wgpu::ColorWrites::ALL,
        };

        let terrain_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("terrain"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[vs_layout],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(target.clone())],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });

        let hl_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("highlight"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_hl"),
                compilation_options: Default::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: 12,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x3,
                        offset: 0,
                        shader_location: 0,
                    }],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_hl"),
                compilation_options: Default::default(),
                targets: &[Some(target)],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });

        let depth = Self::make_depth(&device, config.width, config.height);
        let depth_view = depth.create_view(&wgpu::TextureViewDescriptor::default());

        let ppp = window.scale_factor() as f32;
        let egui_state = egui_winit::State::new(
            egui::Context::default(),
            egui::ViewportId::ROOT,
            &*window,
            Some(ppp),
            None,
            None,
        );
        let egui_renderer = egui_wgpu::Renderer::new(&device, format, Default::default());

        Self {
            window,
            surface,
            device,
            queue,
            config,
            format,
            depth,
            depth_view,
            terrain_pipeline,
            hl_pipeline,
            hl_vb,
            bind_group,
            uniform,
            hl_uniform,
            egui_renderer,
            egui_state,
            ppp,
        }
    }

    fn make_depth(device: &wgpu::Device, w: u32, h: u32) -> wgpu::Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("depth"),
            size: wgpu::Extent3d { width: w.max(1), height: h.max(1), depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
    }

    pub fn resize(&mut self, w: u32, h: u32) {
        if w == 0 || h == 0 {
            return;
        }
        self.config.width = w;
        self.config.height = h;
        self.surface.configure(&self.device, &self.config);
        self.depth = Self::make_depth(&self.device, w, h);
        self.depth_view = self.depth.create_view(&wgpu::TextureViewDescriptor::default());
    }

    pub fn egui_ctx(&self) -> &egui::Context {
        self.egui_state.egui_ctx()
    }

    /// Full frame: egui triangulation, terrain pass, highlight, egui pass, present.
    pub fn render_frame(
        &mut self,
        world: &World,
        meshes: &ChunkGpuMap,
        eye: Vec3,
        dir: Vec3,
        highlight: Option<[i32; 3]>,
        hud: &crate::hud::Payload,
    ) -> Result<(), wgpu::SurfaceError> {
        // ---- egui input + triangulate (CPU side) ----
        let raw = self.egui_state.take_egui_input(&self.window);
        let ctx = self.egui_state.egui_ctx().clone();
        let mut out = ctx.run(raw, |ctx| crate::hud::paint(ctx, hud));
        let clipped = ctx.tessellate(std::mem::take(&mut out.shapes), self.ppp);
        let sd = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [self.config.width, self.config.height],
            pixels_per_point: self.ppp,
        };

        let st = self.surface.get_current_texture()?;
        let view = st.texture.create_view(&wgpu::TextureViewDescriptor::default());

        let vp = self.view_proj(eye, dir);
        let fd = world.radius as f32 * CW as f32;
        let uni = Uni {
            view_proj: vp.to_cols_array_2d(),
            cam_pos: [eye.x, eye.y, eye.z, 1.0],
            fog: [fd * 0.62, fd * 0.97, 0.0, 0.0],
            sky: [SKY.x, SKY.y, SKY.z, 0.0],
        };
        self.queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&uni));

        if let Some([bx, by, bz]) = highlight {
            let m = vp
                * Mat4::from_scale(Vec3::splat(1.008))
                * Mat4::from_translation(Vec3::new(
                    bx as f32 + 0.5,
                    by as f32 + 0.5,
                    bz as f32 + 0.5,
                ));
            self.queue
                .write_buffer(&self.hl_uniform, 0, bytemuck::bytes_of(&m.to_cols_array_2d()));
        }

        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("frame") });

        // egui GPU resources first (may emit its own command buffers)
        for (id, delta) in &out.textures_delta.set {
            self.egui_renderer.update_texture(&self.device, &self.queue, *id, delta);
        }
        let egui_cmd_bufs =
            self.egui_renderer
                .update_buffers(&self.device, &self.queue, &mut enc, &clipped, &sd);

        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("terrain"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: SKY.x as f64,
                            g: SKY.y as f64,
                            b: SKY.z as f64,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            let frustum = Frustum::from_view_proj(vp);
            self.draw_stream(&mut pass, meshes, false, &frustum, eye);
            self.draw_stream(&mut pass, meshes, true, &frustum, eye);

            if highlight.is_some() {
                pass.set_pipeline(&self.hl_pipeline);
                pass.set_bind_group(0, &self.bind_group, &[]);
                pass.set_vertex_buffer(0, self.hl_vb.slice(..));
                pass.draw(0..36, 0..1);
            }
        }

        // ---- egui pass: load the terrain output, no depth ----
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("egui"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            self.egui_renderer
                .render(&mut pass.forget_lifetime(), &clipped, &sd);
        }

        self.queue.submit(egui_cmd_bufs.into_iter().chain([enc.finish()]));
        st.present();
        for id in &out.textures_delta.free {
            self.egui_renderer.free_texture(id);
        }
        self.egui_state
            .handle_platform_output(&self.window, out.platform_output);
        Ok(())
    }

    fn view_proj(&self, eye: Vec3, dir: Vec3) -> Mat4 {
        let aspect = self.config.width as f32 / self.config.height.max(1) as f32;
        let proj = Mat4::perspective_rh(std::f32::consts::FRAC_PI_2 * 0.78, aspect, 0.08, 1200.0);
        proj * Mat4::look_to_rh(eye, dir, Vec3::Y)
    }

    fn draw_stream(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        meshes: &ChunkGpuMap,
        water: bool,
        frustum: &Frustum,
        eye: Vec3,
    ) {
        pass.set_pipeline(&self.terrain_pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        let map = meshes.map(water);
        // far-to-near painter order for water minimizes blend overdraw artifacts
        let mut order: Vec<usize> = (0..map.len()).collect();
        if water {
            order.sort_by(|&a, &b| {
                let da = eye.distance_squared(map[a].center());
                let db = eye.distance_squared(map[b].center());
                db.total_cmp(&da)
            });
        }
        let half = Vec3::new(CW as f32 / 2.0, CH as f32 / 2.0, CW as f32 / 2.0);
        for i in order {
            let m = &map[i];
            if m.count == 0 {
                continue;
            }
            if !frustum.aabb_overlap(m.center(), half) {
                continue;
            }
            pass.set_vertex_buffer(0, m.buf.slice(..));
            pass.draw(0..m.count * 6, 0..1);
        }
    }
}

/// Gribb-Hartmann frustum planes from a view-projection matrix.
/// Planes point inward; a point is inside when dot >= 0 for all six.
pub struct Frustum {
    planes: [[f32; 4]; 6],
}

impl Frustum {
    pub fn from_view_proj(m: Mat4) -> Self {
        let c = m.to_cols_array_2d(); // c[col][row]
        // rows of the matrix: r0..r3
        let row = |i: usize| [c[0][i], c[1][i], c[2][i], c[3][i]];
        let (r0, r1, r2, r3) = (row(0), row(1), row(2), row(3));
        let add = |a: [f32; 4], b: [f32; 4]| [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]];
        let sub = |a: [f32; 4], b: [f32; 4]| [a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]];
        let norm = |p: [f32; 4]| {
            let l = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
            [p[0] / l, p[1] / l, p[2] / l, p[3] / l]
        };
        Self {
            planes: [
                norm(add(r3, r0)), // left
                norm(sub(r3, r0)), // right
                norm(add(r3, r1)), // bottom
                norm(sub(r3, r1)), // top
                norm(r2),          // near
                norm(sub(r3, r2)), // far
            ],
        }
    }

    /// Conservative AABB test: reject only when the box lies fully outside a plane.
    pub fn aabb_overlap(&self, center: Vec3, half: Vec3) -> bool {
        for p in &self.planes {
            let r = p[0] * half.x + p[1] * half.y + p[2] * half.z;
            let d = p[0] * center.x + p[1] * center.y + p[2] * center.z + p[3];
            if d + r < 0.0 {
                return false;
            }
        }
        true
    }
}

/// Highlight cube: unit cube centered at origin, 36 CCW verts.
pub fn highlight_vertices() -> Vec<[f32; 3]> {
    use crate::mesh::FACES;
    let mut v = Vec::with_capacity(36);
    for f in FACES.iter() {
        for k in [0usize, 1, 2, 0, 2, 3] {
            let c = f.c[k];
            v.push([c[0] as f32 - 0.5, c[1] as f32 - 0.5, c[2] as f32 - 0.5]);
        }
    }
    v
}

/// One chunk's GPU vertex buffer.
pub struct ChunkBuf {
    pub origin: (i32, i32),
    pub buf: wgpu::Buffer,
    pub count: u32, // faces; 6 verts each
}

impl ChunkBuf {
    pub fn center(&self) -> Vec3 {
        Vec3::new(
            self.origin.0 as f32 * CW as f32 + CW as f32 / 2.0,
            CH as f32 / 2.0,
            self.origin.1 as f32 * CW as f32 + CW as f32 / 2.0,
        )
    }
}

pub struct ChunkGpuMap {
    pub opaque: Vec<ChunkBuf>,
    pub water: Vec<ChunkBuf>,
}

impl ChunkGpuMap {
    pub fn new() -> Self {
        Self { opaque: Vec::new(), water: Vec::new() }
    }
    pub fn map(&self, water: bool) -> &Vec<ChunkBuf> {
        if water { &self.water } else { &self.opaque }
    }
    pub fn upsert(&mut self, device: &wgpu::Device, origin: (i32, i32), water: bool, verts: &[Vertex]) {
        let map = if water { &mut self.water } else { &mut self.opaque };
        map.retain(|m| m.origin != origin);
        if verts.is_empty() {
            return;
        }
        let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("chunkv"),
            contents: bytemuck::cast_slice(verts),
            usage: wgpu::BufferUsages::VERTEX,
        });
        map.push(ChunkBuf { origin, buf, count: (verts.len() / 6) as u32 });
    }
    pub fn drop_chunk(&mut self, origin: (i32, i32)) {
        self.opaque.retain(|m| m.origin != origin);
        self.water.retain(|m| m.origin != origin);
    }
}
