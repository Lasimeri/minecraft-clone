// Terrain pipeline: world-baked vertices, atlas sampling, AO x face x sky light, distance fog.

struct Uni {
    view_proj: mat4x4<f32>,
    cam_pos: vec4<f32>,
    fog: vec4<f32>,   // x: fog_start, y: fog_end, zw: reserved
    sky: vec4<f32>,   // rgb: sky color, w: time
};

@group(0) @binding(0) var<uniform> uni: Uni;
@group(0) @binding(1) var atlas: texture_2d<f32>;
@group(0) @binding(2) var atlas_samp: sampler;

struct VSIn {
    @location(0) pos: vec4<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) light: vec2<f32>, // x: face_brightness * ao, y: sky
};

struct VSOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) light: vec2<f32>,
};

@vertex
fn vs(v: VSIn) -> VSOut {
    var o: VSOut;
    o.clip = uni.view_proj * v.pos;
    o.world = v.pos.xyz;
    o.uv = v.uv;
    o.light = v.light;
    return o;
}

@fragment
fn fs(f: VSOut) -> @location(0) vec4<f32> {
    let c = textureSample(atlas, atlas_samp, f.uv);
    if (c.a < 0.05) { discard; }
    var rgb = c.rgb * f.light.x * f.light.y;
    // cheap hemispheric ambient: downward faces get a touch of sky bounce
    let d = distance(f.world, uni.cam_pos.xyz);
    let fog_t = clamp((d - uni.fog.x) / max(uni.fog.y - uni.fog.x, 1.0), 0.0, 1.0);
    rgb = mix(rgb, uni.sky.rgb, fog_t * fog_t);
    return vec4<f32>(rgb, c.a);
}

// Highlight pass: pulsing wireframe-ish box around the targeted block.
@group(0) @binding(3) var<uniform> hl_mvp: mat4x4<f32>;

@vertex
fn vs_hl(@location(0) p: vec3<f32>) -> @builtin(position) vec4<f32> {
    return hl_mvp * vec4<f32>(p, 1.0);
}

@fragment
fn fs_hl() -> @location(0) vec4<f32> {
    return vec4<f32>(0.03, 0.03, 0.03, 0.55);
}
