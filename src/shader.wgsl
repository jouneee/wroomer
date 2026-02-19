struct CameraUniform {
    zoom: f32,
    offset: vec2<f32>,
};
@group(1) @binding(0)
var<uniform> camera: CameraUniform;

struct FlUniform {
    cur_pos: vec2<f32>,
    radius: f32,
    alpha: f32
};
@group(2) @binding(0)
var<uniform> flashlight: FlUniform;


struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) tex_coords: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
}

@vertex
fn vs_main(
    tex: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;

    var pos = tex.position.xy * camera.zoom + camera.offset;

    out.tex_coords = tex.tex_coords;
    out.clip_position = vec4<f32>(pos, tex.position.z, 1.0);
    return out;
}

@group(0) @binding(0)
var t_diffuse: texture_2d<f32>;
@group(0) @binding(1)
var s_diffuse: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let screen = textureSample(t_diffuse, s_diffuse, in.tex_coords);

    let dist = length(flashlight.cur_pos - in.clip_position.xy);
    let radius = flashlight.radius * camera.zoom;
    
    let is_inside = dist < radius;
    let mix_factor = select(flashlight.alpha, 0.0, is_inside);
    
    let color = mix(screen, vec4<f32>(0.0, 0.0, 0.0, 1.0), mix_factor);
    return color;
}