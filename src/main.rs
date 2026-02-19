use core::f32;
use std::sync::OnceLock;
use std::{f32::consts::PI, sync::Arc, env};
use std::time::Instant;
use wgpu::util::DeviceExt;
use winit::{
    application::ApplicationHandler, dpi::PhysicalPosition, event::*, event_loop::{
        ActiveEventLoop, EventLoop
    }, keyboard::{
        KeyCode, PhysicalKey
    }, window::{Fullscreen, Window}
};

use image::RgbaImage;
use xcap::Monitor;
#[cfg(target_os="linux")]
use libwayshot::WayshotConnection;

static SCREENSHOT: OnceLock<RgbaImage> = OnceLock::new();

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    tex_coords: [f32; 2],
}

impl Vertex {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                },
            ]
        }
    }
}

const SCREEN_VERTICES: &[Vertex] = &[
    Vertex { position: [-1.0, 1.0, 0.0], tex_coords: [0.0, 0.0] },
    Vertex { position: [-1.0, -1.0, 0.0],tex_coords: [0.0, 1.0] },
    Vertex { position: [1.0, -1.0, 0.0], tex_coords: [1.0, 1.0] },
    Vertex { position: [-1.0, 1.0, 0.0], tex_coords: [0.0, 0.0] },
    Vertex { position: [1.0, -1.0, 0.0], tex_coords: [1.0, 1.0] },
    Vertex { position: [1.0, 1.0, 0.0],  tex_coords: [1.0, 0.0] },
];

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    zoom: f32,
    _padding: f32,
    offset: [f32; 2],
}

impl CameraUniform {
    fn new() -> Self {
        Self {
            zoom: 1.0,
            _padding: 0.0,
            offset: [0.0, 0.0],
        }
    }
    
    fn update_buffer(&self, _device: &wgpu::Device, queue: &wgpu::Queue, buffer: &wgpu::Buffer) {
        queue.write_buffer(buffer, 0, bytemuck::cast_slice(&[*self]));
    }
}


#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct FlUniform {
    pos: [f32; 2],
    radius: f32,
    alpha: f32,
}

impl FlUniform {
    fn new() -> Self {
        Self {
            pos: [0.0, 0.0],
            radius: 200.0,
            alpha: 0.0,
        }
    }

    fn update_buffer(&self, _device: &wgpu::Device, queue: &wgpu::Queue, buffer: &wgpu::Buffer) {
        queue.write_buffer(buffer, 0, bytemuck::cast_slice(&[*self]));
    }
}

struct State {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu:: Queue,
    config: wgpu::SurfaceConfiguration,
    render_pipeline: wgpu::RenderPipeline,
    window: Arc<Window>,
    screen_vertex_buffer: wgpu::Buffer,
    num_vertices: u32,
    diffuse_bind_group: wgpu::BindGroup,
    is_surface_configured: bool,
    camera: CameraUniform,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    flashlight: FlUniform,
    flashlight_buffer: wgpu::Buffer,
    flashlight_bind_group: wgpu::BindGroup,
    is_mouse_down: bool,
    enable_flashlight: bool,
    last_mouse_position: [f64; 2],
    normalized_mouse_coords: [f32; 2],
    velocity: [f32; 2],
    delta_scale: f32,
    scale_pivot: [f32; 2],
    last_frame_time: Instant,
}

impl State {
    async fn new(window: Arc<Window>) -> anyhow::Result<Self> {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(
            &wgpu::InstanceDescriptor {
                backends: wgpu::Backends::PRIMARY,
                ..Default::default()
            }
        );

        let surface = instance.create_surface(window.clone()).unwrap();

        let adapter = instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            }
        ).await?;

        let (device, queue) = adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                required_limits: wgpu::Limits::default(),
                memory_hints: Default::default(),
                trace: wgpu::Trace::Off,
            }
        ).await?;

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps.formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        let screen_vertex_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Vertex Buffer"),
                contents: bytemuck::cast_slice(SCREEN_VERTICES),
                usage: wgpu::BufferUsages::VERTEX,
            }
        );

        let num_vertices = SCREEN_VERTICES.len() as u32;

        let dimensions = SCREENSHOT.get().unwrap().dimensions();
        
        let texture_size = wgpu::Extent3d {
            width: dimensions.0,
            height: dimensions.1,
            depth_or_array_layers: 1,
        };

        let diffuse_texture = device.create_texture(
            &wgpu::TextureDescriptor {
                size: texture_size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                label: Some("Diffuse Texture"),
                view_formats: &[],
            }
        );

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &diffuse_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            }, 
            &SCREENSHOT.get().unwrap(), 
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * dimensions.0),
                rows_per_image: Some(dimensions.1),
            }, 
            texture_size,
        );

        let diffuse_texture_view = diffuse_texture.create_view(
            &wgpu::TextureViewDescriptor::default(),
        );
        let diffuse_sampler = device.create_sampler(
            &wgpu::SamplerDescriptor {
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Nearest,
                min_filter: wgpu::FilterMode::Nearest,
                mipmap_filter: wgpu::FilterMode::Nearest,
                ..Default::default()
            }
        );

        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
                label: Some("texture_bind_group_layout"),
            });

        let diffuse_bind_group = device.create_bind_group(
        &wgpu::BindGroupDescriptor {
                layout: &texture_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&diffuse_texture_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&diffuse_sampler),
                    }
                ],
                label: Some("diffuse_bind_group"),
            }
        );

        let camera_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Camera Buffer"),
                contents: bytemuck::cast_slice(&[CameraUniform::new()]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            }
        );

        let camera_bind_group_layout = device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
                label: Some("camera_bind_group_layout"),
            }
        );

        let camera_bind_group = device.create_bind_group(
            &wgpu::BindGroupDescriptor {
                layout: &camera_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: camera_buffer.as_entire_binding(),
                    },
                ],
                label: Some("camera_bind_group"),
            }
        );

        let flashlight_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Fl Buffer"),
                contents: bytemuck::cast_slice(&[FlUniform::new()]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            }
        );

        let flashlight_bind_group_layout = device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
                label: None,
            }
        );

        let flashlight_bind_group = device.create_bind_group(
            &wgpu::BindGroupDescriptor {
                layout: &flashlight_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: flashlight_buffer.as_entire_binding(),
                    },
                ],
                label: Some("flashlight_bind_group"),
            }
        );

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });
        
        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[
                    &texture_bind_group_layout,
                    &camera_bind_group_layout,
                    &flashlight_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[
                    Vertex::desc(),
                ],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });


        Ok(Self {
            surface,
            device,
            queue,
            config,
            render_pipeline,
            window,
            screen_vertex_buffer,
            num_vertices,
            diffuse_bind_group,
            is_surface_configured: false,
            camera: CameraUniform::new(),
            camera_buffer,
            camera_bind_group,
            flashlight: FlUniform::new(),
            flashlight_buffer,
            flashlight_bind_group,
            is_mouse_down: false,
            enable_flashlight: false,
            last_mouse_position: [0.0, 0.0],
            normalized_mouse_coords: [0.0, 0.0],
            velocity: [0.0, 0.0],
            delta_scale: 0.0,
            scale_pivot: [0.0, 0.0],
            last_frame_time: Instant::now(),
        })
    }

    fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.device, &self.config);
            self.is_surface_configured = true;
        }
    }

    fn update(&mut self) {
        let frame_time = self.last_frame_time.elapsed().as_secs_f32();
        self.last_frame_time = Instant::now();
        const VELOCITY_THRESHOLD: f32 = 0.0001;
        const DRAG_FRICTION: f32 = 2.0;
        const MIN_ZOOM: f32 = 0.001;
        const MAX_ZOOM: f32 = 10.0;
        const ZOOM_FRICTION: f32 = 8.0;

        if self.is_mouse_down {
            self.velocity = [0.0, 0.0];
        } else {
            if self.velocity[0].abs() > VELOCITY_THRESHOLD {
                self.camera.offset[0] += self.velocity[0] * frame_time;
                self.velocity[0]      -= self.velocity[0] * DRAG_FRICTION * frame_time;
            }
            
            if self.velocity[1].abs() > VELOCITY_THRESHOLD {
                self.camera.offset[1] += self.velocity[1] * frame_time;
                self.velocity[1]      -= self.velocity[1] * DRAG_FRICTION * frame_time;
            }
        }
        
        if (self.delta_scale).abs() > 0.001 {
            if self.enable_flashlight {
                let old_r = self.flashlight.radius;
                let step = self.delta_scale / self.camera.zoom * frame_time;
                let new_r = old_r - step;
                
                let area_diff = (old_r * old_r - new_r * new_r).abs() * PI;
                let safe_diff = area_diff.max(0.0001);
                let log_val = safe_diff.log2();
                let a = log_val * log_val * log_val * log_val;
                
                let change = step * a;
                self.flashlight.radius = (old_r - change).clamp(10.0, 2000.0);
                
                self.delta_scale *= (1.0 - ZOOM_FRICTION * frame_time).max(0.0);
            } else {
                let old_zoom = self.camera.zoom;
                self.camera.zoom += self.delta_scale * frame_time;
                self.camera.zoom = self.camera.zoom.clamp(MIN_ZOOM, MAX_ZOOM);
                let new_zoom = self.camera.zoom;
    
                if (new_zoom - old_zoom).abs() > 0.0 {
                    let f = new_zoom / old_zoom;
                    self.camera.offset[0] = self.camera.offset[0] * f + self.scale_pivot[0] as f32 * (1.0 - f);
                    self.camera.offset[1] = self.camera.offset[1] * f + self.scale_pivot[1] as f32 * (1.0 - f);
                }
    
                self.delta_scale *= (1.0 - ZOOM_FRICTION * frame_time).max(0.0);
            }
            if self.delta_scale.abs() < 0.001 {
                self.delta_scale = 0.0;
            }
        }
        self.camera.update_buffer(&self.device, &self.queue, &self.camera_buffer);
        self.flashlight.update_buffer(&self.device, &self.queue, &self.flashlight_buffer);
    }

    fn handle_cursor_moved(&mut self, position: PhysicalPosition<f64>) {
        self.flashlight.pos = [position.x as f32, position.y as f32];
        let window_size = self.window.inner_size();
        let frame_time = self.last_frame_time.elapsed().as_secs_f32();
        let frame_time = if frame_time > 0.0 { frame_time } else { 0.01 };
    
        self.normalized_mouse_coords[0] = position.x as f32 / (window_size.width as f32 * 0.5) - 1.0;
        self.normalized_mouse_coords[1] = 1.0 - (position.y as f32 / (window_size.height as f32 * 0.5));
        let last_normalized_mouse_coords_x = self.last_mouse_position[0] as f32 / (window_size.width as f32 * 0.5) - 1.0;
        let last_normalized_mouse_coords_y = 1.0 - (self.last_mouse_position[1] as f32 / (window_size.height as f32 * 0.5));

        if self.is_mouse_down {
            let delta_x = (self.normalized_mouse_coords[0] - last_normalized_mouse_coords_x) as f32;
            let delta_y = (self.normalized_mouse_coords[1] - last_normalized_mouse_coords_y) as f32;

            let travel_x = (position.x - self.last_mouse_position[0]).abs() as f32;
            let travel_y = (position.y - self.last_mouse_position[1]).abs() as f32;
            let speed_x = travel_x / frame_time * 0.5;
            let speed_y = travel_y / frame_time * 0.5;

            self.velocity[0] = delta_x * speed_x;
            self.velocity[1] = delta_y * speed_y;

            self.camera.offset[0] += delta_x;
            self.camera.offset[1] += delta_y;
        }
    
        self.last_mouse_position = [position.x, position.y];
    }
    
    fn handle_mouse_wheel(&mut self, delta: MouseScrollDelta) {
        const ZOOM_SPEED: f32 = 3.0;
        let scroll_amount = match delta {
            MouseScrollDelta::LineDelta(_, y) => y as f32,
            MouseScrollDelta::PixelDelta(pos) => pos.y as f32,
        };
        let direction = if scroll_amount > 0.0 { 1.0 } else { -1.0 };
        
        self.delta_scale = direction * ZOOM_SPEED * self.camera.zoom;
        self.scale_pivot = self.normalized_mouse_coords;
    }

    fn handle_keyboard_input(&mut self, event_loop: &ActiveEventLoop, code: KeyCode, is_pressed: bool) {
        if code == KeyCode::Escape && is_pressed {
            event_loop.exit();
        } else 
        if code == KeyCode::ShiftLeft {
            self.enable_flashlight = is_pressed;
            self.flashlight.alpha = if is_pressed { 0.75 } else { 0.0 };
        } 
    }

    fn handle_mouse_input(&mut self, event_loop: &ActiveEventLoop, button: MouseButton, state: ElementState) {
        if button == MouseButton::Left {
            self.is_mouse_down = state == ElementState::Pressed;
        } else 
        if button == MouseButton::Right {
            event_loop.exit()
        }
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        self.window.request_redraw();

        if !self.is_surface_configured {
            return Ok(());
        }

        let output = self.surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder")
            });
        {
            let mut render_pass = encoder.begin_render_pass(
                &wgpu::RenderPassDescriptor {
                    label: Some("Render Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.0,
                                g: 0.0,
                                b: 0.0,
                                a: 0.25,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    occlusion_query_set: None,
                    timestamp_writes: None,
                }
            );

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.diffuse_bind_group, &[]);
            render_pass.set_bind_group(1, &self.camera_bind_group, &[]);
            render_pass.set_bind_group(2, &self.flashlight_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.screen_vertex_buffer.slice(..));

            render_pass.draw(0..self.num_vertices, 0..1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

pub struct App {
    state: Option<State>,
}

impl App {
    pub fn new() -> Self {
        Self {
            state: None,
        }
    }
}

impl ApplicationHandler<State> for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        #[allow(unused_mut)]
        let fullscreen = Some(Fullscreen::Borderless(None));
        let window_attributes = Window::default_attributes().with_transparent(true).with_fullscreen(fullscreen);
        let window = Arc::new(event_loop.create_window(window_attributes).unwrap());

        self.state = Some(pollster::block_on(State::new(window)).unwrap());
    }

    #[allow(unused_mut)]
    fn user_event(&mut self, _event_loop: &ActiveEventLoop, mut event: State) {
        self.state = Some(event);
    }

    fn window_event(
        &mut self, 
        event_loop: &ActiveEventLoop, 
        _window_id: winit::window::WindowId, 
        event: WindowEvent) 
    {
        let state = match &mut self.state {
            Some(canvas) => canvas,
            None => return,
        };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => state.resize(size.width, size.height),
            WindowEvent::KeyboardInput { 
                event: KeyEvent {
                    physical_key: PhysicalKey::Code(code),
                    state: key_state,
                    ..
                },
                ..
            } => state.handle_keyboard_input(event_loop, code, key_state.is_pressed()),
            WindowEvent::MouseInput { 
                button, 
                state: mouse_state, .. 
            } => state.handle_mouse_input(event_loop, button, mouse_state),
            WindowEvent::CursorMoved { 
                position, 
                .. 
            } => state.handle_cursor_moved(position),
            WindowEvent::MouseWheel { 
                delta, 
                .. 
            } => state.handle_mouse_wheel(delta),
            WindowEvent::RedrawRequested => {
                state.update();
                match state.render() {
                    Ok(_) => {}
                    Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                        let size = state.window.inner_size();
                        state.resize(size.width, size.height);
                    }
                    Err(e) => {
                        eprintln!("Unable to render {}", e);
                    }
                }
            },
            _ => {}
        }
    }
}

pub fn take_screenshot(monitor: Option<String>) {
    let is_wayland = std::env::var("WAYLAND_DISPLAY").is_ok();
    
    if is_wayland {
        #[cfg(target_os="linux")]
        {
            let wayshot_connection = Some(WayshotConnection::new()
                .expect("Failed to connect to wayland"));
            let wayshot_connection = wayshot_connection.unwrap();
            let outputs = wayshot_connection.get_all_outputs();
            if outputs.is_empty() {
                eprintln!("No outputs found.");
                std::process::exit(1);
            }
            
            let idx = match monitor {
                None => 0,
                Some(ref name) => outputs
                    .iter()
                    .position(|out| &out.name == name)
                    .unwrap_or_else(|| {
                        eprintln!("Output {} was not found.", name);
                        std::process::exit(1);
                    })
            };
        
            let sel_mon = &outputs[idx];
        
            SCREENSHOT.get_or_init(||
                wayshot_connection
                    .screenshot_single_output(sel_mon,false)
                    .expect("Failed to take a screenshot")
                    .to_rgba8()
            );
        }
    }
    else {
        let outputs = Monitor::all().unwrap();
        if outputs.is_empty() {
            eprintln!("No outputs found.");
            std::process::exit(1);
        };

        let idx = match monitor {
            None => 0,
            Some(ref name) => outputs
                .iter()
                .position(|out| &out.name().unwrap() == name)
                .unwrap_or_else(|| {
                    eprintln!("Output {} was not found.", name);
                    std::process::exit(1);
                })
        };

        let sel_mon = &outputs[idx];
        SCREENSHOT.get_or_init(||
            sel_mon.capture_image().unwrap()
        );
    }
}

fn run() -> anyhow::Result<()> {
    let event_loop = EventLoop::with_user_event().build()?;
    let mut app = App::new();
    event_loop.run_app(&mut app)?;

    Ok(())
}

fn main() {
    let mut args = env::args();
    let _program = args.next().unwrap();
    let monitor = args.next();
    take_screenshot(monitor);
    let _ = run();
}