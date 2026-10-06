//! GPU pipelines and buffers for instances and lines.

use std::mem::size_of;
use std::num::NonZeroU64;

use crate::engine::Body;
use bytemuck::{Pod, Zeroable};

use crate::ui::gpu::{CamUniforms, GpuBody};
use crate::ui::scene::{Instance, LineVert};
use crate::ui::theme::BG;

/// Viewport size for the vertex shader. Exactly 16 bytes.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Uniforms {
    /// Viewport width, in physical pixels.
    vp_w: f32,
    /// Viewport height, in physical pixels.
    vp_h: f32,
    /// Pad to the 16 byte uniform size. Never read.
    pad: [f32; 2],
}

// Two floats plus padding. Size growth would break the WGSL layout.
const _: () = assert!(size_of::<Uniforms>() == 16);

/// Size of one instance, in bytes.
const INSTANCE_SIZE: u64 = size_of::<Instance>() as u64;
/// Size of one line vertex, in bytes.
const LINE_VERT_SIZE: u64 = size_of::<LineVert>() as u64;
/// Size of one GPU body, in bytes.
const GPU_BODY_SIZE: u64 = size_of::<GpuBody>() as u64;

/// One frame's draw inputs. The pass clears `target`, egui paints after.
pub(crate) struct Frame<'a> {
    pub(crate) device: &'a wgpu::Device,
    pub(crate) queue: &'a wgpu::Queue,
    pub(crate) encoder: &'a mut wgpu::CommandEncoder,
    pub(crate) target: &'a wgpu::TextureView,
    /// Draw bodies from the GPU storage buffer instead of the instances.
    pub(crate) bodies_gpu: bool,
    pub(crate) instances: &'a [Instance],
    pub(crate) lines: &'a [LineVert],
    /// Target width, in physical pixels.
    pub(crate) w: f32,
    /// Target height, in physical pixels.
    pub(crate) h: f32,
}

/// Pipelines, buffers, and the uniform bind group.
pub(crate) struct Renderer {
    /// Pipeline for one quad per instance. Triangle list, six corners.
    pipeline_instances: wgpu::RenderPipeline,
    /// Pipeline for one quad per body, read from the storage buffer.
    pipeline_bodies: wgpu::RenderPipeline,
    /// Pipeline for flat colored line segments.
    pipeline_lines: wgpu::RenderPipeline,
    /// Shared uniform bind group. Both pipelines use group 0.
    bind_group: wgpu::BindGroup,
    /// Viewport uniform. Written once per frame.
    uniform_buf: wgpu::Buffer,
    /// Instance buffer. Grown only when a frame needs more space.
    instance_buf: wgpu::Buffer,
    /// Instance slots the instance buffer holds.
    instance_capacity: u32,
    /// Line vertex buffer. Grown only when a frame needs more space.
    line_buf: wgpu::Buffer,
    /// Line vertices the line buffer holds.
    line_capacity: u32,
    /// Layout for the vertex-pull bind group. Kept for buffer recreation.
    bgl_bodies: wgpu::BindGroupLayout,
    /// Camera uniform for the vertex-pull path. Written once per frame.
    cam_buf: wgpu::Buffer,
    /// GPU body storage. Grown only when a frame needs more space.
    bodies_buf: wgpu::Buffer,
    /// Body slots the storage buffer holds.
    bodies_capacity: u32,
    /// Bodies the last upload stored. Zero until the first upload.
    bodies_uploaded: u32,
    /// Bind group for the camera uniform and the body storage.
    bind_group_bodies: wgpu::BindGroup,
    /// Reused mirror of the body list. Filled by [`Self::upload_bodies`].
    gpu_scratch: Vec<GpuBody>,
    /// Bind group and body count when the physics keeps the bodies
    /// resident in its own buffer. The draw uses this instead of the
    /// uploaded copy.
    external_bodies: Option<(wgpu::BindGroup, u32)>,
}

impl Renderer {
    /// Builds both pipelines. `format` is the surface texture format.
    pub(crate) fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("particle.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("particle.wgsl").into()),
        });

        // One layout for both pipelines, so one bind group serves both draws.
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("uniform layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: Some(
                        NonZeroU64::new(size_of::<Uniforms>() as u64)
                            .expect("uniform size is not zero"),
                    ),
                },
                count: None,
            }],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("scene layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

        let targets = [Some(wgpu::ColorTargetState {
            format,
            blend: Some(wgpu::BlendState::ALPHA_BLENDING),
            write_mask: wgpu::ColorWrites::ALL,
        })];

        // One quad per instance: pos_radius at 0, color at 16.
        let instance_layout = wgpu::VertexBufferLayout {
            array_stride: INSTANCE_SIZE,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 0,
                    shader_location: 0,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 16,
                    shader_location: 1,
                },
            ],
        };
        let pipeline_instances = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("instances"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[Some(instance_layout)],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &targets,
            }),
            multiview_mask: None,
            cache: None,
        });

        // One quad per body, read from storage. No vertex buffers: the
        // body index comes from the instance index.
        let bodies_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bodies.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("bodies.wgsl").into()),
        });
        let bgl_bodies = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bodies layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: Some(
                            NonZeroU64::new(size_of::<CamUniforms>() as u64)
                                .expect("uniform size is not zero"),
                        ),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: Some(
                            NonZeroU64::new(GPU_BODY_SIZE).expect("body size is not zero"),
                        ),
                    },
                    count: None,
                },
            ],
        });
        let layout_bodies = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("bodies layout"),
            bind_group_layouts: &[Some(&bgl_bodies)],
            immediate_size: 0,
        });
        let pipeline_bodies = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("bodies vertex pull"),
            layout: Some(&layout_bodies),
            vertex: wgpu::VertexState {
                module: &bodies_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &targets,
            }),
            multiview_mask: None,
            cache: None,
        });

        // One expanded line vertex: xy at 0, color at 8.
        let line_layout = wgpu::VertexBufferLayout {
            array_stride: LINE_VERT_SIZE,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x2,
                    offset: 0,
                    shader_location: 0,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 8,
                    shader_location: 1,
                },
            ],
        };
        let pipeline_lines = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("lines"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_line"),
                compilation_options: Default::default(),
                buffers: &[Some(line_layout)],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_line"),
                compilation_options: Default::default(),
                targets: &targets,
            }),
            multiview_mask: None,
            cache: None,
        });

        // Seed both buffers with one slot. Draw grows them on demand.
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("viewport uniform"),
            size: size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("viewport bind group"),
            layout: &bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &uniform_buf,
                    offset: 0,
                    size: None,
                }),
            }],
        });
        let instance_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("instances"),
            size: INSTANCE_SIZE,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::VERTEX,
            mapped_at_creation: false,
        });
        let line_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("line vertices"),
            size: LINE_VERT_SIZE,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::VERTEX,
            mapped_at_creation: false,
        });
        let cam_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("camera uniform"),
            size: size_of::<CamUniforms>() as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: false,
        });
        // Seed the storage with one slot. The upload grows it on demand.
        let bodies_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu bodies"),
            size: GPU_BODY_SIZE,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let bind_group_bodies = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bodies bind group"),
            layout: &bgl_bodies,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &cam_buf,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &bodies_buf,
                        offset: 0,
                        size: None,
                    }),
                },
            ],
        });

        Self {
            pipeline_instances,
            pipeline_bodies,
            pipeline_lines,
            bind_group,
            uniform_buf,
            instance_buf,
            instance_capacity: 1,
            line_buf,
            line_capacity: 1,
            bgl_bodies,
            cam_buf,
            bodies_buf,
            bodies_capacity: 1,
            bodies_uploaded: 0,
            bind_group_bodies,
            gpu_scratch: Vec::new(),
            external_bodies: None,
        }
    }

    /// Uploads the camera uniform and the body storage for the vertex-pull
    /// path. The CPU stays authoritative: bodies move every step, so both
    /// buffers rewrite every frame. The storage grows only when needed.
    pub(crate) fn upload_bodies(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        cam: &CamUniforms,
        bodies: &[Body],
    ) {
        queue.write_buffer(&self.cam_buf, 0, bytemuck::bytes_of(cam));
        self.gpu_scratch.clear();
        self.gpu_scratch
            .extend(bodies.iter().map(GpuBody::from_body));
        self.bodies_uploaded = self.gpu_scratch.len() as u32;
        if self.gpu_scratch.len() as u64 > self.bodies_capacity as u64 {
            self.bodies_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("gpu bodies"),
                size: self.gpu_scratch.len() as u64 * GPU_BODY_SIZE,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            self.bodies_capacity = self.gpu_scratch.len() as u32;
            // The old bind group points at the dropped buffer. Rebuild it.
            self.bind_group_bodies = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bodies bind group"),
                layout: &self.bgl_bodies,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &self.cam_buf,
                            offset: 0,
                            size: None,
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &self.bodies_buf,
                            offset: 0,
                            size: None,
                        }),
                    },
                ],
            });
        }
        if !self.gpu_scratch.is_empty() {
            queue.write_buffer(&self.bodies_buf, 0, bytemuck::cast_slice(&self.gpu_scratch));
        }
    }

    /// Uploads one vertex buffer. Recreates it only on growth.
    fn upload(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        buffer: &mut wgpu::Buffer,
        capacity: &mut u32,
        element_size: u64,
        label: &str,
        data: &[u8],
    ) {
        let slots = (data.len() as u64 / element_size) as u32;
        if slots > *capacity {
            *buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: data.len() as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::VERTEX,
                mapped_at_creation: false,
            });
            *capacity = slots;
        }
        queue.write_buffer(buffer, 0, data);
    }

    /// Writes the camera uniform for the vertex-pull path.
    pub(crate) fn write_cam(&mut self, queue: &wgpu::Queue, cam: &CamUniforms) {
        queue.write_buffer(&self.cam_buf, 0, bytemuck::bytes_of(cam));
    }

    /// Points the vertex-pull draw at the physics body buffer. The
    /// physics stays resident, so the uploaded copy would draw stale
    /// positions.
    pub(crate) fn use_external_bodies(
        &mut self,
        device: &wgpu::Device,
        buffer: &wgpu::Buffer,
        n: u32,
    ) {
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("external bodies bind group"),
            layout: &self.bgl_bodies,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.cam_buf,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer,
                        offset: 0,
                        size: None,
                    }),
                },
            ],
        });
        self.external_bodies = Some((group, n));
    }

    /// Drops the external override. The physics left the GPU.
    pub(crate) fn clear_external_bodies(&mut self) {
        self.external_bodies = None;
    }

    /// Records the instance pass and the line pass into the encoder.
    /// The pass clears `target` to the background color. egui paints after.
    /// Sizes are physical pixels.
    pub(crate) fn draw(&mut self, frame: Frame<'_>) {
        let Frame {
            device,
            queue,
            encoder,
            target,
            bodies_gpu,
            instances,
            lines,
            w,
            h,
        } = frame;
        let (vp_w, vp_h) = (w, h);
        let uniforms = Uniforms {
            vp_w,
            vp_h,
            pad: [0.0; 2],
        };
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&uniforms));

        if !instances.is_empty() {
            Self::upload(
                device,
                queue,
                &mut self.instance_buf,
                &mut self.instance_capacity,
                INSTANCE_SIZE,
                "instances",
                bytemuck::cast_slice(instances),
            );
        }
        if !lines.is_empty() {
            Self::upload(
                device,
                queue,
                &mut self.line_buf,
                &mut self.line_capacity,
                LINE_VERT_SIZE,
                "line vertices",
                bytemuck::cast_slice(lines),
            );
        }

        // The clear color comes from the one shared background color.
        let clear = wgpu::Color {
            r: BG.r() as f64 / 255.0,
            g: BG.g() as f64 / 255.0,
            b: BG.b() as f64 / 255.0,
            a: 1.0,
        };
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("particles"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(clear),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        if bodies_gpu {
            if let Some((group, n)) = &self.external_bodies {
                if *n > 0 {
                    pass.set_pipeline(&self.pipeline_bodies);
                    pass.set_bind_group(0, group, &[]);
                    // Six corners per quad come from vertex_index, one
                    // body per instance comes from instance_index.
                    pass.draw(0..6, 0..*n);
                }
            } else if self.bodies_uploaded > 0 {
                pass.set_pipeline(&self.pipeline_bodies);
                pass.set_bind_group(0, &self.bind_group_bodies, &[]);
                pass.draw(0..6, 0..self.bodies_uploaded);
            }
        } else if !instances.is_empty() {
            pass.set_pipeline(&self.pipeline_instances);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.set_vertex_buffer(0, self.instance_buf.slice(..));
            // Six corners per quad come from vertex_index. No index buffer.
            pass.draw(0..6, 0..instances.len() as u32);
        }
        if !lines.is_empty() {
            pass.set_pipeline(&self.pipeline_lines);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.set_vertex_buffer(0, self.line_buf.slice(..));
            // Scene emits six expanded vertices per segment.
            pass.draw(0..lines.len() as u32, 0..1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::Shape;
    use crate::ui::camera::Camera;
    use crate::ui::scene::Instance;

    /// Builds one offscreen device. Returns `None` without an adapter.
    fn headless_device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .ok()?;
        Some(
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::default(),
                required_limits: wgpu::Limits::default(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::default(),
                trace: wgpu::Trace::Off,
            }))
            .expect("device request fails"),
        )
    }

    /// Builds one 256 by 256 render target view.
    fn test_target(device: &wgpu::Device) -> wgpu::TextureView {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("test target"),
            size: wgpu::Extent3d {
                width: 256,
                height: 256,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Bgra8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        texture.create_view(&wgpu::TextureViewDescriptor::default())
    }

    /// Builds the renderer and draws one instance offscreen.
    /// Skips when the machine has no wgpu adapter.
    #[test]
    fn offscreen_draw_passes() {
        let Some((device, queue)) = headless_device() else {
            eprintln!("no wgpu adapter; skipping");
            return;
        };
        let mut renderer = Renderer::new(&device, wgpu::TextureFormat::Bgra8Unorm);
        let view = test_target(&device);

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        let dot = Instance {
            x: 128.0,
            y: 128.0,
            radius: 16.0,
            shape: 0.0,
            color: [1.0, 0.2, 0.1, 1.0],
        };
        renderer.draw(Frame {
            device: &device,
            queue: &queue,
            encoder: &mut encoder,
            target: &view,
            bodies_gpu: false,
            instances: &[dot],
            lines: &[],
            w: 256.0,
            h: 256.0,
        });
        queue.submit([encoder.finish()]);
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll fails");
    }

    /// Uploads two bodies and draws them through the storage buffer.
    /// Skips when the machine has no wgpu adapter.
    #[test]
    fn offscreen_gpu_bodies_draw_passes() {
        let Some((device, queue)) = headless_device() else {
            eprintln!("no wgpu adapter; skipping");
            return;
        };
        let mut renderer = Renderer::new(&device, wgpu::TextureFormat::Bgra8Unorm);
        let view = test_target(&device);

        let cam = Camera {
            target: [0.0; 3],
            yaw: 0.0,
            pitch: 0.0,
            dist: 12.0,
        };
        let cam_uniforms = CamUniforms::new(&cam, 256.0, 256.0);
        let sphere = Body {
            pos: [0.0, 0.0, 0.0],
            vel: [0.0; 3],
            radius: 0.1,
            shape: Shape::Sphere,
        };
        let cube = Body {
            pos: [0.5, 0.0, 0.0],
            shape: Shape::Cube,
            ..sphere
        };

        renderer.upload_bodies(&device, &queue, &cam_uniforms, &[sphere, cube]);
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        renderer.draw(Frame {
            device: &device,
            queue: &queue,
            encoder: &mut encoder,
            target: &view,
            bodies_gpu: true,
            instances: &[],
            lines: &[],
            w: 256.0,
            h: 256.0,
        });
        queue.submit([encoder.finish()]);
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll fails");
    }
}
