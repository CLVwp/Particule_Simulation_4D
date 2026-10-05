//! GPU pipelines and buffers for instances and lines.

use std::mem::size_of;
use std::num::NonZeroU64;

use bytemuck::{Pod, Zeroable};

use crate::ui::scene::{Instance, LineVert};

/// Clear color of the scene pass. Dark blue-black, opaque.
const CLEAR_COLOR: wgpu::Color = wgpu::Color {
    r: 0x0b as f64 / 255.0,
    g: 0x0e as f64 / 255.0,
    b: 0x14 as f64 / 255.0,
    a: 1.0,
};

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

/// Pipelines, buffers, and the uniform bind group.
pub(crate) struct Renderer {
    /// Pipeline for one quad per instance. Triangle list, six corners.
    pipeline_instances: wgpu::RenderPipeline,
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
    /// Last recorded width, in physical pixels.
    width: u32,
    /// Last recorded height, in physical pixels.
    height: u32,
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

        Self {
            pipeline_instances,
            pipeline_lines,
            bind_group,
            uniform_buf,
            instance_buf,
            instance_capacity: 1,
            line_buf,
            line_capacity: 1,
            width: 0,
            height: 0,
        }
    }

    /// Records the render size. Sizes are physical pixels.
    /// Keeps no swapchain. Nothing to rebuild.
    pub(crate) fn resize(&mut self, _device: &wgpu::Device, w: u32, h: u32) {
        if self.width == w && self.height == h {
            return;
        }
        self.width = w;
        self.height = h;
    }

    /// Picks the viewport for the pass. Falls back to the recorded size
    /// when the frame size is not positive.
    fn viewport(&self, w: f32, h: f32) -> (f32, f32) {
        if w > 0.0 && h > 0.0 {
            (w, h)
        } else {
            (self.width as f32, self.height as f32)
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

    /// Records the instance pass and the line pass into `encoder`.
    /// The pass clears `target` to the background color. egui paints after.
    /// Sizes are physical pixels.
    #[expect(clippy::too_many_arguments)] // the plan fixes this draw signature
    pub(crate) fn draw(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        instances: &[Instance],
        lines: &[LineVert],
        w: f32,
        h: f32,
    ) {
        let (vp_w, vp_h) = self.viewport(w, h);
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

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("particles"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(CLEAR_COLOR),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        if !instances.is_empty() {
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
    use crate::ui::scene::Instance;

    /// Builds the renderer and draws one instance offscreen.
    /// Skips when the machine has no wgpu adapter.
    #[test]
    fn offscreen_draw_passes() {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }));
        let Ok(adapter) = adapter else {
            eprintln!("no wgpu adapter; skipping");
            return;
        };
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: None,
            required_features: wgpu::Features::default(),
            required_limits: wgpu::Limits::default(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::default(),
            trace: wgpu::Trace::Off,
        }))
        .expect("device request fails");

        let mut renderer = Renderer::new(&device, wgpu::TextureFormat::Bgra8Unorm);
        renderer.resize(&device, 256, 256);

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
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        let dot = Instance {
            x: 128.0,
            y: 128.0,
            radius: 16.0,
            shape: 0.0,
            color: [1.0, 0.2, 0.1, 1.0],
        };
        renderer.draw(
            &device,
            &queue,
            &mut encoder,
            &view,
            &[dot],
            &[],
            256.0,
            256.0,
        );
        queue.submit([encoder.finish()]);
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll fails");
    }
}
