//! Window loop: wgpu init, the winit event handler, and the frame render.

use std::sync::Arc;

use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::Key;
use winit::window::{Window, WindowId};

use crate::ui::camera::Camera;
use crate::ui::gpu::{CamUniforms, GpuBody, GpuState, SimUniforms, TABLE_MASK};
use crate::ui::input::{Drag, apply_drag, zoom};
use crate::ui::renderer::{Frame, Renderer};
use crate::ui::scene::SceneOut;
use crate::ui::theme::{SMOOTH_KEEP, SMOOTH_NEW};
use crate::ui::{App, FIXED_DT, Page, PhysicsMode};

/// Title of the app window.
const WINDOW_TITLE: &str = "Particule Simulation 4D";

/// All GPU, window, and UI state. Set once in [`Handler::resumed`].
struct WindowState {
    /// The app window.
    window: Arc<Window>,
    /// The draw target. Holds its own window handle clone.
    surface: wgpu::Surface<'static>,
    /// Current surface format and size.
    config: wgpu::SurfaceConfiguration,
    /// The wgpu instance. Lives with the surface.
    device: wgpu::Device,
    /// Frame upload queue.
    queue: wgpu::Queue,
    /// egui input state. Owns the egui context.
    egui: egui_winit::State,
    /// egui pipelines and textures.
    egui_paint: egui_wgpu::Renderer,
    /// Particle and line pipelines.
    renderer: Renderer,
    /// Scene output. The vectors stay allocated between frames.
    scene: SceneOut,
    /// Physical window size. Matches the surface config.
    size: (u32, u32),
    /// True while the shift key is held.
    shift: bool,
    /// The GPU physics state. Buffers and pipelines only; the step runs
    /// when the residency handshake below says so.
    gpu: GpuState,
    /// True while the bodies live in the GPU buffer.
    gpu_resident: bool,
    /// Body count the GPU buffer holds.
    gpu_n: usize,
    /// Set by the first device error. The physics falls back to the CPU
    /// for the rest of the session.
    gpu_failed: bool,
    /// Phase timestamps, when the adapter supports the queries.
    ts_set: Option<wgpu::QuerySet>,
    /// Resolved timestamp ticks for the last recorded frame.
    ts_resolve: wgpu::Buffer,
    /// Mapped copy of the resolved ticks.
    ts_stage: wgpu::Buffer,
    /// Nanoseconds per timestamp tick.
    ts_period: f32,
}

impl WindowState {
    /// Builds the window, the GPU, and the egui state. Stores the adapter
    /// name into the app.
    ///
    /// # Errors
    /// Returns the failure reason when the window, the adapter, or the
    /// device cannot be created.
    fn init(event_loop: &ActiveEventLoop, app: &mut App) -> Result<Self, String> {
        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes().with_title(WINDOW_TITLE))
                .map_err(|error| error.to_string())?,
        );
        let instance = wgpu::Instance::default();
        let surface = instance
            .create_surface(window.clone())
            .map_err(|error| error.to_string())?;
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        }))
        .map_err(|error| error.to_string())?;
        let info = adapter.get_info();
        app.adapter_info = format!("{} ({:?})", info.name, info.backend);
        eprintln!("adapter: {}", app.adapter_info);

        // The phase timestamps ride along when the adapter has them.
        let wants_ts = adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY);
        let required = if wants_ts {
            wgpu::Features::TIMESTAMP_QUERY
        } else {
            wgpu::Features::empty()
        };
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("main device"),
            required_features: required,
            required_limits: wgpu::Limits::default(),
            experimental_features: wgpu::ExperimentalFeatures::default(),
            memory_hints: wgpu::MemoryHints::default(),
            trace: wgpu::Trace::default(),
        }))
        .map_err(|error| error.to_string())?;
        let gpu = GpuState::new(&device);
        let ts_set = GpuState::timestamp_set(&device);
        let ts_resolve = GpuState::timestamp_resolve(&device);
        let ts_stage = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("physics phase staging"),
            size: 6 * 8,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let ts_period = queue.get_timestamp_period();

        let raw = window.inner_size();
        let size = (raw.width.max(1), raw.height.max(1));
        let mut config = surface
            .get_default_config(&adapter, size.0, size.1)
            .ok_or("the adapter does not support the surface")?;
        config.present_mode = wgpu::PresentMode::Fifo;
        config.alpha_mode = wgpu::CompositeAlphaMode::Opaque;
        // Colors are tuned for linear targets, like the old D3D11 stack. Prefer
        // a non-sRGB format when one is supported.
        let supported = surface.get_capabilities(&adapter).formats;
        let linear = supported
            .iter()
            .find(|f| {
                matches!(
                    **f,
                    wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Rgba8Unorm
                )
            })
            .copied();
        if let Some(format) = linear {
            config.format = format;
        }
        surface.configure(&device, &config);

        let egui = egui_winit::State::new(
            egui::Context::default(),
            egui::ViewportId::ROOT,
            window.as_ref(),
            Some(window.scale_factor() as f32),
            window.theme(),
            Some(device.limits().max_texture_dimension_2d as usize),
        );
        let egui_paint = egui_wgpu::Renderer::new(
            &device,
            config.format,
            egui_wgpu::RendererOptions::default(),
        );
        let renderer = Renderer::new(&device, config.format);

        Ok(Self {
            window,
            surface,
            config,
            device,
            queue,
            egui,
            egui_paint,
            renderer,
            scene: SceneOut::default(),
            size,
            shift: false,
            gpu,
            gpu_resident: false,
            gpu_n: 0,
            gpu_failed: false,
            ts_set,
            ts_resolve,
            ts_stage,
            ts_period,
        })
    }

    /// Rebuilds the surface for the new size. Skips zero sizes.
    fn resize(&mut self, size: PhysicalSize<u32>) {
        if size.width == 0 || size.height == 0 {
            return;
        }
        self.size = (size.width, size.height);
        self.config.width = size.width;
        self.config.height = size.height;
        self.surface.configure(&self.device, &self.config);
    }

    /// Reconfigures the surface after a lost or outdated frame.
    fn reconfigure(&mut self) {
        self.surface.configure(&self.device, &self.config);
    }

    /// Acquires the frame target. Lost and outdated surfaces reconfigure.
    fn redraw(&mut self, app: &mut App) {
        match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(target) => self.draw(target, app),
            wgpu::CurrentSurfaceTexture::Suboptimal(target) => {
                self.reconfigure();
                self.draw(target, app);
            }
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.reconfigure();
            }
            // Skip the frame. The next redraw retries.
            wgpu::CurrentSurfaceTexture::Timeout
            | wgpu::CurrentSurfaceTexture::Occluded
            | wgpu::CurrentSurfaceTexture::Validation => {}
        }
    }

    /// Enters or leaves GPU residency, then records the physics into the
    /// frame encoder. Returns true when the GPU stepped.
    fn step_or_resident(&mut self, app: &mut App, encoder: &mut wgpu::CommandEncoder) -> bool {
        let want_gpu = !self.gpu_failed
            && app.mode == PhysicsMode::Newton
            && app.tuning.gpu_physics
            && app.world.bodies.len() >= app.tuning.gpu_threshold;
        if want_gpu && !self.gpu_resident {
            let mirror: Vec<GpuBody> = app.world.bodies.iter().map(GpuBody::from_body).collect();
            self.gpu_n = mirror.len();
            self.gpu.upload_bodies(&self.device, &self.queue, &mirror);
            self.gpu_resident = true;
            self.renderer.use_external_bodies(
                &self.device,
                self.gpu.render_buffer(),
                self.gpu_n as u32,
            );
        } else if !want_gpu && self.gpu_resident {
            // Exit: the resident state rides back into the world. Radius
            // and shape never change on the GPU, so the CPU copy stays
            // right about everything but position and velocity.
            let back = self
                .gpu
                .download_bodies(&self.device, &self.queue, self.gpu_n);
            for (dst, src) in app.world.bodies.iter_mut().zip(&back) {
                dst.pos = src.pos;
                dst.vel = src.vel;
            }
            self.gpu_resident = false;
            self.renderer.clear_external_bodies();
        }
        if self.gpu_resident && app.world.bodies.len() != self.gpu_n {
            if app.world.bodies.len() > self.gpu_n {
                // A spawn while resident: the CPU owns only the fresh tail.
                // The resident bodies never move on the CPU, so rewriting
                // the whole buffer from the mirror would teleport them.
                let mirror: Vec<GpuBody> =
                    app.world.bodies.iter().map(GpuBody::from_body).collect();
                let from = self.gpu_n;
                self.gpu.grow_to(&self.device, mirror.len());
                self.gpu.upload_tail(&self.queue, &mirror, from);
                self.gpu_n = mirror.len();
                self.renderer.use_external_bodies(
                    &self.device,
                    self.gpu.render_buffer(),
                    self.gpu_n as u32,
                );
            } else {
                // A clear while resident. The buffers stay.
                self.gpu_n = app.world.bodies.len();
                self.renderer
                    .use_external_bodies(&self.device, self.gpu.render_buffer(), 0);
            }
        }
        if !self.gpu_resident || self.gpu_failed {
            app.step_physics();
            return false;
        }
        app.tick_frame();
        // The same gates the CPU step honors: menu, pause, time scale.
        if app.page != Page::Sim || app.paused {
            return false;
        }
        let sim = SimUniforms::new(
            &app.world.settings,
            FIXED_DT * app.time_scale,
            app.world.cell_size(),
            TABLE_MASK,
            self.gpu_n,
        );
        self.gpu.write_sim(&self.queue, &sim);
        let rounds = app.world.settings.resolve_rounds.max(1);
        self.gpu
            .record_full_step(encoder, self.gpu_n, rounds, self.ts_set.as_ref());
        true
    }

    /// Reads the phase timestamps and the pair counters back. The poll
    /// blocks until the frame finishes, which keeps the panel honest at
    /// the cost of the old synchronous frame shape.
    // ponytail: one sync poll per frame; a two-slot staging ring if the
    // latency ever shows
    fn read_gpu_feedback(&mut self, app: &mut App) {
        self.ts_stage
            .slice(..)
            .map_async(wgpu::MapMode::Read, |_| {});
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("timestamp poll fails");
        let ticks: [u64; 6] = {
            let view = self.ts_stage.get_mapped_range(..).expect("map fails");
            let bytes: Vec<u8> = view.to_vec();
            let mut ticks = [0u64; 6];
            for (slot, chunk) in ticks.iter_mut().zip(bytes.chunks_exact(8)) {
                *slot = u64::from_le_bytes(chunk.try_into().expect("eight bytes"));
            }
            ticks
        };
        self.ts_stage.unmap();
        let mut phases = [0.0f32; 5];
        for i in 0..5 {
            phases[i] = (ticks[i + 1].saturating_sub(ticks[i])) as f32 * self.ts_period / 1.0e6;
        }
        app.gpu_phase_ms = phases;
        let (pairs, dropped) = self
            .gpu
            .pair_counters(&self.device, &self.queue, self.gpu_n);
        app.gpu_pairs = pairs;
        app.gpu_dropped = dropped;
        let total: f32 = phases.iter().sum();
        app.step_ms = app.step_ms * SMOOTH_KEEP + total * SMOOTH_NEW;
    }

    /// Renders one frame: physics, scene, particles, then egui on top.
    fn draw(&mut self, target: wgpu::SurfaceTexture, app: &mut App) {
        let raw_input = self.egui.take_egui_input(&self.window);
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame"),
            });
        let gpu_stepped = self.step_or_resident(app, &mut encoder);
        let ctx = self.egui.egui_ctx().clone();
        ctx.begin_pass(raw_input);
        crate::ui::gui::show(&ctx, app, &self.scene);
        let mut output = ctx.end_pass();
        self.egui
            .handle_platform_output(&self.window, output.platform_output);

        // The scene build works in physical pixels. The shader converts with
        // the same size, so the scaling factor cancels at every DPI.
        let (w, h) = (self.size.0 as f32, self.size.1 as f32);
        app.build_scene(w, h, gpu_stepped, &mut self.scene);

        let paint_jobs = ctx.tessellate(output.shapes, output.pixels_per_point);
        let screen = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [self.size.0, self.size.1],
            pixels_per_point: output.pixels_per_point,
        };

        let view = target
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // Fonts need their textures before the egui pass. Drain, so the Drop
        // guard sees both collections empty.
        for (id, deltas) in output.textures_delta.set.drain() {
            for delta in deltas {
                self.egui_paint
                    .update_texture(&self.device, &self.queue, id, &delta);
            }
        }

        // The vertex-pull path uploads the bodies it draws while the CPU
        // stays authoritative. The camera moves every frame, so the uniform
        // rewrites every frame with it. Residency skips the upload: the
        // draw reads the physics buffer instead.
        let gpu_render = (app.mode == PhysicsMode::Newton && app.tuning.gpu_render) || gpu_stepped;
        if gpu_render && !gpu_stepped {
            let cam_uniforms = CamUniforms::new(&app.cam, w, h);
            self.renderer.upload_bodies(
                &self.device,
                &self.queue,
                &cam_uniforms,
                &app.world.bodies,
            );
        }
        if gpu_stepped {
            let cam_uniforms = CamUniforms::new(&app.cam, w, h);
            self.renderer.write_cam(&self.queue, &cam_uniforms);
        }

        // The particles and lines clear the target.
        self.renderer.draw(Frame {
            device: &self.device,
            queue: &self.queue,
            encoder: &mut encoder,
            target: &view,
            bodies_gpu: gpu_render,
            instances: &self.scene.instances,
            lines: &self.scene.lines,
            w: self.size.0 as f32,
            h: self.size.1 as f32,
        });

        // egui paints over the particles with a load pass.
        self.egui_paint.update_buffers(
            &self.device,
            &self.queue,
            &mut encoder,
            &paint_jobs,
            &screen,
        );
        {
            let mut pass = encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("egui"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                })
                .forget_lifetime();
            self.egui_paint.render(&mut pass, &paint_jobs, &screen);
        }
        for id in output.textures_delta.free.drain() {
            self.egui_paint.free_texture(&id);
        }

        // The resolve rides the same submission, before the finish.
        if gpu_stepped && let Some(set) = &self.ts_set {
            encoder.resolve_query_set(set, 0..6, &self.ts_resolve, 0);
            encoder.copy_buffer_to_buffer(&self.ts_resolve, 0, &self.ts_stage, 0, 6 * 8);
        }
        self.queue.submit([encoder.finish()]);
        if gpu_stepped {
            // The poll waits for the whole frame, so the panel reads real
            // numbers and the old synchronous frame shape stays.
            self.queue.present(target);
            self.read_gpu_feedback(app);
            return;
        }
        self.queue.present(target);
    }
}

/// The winit handler. Holds the app and the per-window GPU state.
struct Handler {
    /// App state shared with the panels.
    app: App,
    /// GPU state. `None` until the first `resumed`.
    state: Option<WindowState>,
}

impl ApplicationHandler for Handler {
    /// Initializes the window and the GPU once. `resumed` can fire again.
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }
        match WindowState::init(event_loop, &mut self.app) {
            Ok(state) => self.state = Some(state),
            Err(error) => {
                eprintln!("failed to open window: {error}");
                std::process::exit(1);
            }
        }
    }

    /// Feeds egui first, then maps the event to the app state.
    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Handler { app, state } = self;
        let Some(state) = state.as_mut() else { return };
        let window = state.window.clone();
        let response = state.egui.on_window_event(&window, &event);

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => state.resize(size),
            WindowEvent::RedrawRequested => state.redraw(app),
            WindowEvent::ModifiersChanged(mods) => state.shift = mods.state().shift_key(),
            // Focus loss hides the key release. Drop the held keys, or the
            // camera keeps sliding after alt-tab.
            WindowEvent::Focused(false) => {
                app.input.keys.clear();
                app.drag = None;
                state.shift = false;
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        logical_key,
                        state: key_state,
                        ..
                    },
                ..
            } => {
                let key = key_name(&logical_key);
                if key.is_empty() {
                    return;
                }
                match key_state {
                    ElementState::Pressed => {
                        // The F1 toggle and the rebind capture ignore egui.
                        if key == "f1" {
                            app.debug = !app.debug;
                        } else if key == "space" && app.page == Page::Sim {
                            app.paused = !app.paused;
                        } else if let Some(action) = app.input.rebinding {
                            // Escape cancels the rebind.
                            if key != "escape" {
                                app.input.bindings[action as usize] = key;
                            }
                            app.input.rebinding = None;
                        } else if !response.consumed
                            && !state.egui.egui_ctx().egui_wants_keyboard_input()
                        {
                            // R resets the camera. Unbound by default; a
                            // custom binding on R fires both.
                            if key == "r" {
                                app.cam = Camera::default();
                            }
                            app.input.keys.insert(key);
                        }
                    }
                    ElementState::Released => {
                        app.input.keys.remove(&key);
                    }
                }
            }
            WindowEvent::MouseInput {
                state: button_state,
                button,
                ..
            } => {
                if button_state == ElementState::Released {
                    app.drag = None;
                } else if !response.consumed {
                    // The cursor position is already in `last_mouse`.
                    match button {
                        MouseButton::Left => {
                            app.drag = Some(if state.shift { Drag::Pan } else { Drag::Orbit });
                        }
                        MouseButton::Middle => app.drag = Some(Drag::Pan),
                        _ => {}
                    }
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                let pos = position.to_logical::<f64>(window.scale_factor());
                let pos = [pos.x as f32, pos.y as f32];
                if let (Some(drag), Some(last)) = (app.drag, app.last_mouse) {
                    let dx = pos[0] - last[0];
                    let dy = pos[1] - last[1];
                    apply_drag(&mut app.cam, drag, dx, dy);
                }
                app.last_mouse = Some(pos);
            }
            // egui marks the wheel consumed over its scroll areas.
            WindowEvent::MouseWheel { delta, .. } if !response.consumed => {
                let dy = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(pixels) => (pixels.y / 20.0) as f32,
                };
                zoom(&mut app.cam, dy);
            }
            _ => {}
        }
    }

    /// Requests one redraw per poll. Drives the game loop.
    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(state) = &self.state {
            state.window.request_redraw();
        }
    }
}

/// Normalizes one logical key to the stored name. Unknown keys return an
/// empty string. Named keys print their variant, so `F1` becomes `f1`.
fn key_name(key: &Key) -> String {
    match key {
        Key::Character(text) => text.to_lowercase(),
        Key::Named(named) => format!("{named:?}").to_lowercase(),
        _ => String::new(),
    }
}

/// Builds the window and runs the frame loop. Exits with code 1 on failure.
pub(crate) fn run() {
    let event_loop: EventLoop<()> = match EventLoop::builder().build() {
        Ok(event_loop) => event_loop,
        Err(error) => open_failed(&error),
    };
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut handler = Handler {
        app: App::new(),
        state: None,
    };
    if let Err(error) = event_loop.run_app(&mut handler) {
        open_failed(&error);
    }
}

/// Prints the failure and exits. Matches the old `open_window` error path.
fn open_failed(error: impl std::fmt::Display) -> ! {
    eprintln!("failed to open window: {error}");
    std::process::exit(1)
}
