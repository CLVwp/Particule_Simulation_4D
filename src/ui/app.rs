//! Window loop: wgpu init, the winit event handler, and the frame render.

use std::sync::Arc;

use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::Key;
use winit::window::{Window, WindowId};

use crate::ui::App;
use crate::ui::input::{Drag, ORBIT_SENSITIVITY, zoom};
use crate::ui::renderer::Renderer;
use crate::ui::scene::SceneOut;

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

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("main device"),
            required_features: wgpu::Features::default(),
            required_limits: wgpu::Limits::default(),
            experimental_features: wgpu::ExperimentalFeatures::default(),
            memory_hints: wgpu::MemoryHints::default(),
            trace: wgpu::Trace::default(),
        }))
        .map_err(|error| error.to_string())?;

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

    /// Renders one frame: physics, scene, particles, then egui on top.
    fn draw(&mut self, target: wgpu::SurfaceTexture, app: &mut App) {
        let raw_input = self.egui.take_egui_input(&self.window);
        app.step_physics();
        let ctx = self.egui.egui_ctx().clone();
        ctx.begin_pass(raw_input);
        crate::ui::gui::show(&ctx, app, &self.scene);
        let mut output = ctx.end_pass();
        self.egui
            .handle_platform_output(&self.window, output.platform_output);

        let (w, h) = (
            self.size.0 as f32 / output.pixels_per_point,
            self.size.1 as f32 / output.pixels_per_point,
        );
        app.build_scene(w, h, &mut self.scene);

        let paint_jobs = ctx.tessellate(output.shapes, output.pixels_per_point);
        let screen = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [self.size.0, self.size.1],
            pixels_per_point: output.pixels_per_point,
        };

        let view = target
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame"),
            });

        // Fonts need their textures before the egui pass. Drain, so the Drop
        // guard sees both collections empty.
        for (id, deltas) in output.textures_delta.set.drain() {
            for delta in deltas {
                self.egui_paint
                    .update_texture(&self.device, &self.queue, id, &delta);
            }
        }

        // The particles and lines clear the target.
        self.renderer.draw(
            &self.device,
            &self.queue,
            &mut encoder,
            &view,
            &self.scene.instances,
            &self.scene.lines,
            self.size.0 as f32,
            self.size.1 as f32,
        );

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

        self.queue.submit([encoder.finish()]);
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
                        } else if let Some(action) = app.input.rebinding {
                            // Escape cancels the rebind.
                            if key != "escape" {
                                app.input.bindings[action as usize] = key;
                            }
                            app.input.rebinding = None;
                        } else if !response.consumed
                            && !state.egui.egui_ctx().egui_wants_keyboard_input()
                        {
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
                    match drag {
                        Drag::Orbit => {
                            app.cam.yaw += dx * ORBIT_SENSITIVITY;
                            app.cam.pitch =
                                (app.cam.pitch + dy * ORBIT_SENSITIVITY).clamp(-1.4, 1.4);
                        }
                        Drag::Pan => {
                            let scale = app.cam.dist * 0.0015;
                            let right = app.cam.right();
                            let up = app.cam.up();
                            for a in 0..3 {
                                app.cam.target[a] -= right[a] * dx * scale;
                                app.cam.target[a] += up[a] * dy * scale;
                            }
                        }
                    }
                }
                app.last_mouse = Some(pos);
            }
            WindowEvent::MouseWheel { delta, .. } => {
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
