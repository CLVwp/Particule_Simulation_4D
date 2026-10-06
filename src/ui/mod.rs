//! App state: pages, physics modes, the spawn panel, and the frame step.
//!
//! Two items stay public for benches, examples, and the binary: [`run`]
//! starts the window, and [`emit_scene_cpu`] builds one frame's scene
//! headless. Everything else is crate-private.

pub(crate) mod app;
pub(crate) mod camera;
pub(crate) mod gpu;
pub(crate) mod gui;
pub(crate) mod input;
pub(crate) mod renderer;
pub(crate) mod scene;
pub(crate) mod theme;
pub(crate) mod widgets;

use std::time::Instant;

use crate::engine::fluid::Fluid;
use crate::engine::{BODY_RADIUS, MIN_RADIUS, Shape, World};

use crate::ui::input::{InputState, apply_moves};
use crate::ui::theme::{SMOOTH_KEEP, SMOOTH_NEW};

pub use crate::ui::camera::Camera;
pub use crate::ui::scene::{SceneOut, TileCache, Tuning};

/// Starts the particle simulation window.
pub fn run() {
    app::run();
}

/// Builds one frame's scene headless: the CPU instance path plus the grid
/// and axes. This is the phase the GPU vertex-pull path replaces, so the
/// benches and the phase table example measure it as their baseline.
/// The GPU path needs a device and has no headless form.
#[allow(clippy::too_many_arguments)]
pub fn emit_scene_cpu(
    cam: &Camera,
    bodies: &[crate::engine::Body],
    w: f32,
    h: f32,
    dist: f32,
    tuning: &Tuning,
    par_min: usize,
    tiles: &mut TileCache,
    out: &mut SceneOut,
) {
    out.instances.clear();
    out.lines.clear();
    out.axis_labels.clear();
    scene::emit_bodies(cam, bodies, w, h, dist, tuning, par_min, tiles, out);
    scene::emit_lines(cam, w, h, out);
}

/// Fixed physics step. One step runs per rendered frame.
pub(crate) const FIXED_DT: f32 = 1.0 / 60.0;
/// Origin for every spawned wave.
pub(crate) const SPAWN_ORIGIN: [f32; 3] = [0.0, 4.0, 0.0];
/// Speed for every spawned wave.
pub(crate) const SPAWN_SPEED: f32 = 4.0;

/// One page of the app.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Page {
    /// Title page with the entry buttons.
    Menu,
    /// Layout preset and move bindings.
    Settings,
    /// The simulation viewport.
    Sim,
}

/// Which law set the sim steps. Newton: rigid bodies. Fluid: Navier-Stokes.
#[derive(Clone, Copy, PartialEq)]
pub(crate) enum PhysicsMode {
    /// Rigid bodies with contacts.
    Newton,
    /// Navier-Stokes grid.
    Fluid,
}

/// How the spawn panel builds its bodies.
#[derive(Clone, Copy, PartialEq)]
pub(crate) enum SpawnLayout {
    /// Random cloud with launch speeds. Honors the seed field.
    Fountain,
    /// Velocity-free lattice cube. Side cubed bodies, no randomness.
    Lattice,
}

/// One app instance. The state is 1:1 with the old `SimView` state.
pub(crate) struct App {
    /// Active page.
    pub(crate) page: Page,
    /// Newton bodies.
    pub(crate) world: World,
    /// Held keys, bindings, and the pending rebind.
    pub(crate) input: InputState,
    /// Which law set the sim steps.
    pub(crate) mode: PhysicsMode,
    /// Navier-Stokes grid for the fluid mode.
    pub(crate) fluid: Fluid,
    /// Spawn panel: shape of the spawned bodies.
    pub(crate) spawn_shape: Shape,
    /// Spawn panel: body count.
    pub(crate) spawn_count: usize,
    /// Spawn panel: body radius.
    pub(crate) spawn_radius: f32,
    /// Spawn panel: launch speed.
    pub(crate) spawn_speed: f32,
    /// Spawn panel: cloud or lattice build.
    pub(crate) spawn_layout: SpawnLayout,
    /// Spawn panel: explicit fountain seed. Zero keeps the auto stream.
    pub(crate) spawn_seed: u64,
    /// True while the step is held. Space toggles it.
    pub(crate) paused: bool,
    /// Multiplier on the fixed step. 0.1 crawls, 4.0 races.
    pub(crate) time_scale: f32,
    /// Orbit camera.
    pub(crate) cam: Camera,
    /// Active mouse drag gesture.
    pub(crate) drag: Option<crate::ui::input::Drag>,
    /// Last mouse position, in logical pixels.
    pub(crate) last_mouse: Option<[f32; 2]>,
    /// True while the F1 overlay shows.
    pub(crate) debug: bool,
    /// Smoothed frames per second.
    pub(crate) fps: f32,
    /// Smoothed physics step time, in milliseconds.
    pub(crate) step_ms: f32,
    /// Smoothed scene build time, in milliseconds.
    pub(crate) scene_ms: f32,
    /// Reused tile merge bins.
    pub(crate) tiles: TileCache,
    /// Live render tuning. The F1 panel edits these.
    pub(crate) tuning: Tuning,
    /// GPU adapter name. Set after wgpu init.
    pub(crate) adapter_info: String,
    /// Time of the previous frame. Drives the move scale.
    last_frame: Option<Instant>,
}

impl App {
    /// Builds the app. Defaults match the old `SimView::new`.
    pub(crate) fn new() -> Self {
        let mut world = World::new();
        world.spawn_wave(1000, SPAWN_ORIGIN, SPAWN_SPEED);
        Self {
            page: Page::Menu,
            world,
            input: InputState::default(),
            mode: PhysicsMode::Newton,
            fluid: Fluid::new(64),
            spawn_shape: Shape::Sphere,
            spawn_count: 1000,
            spawn_radius: BODY_RADIUS,
            spawn_speed: SPAWN_SPEED,
            spawn_layout: SpawnLayout::Fountain,
            spawn_seed: 0,
            paused: false,
            time_scale: 1.0,
            cam: Camera::default(),
            drag: None,
            last_mouse: None,
            debug: false,
            fps: 60.0,
            step_ms: 0.0,
            scene_ms: 0.0,
            tiles: TileCache::default(),
            tuning: Tuning::default(),
            adapter_info: "unknown".to_string(),
            last_frame: None,
        }
    }

    /// Advances one frame. Smooths the fps, applies the camera moves, and
    /// steps the active physics mode once at [`FIXED_DT`] times
    /// [`App::time_scale`]. The step runs only in the viewport, and only
    /// while the sim is not paused.
    // ponytail: fixed dt decoupled from real time; wall-clock dt if physics gets speed-sensitive
    pub(crate) fn step_physics(&mut self) {
        let now = Instant::now();
        let dt = match self.last_frame {
            Some(last) => now.duration_since(last).as_secs_f32(),
            None => FIXED_DT,
        };
        self.last_frame = Some(now);
        if dt > 0.0 {
            self.fps = self.fps * SMOOTH_KEEP + (1.0 / dt) * SMOOTH_NEW;
        }
        // The camera slide scales with the frame rate. Physics does not.
        let scale = (dt / FIXED_DT).clamp(0.25, 4.0);
        apply_moves(&self.input, &mut self.cam, scale);
        // The sim waits behind the menu, and pause holds the step.
        if self.page != Page::Sim || self.paused {
            return;
        }
        let t = Instant::now();
        let step_dt = FIXED_DT * self.time_scale;
        match self.mode {
            PhysicsMode::Newton => self.world.step(step_dt),
            PhysicsMode::Fluid => self.fluid.step(step_dt),
        }
        let ms = (t.elapsed().as_secs_f32() * 1000.0).min(1000.0);
        self.step_ms = self.step_ms * SMOOTH_KEEP + ms * SMOOTH_NEW;
    }

    /// Points the camera at the whole body cloud. Fits the bounding sphere
    /// of the bodies into view. No-op on an empty world.
    pub(crate) fn frame_scene(&mut self) {
        if self.world.bodies.is_empty() {
            return;
        }
        let mut lo = [f32::MAX; 3];
        let mut hi = [f32::MIN; 3];
        for b in &self.world.bodies {
            for k in 0..3 {
                lo[k] = lo[k].min(b.pos[k]);
                hi[k] = hi[k].max(b.pos[k]);
            }
        }
        let center = [
            (lo[0] + hi[0]) * 0.5,
            (lo[1] + hi[1]) * 0.5,
            (lo[2] + hi[2]) * 0.5,
        ];
        let radius = self
            .world
            .bodies
            .iter()
            .map(|b| {
                let d = [
                    b.pos[0] - center[0],
                    b.pos[1] - center[1],
                    b.pos[2] - center[2],
                ];
                (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt() + b.radius
            })
            .fold(0.0, f32::max);
        self.cam.target = center;
        // 2.2 keeps the sphere clear of the viewport edge at any aspect.
        self.cam.dist = (radius * 2.2).clamp(3.0, 200.0);
    }

    /// Spawns `n` bodies with the panel parameters. Clamps the radius to the
    /// engine floor. The lattice maps the count to a cube side.
    pub(crate) fn spawn_from_panel(&mut self, n: usize) {
        let radius = self.spawn_radius.max(MIN_RADIUS);
        match self.spawn_layout {
            SpawnLayout::Fountain => {
                // Seed zero keeps the auto stream, which derives from the
                // count alone. Any other value replays the same cloud.
                if self.spawn_seed == 0 {
                    self.world
                        .spawn(n, SPAWN_ORIGIN, self.spawn_speed, self.spawn_shape, radius);
                } else {
                    self.world.spawn_seeded(
                        n,
                        SPAWN_ORIGIN,
                        self.spawn_speed,
                        self.spawn_shape,
                        radius,
                        self.spawn_seed,
                    );
                }
            }
            SpawnLayout::Lattice => {
                let side = ((n as f32).cbrt().round() as usize).max(1);
                self.world
                    .spawn_grid(side, SPAWN_ORIGIN, 2.0 * radius, self.spawn_shape, radius);
            }
        }
    }
}
