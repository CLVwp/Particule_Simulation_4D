//! App state: pages, physics modes, the spawn panel, and the frame step.

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

use particule_simulation_4d::engine::fluid::Fluid;
use particule_simulation_4d::engine::{BODY_RADIUS, MIN_RADIUS, Shape, World};

use crate::ui::camera::Camera;
use crate::ui::input::{InputState, apply_moves};
use crate::ui::scene::{TileCache, Tuning};
use crate::ui::theme::{SMOOTH_KEEP, SMOOTH_NEW};

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
    /// steps the active physics mode once at [`FIXED_DT`].
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
        let t = Instant::now();
        match self.mode {
            PhysicsMode::Newton => self.world.step(FIXED_DT),
            PhysicsMode::Fluid => self.fluid.step(FIXED_DT),
        }
        let ms = (t.elapsed().as_secs_f32() * 1000.0).min(1000.0);
        self.step_ms = self.step_ms * SMOOTH_KEEP + ms * SMOOTH_NEW;
    }

    /// Spawns `n` bodies with the panel parameters. Clamps the radius to the
    /// engine floor.
    pub(crate) fn spawn_from_panel(&mut self, n: usize) {
        let radius = self.spawn_radius.max(MIN_RADIUS);
        self.world
            .spawn(n, SPAWN_ORIGIN, self.spawn_speed, self.spawn_shape, radius);
    }
}
