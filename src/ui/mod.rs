//! Pages: menu, key settings, and the simulation view.
//! The simulation view owns the camera, the axes, the floor grid, and the HUD.

/// Camera math helpers.
pub mod camera;
/// Input handling: move actions, key bindings, and camera moves.
pub mod input;
/// Scene painting: floor grid, axes, and projected quads.
pub mod paint;
/// Panel pages: menu, settings, HUD, debug, and the spawn panels.
pub mod panel;
/// Scene projection: world objects to screen quads.
pub mod scene;
/// Shared colors and smoothing weights.
pub mod theme;

use std::collections::HashSet;
use std::time::Instant;

use gpui_kit::*;
use particule_simulation_4d::engine::fluid::Fluid;
use particule_simulation_4d::engine::{BODY_RADIUS, Shape, World};

use crate::ui::input::{Drag, KeyLayout, MoveAction, ORBIT_SENSITIVITY};
use crate::ui::paint::{axis_label_divs, paint_scene};
use crate::ui::theme::{BG, SMOOTH_KEEP, SMOOTH_NEW};

const FIXED_DT: f32 = 1.0 / 60.0;
/// Origin for every spawned wave.
pub(crate) const SPAWN_ORIGIN: [f32; 3] = [0.0, 4.0, 0.0];
/// Speed for every spawned wave.
pub(crate) const SPAWN_SPEED: f32 = 4.0;

/// One page of the app.
#[derive(Clone, Copy, PartialEq)]
pub(crate) enum Page {
    Menu,
    Settings,
    Sim,
}

/// Which law set the sim steps. Newton: rigid bodies. Fluid: Navier-Stokes.
#[derive(Clone, Copy, PartialEq)]
pub(crate) enum PhysicsMode {
    Newton,
    Fluid,
}

pub struct SimView {
    pub(crate) page: Page,
    pub(crate) world: World,
    // Key bindings, one key per `MoveAction`, indexed by `action as usize`.
    pub(crate) bindings: [String; 6],
    pub(crate) rebinding: Option<MoveAction>,
    pub(crate) layout: KeyLayout,
    // Physics mode and its state.
    pub(crate) mode: PhysicsMode,
    pub(crate) fluid: Fluid,
    // Spawn panel parameters.
    pub(crate) spawn_shape: Shape,
    pub(crate) spawn_count: usize,
    pub(crate) spawn_radius: f32,
    pub(crate) spawn_speed: f32,
    // Orbit camera: looks at `target` from `dist` along the rotated +z axis.
    pub(crate) target: [f32; 3],
    pub(crate) yaw: f32,
    pub(crate) pitch: f32,
    pub(crate) dist: f32,
    // Input state.
    pub(crate) drag: Option<Drag>,
    last_mouse: Option<Point<Pixels>>,
    pub(crate) keys: HashSet<String>,
    pub(crate) focus: FocusHandle,
    // Debug HUD values. F1 toggles the overlay.
    pub(crate) debug: bool,
    pub(crate) fps: f32,
    pub(crate) step_ms: f32,
    pub(crate) scene_ms: f32,
    last_frame: Option<Instant>,
}

impl SimView {
    pub fn new(cx: &mut Context<Self>) -> Self {
        let mut world = World::new();
        world.spawn_wave(1000, SPAWN_ORIGIN, SPAWN_SPEED);
        SimView {
            page: Page::Menu,
            world,
            bindings: MoveAction::all().map(|a| a.default_key().to_string()),
            rebinding: None,
            layout: KeyLayout::Qwerty,
            mode: PhysicsMode::Newton,
            fluid: Fluid::new(64),
            spawn_shape: Shape::Sphere,
            spawn_count: 1000,
            spawn_radius: BODY_RADIUS,
            spawn_speed: SPAWN_SPEED,
            target: [0.0, 1.0, 0.0],
            yaw: 0.6,
            pitch: 0.35,
            dist: 12.0,
            drag: None,
            last_mouse: None,
            keys: HashSet::new(),
            focus: cx.focus_handle(),
            debug: false,
            fps: 60.0,
            step_ms: 0.0,
            scene_ms: 0.0,
            last_frame: None,
        }
    }

    /// True while the key bound to `action` is held down.
    pub(crate) fn is_held(&self, action: MoveAction) -> bool {
        self.keys.contains(&self.bindings[action as usize])
    }
}

impl Render for SimView {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        match self.page {
            Page::Menu => self.render_menu(cx).into_any_element(),
            Page::Settings => self.render_settings(cx).into_any_element(),
            Page::Sim => self.render_sim(window, cx).into_any_element(),
        }
    }
}

impl SimView {
    fn render_sim(&mut self, window: &mut Window, cx: &mut Context<Self>) -> Div {
        // ponytail: fixed dt decoupled from real time; wall-clock dt if physics gets speed-sensitive
        cx.on_next_frame(window, |this, _window, cx| {
            let now = Instant::now();
            if let Some(last) = this.last_frame {
                let dt = (now - last).as_secs_f32();
                if dt > 0.0 {
                    this.fps = this.fps * SMOOTH_KEEP + (1.0 / dt) * SMOOTH_NEW;
                }
            }
            this.last_frame = Some(now);

            this.apply_keys();
            let t = Instant::now();
            match this.mode {
                PhysicsMode::Newton => this.world.step(FIXED_DT),
                PhysicsMode::Fluid => this.fluid.step(FIXED_DT),
            }
            let ms = (t.elapsed().as_secs_f32() * 1000.0).min(1000.0);
            this.step_ms = this.step_ms * SMOOTH_KEEP + ms * SMOOTH_NEW;
            cx.notify();
        });

        let viewport = window.viewport_size();
        let (w, h) = (f32::from(viewport.width), f32::from(viewport.height));
        let dist = self.dist;

        let (points, fluid_quads) = self.build_scene(w, h);
        let quads_drawn = points.len() + fluid_quads.len();
        let grid = self.grid_lines(w, h);
        let (axis_lines, axis_labels) = self.axes(w, h);

        let hud = self.render_hud(cx);
        let toolbar = self.render_toolbar(cx);

        div()
            .size_full()
            .relative()
            .overflow_hidden()
            .bg(rgb(BG))
            .track_focus(&self.focus)
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(|this, ev: &MouseDownEvent, window, cx| {
                    this.drag = Some(if ev.modifiers.shift {
                        Drag::Pan
                    } else {
                        Drag::Orbit
                    });
                    this.last_mouse = Some(ev.position);
                    window.focus(&this.focus, cx);
                }),
            )
            .on_mouse_down(
                MouseButton::Middle,
                cx.listener(|this, ev: &MouseDownEvent, window, cx| {
                    this.drag = Some(Drag::Pan);
                    this.last_mouse = Some(ev.position);
                    window.focus(&this.focus, cx);
                }),
            )
            .on_mouse_up(
                MouseButton::Left,
                cx.listener(|this, _: &MouseUpEvent, _w, _cx| {
                    this.drag = None;
                }),
            )
            .on_mouse_up(
                MouseButton::Middle,
                cx.listener(|this, _: &MouseUpEvent, _w, _cx| {
                    this.drag = None;
                }),
            )
            .on_mouse_move(cx.listener(|this, ev: &MouseMoveEvent, _w, _cx| {
                let Some(drag) = this.drag else { return };
                let Some(last) = this.last_mouse else { return };
                let dx = f32::from(ev.position.x - last.x);
                let dy = f32::from(ev.position.y - last.y);
                this.last_mouse = Some(ev.position);
                match drag {
                    Drag::Orbit => {
                        this.yaw += dx * ORBIT_SENSITIVITY;
                        this.pitch = (this.pitch + dy * ORBIT_SENSITIVITY).clamp(-1.4, 1.4);
                    }
                    Drag::Pan => {
                        let s = this.dist * 0.0015;
                        let right = this.right();
                        let up = this.up();
                        for a in 0..3 {
                            this.target[a] -= right[a] * dx * s;
                            this.target[a] += up[a] * dy * s;
                        }
                    }
                }
            }))
            .on_scroll_wheel(cx.listener(|this, ev: &ScrollWheelEvent, _w, _cx| {
                let dy = match ev.delta {
                    ScrollDelta::Lines(p) => p.y,
                    ScrollDelta::Pixels(p) => f32::from(p.y) / 20.0,
                };
                this.dist = (this.dist * (1.0 - 0.1 * dy)).clamp(3.0, 40.0);
            }))
            .on_key_down(cx.listener(|this, ev: &KeyDownEvent, _w, _cx| {
                this.handle_key_down(ev);
            }))
            .on_key_up(cx.listener(|this, ev: &KeyUpEvent, _w, _cx| {
                this.keys.remove(&ev.keystroke.key);
            }))
            .child(canvas(
                |_, _, _| (),
                move |_, _, window, _| {
                    paint_scene(window, dist, points, fluid_quads, grid, axis_lines);
                },
            ))
            .children(axis_label_divs(axis_labels))
            .child(hud)
            .child(toolbar)
            .child(self.render_panels(cx))
            .children(self.debug.then(|| self.render_debug(quads_drawn, w, h)))
    }
}
