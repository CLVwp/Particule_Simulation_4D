//! Pages: menu, key settings, and the simulation view.
//! The simulation view owns the camera, the axes, the floor grid, and the HUD.

use std::collections::HashSet;
use std::mem::size_of;
use std::time::Instant;

use gpui_kit::component::button::{Button, ButtonVariants};
use gpui_kit::*;
use rayon::prelude::*;

use particule_simulation_4d::engine::fluid::Fluid;
use particule_simulation_4d::engine::{BODY_RADIUS, Body, PAR_MIN, Shape, World, thread_count};
use particule_simulation_4d::perf::{allocated_bytes, peak_bytes};

const FIXED_DT: f32 = 1.0 / 60.0;
const SPAWN_ORIGIN: [f32; 3] = [0.0, 4.0, 0.0];
const SPAWN_SPEED: f32 = 4.0;
const GRID_HALF: i32 = 10; // floor grid spans -10..=10 units
const GRID_STEP: i32 = 1;
const AXIS_LEN: f32 = 2.0;
/// The fluid plane spans `FLUID_SPAN` world units and starts at `FLUID_LEFT`.
const FLUID_LEFT: f32 = -8.0;
const FLUID_SPAN: f32 = 16.0;

const BG: u32 = 0x0b0e14;
const FG: Hsla = hsla(0.58, 0.15, 0.9, 1.0);
const FAINT: Hsla = hsla(0.58, 0.15, 0.85, 0.9);

/// A projected line segment: two screen points and a color.
type ProjectedLine = ((f32, f32, f32, f32), (f32, f32, f32, f32), Hsla);

/// A projected body: screen x, screen y, radius in px, depth, shape.
type ProjectedBody = (f32, f32, f32, f32, Shape);

/// A projected fluid cell: screen x, screen y, radius in px, depth, density.
type ProjectedCell = (f32, f32, f32, f32, f32);

#[derive(Clone, Copy, PartialEq)]
enum Page {
    Menu,
    Settings,
    Sim,
}

/// One move action. Each action holds one bound key.
#[derive(Clone, Copy, PartialEq)]
enum MoveAction {
    Forward,
    Back,
    Left,
    Right,
    Rise,
    Sink,
}

impl MoveAction {
    fn all() -> [MoveAction; 6] {
        [
            MoveAction::Forward,
            MoveAction::Back,
            MoveAction::Left,
            MoveAction::Right,
            MoveAction::Rise,
            MoveAction::Sink,
        ]
    }

    fn label(self) -> &'static str {
        match self {
            MoveAction::Forward => "Slide forward",
            MoveAction::Back => "Slide backward",
            MoveAction::Left => "Slide left",
            MoveAction::Right => "Slide right",
            MoveAction::Rise => "Rise",
            MoveAction::Sink => "Sink",
        }
    }

    fn default_key(self) -> &'static str {
        match self {
            MoveAction::Forward => "w",
            MoveAction::Back => "s",
            MoveAction::Left => "a",
            MoveAction::Right => "d",
            MoveAction::Rise => "e",
            MoveAction::Sink => "q",
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Drag {
    Orbit,
    Pan,
}

/// Keyboard layout preset for the move bindings.
#[derive(Clone, Copy, PartialEq)]
enum KeyLayout {
    Qwerty,
    Azerty,
}

impl KeyLayout {
    fn label(self) -> &'static str {
        match self {
            KeyLayout::Qwerty => "QWERTY",
            KeyLayout::Azerty => "AZERTY",
        }
    }

    /// One key per `MoveAction`, in `MoveAction::all()` order.
    fn keys(self) -> [&'static str; 6] {
        match self {
            KeyLayout::Qwerty => ["w", "s", "a", "d", "e", "q"],
            KeyLayout::Azerty => ["z", "s", "q", "d", "e", "a"],
        }
    }
}

/// Which law set the sim steps. Newton: rigid bodies. Fluid: Navier-Stokes.
#[derive(Clone, Copy, PartialEq)]
enum PhysicsMode {
    Newton,
    Fluid,
}

pub struct SimView {
    page: Page,
    world: World,
    // Key bindings, one key per `MoveAction`, indexed by `action as usize`.
    bindings: [String; 6],
    rebinding: Option<MoveAction>,
    layout: KeyLayout,
    // Physics mode and its state.
    mode: PhysicsMode,
    fluid: Fluid,
    // Spawn panel parameters.
    spawn_shape: Shape,
    spawn_count: usize,
    spawn_radius: f32,
    spawn_speed: f32,
    // Orbit camera: looks at `target` from `dist` along the rotated +z axis.
    target: [f32; 3],
    yaw: f32,
    pitch: f32,
    dist: f32,
    // Input state.
    drag: Option<Drag>,
    last_mouse: Option<Point<Pixels>>,
    keys: HashSet<String>,
    focus: FocusHandle,
    // Debug HUD values. F1 toggles the overlay.
    debug: bool,
    fps: f32,
    step_ms: f32,
    scene_ms: f32,
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
    fn held(&self, action: MoveAction) -> bool {
        self.keys.contains(&self.bindings[action as usize])
    }

    /// Applies the active layout preset to every binding.
    fn apply_layout(&mut self) {
        self.bindings = self.layout.keys().map(str::to_string);
    }

    /// Spawns `n` bodies with the panel parameters.
    fn spawn_from_panel(&mut self, n: usize) {
        self.world.spawn(
            n,
            SPAWN_ORIGIN,
            self.spawn_speed,
            self.spawn_shape,
            self.spawn_radius,
        );
    }

    /// Applies the held keys: slides follow the camera, rise and sink use the world axis.
    fn apply_keys(&mut self) {
        let speed = self.dist * 0.03;
        let fwd = self.forward();
        let right = self.right();
        let mut t = self.target;
        if self.held(MoveAction::Forward) {
            for a in 0..3 {
                t[a] += fwd[a] * speed;
            }
        }
        if self.held(MoveAction::Back) {
            for a in 0..3 {
                t[a] -= fwd[a] * speed;
            }
        }
        if self.held(MoveAction::Right) {
            for a in 0..3 {
                t[a] += right[a] * speed;
            }
        }
        if self.held(MoveAction::Left) {
            for a in 0..3 {
                t[a] -= right[a] * speed;
            }
        }
        if self.held(MoveAction::Rise) {
            t[1] += speed;
        }
        if self.held(MoveAction::Sink) {
            t[1] -= speed;
        }
        self.target = t;
    }

    /// Stores a pressed key. While a rebind waits, the next key becomes the binding.
    fn handle_key_down(&mut self, ev: &KeyDownEvent) {
        let key = ev.keystroke.key.clone();
        if key == "f1" {
            self.debug = !self.debug;
            return;
        }
        if let Some(action) = self.rebinding {
            if key != "escape" {
                self.bindings[action as usize] = key;
            }
            self.rebinding = None;
            return;
        }
        self.keys.insert(key);
    }

    /// World point → (screen_x, screen_y, focal_px, camera_depth).
    fn project(&self, p: [f32; 3], w: f32, h: f32) -> (f32, f32, f32, f32) {
        let p = [
            p[0] - self.target[0],
            p[1] - self.target[1],
            p[2] - self.target[2],
        ];
        let (cy, sy) = (self.yaw.cos(), self.yaw.sin());
        let (cp, sp) = (self.pitch.cos(), self.pitch.sin());
        let x = cy * p[0] + sy * p[2];
        let z1 = -sy * p[0] + cy * p[2];
        let y = cp * p[1] - sp * z1;
        let z = sp * p[1] + cp * z1 + self.dist;
        let focal = w.min(h);
        (w / 2.0 + x * focal / z, h / 2.0 - y * focal / z, focal, z)
    }

    /// Camera right axis in world space.
    fn right(&self) -> [f32; 3] {
        [self.yaw.cos(), 0.0, self.yaw.sin()]
    }

    /// Camera up axis in world space.
    fn up(&self) -> [f32; 3] {
        let (cp, sp) = (self.pitch.cos(), self.pitch.sin());
        [self.yaw.sin() * sp, cp, -self.yaw.cos() * sp]
    }

    /// Horizontal look direction in world space. The camera looks toward +depth,
    /// and depth 0 sits at the camera, so this is the positive depth axis.
    fn forward(&self) -> [f32; 3] {
        [-self.yaw.sin(), 0.0, self.yaw.cos()]
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
    fn render_menu(&mut self, cx: &mut Context<Self>) -> Div {
        div()
            .size_full()
            .bg(rgb(BG))
            .flex()
            .flex_col()
            .items_center()
            .justify_center()
            .gap_4()
            .track_focus(&self.focus)
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(|this, _: &MouseDownEvent, window, cx| {
                    window.focus(&this.focus, cx);
                }),
            )
            .on_key_down(cx.listener(|this, ev: &KeyDownEvent, _w, _cx| {
                this.handle_key_down(ev);
            }))
            .on_key_up(cx.listener(|this, ev: &KeyUpEvent, _w, _cx| {
                this.keys.remove(&ev.keystroke.key);
            }))
            .child(
                div()
                    .text_size(px(28.0))
                    .text_color(FG)
                    .child("Particule Simulation 4D"),
            )
            .child(
                Button::new("enter-sim")
                    .primary()
                    .label("Enter simulation")
                    .on_click(cx.listener(|this, _: &ClickEvent, _, _| {
                        this.page = Page::Sim;
                    })),
            )
            .child(
                Button::new("open-settings")
                    .label("Settings")
                    .on_click(cx.listener(|this, _: &ClickEvent, _, _| {
                        this.page = Page::Settings;
                    })),
            )
    }

    fn render_settings(&mut self, cx: &mut Context<Self>) -> Div {
        let (qwerty, azerty) = (
            Button::new("layout-qwerty").label(KeyLayout::Qwerty.label()),
            Button::new("layout-azerty").label(KeyLayout::Azerty.label()),
        );
        let (qwerty, azerty) = (
            if self.layout == KeyLayout::Qwerty {
                qwerty.primary()
            } else {
                qwerty
            },
            if self.layout == KeyLayout::Azerty {
                azerty.primary()
            } else {
                azerty
            },
        );

        let mut rows = div().flex().flex_col().gap_2();
        for action in MoveAction::all() {
            let listening = self.rebinding == Some(action);
            let key = if listening {
                "press a key...".to_string()
            } else {
                self.bindings[action as usize].to_uppercase()
            };
            rows = rows.child(
                div()
                    .flex()
                    .items_center()
                    .gap_6()
                    .child(div().w(px(110.0)).text_color(FAINT).child(action.label()))
                    .child(Button::new(("bind", action as usize)).label(key).on_click(
                        cx.listener(move |this, _: &ClickEvent, window, cx| {
                            this.rebinding = Some(action);
                            window.focus(&this.focus, cx);
                        }),
                    )),
            );
        }

        div()
            .size_full()
            .bg(rgb(BG))
            .flex()
            .items_center()
            .justify_center()
            .track_focus(&self.focus)
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(|this, _: &MouseDownEvent, window, cx| {
                    window.focus(&this.focus, cx);
                }),
            )
            .on_key_down(cx.listener(|this, ev: &KeyDownEvent, _w, _cx| {
                this.handle_key_down(ev);
            }))
            .on_key_up(cx.listener(|this, ev: &KeyUpEvent, _w, _cx| {
                this.keys.remove(&ev.keystroke.key);
            }))
            .child(
                div()
                    .flex()
                    .flex_col()
                    .gap_4()
                    .p_8()
                    .rounded_lg()
                    .bg(rgb(0x151b23))
                    .child(div().text_size(px(20.0)).text_color(FG).child("Settings"))
                    .child(
                        div()
                            .text_color(FAINT)
                            .text_size(px(12.0))
                            .child("Pick an action. Then press the new key. Escape cancels."),
                    )
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .gap_6()
                            .child(div().w(px(110.0)).text_color(FAINT).child("Key layout"))
                            .child(qwerty.on_click(cx.listener(|this, _: &ClickEvent, _, _| {
                                this.layout = KeyLayout::Qwerty;
                                this.apply_layout();
                            })))
                            .child(azerty.on_click(cx.listener(|this, _: &ClickEvent, _, _| {
                                this.layout = KeyLayout::Azerty;
                                this.apply_layout();
                            }))),
                    )
                    .child(rows)
                    .child(Button::new("back").label("Back").on_click(cx.listener(
                        |this, _: &ClickEvent, _, _| {
                            this.page = Page::Menu;
                            this.rebinding = None;
                        },
                    ))),
            )
    }

    fn render_sim(&mut self, window: &mut Window, cx: &mut Context<Self>) -> Div {
        // ponytail: fixed dt decoupled from real time; wall-clock dt if physics gets speed-sensitive
        cx.on_next_frame(window, |this, _window, cx| {
            let now = Instant::now();
            if let Some(last) = this.last_frame {
                let dt = (now - last).as_secs_f32();
                if dt > 0.0 {
                    this.fps = this.fps * 0.9 + (1.0 / dt) * 0.1;
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
            this.step_ms = this.step_ms * 0.9 + ms * 0.1;
            cx.notify();
        });

        let viewport = window.viewport_size();
        let (w, h) = (f32::from(viewport.width), f32::from(viewport.height));
        let dist = self.dist;

        // Project everything up-front; the paint closure only draws.
        let scene = Instant::now();
        // Newton: (x, y, radius_px, depth, shape) per body.
        // Fluid: (x, y, radius_px, density, depth) per lit cell.
        let (points, fluid_quads): (Vec<ProjectedBody>, Vec<ProjectedCell>) =
            if self.mode == PhysicsMode::Newton {
                let project = |b: &Body| {
                    let (x, y, focal, z) = self.project(b.pos, w, h);
                    (x, y, (b.radius * focal / z).max(1.5), z, b.shape)
                };
                let bodies = &self.world.bodies;
                let mut points: Vec<ProjectedBody> = if bodies.len() >= PAR_MIN {
                    bodies.par_iter().map(project).collect()
                } else {
                    bodies.iter().map(project).collect()
                };
                // painter's algorithm: far first
                if points.len() >= PAR_MIN {
                    points.par_sort_unstable_by(|a, b| b.3.total_cmp(&a.3));
                } else {
                    points.sort_unstable_by(|a, b| b.3.total_cmp(&a.3));
                }
                (points, Vec::new())
            } else {
                let n = self.fluid.n;
                let cell = FLUID_SPAN / n as f32;
                let mut quads: Vec<ProjectedCell> = Vec::new();
                for j in 1..=n {
                    for i in 1..=n {
                        let d = self.fluid.dens[i + (n + 2) * j];
                        if d <= 0.02 {
                            continue;
                        }
                        let xw = FLUID_LEFT + (i as f32 - 0.5) * cell;
                        let yw = (j as f32 - 0.5) * cell;
                        let (x, y, focal, z) = self.project([xw, yw, 0.0], w, h);
                        let rad = (cell * focal / z * 0.5).max(1.0);
                        quads.push((x, y, rad, d, z));
                    }
                }
                quads.sort_unstable_by(|a, b| b.4.total_cmp(&a.4));
                (Vec::new(), quads)
            };
        let ms = (scene.elapsed().as_secs_f32() * 1000.0).min(1000.0);
        self.scene_ms = self.scene_ms * 0.9 + ms * 0.1;
        let quads_drawn = points.len() + fluid_quads.len();

        // Floor grid.
        let span = (GRID_HALF * GRID_STEP) as f32;
        let mut grid: Vec<([f32; 3], [f32; 3])> = Vec::with_capacity(42);
        for gi in -GRID_HALF..=GRID_HALF {
            let g = (gi * GRID_STEP) as f32;
            grid.push(([g, 0.0, -span], [g, 0.0, span]));
            grid.push(([-span, 0.0, g], [span, 0.0, g]));
        }
        let grid = grid
            .into_iter()
            .map(|(a, b)| (self.project(a, w, h), self.project(b, w, h)))
            .collect::<Vec<_>>();

        // Orthonormal frame: one colored arm per axis, plus a text label at each tip.
        const AXES: [([f32; 3], Hsla, &str); 3] = [
            ([AXIS_LEN, 0.0, 0.0], hsla(0.0, 0.8, 0.55, 1.0), "X"),
            ([0.0, AXIS_LEN, 0.0], hsla(0.33, 0.8, 0.5, 1.0), "Y"),
            ([0.0, 0.0, AXIS_LEN], hsla(0.58, 0.8, 0.6, 1.0), "Z"),
        ];
        let mut axis_lines: Vec<ProjectedLine> = Vec::with_capacity(3);
        let mut axis_labels: Vec<(f32, f32, Hsla, &'static str)> = Vec::with_capacity(3);
        for (tip, color, label) in AXES {
            let a = self.project([0.0, 0.0, 0.0], w, h);
            let b = self.project(tip, w, h);
            axis_lines.push((a, b, color));
            axis_labels.push((b.0, b.1, color, label));
        }

        let (fwd, back, left, right) = (
            self.bindings[MoveAction::Forward as usize].to_uppercase(),
            self.bindings[MoveAction::Back as usize].to_uppercase(),
            self.bindings[MoveAction::Left as usize].to_uppercase(),
            self.bindings[MoveAction::Right as usize].to_uppercase(),
        );
        let (rise, sink) = (
            self.bindings[MoveAction::Rise as usize].to_uppercase(),
            self.bindings[MoveAction::Sink as usize].to_uppercase(),
        );

        let hud = div()
            .absolute()
            .top_2()
            .left_2()
            .flex()
            .flex_col()
            .gap_1()
            .text_size(px(12.0))
            .text_color(FAINT)
            .child(format!("Bodies: {}", self.world.bodies.len()))
            .child(format!("Threads: {}", thread_count()))
            .child(format!("FPS: {:.0}", self.fps))
            .child(format!(
                "Camera: dist {:.1}  yaw {:.2}  pitch {:.2}",
                self.dist, self.yaw, self.pitch
            ))
            .child(format!(
                "Slide: {fwd} {back} {left} {right}.  Rise: {rise}.  Sink: {sink}."
            ))
            .child("Drag: orbit. Shift+drag or middle-drag: pan. Wheel: zoom.")
            .child("Movement follows the camera.")
            .child("F1: engine stats.")
            .child(Button::new("menu").label("Menu").on_click(cx.listener(
                |this, _: &ClickEvent, _, _| {
                    this.page = Page::Menu;
                    this.drag = None;
                },
            )));

        let toolbar = div()
            .absolute()
            .top_2()
            .left_1_2()
            .flex()
            .gap_2()
            .child(Button::new("add-100").label("+100").on_click(cx.listener(
                |this, _: &ClickEvent, _, _| {
                    this.spawn_from_panel(100);
                },
            )))
            .child(Button::new("add-1000").label("+1000").on_click(cx.listener(
                |this, _: &ClickEvent, _, _| {
                    this.spawn_from_panel(1000);
                },
            )))
            .child(Button::new("clear").label("Clear").on_click(cx.listener(
                |this, _: &ClickEvent, _, _| {
                    this.world.clear();
                },
            )));

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
                        this.yaw += dx * 0.01;
                        this.pitch = (this.pitch + dy * 0.01).clamp(-1.4, 1.4);
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
                    for (a, b, color) in axis_lines {
                        paint_line(window, a, b, color, 1.5);
                    }
                    for (a, b) in grid {
                        paint_line(window, a, b, hsla(0.55, 0.4, 0.5, 0.35), 0.7);
                    }
                    for (x, y, rad, d, _z) in &fluid_quads {
                        let alpha = (0.15 + 0.75 * d).clamp(0.15, 0.9);
                        let light = (0.45 + 0.2 * d).clamp(0.4, 0.75);
                        window.paint_quad(fill(
                            Bounds::new(
                                point(px(x - rad), px(y - rad)),
                                size(px(rad * 2.0), px(rad * 2.0)),
                            ),
                            hsla(0.55, 0.85, light, alpha),
                        ));
                    }
                    for (x, y, rad, z, shape) in &points {
                        let alpha = (1.5 - z / dist).clamp(0.25, 1.0);
                        let color = match shape {
                            Shape::Sphere => hsla(0.53, 0.9, 0.6, alpha),
                            Shape::Cube => hsla(0.08, 0.9, 0.6, alpha),
                        };
                        let round = match shape {
                            Shape::Sphere => px(*rad),
                            Shape::Cube => px(0.0),
                        };
                        window.paint_quad(
                            fill(
                                Bounds::new(
                                    point(px(x - rad), px(y - rad)),
                                    size(px(rad * 2.0), px(rad * 2.0)),
                                ),
                                color,
                            )
                            .corner_radii(round),
                        );
                    }
                },
            ))
            .children(axis_labels.into_iter().map(|(x, y, color, label)| {
                div()
                    .absolute()
                    .left(px(x - 4.0))
                    .top(px(y - 14.0))
                    .text_size(px(11.0))
                    .text_color(color)
                    .child(label)
            }))
            .child(hud)
            .child(toolbar)
            .child(self.render_panels(cx))
            .children(self.debug.then(|| self.render_debug(quads_drawn, w, h)))
    }

    /// Right-side panels: object spawn and physics laws.
    fn render_panels(&mut self, cx: &mut Context<Self>) -> Div {
        let (sphere, cube) = (
            Button::new("shape-sphere").label("Sphere"),
            Button::new("shape-cube").label("Cube"),
        );
        let (sphere, cube) = (
            if self.spawn_shape == Shape::Sphere {
                sphere.primary()
            } else {
                sphere
            },
            if self.spawn_shape == Shape::Cube {
                cube.primary()
            } else {
                cube
            },
        );
        let (newton, fluid) = (
            Button::new("mode-newton").label("Newton"),
            Button::new("mode-fluid").label("Fluid (N-S)"),
        );
        let (newton, fluid) = (
            if self.mode == PhysicsMode::Newton {
                newton.primary()
            } else {
                newton
            },
            if self.mode == PhysicsMode::Fluid {
                fluid.primary()
            } else {
                fluid
            },
        );

        let spawn_panel = div()
            .flex()
            .flex_col()
            .gap_2()
            .w(px(250.0))
            .p_3()
            .rounded_lg()
            .bg(rgb(0x151b23))
            .child(div().text_size(px(13.0)).text_color(FG).child("Spawn"))
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap_2()
                    .child(div().w(px(64.0)).text_color(FAINT).child("Shape"))
                    .child(sphere.on_click(cx.listener(|this, _: &ClickEvent, _, _| {
                        this.spawn_shape = Shape::Sphere;
                    })))
                    .child(cube.on_click(cx.listener(|this, _: &ClickEvent, _, _| {
                        this.spawn_shape = Shape::Cube;
                    }))),
            )
            .child(stepper(
                "count",
                "Count",
                self.spawn_count.to_string(),
                cx,
                count_dec,
                count_inc,
            ))
            .child(stepper(
                "size",
                "Size",
                format!("{:.2}", self.spawn_radius),
                cx,
                radius_dec,
                radius_inc,
            ))
            .child(stepper(
                "speed",
                "Speed",
                format!("{:.1}", self.spawn_speed),
                cx,
                speed_dec,
                speed_inc,
            ))
            .child(
                Button::new("spawn")
                    .primary()
                    .label("Spawn")
                    .on_click(cx.listener(|this, _: &ClickEvent, _, _| {
                        this.spawn_from_panel(this.spawn_count);
                    })),
            );

        let law_panel = div()
            .flex()
            .flex_col()
            .gap_2()
            .w(px(250.0))
            .p_3()
            .rounded_lg()
            .bg(rgb(0x151b23))
            .child(div().text_size(px(13.0)).text_color(FG).child("Physics"))
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap_2()
                    .child(div().w(px(64.0)).text_color(FAINT).child("Laws"))
                    .child(newton.on_click(cx.listener(|this, _: &ClickEvent, _, _| {
                        this.mode = PhysicsMode::Newton;
                    })))
                    .child(fluid.on_click(cx.listener(|this, _: &ClickEvent, _, _| {
                        this.mode = PhysicsMode::Fluid;
                    }))),
            );

        let law_panel = match self.mode {
            PhysicsMode::Newton => law_panel
                .child(stepper(
                    "gravity",
                    "Gravity",
                    format!("{:.1}", self.world.settings.gravity),
                    cx,
                    gravity_dec,
                    gravity_inc,
                ))
                .child(stepper(
                    "bounce",
                    "Bounce",
                    format!("{:.2}", self.world.settings.floor_restitution),
                    cx,
                    bounce_dec,
                    bounce_inc,
                ))
                .child(stepper(
                    "friction",
                    "Friction",
                    format!("{:.2}", self.world.settings.ground_friction),
                    cx,
                    friction_dec,
                    friction_inc,
                )),
            PhysicsMode::Fluid => law_panel
                .child(stepper(
                    "viscosity",
                    "Viscosity",
                    fmt_rate(self.fluid.viscosity),
                    cx,
                    visc_dec,
                    visc_inc,
                ))
                .child(stepper(
                    "diffusion",
                    "Diffusion",
                    fmt_rate(self.fluid.diffusion),
                    cx,
                    diff_dec,
                    diff_inc,
                ))
                .child(stepper(
                    "emit",
                    "Emit",
                    format!("{:.1}", self.fluid.emit),
                    cx,
                    emit_dec,
                    emit_inc,
                )),
        };

        div()
            .absolute()
            .top_2()
            .right_2()
            .flex()
            .flex_col()
            .gap_2()
            .child(spawn_panel)
            .child(law_panel)
    }

    /// Bottom-left overlay with CPU, GPU, and memory stats.
    // ponytail: per-thread OS load needs Win32 FFI; the phase split stands in for it
    fn render_debug(&self, quads: usize, w: f32, h: f32) -> Div {
        let head = |t: &str| {
            div()
                .text_color(FG)
                .text_size(px(12.0))
                .child(t.to_string())
        };
        let line = |t: String| div().text_color(FAINT).child(t);
        let p = self.world.phase_ms;
        let step_total: f32 = p.iter().sum();
        let phase = |name: &str, ms: f32| {
            let share = if step_total > 0.0 {
                100.0 * ms / step_total
            } else {
                0.0
            };
            format!("{name:<9} {ms:7.3} ms {share:5.1} %")
        };
        let path = if self.mode == PhysicsMode::Fluid {
            format!("fluid grid {} x {}", self.fluid.n, self.fluid.n)
        } else if self.world.bodies.len() >= PAR_MIN {
            "parallel".to_string()
        } else {
            "inline".to_string()
        };

        div()
            .absolute()
            .bottom_2()
            .left_2()
            .flex()
            .flex_col()
            .gap_3()
            .p_3()
            .rounded_lg()
            .bg(rgb(0x10141b))
            .text_size(px(11.0))
            .child(head("CPU"))
            .child(line(format!(
                "FPS {:.0}   frame {:.1} ms   step {:.3} ms",
                self.fps,
                1000.0 / self.fps,
                self.step_ms
            )))
            .child(line(format!(
                "bodies {}   contacts {}   threads {}",
                self.world.bodies.len(),
                self.world.contact_count(),
                thread_count()
            )))
            .child(line(format!("path: {path}   PAR_MIN {PAR_MIN}")))
            .child(line(phase("integrate", p[0])))
            .child(line(phase("grid", p[1])))
            .child(line(phase("contacts", p[2])))
            .child(line(phase("resolve", p[3])))
            .child(line(phase("floor", p[4])))
            .child(head("GPU"))
            .child(line(format!("viewport {:.0} x {:.0} px", w, h)))
            .child(line(format!("quads painted {quads}")))
            .child(line(format!(
                "scene (project + sort) {:.3} ms",
                self.scene_ms
            )))
            .child(line("gpui does not expose GPU timers.".to_string()))
            .child(head("Memory"))
            .child(line(format!(
                "in use {:.1} MB   peak {:.1} MB",
                allocated_bytes() as f32 / 1048576.0,
                peak_bytes() as f32 / 1048576.0
            )))
            .child(line(format!(
                "bodies array {:.2} MB",
                (size_of::<Body>() * self.world.bodies.len()) as f32 / 1048576.0
            )))
    }
}

fn fmt_rate(x: f32) -> String {
    if x <= 0.0 {
        "0".to_string()
    } else {
        format!("{x:.1e}")
    }
}

fn count_dec(v: &mut SimView) {
    v.spawn_count = v.spawn_count.saturating_sub(100);
}

fn count_inc(v: &mut SimView) {
    v.spawn_count = (v.spawn_count + 100).min(100_000);
}

fn radius_dec(v: &mut SimView) {
    v.spawn_radius = (v.spawn_radius - 0.05).max(0.05);
}

fn radius_inc(v: &mut SimView) {
    v.spawn_radius = (v.spawn_radius + 0.05).min(2.0);
}

fn speed_dec(v: &mut SimView) {
    v.spawn_speed = (v.spawn_speed - 1.0).max(0.0);
}

fn speed_inc(v: &mut SimView) {
    v.spawn_speed = (v.spawn_speed + 1.0).min(30.0);
}

fn gravity_dec(v: &mut SimView) {
    v.world.settings.gravity = (v.world.settings.gravity - 1.0).max(-40.0);
}

fn gravity_inc(v: &mut SimView) {
    v.world.settings.gravity = (v.world.settings.gravity + 1.0).min(0.0);
}

fn bounce_dec(v: &mut SimView) {
    v.world.settings.floor_restitution = (v.world.settings.floor_restitution - 0.05).max(0.0);
}

fn bounce_inc(v: &mut SimView) {
    v.world.settings.floor_restitution = (v.world.settings.floor_restitution + 0.05).min(1.0);
}

fn friction_dec(v: &mut SimView) {
    v.world.settings.ground_friction = (v.world.settings.ground_friction - 0.05).max(0.0);
}

fn friction_inc(v: &mut SimView) {
    v.world.settings.ground_friction = (v.world.settings.ground_friction + 0.05).min(1.0);
}

fn visc_dec(v: &mut SimView) {
    let x = v.fluid.viscosity;
    v.fluid.viscosity = if x <= 1e-6 { 0.0 } else { x / 2.0 };
}

fn visc_inc(v: &mut SimView) {
    let x = v.fluid.viscosity;
    v.fluid.viscosity = if x <= 0.0 { 1e-6 } else { (x * 2.0).min(1e-2) };
}

fn diff_dec(v: &mut SimView) {
    let x = v.fluid.diffusion;
    v.fluid.diffusion = if x <= 1e-6 { 0.0 } else { x / 2.0 };
}

fn diff_inc(v: &mut SimView) {
    let x = v.fluid.diffusion;
    v.fluid.diffusion = if x <= 0.0 { 1e-6 } else { (x * 2.0).min(1e-2) };
}

fn emit_dec(v: &mut SimView) {
    v.fluid.emit = (v.fluid.emit - 0.5).max(0.0);
}

fn emit_inc(v: &mut SimView) {
    v.fluid.emit = (v.fluid.emit + 0.5).min(10.0);
}

/// One parameter row: label, minus, value, plus. `dec` and `inc` apply one step.
fn stepper(
    id: &'static str,
    label: &str,
    value: String,
    cx: &mut Context<SimView>,
    dec: fn(&mut SimView),
    inc: fn(&mut SimView),
) -> Div {
    div()
        .flex()
        .items_center()
        .gap_2()
        .child(div().w(px(64.0)).text_color(FAINT).child(label.to_string()))
        .child(
            Button::new((id, 1usize))
                .label("-")
                .on_click(cx.listener(move |this, _: &ClickEvent, _, _| dec(this))),
        )
        .child(div().w(px(64.0)).text_color(FG).child(value))
        .child(
            Button::new((id, 2usize))
                .label("+")
                .on_click(cx.listener(move |this, _: &ClickEvent, _, _| inc(this))),
        )
}

/// Paints a world-space line segment as a thin filled quad.
fn paint_line(
    window: &mut Window,
    a: (f32, f32, f32, f32),
    b: (f32, f32, f32, f32),
    color: Hsla,
    thickness: f32,
) {
    let (x1, y1) = (a.0, a.1);
    let (x2, y2) = (b.0, b.1);
    let dx = x2 - x1;
    let dy = y2 - y1;
    let len = (dx * dx + dy * dy).sqrt().max(1e-6);
    let nx = -dy / len * thickness;
    let ny = dx / len * thickness;
    let mut path = PathBuilder::default();
    path.move_to(point(px(x1 + nx), px(y1 + ny)));
    path.line_to(point(px(x2 + nx), px(y2 + ny)));
    path.line_to(point(px(x2 - nx), px(y2 - ny)));
    path.line_to(point(px(x1 - nx), px(y1 - ny)));
    if let Ok(p) = path.build() {
        window.paint_path(p, color);
    }
}
