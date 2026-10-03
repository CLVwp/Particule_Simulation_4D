//! Pages: menu, key settings, and the simulation view.
//! The simulation view owns the camera, the axes, the floor grid, and the HUD.

use std::collections::HashSet;
use std::time::Instant;

use gpui_kit::component::button::{Button, ButtonVariants};
use gpui_kit::*;

use crate::engine::{BODY_RADIUS, World};

const FIXED_DT: f32 = 1.0 / 60.0;
const SPAWN_ORIGIN: [f32; 3] = [0.0, 4.0, 0.0];
const SPAWN_SPEED: f32 = 4.0;
const GRID_HALF: i32 = 10; // floor grid spans -10..=10 units
const GRID_STEP: i32 = 1;
const AXIS_LEN: f32 = 2.0;

const BG: u32 = 0x0b0e14;
const FG: Hsla = hsla(0.58, 0.15, 0.9, 1.0);
const FAINT: Hsla = hsla(0.58, 0.15, 0.85, 0.9);

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

pub struct SimView {
    page: Page,
    world: World,
    // Key bindings, one key per `MoveAction`, indexed by `action as usize`.
    bindings: [String; 6],
    rebinding: Option<MoveAction>,
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
    // Debug HUD values.
    fps: f32,
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
            target: [0.0, 1.0, 0.0],
            yaw: 0.6,
            pitch: 0.35,
            dist: 12.0,
            drag: None,
            last_mouse: None,
            keys: HashSet::new(),
            focus: cx.focus_handle(),
            fps: 60.0,
            last_frame: None,
        }
    }

    /// True while the key bound to `action` is held down.
    fn held(&self, action: MoveAction) -> bool {
        self.keys.contains(&self.bindings[action as usize])
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
            .on_mouse_down(MouseButton::Left, cx.listener(|this, _: &MouseDownEvent, window, cx| {
                window.focus(&this.focus, cx);
            }))
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
                    .child(
                        div()
                            .w(px(110.0))
                            .text_color(FAINT)
                            .child(action.label()),
                    )
                    .child(
                        Button::new(("bind", action as usize))
                            .label(key)
                            .on_click(cx.listener(move |this, _: &ClickEvent, window, cx| {
                                this.rebinding = Some(action);
                                window.focus(&this.focus, cx);
                            })),
                    ),
            );
        }

        div()
            .size_full()
            .bg(rgb(BG))
            .flex()
            .items_center()
            .justify_center()
            .track_focus(&self.focus)
            .on_mouse_down(MouseButton::Left, cx.listener(|this, _: &MouseDownEvent, window, cx| {
                window.focus(&this.focus, cx);
            }))
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
                    .child(rows)
                    .child(
                        Button::new("back")
                            .label("Back")
                            .on_click(cx.listener(|this, _: &ClickEvent, _, _| {
                                this.page = Page::Menu;
                                this.rebinding = None;
                            })),
                    ),
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
            this.world.step(FIXED_DT);
            cx.notify();
        });

        let viewport = window.viewport_size();
        let (w, h) = (f32::from(viewport.width), f32::from(viewport.height));
        let dist = self.dist;

        // Project everything up-front; the paint closure only draws.
        let mut points: Vec<(f32, f32, f32, f32)> = self
            .world
            .bodies
            .iter()
            .map(|b| self.project(b.pos, w, h))
            .collect();
        points.sort_by(|a, b| b.3.total_cmp(&a.3)); // painter's algorithm: far first

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
            (
                [AXIS_LEN, 0.0, 0.0],
                hsla(0.0, 0.8, 0.55, 1.0),
                "X",
            ),
            (
                [0.0, AXIS_LEN, 0.0],
                hsla(0.33, 0.8, 0.5, 1.0),
                "Y",
            ),
            (
                [0.0, 0.0, AXIS_LEN],
                hsla(0.58, 0.8, 0.6, 1.0),
                "Z",
            ),
        ];
        let mut axis_lines: Vec<((f32, f32, f32, f32), (f32, f32, f32, f32), Hsla)> =
            Vec::with_capacity(3);
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
            .child(
                Button::new("menu")
                    .label("Menu")
                    .on_click(cx.listener(|this, _: &ClickEvent, _, _| {
                        this.page = Page::Menu;
                        this.drag = None;
                    })),
            );

        let toolbar = div()
            .absolute()
            .top_2()
            .left_1_2()
            .flex()
            .gap_2()
            .child(
                Button::new("add-100")
                    .label("+100")
                    .on_click(cx.listener(|this, _: &ClickEvent, _, _| {
                        this.world.spawn_wave(100, SPAWN_ORIGIN, SPAWN_SPEED);
                    })),
            )
            .child(
                Button::new("add-1000")
                    .label("+1000")
                    .on_click(cx.listener(|this, _: &ClickEvent, _, _| {
                        this.world.spawn_wave(1000, SPAWN_ORIGIN, SPAWN_SPEED);
                    })),
            )
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
            .on_mouse_down(MouseButton::Left, cx.listener(|this, ev: &MouseDownEvent, window, cx| {
                this.drag = Some(if ev.modifiers.shift {
                    Drag::Pan
                } else {
                    Drag::Orbit
                });
                this.last_mouse = Some(ev.position);
                window.focus(&this.focus, cx);
            }))
            .on_mouse_down(MouseButton::Middle, cx.listener(|this, ev: &MouseDownEvent, window, cx| {
                this.drag = Some(Drag::Pan);
                this.last_mouse = Some(ev.position);
                window.focus(&this.focus, cx);
            }))
            .on_mouse_up(MouseButton::Left, cx.listener(|this, _: &MouseUpEvent, _w, _cx| {
                this.drag = None;
            }))
            .on_mouse_up(MouseButton::Middle, cx.listener(|this, _: &MouseUpEvent, _w, _cx| {
                this.drag = None;
            }))
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
                    for (x, y, focal, z) in points {
                        let rad = (BODY_RADIUS * focal / z).max(1.5);
                        let alpha = (1.5 - z / dist).clamp(0.25, 1.0);
                        window.paint_quad(
                            fill(
                                Bounds::new(
                                    point(px(x - rad), px(y - rad)),
                                    size(px(rad * 2.0), px(rad * 2.0)),
                                ),
                                hsla(0.53, 0.9, 0.6, alpha),
                            )
                            .corner_radii(px(rad)),
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
    }
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
