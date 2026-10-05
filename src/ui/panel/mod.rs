//! Right-side panels: object spawn and physics laws.

/// Debug overlay page.
pub mod debug;

/// Top-left HUD and spawn toolbar.
pub mod hud;

/// Menu page.
pub mod menu;

/// Settings page: layout presets and move bindings.
pub mod settings;

use gpui_kit::component::button::{Button, ButtonVariants};
use gpui_kit::*;
use particule_simulation_4d::engine::Shape;

use self::debug::fmt_rate;
use crate::ui::theme::{FAINT, FG};
use crate::ui::{PhysicsMode, SimView};

impl SimView {
    /// Right-side panels: object spawn and physics laws.
    pub(crate) fn render_panels(&mut self, cx: &mut Context<Self>) -> Div {
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
                Setting::Count,
            ))
            .child(stepper(
                "size",
                "Size",
                format!("{:.2}", self.spawn_radius),
                cx,
                Setting::Radius,
            ))
            .child(stepper(
                "speed",
                "Speed",
                format!("{:.1}", self.spawn_speed),
                cx,
                Setting::Speed,
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
                    Setting::Gravity,
                ))
                .child(stepper(
                    "bounce",
                    "Bounce",
                    format!("{:.2}", self.world.settings.floor_restitution),
                    cx,
                    Setting::Bounce,
                ))
                .child(stepper(
                    "friction",
                    "Friction",
                    format!("{:.2}", self.world.settings.ground_friction),
                    cx,
                    Setting::Friction,
                )),
            PhysicsMode::Fluid => law_panel
                .child(stepper(
                    "viscosity",
                    "Viscosity",
                    fmt_rate(self.fluid.viscosity),
                    cx,
                    Setting::Viscosity,
                ))
                .child(stepper(
                    "diffusion",
                    "Diffusion",
                    fmt_rate(self.fluid.diffusion),
                    cx,
                    Setting::Diffusion,
                ))
                .child(stepper(
                    "emit",
                    "Emit",
                    format!("{:.1}", self.fluid.emit),
                    cx,
                    Setting::Emit,
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
}

/// One stepper-controlled panel value.
#[derive(Clone, Copy)]
enum Setting {
    Count,
    Radius,
    Speed,
    Gravity,
    Bounce,
    Friction,
    Viscosity,
    Diffusion,
    Emit,
}

/// Applies one step to `setting`. A negative `dir` lowers the value.
fn adjust(v: &mut SimView, setting: Setting, dir: i32) {
    let down = dir < 0;
    match setting {
        Setting::Count => {
            v.spawn_count = if down {
                v.spawn_count.saturating_sub(100)
            } else {
                (v.spawn_count + 100).min(100_000)
            };
        }
        Setting::Radius => v.spawn_radius = linear(v.spawn_radius, down, 0.05, 0.05, 2.0),
        Setting::Speed => v.spawn_speed = linear(v.spawn_speed, down, 1.0, 0.0, 30.0),
        Setting::Gravity => {
            v.world.settings.gravity = linear(v.world.settings.gravity, down, 1.0, -40.0, 0.0);
        }
        Setting::Bounce => {
            v.world.settings.floor_restitution =
                linear(v.world.settings.floor_restitution, down, 0.05, 0.0, 1.0);
        }
        Setting::Friction => {
            v.world.settings.ground_friction =
                linear(v.world.settings.ground_friction, down, 0.05, 0.0, 1.0);
        }
        Setting::Viscosity => rate(&mut v.fluid.viscosity, down),
        Setting::Diffusion => rate(&mut v.fluid.diffusion, down),
        Setting::Emit => v.fluid.emit = linear(v.fluid.emit, down, 0.5, 0.0, 10.0),
    }
}

/// Steps `x` by `step` and clamps the result to `min` or `max`.
fn linear(x: f32, down: bool, step: f32, min: f32, max: f32) -> f32 {
    if down {
        (x - step).max(min)
    } else {
        (x + step).min(max)
    }
}

/// Halves or doubles a rate. Near-zero turns off; growth stops at the cap.
fn rate(x: &mut f32, down: bool) {
    if down {
        *x = if *x <= 1e-6 { 0.0 } else { *x / 2.0 };
    } else {
        *x = if *x <= 0.0 {
            1e-6
        } else {
            (*x * 2.0).min(1e-2)
        };
    }
}

/// One parameter row: label, minus, value, plus. The buttons apply one step.
fn stepper(
    id: &'static str,
    label: &str,
    value: String,
    cx: &mut Context<SimView>,
    setting: Setting,
) -> Div {
    div()
        .flex()
        .items_center()
        .gap_2()
        .child(div().w(px(64.0)).text_color(FAINT).child(label.to_string()))
        .child(Button::new((id, 1usize)).label("-").on_click(cx.listener(
            move |this, _: &ClickEvent, _, _| {
                adjust(this, setting, -1);
            },
        )))
        .child(div().w(px(64.0)).text_color(FG).child(value))
        .child(Button::new((id, 2usize)).label("+").on_click(cx.listener(
            move |this, _: &ClickEvent, _, _| {
                adjust(this, setting, 1);
            },
        )))
}
