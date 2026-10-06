//! Menu page and the settings page.

use egui::{CentralPanel, Color32, Context, Frame, Grid, RichText};

use super::root;
use crate::ui::input::{KeyLayout, MoveAction};
use crate::ui::theme::FG;
use crate::ui::widgets::{enum_toggle, faint, faint_px};
use crate::ui::{App, Page};

/// Card fill for the settings page and the side panels.
const CARD: Color32 = Color32::from_rgb(0x15, 0x1b, 0x23);

/// Menu page. Big title and the entry buttons, centered.
pub(crate) fn menu(ctx: &Context, app: &mut App) {
    let mut ui = root(ctx);
    CentralPanel::default().show(&mut ui, |ui| {
        ui.centered_and_justified(|ui| {
            // Plain top-down flow. The parent centers this block once; a
            // Center main-align here would re-center every widget instead.
            ui.with_layout(egui::Layout::top_down(egui::Align::Center), |ui| {
                ui.vertical_centered(|ui| {
                    ui.label(
                        RichText::new("Particule Simulation 4D")
                            .size(28.0)
                            .color(FG),
                    );
                    ui.add_space(16.0);
                    if ui
                        .button(RichText::new("Enter simulation").strong())
                        .clicked()
                    {
                        app.page = Page::Sim;
                    }
                    if ui.button("Settings").clicked() {
                        app.page = Page::Settings;
                    }
                });
            });
        });
    });
}

/// Settings page. Layout preset, one binding row per move action, and back.
pub(crate) fn settings(ctx: &Context, app: &mut App) {
    let mut ui = root(ctx);
    CentralPanel::default().show(&mut ui, |ui| {
        ui.centered_and_justified(|ui| {
            Frame::default()
                .fill(CARD)
                .inner_margin(32.0)
                .corner_radius(8.0)
                .show(ui, |ui| {
                    // Plain top-down flow inside the card. A Center main-align
                    // inherited from the parent re-centers every widget in the
                    // remaining space and pushes rows off-screen.
                    ui.with_layout(egui::Layout::top_down(egui::Align::Center), |ui| {
                        ui.set_min_width(340.0);
                        ui.label(RichText::new("Settings").size(20.0).color(FG));
                        faint_px(
                            ui,
                            "Pick an action. Then press the new key. Escape cancels.",
                            12.0,
                        );
                        ui.add_space(8.0);
                        ui.horizontal(|ui| {
                            faint(ui, "Key layout");
                            if enum_toggle(
                                ui,
                                &mut app.input.layout,
                                &[
                                    (KeyLayout::Qwerty, KeyLayout::Qwerty.label()),
                                    (KeyLayout::Azerty, KeyLayout::Azerty.label()),
                                ],
                            ) {
                                app.input.apply_layout();
                            }
                        });
                        Grid::new("bind rows")
                            .num_columns(2)
                            .spacing([24.0, 8.0])
                            .show(ui, |ui| {
                                for action in MoveAction::all() {
                                    let listening = app.input.rebinding == Some(action);
                                    let key = if listening {
                                        "press a key...".to_string()
                                    } else {
                                        app.input.bindings[action as usize].to_uppercase()
                                    };
                                    faint(ui, action.label());
                                    if ui.button(key).clicked() {
                                        app.input.rebinding = Some(action);
                                    }
                                    ui.end_row();
                                }
                            });
                        if ui.button("Back").clicked() {
                            app.page = Page::Menu;
                            app.input.rebinding = None;
                        }
                    });
                });
        });
    });
}
