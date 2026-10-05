//! Move actions, key bindings, and camera slide input.

use gpui_kit::*;

use crate::ui::{SPAWN_ORIGIN, SimView};

/// One pixel of drag turns the camera by this many radians.
pub(crate) const ORBIT_SENSITIVITY: f32 = 0.01;
/// Slide speed as a fraction of the camera distance.
const MOVE_SPEED_FRACTION: f32 = 0.03;

/// One move action. Each action holds one bound key.
#[derive(Clone, Copy, PartialEq)]
pub(crate) enum MoveAction {
    Forward,
    Back,
    Left,
    Right,
    Rise,
    Sink,
}

impl MoveAction {
    /// Every action, in binding order.
    pub(crate) fn all() -> [MoveAction; 6] {
        [
            MoveAction::Forward,
            MoveAction::Back,
            MoveAction::Left,
            MoveAction::Right,
            MoveAction::Rise,
            MoveAction::Sink,
        ]
    }

    /// Menu label for this action.
    pub(crate) fn label(self) -> &'static str {
        match self {
            MoveAction::Forward => "Slide forward",
            MoveAction::Back => "Slide backward",
            MoveAction::Left => "Slide left",
            MoveAction::Right => "Slide right",
            MoveAction::Rise => "Rise",
            MoveAction::Sink => "Sink",
        }
    }

    /// Default bound key for this action.
    pub(crate) fn default_key(self) -> &'static str {
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

/// Active mouse drag gesture.
#[derive(Clone, Copy, PartialEq)]
pub(crate) enum Drag {
    Orbit,
    Pan,
}

/// Keyboard layout preset for the move bindings.
#[derive(Clone, Copy, PartialEq)]
pub(crate) enum KeyLayout {
    Qwerty,
    Azerty,
}

impl KeyLayout {
    /// Menu label for this layout.
    pub(crate) fn label(self) -> &'static str {
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

impl SimView {
    /// Applies the active layout preset to every binding.
    pub(crate) fn apply_layout(&mut self) {
        self.bindings = self.layout.keys().map(str::to_string);
    }

    /// Spawns `n` bodies with the panel parameters.
    pub(crate) fn spawn_from_panel(&mut self, n: usize) {
        self.world.spawn(
            n,
            SPAWN_ORIGIN,
            self.spawn_speed,
            self.spawn_shape,
            self.spawn_radius,
        );
    }

    /// Applies the held keys: slides follow the camera, rise and sink use the world axis.
    pub(crate) fn apply_keys(&mut self) {
        let speed = self.dist * MOVE_SPEED_FRACTION;
        let fwd = self.forward();
        let right = self.right();
        let mut t = self.target;
        if self.is_held(MoveAction::Forward) {
            for a in 0..3 {
                t[a] += fwd[a] * speed;
            }
        }
        if self.is_held(MoveAction::Back) {
            for a in 0..3 {
                t[a] -= fwd[a] * speed;
            }
        }
        if self.is_held(MoveAction::Right) {
            for a in 0..3 {
                t[a] += right[a] * speed;
            }
        }
        if self.is_held(MoveAction::Left) {
            for a in 0..3 {
                t[a] -= right[a] * speed;
            }
        }
        if self.is_held(MoveAction::Rise) {
            t[1] += speed;
        }
        if self.is_held(MoveAction::Sink) {
            t[1] -= speed;
        }
        self.target = t;
    }

    /// Stores a pressed key. While a rebind waits, the next key becomes the binding.
    pub(crate) fn handle_key_down(&mut self, ev: &KeyDownEvent) {
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
}
