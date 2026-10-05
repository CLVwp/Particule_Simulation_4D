//! Move actions, key bindings, and camera slide input.

use std::collections::HashSet;

use crate::ui::camera::Camera;

/// One pixel of drag turns the camera by this many radians.
pub(crate) const ORBIT_SENSITIVITY: f32 = 0.01;
/// Slide speed as a fraction of the camera distance.
const MOVE_SPEED_FRACTION: f32 = 0.03;

/// One move action. Each action holds one bound key.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum MoveAction {
    /// Slide toward the camera look direction.
    Forward,
    /// Slide away from the camera look direction.
    Back,
    /// Slide toward the camera left.
    Left,
    /// Slide toward the camera right.
    Right,
    /// Rise along the world up axis.
    Rise,
    /// Sink along the world down axis.
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
    /// Left drag: turn the camera.
    Orbit,
    /// Shift+left or middle drag: slide the target.
    Pan,
}

/// Keyboard layout preset for the move bindings.
#[derive(Clone, Copy, PartialEq)]
pub(crate) enum KeyLayout {
    /// Standard "w s a d" preset.
    Qwerty,
    /// Standard "z q s d" preset.
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

/// Held keys, bindings, and the pending rebind. Keys are normalized names
/// such as "w", "f1", or "escape". No window toolkit types live here.
pub(crate) struct InputState {
    /// One key per `MoveAction`, indexed by `action as usize`.
    pub(crate) bindings: [String; 6],
    /// Keys held down right now.
    pub(crate) keys: HashSet<String>,
    /// Action that waits for its next key. Escape cancels.
    pub(crate) rebinding: Option<MoveAction>,
    /// Active layout preset.
    pub(crate) layout: KeyLayout,
}

impl Default for InputState {
    /// Defaults match the old `SimView::new`: QWERTY and the default keys.
    fn default() -> Self {
        Self {
            bindings: MoveAction::all().map(|a| a.default_key().to_string()),
            keys: HashSet::new(),
            rebinding: None,
            layout: KeyLayout::Qwerty,
        }
    }
}

impl InputState {
    /// True while the key bound to `action` is held down.
    pub(crate) fn is_held(&self, action: MoveAction) -> bool {
        self.keys.contains(&self.bindings[action as usize])
    }

    /// Applies the active layout preset to every binding.
    pub(crate) fn apply_layout(&mut self) {
        self.bindings = self.layout.keys().map(str::to_string);
    }
}

/// Applies the held keys. Slides follow the camera. Rise and sink use the
/// world axis. `scale` converts one fixed step of motion into this frame.
pub(crate) fn apply_moves(input: &InputState, cam: &mut Camera, scale: f32) {
    let speed = cam.dist * MOVE_SPEED_FRACTION * scale;
    let fwd = cam.forward();
    let right = cam.right();
    let mut t = cam.target;
    if input.is_held(MoveAction::Forward) {
        for a in 0..3 {
            t[a] += fwd[a] * speed;
        }
    }
    if input.is_held(MoveAction::Back) {
        for a in 0..3 {
            t[a] -= fwd[a] * speed;
        }
    }
    if input.is_held(MoveAction::Right) {
        for a in 0..3 {
            t[a] += right[a] * speed;
        }
    }
    if input.is_held(MoveAction::Left) {
        for a in 0..3 {
            t[a] -= right[a] * speed;
        }
    }
    if input.is_held(MoveAction::Rise) {
        t[1] += speed;
    }
    if input.is_held(MoveAction::Sink) {
        t[1] -= speed;
    }
    cam.target = t;
}

/// Zooms one wheel step. `dy` is the scroll amount in lines. Clamped.
pub(crate) fn zoom(cam: &mut Camera, dy: f32) {
    cam.dist = (cam.dist * (1.0 - 0.1 * dy)).clamp(3.0, 40.0);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_swap_rewrites_bindings() {
        let mut input = InputState::default();
        assert_eq!(input.bindings[MoveAction::Forward as usize], "w");
        assert_eq!(input.bindings[MoveAction::Sink as usize], "q");
        input.layout = KeyLayout::Azerty;
        input.apply_layout();
        assert_eq!(input.bindings[MoveAction::Forward as usize], "z");
        assert_eq!(input.bindings[MoveAction::Sink as usize], "a");
    }

    #[test]
    fn is_held_tracks_pressed_keys() {
        let mut input = InputState::default();
        assert!(!input.is_held(MoveAction::Forward));
        input.keys.insert("w".to_string());
        assert!(input.is_held(MoveAction::Forward));
        assert!(!input.is_held(MoveAction::Back));
    }

    #[test]
    fn apply_moves_scales_slide_speed() {
        let mut input = InputState::default();
        input.keys.insert("w".to_string());
        let mut cam = Camera {
            target: [0.0; 3],
            yaw: 0.0,
            pitch: 0.0,
            dist: 10.0,
        };
        apply_moves(&input, &mut cam, 2.0);
        // Forward is +z at yaw 0. One step covers dist * MOVE_SPEED_FRACTION.
        let expected = 10.0 * MOVE_SPEED_FRACTION * 2.0;
        assert!((cam.target[2] - expected).abs() < 1e-5);
    }

    #[test]
    fn zoom_clamps_distance() {
        let mut cam = Camera {
            target: [0.0; 3],
            yaw: 0.0,
            pitch: 0.0,
            dist: 30.0,
        };
        zoom(&mut cam, -10.0);
        assert_eq!(cam.dist, 40.0);
        cam.dist = 4.0;
        zoom(&mut cam, 10.0);
        assert_eq!(cam.dist, 3.0);
    }
}
