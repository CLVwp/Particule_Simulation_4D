# Particule_Simulation_4D

Real-time particle simulation in **3D — x, y, z, with time as the fourth dimension** — written in Rust on top of [GPUI](https://gpui-kit.com) (Zed's GPU-accelerated UI framework).

A remake of my C++ / Raylib project [Cpp-Raylib-Particule-Simulation](https://github.com/CLVwp/Cpp-Raylib-Particule-Simulation).

> 🚧 Early development — the repo currently holds the gpui-kit scaffold.

## Stack

| | |
|---|---|
| Language | Rust 1.92+ (edition 2024) |
| UI & rendering | [gpui-kit](https://gpui-kit.com) 0.7 (re-exports GPUI + component library) |
| Build system | Cargo |

## Prerequisites

Follow the [gpui-kit installation guide](https://gpui-kit.com/versions/main/docs/installation/):

- **All platforms:** [Rust 1.92+](https://rustup.rs) and CMake on PATH (`cmake --version`)
- **Windows:** Visual Studio 2022 Build Tools or Community with the **C++ desktop workload** (MSVC + Windows SDK), and the `msvc` Rust toolchain (`rustup show active-toolchain`)
- **macOS:** macOS 15+, Xcode Command Line Tools (`xcode-select --install`)
- **Linux** (verified on Ubuntu 24.04):

```bash
sudo apt update
sudo apt install -y gcc g++ clang libfontconfig-dev libwayland-dev \
  libwebkit2gtk-4.1-dev libxkbcommon-x11-dev libx11-xcb-dev \
  libssl-dev libzstd-dev libasound2-dev vulkan-validationlayers libvulkan1
```

## Run

```bash
git clone https://github.com/CLVwp/Particule_Simulation_4D.git
cd Particule_Simulation_4D
cargo run
```

The `Cargo.toml` ships `[profile.dev.package]` overrides (`opt-level = 3` on the gpui stack), which keeps debug builds fast at *runtime* — it does not speed up compilation.

## Roadmap

- [ ] 3D particle system (positions & velocities in x, y, z)
- [ ] Time integration — the 4th dimension
- [ ] Gravity & forces
- [ ] Particle interactions
- [ ] Camera controls (orbit / zoom)
- [ ] UI panel (spawn count, forces, time scale)

## References

- [gpui-kit — Getting started (main)](https://gpui-kit.com/versions/main/docs/getting-started/)
- [gpui-kit — Installation (main)](https://gpui-kit.com/versions/main/docs/installation/)
- Original project: [Cpp-Raylib-Particule-Simulation](https://github.com/CLVwp/Cpp-Raylib-Particule-Simulation)
