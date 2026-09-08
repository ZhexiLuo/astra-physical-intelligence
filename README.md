# Astra Physical Intelligence

[Explore the project](https://zhexiluo.github.io/astra-physical-intelligence/).

This repository currently publishes a frozen tabletop-cleaning prototype: a six-second reference video informs a scripted task and inverse-kinematics controller for a TIAGo++ dual-arm robot model. MuJoCo executes the controls, and Blender replays the saved motion. The audited reference run collects six of six blocks; disabling brush–block contact with the same initial state and controls collects zero.

The current result uses estimated dimensions and physical parameters, a fixed base, rigidly attached pre-grasped tools, and a rigid bristle collision proxy. It does not include policy training or learned-policy results. The browser's 3D viewer displays recorded geometry and motion, rather than executing physics.

## Contents

- `docs/`: the static GitHub Pages site, comparison videos, interactive 3D replay, complete GUI recording, and audit records.
- `docs/downloads/reproduce.zip`: frozen simulation source, robot assets, saved controls and contacts, independent tests, and reproduction commands.
- `docs/downloads/scene.blend`: the self-contained Blender scene and three camera views.
- `docs/downloads/web-viewer-source.zip`: the separate glTF exporter, browser viewer, pinned dependencies, and transfer audit.

The reference simulation archive and Blender file retain their original hashes. The site is published from `main:/docs`, leaving the repository root available for subsequent research code.

## Run and inspect

Follow the [engineering walkthrough](https://zhexiluo.github.io/astra-physical-intelligence/workflow.html) for the full simulation and rendering pipeline, or the [viewer guide](https://zhexiluo.github.io/astra-physical-intelligence/downloads/web-viewer-guide.md) to export the saved scene for a browser.

## Model and licenses

TIAGo++ is a PAL Robotics platform. The model is from [MuJoCo Menagerie at revision 8161bba264d7fa7c99ca301e91e7fb44737676ad](https://github.com/google-deepmind/mujoco_menagerie/tree/8161bba264d7fa7c99ca301e91e7fb44737676ad/pal_tiago_dual); its Apache-2.0 license is preserved with the downloads. The pinned model-viewer bundle and its dependency licenses are in `docs/vendor/`. Those licenses apply to their respective upstream components.
