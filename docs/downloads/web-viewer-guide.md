# Open the Blender replay in a browser

The interactive scene is exported from the same self-contained Blender replay used for the three rendered videos. MuJoCo computes the physics first. The browser displays the saved geometry and animation, with free camera controls and a timeline.

Download [the viewer source](https://zhexiluo.github.io/astra-physical-intelligence/downloads/web-viewer-source.zip) and [the original Blender scene](https://zhexiluo.github.io/astra-physical-intelligence/downloads/scene.blend). The source package contains the export script, viewer controls, a minimal HTML page, and a local copy of `@google/model-viewer` 4.3.1 with its license and package provenance.

## Export

Extract the source ZIP, put `scene.blend` in its root, and run Blender 4.2.1:

```bash
/absolute/path/to/blender --background --threads 4 --python-exit-code 1 \
  --python src/export_web_scene.py -- \
  --blend scene.blend --output media/scene.glb
```

The script opens the scene in memory, bakes the procedural table color into an embedded texture, and exports the meshes and the recorded motion as a single glTF animation. It does not save changes to the source `.blend`. The export uses four CPU threads; a GPU is not required.

The export explicitly merges objects into one scene clip and retains all sampled transform channels. Blender's default animation-size optimization can remove nearly constant motion; preserving the samples keeps those small movements available to the browser replay.

## Explore

Serve the extracted folder with a local HTTP server:

```bash
python3 -m http.server 8879 --bind 127.0.0.1
```

Open `http://127.0.0.1:8879/`, then click **Load interactive scene**. Drag to orbit, shift-drag to pan, and scroll to zoom. Use Play, the timeline, or the three camera presets to inspect the recorded motion. Playback pauses when the viewer leaves the viewport or the tab is hidden.

The local viewer bundle and model load only after that click. The scene contains uncompressed geometry and embedded textures, so it does not need a remote decoder service. Browser materials approximate Blender's lighting and shader appearance; the original scene remains the reference for Cycles rendering.

## Verify the transfer

Run the included audit using Blender's bundled Python and NumPy:

```bash
/absolute/path/to/blender --background --threads 4 --python-exit-code 1 \
  --python agent/test/web_scene_contract.py -- \
  --blend "$PWD/scene.blend" --glb "$PWD/media/scene.glb" \
  --output "$PWD/media/web-scene-audit.json"
```

The published transfer audit compares all 58 meshes' world transforms at all 174 frames after glTF export and Blender reimport. These are rigid transforms, without mesh deformation. At frame zero, it compares vertex sets and matches each triangle's three corners to a triangle in the other mesh, testing all six corner permutations in double precision. Both checks run in both directions. The largest corner or vertex-set distance is `1.3411045e-7 m`; the largest world-matrix element difference is `5.9604645e-7`.

Some source STL meshes contain repeated faces; glTF export removes some of those duplicate instances. The audit preserves raw and unique triangle counts and compares geometry independently of vertex numbering. An earlier float32 surface-distance check failed even when comparing a mesh with itself; the final triangle-corner check avoids that numerical issue. Consult the included audit JSON for per-mesh results and the exact input and output hashes.

The original Blender scene's SHA-256 is:

```text
9823c5bc2ef13cd483f413613c7f84f67179cbc6653b0e34e5f3af707ef7e98d
```

The physics reproduction package is distributed separately as [reproduce.zip](https://zhexiluo.github.io/astra-physical-intelligence/downloads/reproduce.zip). It contains the original control rollout, contact evidence, simulation source, and renderer pipeline.

Sources: [Blender glTF exporter](https://docs.blender.org/manual/en/4.2/addons/import_export/scene_gltf2.html), [model-viewer controls and animation](https://modelviewer.dev/docs/).

The TIAGo++ robot geometry comes from [the pinned MuJoCo Menagerie model](https://github.com/google-deepmind/mujoco_menagerie/tree/8161bba264d7fa7c99ca301e91e7fb44737676ad/pal_tiago_dual). Its original Apache-2.0 license is included in the source package.
