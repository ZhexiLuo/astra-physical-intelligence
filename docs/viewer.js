const mount = document.querySelector('#scene-viewer');
const views = {
  overview: {target: '0.36m 0.68m 0.05m', orbit: '30deg 66deg 3.9m'},
  tabletop: {target: '0.6m 0.78m 0.15m', orbit: '20deg 24deg 1.45m'},
  side: {target: '0.45m 0.85m 0m', orbit: '100deg 75deg 2.6m'},
};

mount.innerHTML = `
  <div class="scene-stage">
    <img class="scene-poster" src="${mount.dataset.poster}" alt="Preview of the robot cleaning scene" loading="lazy">
    <div class="scene-entry">
      <button id="scene-load" type="button">Load interactive scene <span>7.2 MB</span></button>
      <p>No 3D rendering starts until you click.</p>
    </div>
  </div>
  <div class="scene-controls" hidden>
    <div class="scene-transport">
      <button id="scene-play" type="button" aria-label="Play animation">Play</button>
      <input id="scene-timeline" type="range" min="0" max="1" step="0.001" value="0" aria-label="Animation time">
      <output id="scene-time" for="scene-timeline" aria-live="off">0.00 / 0.00 s</output>
    </div>
    <div class="scene-views" role="group" aria-label="Camera presets">
      <button type="button" data-view="overview" aria-pressed="true">Overview</button>
      <button type="button" data-view="tabletop" aria-pressed="false">Tabletop</button>
      <button type="button" data-view="side" aria-pressed="false">Side</button>
      <button id="scene-reset" type="button">Reset view &amp; time</button>
    </div>
    <p class="scene-help">Drag to orbit · Shift-drag or right-drag to pan · Scroll or pinch to zoom</p>
  </div>
  <p id="scene-status" class="scene-status" role="status" aria-live="polite">The browser copy uses the frozen scene's geometry and recorded animation.</p>
`;

const loadButton = mount.querySelector('#scene-load');
const stage = mount.querySelector('.scene-stage');
const status = mount.querySelector('#scene-status');

function connectControls(model) {
  const playButton = mount.querySelector('#scene-play');
  const timeline = mount.querySelector('#scene-timeline');
  const clock = mount.querySelector('#scene-time');
  let tick = 0;
  timeline.max = model.duration;

  function showTime() {
    timeline.value = model.currentTime;
    clock.value = `${model.currentTime.toFixed(2)} / ${model.duration.toFixed(2)} s`;
    timeline.setAttribute('aria-valuetext', `${model.currentTime.toFixed(2)} seconds of ${model.duration.toFixed(2)} seconds`);
    if (!model.paused && model.currentTime >= model.duration) model.pause();
    if (!model.paused) tick = requestAnimationFrame(showTime);
  }

  function showPlayback() {
    cancelAnimationFrame(tick);
    playButton.textContent = model.paused ? 'Play' : 'Pause';
    playButton.setAttribute('aria-label', model.paused ? 'Play animation' : 'Pause animation');
    showTime();
  }

  function setView(name) {
    model.cameraTarget = views[name].target;
    model.cameraOrbit = views[name].orbit;
    mount.querySelectorAll('[data-view]').forEach(button => {
      button.setAttribute('aria-pressed', String(button.dataset.view === name));
    });
  }

  playButton.addEventListener('click', () => {
    if (model.paused) {
      if (model.currentTime >= model.duration - 0.01) model.currentTime = 0;
      model.play({repetitions: 1});
    } else {
      model.pause();
    }
  });
  timeline.addEventListener('input', () => {
    const time = Number(timeline.value);
    model.pause();
    model.currentTime = time;
    showTime();
  });
  mount.querySelectorAll('[data-view]').forEach(button => {
    button.addEventListener('click', () => setView(button.dataset.view));
  });
  mount.querySelector('#scene-reset').addEventListener('click', () => {
    model.pause();
    model.currentTime = 0;
    setView('overview');
    model.jumpCameraToGoal();
    showTime();
  });
  model.addEventListener('play', showPlayback);
  model.addEventListener('pause', showPlayback);
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) model.pause();
  });
  new IntersectionObserver(entries => {
    if (!entries[0].isIntersecting) model.pause();
  }).observe(mount);
  showPlayback();
}

loadButton.addEventListener('click', async () => {
  loadButton.disabled = true;
  loadButton.textContent = 'Loading scene…';
  status.textContent = 'Loading the local 3D viewer and scene…';
  await import('./vendor/model-viewer-4.3.1.min.js');
  const model = document.createElement('model-viewer');
  model.id = 'scene-model';
  for (const [name, value] of Object.entries({
    'alt': 'Interactive TIAGo++ robot sweeping six blocks into a dustpan',
    'camera-controls': '', 'touch-action': 'pan-y', 'interaction-prompt': 'none',
    'camera-target': views.overview.target, 'camera-orbit': views.overview.orbit,
    'min-camera-orbit': 'auto 5deg 0.15m', 'max-camera-orbit': 'auto 88deg 8m',
    'field-of-view': '35deg', 'shadow-intensity': '0', 'exposure': '1',
    'animation-crossfade-duration': '0',
  })) model.setAttribute(name, value);
  model.addEventListener('load', async () => {
    model.animationName = model.availableAnimations[0];
    await model.updateComplete;
    model.currentTime = 0;
    connectControls(model);
    stage.classList.add('is-loaded');
    mount.querySelector('.scene-controls').hidden = false;
    status.textContent = 'Scene loaded. Rotate the view or press Play to explore the recorded motion.';
  }, {once: true});
  model.addEventListener('error', () => {
    status.textContent = 'The 3D scene could not load. The original Blender download remains available.';
  });
  stage.append(model);
  model.src = mount.dataset.scene;
}, {once: true});
