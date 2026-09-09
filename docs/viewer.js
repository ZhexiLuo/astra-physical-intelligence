const referenceViews = [
  {id: 'overview', label: 'Overview', target: '0.36m 0.68m 0.05m', orbit: '30deg 66deg 3.9m'},
  {id: 'tabletop', label: 'Tabletop', target: '0.6m 0.78m 0.15m', orbit: '20deg 24deg 1.45m'},
  {id: 'side', label: 'Side', target: '0.45m 0.85m 0m', orbit: '100deg 75deg 2.6m'},
];

function buildSceneShell(mount, config) {
  const timelineId = `${mount.id}-timeline`;
  mount.innerHTML = `
    <div class="scene-stage">
      <img class="scene-poster" src="${config.poster}" alt="${config.alt}" loading="lazy">
      <div class="scene-entry">
        <button class="scene-load" type="button">Load interactive scene <span>${(config.bytes / 1e6).toFixed(1)} MB</span></button>
        <p>Explore the scene in 3D when you are ready.</p>
      </div>
    </div>
    <div class="scene-controls" hidden>
      <div class="scene-transport">
        <button class="scene-play" type="button" aria-label="Play animation">Play</button>
        <input class="scene-timeline" id="${timelineId}" type="range" min="0" max="1" step="0.001" value="0" aria-label="Animation time">
        <output class="scene-time" for="${timelineId}" aria-live="off">0.00 / 0.00 s</output>
      </div>
      <div class="scene-views" role="group" aria-label="Camera presets">
        ${config.views.map(view => `<button type="button" data-view="${view.id}" aria-pressed="${view.id === config.views[0].id}">${view.label}</button>`).join('')}
        <button class="scene-reset" type="button">Reset view &amp; time</button>
      </div>
      ${config.stations.length ? `<label class="scene-station-label">Inspect a workstation
        <select class="scene-stations"><option value="">Choose a station</option>
          ${config.stations.map(station => `<option value="${station.id}">${station.label}</option>`).join('')}
        </select></label>` : ''}
      <p class="scene-help">Drag to orbit · Shift-drag or right-drag to pan · Scroll or pinch to zoom</p>
    </div>
    <p class="scene-status" role="status" aria-live="polite">${config.description}</p>`;
}

function setCamera(mount, model, view) {
  model.cameraTarget = view.target;
  model.cameraOrbit = view.orbit;
  mount.querySelectorAll('[data-view]').forEach(button => {
    button.setAttribute('aria-pressed', String(button.dataset.view === view.id));
  });
}

function connectTimeline(mount, model) {
  const play = mount.querySelector('.scene-play');
  const timeline = mount.querySelector('.scene-timeline');
  const clock = mount.querySelector('.scene-time');
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
    play.textContent = model.paused ? 'Play' : 'Pause';
    play.setAttribute('aria-label', model.paused ? 'Play animation' : 'Pause animation');
    showTime();
  }
  play.addEventListener('click', () => {
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
  model.addEventListener('play', showPlayback);
  model.addEventListener('pause', showPlayback);
  showPlayback();
  return showTime;
}

function connectViews(mount, model, config, showTime) {
  const stations = mount.querySelector('.scene-stations');
  mount.querySelectorAll('[data-view]').forEach(button => {
    button.addEventListener('click', () => {
      setCamera(mount, model, config.views.find(view => view.id === button.dataset.view));
      if (stations) stations.value = '';
    });
  });
  if (stations) stations.addEventListener('change', () => {
    const view = stations.value ? config.stations.find(station => station.id === stations.value) : config.views[0];
    setCamera(mount, model, view);
  });
  mount.querySelector('.scene-reset').addEventListener('click', () => {
    model.pause();
    model.currentTime = 0;
    setCamera(mount, model, config.views[0]);
    if (stations) stations.value = '';
    model.jumpCameraToGoal();
    showTime();
  });
}

function pauseWhenHidden(mount, model) {
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) model.pause();
  });
  new IntersectionObserver(entries => {
    if (!entries[0].isIntersecting) model.pause();
  }).observe(mount);
}

async function loadScene(mount, config) {
  const load = mount.querySelector('.scene-load');
  const stage = mount.querySelector('.scene-stage');
  const status = mount.querySelector('.scene-status');
  load.disabled = true;
  load.textContent = 'Loading scene…';
  status.textContent = 'Loading the 3D scene…';
  await import('./vendor/model-viewer-4.3.1.min.js');
  const model = document.createElement('model-viewer');
  model.id = `${mount.id}-model`;
  for (const [name, value] of Object.entries({
    'alt': config.alt, 'camera-controls': '', 'touch-action': 'pan-y', 'interaction-prompt': 'none',
    'camera-target': config.views[0].target, 'camera-orbit': config.views[0].orbit,
    'min-camera-orbit': 'auto 5deg 0.15m', 'max-camera-orbit': config.maxOrbit,
    'field-of-view': '35deg', 'shadow-intensity': '0', 'exposure': String(config.exposure),
    'animation-crossfade-duration': '0',
  })) model.setAttribute(name, value);
  model.addEventListener('load', async () => {
    model.animationName = model.availableAnimations[0];
    await model.updateComplete;
    model.currentTime = 0;
    connectViews(mount, model, config, connectTimeline(mount, model));
    pauseWhenHidden(mount, model);
    stage.classList.add('is-loaded');
    mount.querySelector('.scene-controls').hidden = false;
    status.textContent = 'Scene loaded. Rotate the view or press Play to explore the recorded motion.';
  }, {once: true});
  model.addEventListener('error', () => {
    status.textContent = 'The 3D scene could not load. The Blender download remains available.';
  });
  stage.append(model);
  model.src = config.scene;
}

async function initializeScene(mount) {
  const config = mount.dataset.config ? await (await fetch(mount.dataset.config)).json() : {
    scene: mount.dataset.scene, poster: mount.dataset.poster, bytes: 7199444,
    alt: 'Interactive TIAGo++ robot sweeping six blocks into a dustpan',
    description: "The browser copy uses the frozen scene's geometry and recorded animation.",
    exposure: 1, views: referenceViews, stations: [], maxOrbit: 'auto 88deg 8m',
  };
  buildSceneShell(mount, config);
  mount.querySelector('.scene-load').addEventListener('click', () => loadScene(mount, config), {once: true});
}

document.querySelectorAll('.scene-viewer').forEach(initializeScene);
