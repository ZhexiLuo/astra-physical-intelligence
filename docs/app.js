"use strict";

const sourceVideo = document.querySelector("#source-video");
const simulationVideo = document.querySelector("#simulation-video");
const videos = [sourceVideo, simulationVideo];
const playButton = document.querySelector("#play-button");
const restartButton = document.querySelector("#restart-button");
const timeline = document.querySelector("#timeline");
const speedSelect = document.querySelector("#playback-speed");
const message = document.querySelector("#playback-message");
const cameraLabels = { global: "Global view", ego: "Ego view", gripper: "Gripper view" };
let activeCamera = "global";
let progress = 0;
let playing = false;
let resumeAfterCameraChange = false;
let animationFrame = 0;
let playbackRequest = 0;

function formatTime(seconds) {
  const wholeSeconds = Math.floor(seconds);
  return `${String(Math.floor(wholeSeconds / 60)).padStart(2, "0")}:${String(wholeSeconds % 60).padStart(2, "0")}`;
}

function videoReady(video) {
  return video.readyState >= 1 && Number.isFinite(video.duration) && video.duration > 0 && !video.error;
}

function updateControls() {
  const ready = videos.every(videoReady);
  playButton.disabled = !ready;
  restartButton.disabled = !ready;
  timeline.disabled = !ready;
  document.querySelector("#play-label").textContent = playing ? "Pause" : "Play";
  document.querySelector("#play-icon").textContent = playing ? "Ⅱ" : "▶";
  playButton.setAttribute("aria-label", playing ? "Pause both videos" : "Play both videos");
}

function updateReadouts() {
  for (const [index, video] of videos.entries()) {
    const time = document.querySelector(index === 0 ? "#source-time" : "#simulation-time");
    time.textContent = videoReady(video) ? `${formatTime(video.currentTime)} / ${formatTime(video.duration)}` : "00:00 / --:--";
  }
  timeline.value = String(Math.round(progress * 1000));
  document.querySelector("#progress-label").textContent = `${Math.round(progress * 100)}%`;
  timeline.setAttribute("aria-valuetext", `${Math.round(progress * 100)}%`);
}

function seekProgress(value) {
  progress = Math.max(0, Math.min(1, value));
  for (const video of videos) {
    if (videoReady(video)) video.currentTime = progress * video.duration;
  }
  updateReadouts();
}

function pausePlayback() {
  playbackRequest += 1;
  playing = false;
  resumeAfterCameraChange = false;
  cancelAnimationFrame(animationFrame);
  videos.forEach(video => video.pause());
  updateControls();
}

function setPlaybackRates() {
  const rate = Number(speedSelect.value);
  simulationVideo.playbackRate = rate;
  sourceVideo.playbackRate = sourceVideo.duration / simulationVideo.duration * rate;
}

function tick() {
  progress = simulationVideo.currentTime / simulationVideo.duration;
  const sourceTarget = progress * sourceVideo.duration;
  if (Math.abs(sourceVideo.currentTime - sourceTarget) > 0.12) sourceVideo.currentTime = sourceTarget;
  updateReadouts();
  if (playing) animationFrame = requestAnimationFrame(tick);
}

async function startPlayback() {
  if (!videos.every(videoReady)) return;
  if (progress >= 0.995) seekProgress(0);
  const request = ++playbackRequest;
  setPlaybackRates();
  playing = true;
  updateControls();
  message.textContent = "";
  try {
    await Promise.all(videos.map(video => video.play()));
    if (request !== playbackRequest) return;
    animationFrame = requestAnimationFrame(tick);
  } catch (error) {
    if (request !== playbackRequest) return;
    pausePlayback();
    message.textContent = `Playback could not start: ${error.message}`;
  }
}

function togglePlayback() {
  if (playing) pausePlayback();
  else startPlayback();
}

function selectCamera(camera) {
  if (camera === activeCamera) return;
  const resume = playing || resumeAfterCameraChange;
  pausePlayback();
  resumeAfterCameraChange = resume;
  activeCamera = camera;
  document.querySelectorAll("[data-camera]").forEach(button => {
    button.setAttribute("aria-pressed", String(button.dataset.camera === camera));
  });
  document.querySelector("#camera-label").textContent = cameraLabels[camera];
  simulationVideo.setAttribute("aria-label", `Robot simulation: ${cameraLabels[camera]}`);
  simulationVideo.closest(".video-stage").classList.remove("is-ready");
  document.querySelector("#simulation-state > span:last-child").textContent = "Loading this camera view";
  simulationVideo.src = `media/${camera}.mp4`;
  simulationVideo.load();
  updateControls();
}

function connectVideo(video) {
  const stage = video.closest(".video-stage");
  const showVideo = () => {
    stage.classList.add("is-ready");
  };
  video.addEventListener("loadeddata", showVideo);
  if (video.readyState >= 2) showVideo();
  video.addEventListener("error", () => {
    stage.classList.remove("is-ready");
    stage.querySelector(".media-state > span:last-child").textContent = "This video is unavailable";
    if (videos.includes(video)) {
      resumeAfterCameraChange = false;
      pausePlayback();
    }
  });
  if (!videos.includes(video)) return;
  video.addEventListener("loadedmetadata", () => {
    seekProgress(progress);
    updateControls();
    if (video === simulationVideo && resumeAfterCameraChange) {
      resumeAfterCameraChange = false;
      startPlayback();
    }
  });
}

playButton.addEventListener("click", togglePlayback);
restartButton.addEventListener("click", () => seekProgress(0));
timeline.addEventListener("input", () => seekProgress(Number(timeline.value) / 1000));
speedSelect.addEventListener("change", () => {
  if (videos.every(videoReady)) setPlaybackRates();
});
document.querySelectorAll("[data-camera]").forEach(button => {
  button.addEventListener("click", () => selectCamera(button.dataset.camera));
});
document.querySelectorAll("video").forEach(connectVideo);
simulationVideo.addEventListener("ended", () => {
  pausePlayback();
  seekProgress(1);
});
document.addEventListener("visibilitychange", () => {
  if (document.hidden) pausePlayback();
});
document.querySelector(".viewer").addEventListener("keydown", event => {
  if (event.target.matches("input, select, button, a")) return;
  if (event.key === " ") {
    event.preventDefault();
    togglePlayback();
  } else if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
    event.preventDefault();
    seekProgress(progress + (event.key === "ArrowRight" ? 0.05 : -0.05));
  } else if (["1", "2", "3"].includes(event.key)) {
    selectCamera(["global", "ego", "gripper"][Number(event.key) - 1]);
  }
});

async function loadMetrics() {
  const response = await fetch("media/metrics.json");
  if (response.status === 404) return;
  if (!response.ok) throw new Error(`Metrics request failed: HTTP ${response.status}`);
  const metrics = await response.json();
  const metricsLink = document.querySelector("#metrics-link");
  metricsLink.href = "media/metrics.json";
  metricsLink.setAttribute("aria-disabled", "false");
  const status = document.querySelector("#run-status");
  const success = metrics.success === true;
  status.classList.toggle("is-success", success);
  status.replaceChildren();
  const dot = document.createElement("i");
  dot.className = "status-dot orange";
  status.append(dot, success ? "Run verified" : "Run did not pass");
  document.querySelector("#result-description").textContent = success
    ? "This recorded run passed its independent audit. Inspect the collection count, contact measurements, and original logs below."
    : "This run did not pass its audit. The actual measurements remain available for inspecting contact, motion, and uncollected blocks.";
  document.querySelector("#metric-collected").textContent = metrics.collected ?? "—";
  document.querySelector("#metric-total").textContent = metrics.total_blocks ?? "—";
  document.querySelector("#metric-duration").textContent = metrics.duration_s?.toFixed(1) ?? "—";
  document.querySelector("#metric-penetration").textContent = metrics.max_penetration_m == null ? "—" : (metrics.max_penetration_m * 1000).toFixed(2);
  document.querySelector("#metric-impulse").textContent = metrics.contact_impulse_ns?.toFixed(3) ?? "—";
  document.querySelector("#run-id").textContent = metrics.run_id ?? "No run ID provided";
}

async function connectDownload(link) {
  const response = await fetch(link.dataset.download, { method: "HEAD" });
  if (!response.ok) return;
  link.href = link.dataset.download;
  link.setAttribute("aria-disabled", "false");
  link.querySelector(".file-state").textContent = "Download";
}

loadMetrics().catch(error => {
  document.querySelector("#run-status").textContent = "Could not load metrics";
  document.querySelector("#result-description").textContent = error.message;
});
document.querySelectorAll("[data-download]").forEach(link => {
  connectDownload(link).catch(() => {
    link.querySelector(".file-state").textContent = "Unavailable";
  });
});
updateControls();
updateReadouts();
