const sections = document.querySelectorAll('article section');
const links = document.querySelectorAll('.contents a');

const sectionObserver = new IntersectionObserver(entries => {
  for (const entry of entries) {
    if (entry.isIntersecting) {
      for (const link of links) {
        link.setAttribute('aria-current', String(link.hash === `#${entry.target.id}`));
      }
    }
  }
}, {rootMargin: '-5% 0px -75% 0px'});
sections.forEach(section => sectionObserver.observe(section));

const observer = new IntersectionObserver(entries => {
  for (const entry of entries) {
    if (!entry.isIntersecting) entry.target.pause();
  }
});
document.querySelectorAll('video').forEach(video => observer.observe(video));
document.addEventListener('visibilitychange', () => {
  if (document.hidden) document.querySelectorAll('video').forEach(video => video.pause());
});

const heroVideo = document.querySelector('#hero-video');
const cameraButtons = document.querySelectorAll('[data-camera]');
for (const button of cameraButtons) {
  button.addEventListener('click', () => {
    const progress = heroVideo.readyState ? heroVideo.currentTime / heroVideo.duration : 0;
    const playing = !heroVideo.paused;
    for (const camera of cameraButtons) camera.disabled = true;
    heroVideo.addEventListener('loadedmetadata', () => {
      heroVideo.currentTime = progress * heroVideo.duration;
      for (const camera of cameraButtons) camera.disabled = false;
      if (playing) heroVideo.play();
    }, {once: true});
    heroVideo.src = `media/${button.dataset.camera}.mp4`;
    for (const camera of cameraButtons) {
      camera.setAttribute('aria-pressed', String(camera === button));
    }
  });
}
