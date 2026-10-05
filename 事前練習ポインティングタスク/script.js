const TARGET_COUNT = 12;
const START_OFFSET = 2;
const TRIAL_COUNT = window.SteeringExperimentConfig.conditions.reduce(
  (total, condition) => total + Number(condition.trials || 0),
  0,
);

const taskArea = document.getElementById("taskArea");
const statusText = document.getElementById("statusText");
const stepValue = document.getElementById("stepValue");
const hitValue = document.getElementById("hitValue");
const missValue = document.getElementById("missValue");
const lapValue = document.getElementById("lapValue");
const timeValue = document.getElementById("timeValue");
const fullscreenButton = document.getElementById("fullscreenButton");
const resetButton = document.getElementById("resetButton");

const sequence = buildIsoSequence(TARGET_COUNT).map(
  (index) => (index + START_OFFSET) % TARGET_COUNT,
);

const state = {
  activeStep: 0,
  hits: 0,
  misses: 0,
  laps: 0,
  completed: false,
  startedAt: performance.now(),
};

const targets = Array.from({ length: TARGET_COUNT }, (_, index) => {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "target";
  button.dataset.state = "idle";
  button.setAttribute("aria-label", `\u30bf\u30fc\u30b2\u30c3\u30c8 ${index + 1}`);
  button.addEventListener("click", () => handleTargetClick(index, button));
  taskArea.append(button);
  return button;
});

function buildIsoSequence(count) {
  const half = count / 2;
  const order = [];

  for (let index = 0; index < half; index += 1) {
    order.push(index, index + half);
  }

  return order;
}

function layoutTargets() {
  const bounds = taskArea.getBoundingClientRect();
  const baseSize = Math.min(bounds.width, bounds.height);
  const targetSize = Math.max(72, Math.min(108, baseSize * 0.125));
  const maxRadiusX = bounds.width / 2 - targetSize / 2 - 24;
  const maxRadiusY = bounds.height / 2 - targetSize / 2 - 24;
  const radius = Math.max(0, Math.min(baseSize * 0.39, maxRadiusX, maxRadiusY));
  const centerX = bounds.width / 2;
  const centerY = bounds.height / 2;

  taskArea.style.setProperty("--target-size", `${targetSize}px`);
  taskArea.style.setProperty("--guide-size", `${radius * 2}px`);

  targets.forEach((button, index) => {
    const angle = ((360 / TARGET_COUNT) * index - 90) * (Math.PI / 180);
    const x = centerX + Math.cos(angle) * radius;
    const y = centerY + Math.sin(angle) * radius;
    button.style.left = `${x}px`;
    button.style.top = `${y}px`;
  });
}

function handleTargetClick(index, button) {
  if (state.completed) {
    return;
  }

  if (sequence[state.activeStep] === index) {
    state.hits += 1;

    if (state.hits >= TRIAL_COUNT) {
      state.completed = true;
      statusText.textContent = `練習を${TRIAL_COUNT}試行完了しました。`;
    } else if (state.activeStep + 1 >= sequence.length) {
      state.activeStep = 0;
      state.laps += 1;
      statusText.textContent =
        "1 \u5468\u5b8c\u4e86\u3057\u307e\u3057\u305f\u3002\u5f15\u304d\u7d9a\u304d\u9752\u3044\u30bf\u30fc\u30b2\u30c3\u30c8\u3092\u30af\u30ea\u30c3\u30af\u3057\u3066\u7df4\u7fd2\u3067\u304d\u307e\u3059\u3002";
    } else {
      state.activeStep += 1;
      statusText.textContent =
        "\u305d\u306e\u307e\u307e\u6b21\u306e\u9752\u3044\u30bf\u30fc\u30b2\u30c3\u30c8\u3092\u30af\u30ea\u30c3\u30af\u3057\u3066\u304f\u3060\u3055\u3044\u3002";
    }

    syncTargets();
    renderStats();
    return;
  }

  state.misses += 1;
  button.dataset.state = "miss";
  statusText.textContent =
    "\u7070\u8272\u306e\u30bf\u30fc\u30b2\u30c3\u30c8\u304c\u62bc\u3055\u308c\u307e\u3057\u305f\u3002\u9752\u3044\u30bf\u30fc\u30b2\u30c3\u30c8\u3060\u3051\u3092\u72d9\u3063\u3066\u304f\u3060\u3055\u3044\u3002";
  renderStats();

  window.setTimeout(() => {
    syncTargets();
  }, 180);
}

function syncTargets() {
  const activeIndex = state.completed ? -1 : sequence[state.activeStep];

  targets.forEach((button, index) => {
    button.dataset.state = index === activeIndex ? "active" : "idle";
  });
}

function renderStats() {
  const currentTrial = state.completed ? TRIAL_COUNT : state.hits + 1;
  stepValue.textContent = `${currentTrial} / ${TRIAL_COUNT}`;
  hitValue.textContent = String(state.hits);
  missValue.textContent = String(state.misses);
  lapValue.textContent = String(state.laps);
}

function tick() {
  const elapsedSeconds = (performance.now() - state.startedAt) / 1000;
  timeValue.textContent = `${elapsedSeconds.toFixed(1)} \u79d2`;
  window.requestAnimationFrame(tick);
}

function resetTask() {
  state.activeStep = 0;
  state.hits = 0;
  state.misses = 0;
  state.laps = 0;
  state.completed = false;
  state.startedAt = performance.now();
  statusText.textContent =
    "\u9752\u3044\u5186\u3092\u30af\u30ea\u30c3\u30af\u3057\u3066\u7df4\u7fd2\u3092\u59cb\u3081\u3066\u304f\u3060\u3055\u3044\u3002";
  syncTargets();
  renderStats();
}

async function toggleFullscreen() {
  try {
    if (!document.fullscreenElement) {
      await document.documentElement.requestFullscreen();
    } else {
      await document.exitFullscreen();
    }
  } catch (error) {
    statusText.textContent =
      "\u5168\u753b\u9762\u8868\u793a\u3092\u958b\u59cb\u3067\u304d\u307e\u305b\u3093\u3067\u3057\u305f\u3002";
  }
}

function syncFullscreenButton() {
  if (!fullscreenButton) {
    return;
  }

  fullscreenButton.textContent = document.fullscreenElement
    ? "\u5168\u753b\u9762\u3092\u7d42\u4e86"
    : "\u5168\u753b\u9762\u8868\u793a";
}

fullscreenButton.addEventListener("click", toggleFullscreen);
resetButton.addEventListener("click", resetTask);
document.addEventListener("fullscreenchange", () => {
  syncFullscreenButton();
  layoutTargets();
});
window.addEventListener("resize", layoutTargets);

layoutTargets();
resetTask();
syncFullscreenButton();
window.requestAnimationFrame(tick);
