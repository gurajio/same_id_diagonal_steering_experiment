const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const root = path.join(__dirname, '..');
const html = fs.readFileSync(path.join(root, 'same_id_diagonal_steering_experiment.html'), 'utf8');
const script = html.match(/<script>\s*([\s\S]*?)<\/script>/)[1];
new vm.Script(script); // Check syntax of the complete application script.

function session({ trials = 400, postTrials = 75, duration = 60, mode = 'main', conditions, random = false } = {}) {
  let now = 100000;
  const context = vm.createContext({
    window: { setInterval: () => 1, clearInterval() {} },
    document: { addEventListener() {} },
    Date: { now: () => now },
    performance: { now: () => now }
  });
  vm.runInContext(fs.readFileSync(path.join(root, 'steering_experiment_config.js'), 'utf8'), context);
  vm.runInContext(script.replace('document.addEventListener("DOMContentLoaded", init);', `
    collectElements = () => {};
    updateUi = () => {};
    setMessage = () => {};
    prepareTrial = () => {};
    finishExperiment = () => { state.phase = "ended"; };
    recordCoreTiming = () => {};
    makeResultRow = (trial, outcome) => outcome;
    Renderer.draw = () => {};
    for (const key of ["breakModal", "breakText", "breakTimerText", "modalResumeButton", "resumeButton"]) {
      elements[key] = { classList: { add() {}, remove() {} } };
    }
    window.test = { state, elements, advanceAfterCountedTrial, resumeAfterBreak,
      updateBreakTimer, retryTrial, handleKeyDown, startBreak };
  `), context);
  const api = context.window.test;
  Object.assign(api.state, {
    mode, phase: mode === 'pilot' ? 'pilot' : 'practice',
    selectedConditions: conditions || [{ id: 'C1', trials }],
    postSequence: Array.from({ length: postTrials }, () => ({ id: 'C1' })),
    breakDurationSeconds: duration,
    conditionOrderMode: random ? 'random' : 'sequential'
  });
  if (random) {
    api.state.practiceSequence = Array.from({ length: trials }, (_, i) => ({
      id: 'C1', randomizedTrialInCondition: i + 1
    }));
  }
  api.elapse = ms => { now += ms; api.updateBreakTimer(); };
  return api;
}

function runPractice(api) {
  const breaks = [];
  while (['practice', 'pilot'].includes(api.state.phase)) {
    api.advanceAfterCountedTrial();
    if (api.state.phase === 'break') {
      breaks.push(api.state.practiceCompleted || api.state.pilotCompleted);
      api.elapse(api.state.breakDurationSeconds * 1000);
      api.resumeAfterBreak();
    }
  }
  return breaks;
}

assert.deepEqual(runPractice(session()), [100, 200, 300, 400]);
assert.deepEqual(runPractice(session({ trials: 250 })), [100, 200, 250]);
assert.deepEqual(runPractice(session({ trials: 100 })), [100]);
assert.deepEqual(runPractice(session({ postTrials: 0 })), [100, 200, 300]);
assert.deepEqual(runPractice(session({ conditions: [{ id: 'C1', trials: 150 }, { id: 'C2', trials: 150 }] })), [100, 200, 300]);
assert.deepEqual(runPractice(session({ conditions: [{ id: 'C1', trials: 100 }, { id: 'C2', trials: 100 }] })), [100, 200]);
assert.deepEqual(runPractice(session({ random: true })), [100, 200, 300, 400]);
assert.deepEqual(runPractice(session({ mode: 'pilot', trials: 300 })), [100, 200]);

const timed = session({ duration: 90 });
for (let i = 0; i < 100; i++) timed.advanceAfterCountedTrial();
assert.equal(timed.elements.modalResumeButton.disabled, true);
assert.equal(timed.elements.resumeButton.disabled, true);
timed.resumeAfterBreak();
timed.handleKeyDown({ key: 'Enter', preventDefault() {} });
assert.equal(timed.state.phase, 'break');
timed.elapse(89999);
timed.resumeAfterBreak();
assert.equal(timed.state.phase, 'break');
timed.elapse(1);
assert.equal(timed.elements.modalResumeButton.disabled, false);
assert.equal(timed.elements.resumeButton.disabled, false);
timed.handleKeyDown({ key: 'Enter', preventDefault() {} });
assert.equal(timed.state.phase, 'practice');

const invalid = session();
for (let i = 0; i < 99; i++) invalid.advanceAfterCountedTrial();
invalid.state.activeTrial = { startedPerf: 100000, path: { start: { x: 0, y: 0 } } };
invalid.retryTrial('manual_abort', { x: 0, y: 0 });
assert.equal(invalid.state.results[0].countedAsTrial, false);
assert.equal(invalid.state.practiceCompleted, 99);
assert.equal(invalid.state.countedTotal, 99);
assert.equal(invalid.state.phase, 'practice');
invalid.advanceAfterCountedTrial();
assert.equal(invalid.state.phase, 'break');

const post = session({ trials: 1, postTrials: 200 });
assert.deepEqual(runPractice(post), [1]);
assert.equal(post.state.phase, 'post');
for (let i = 0; i < 100; i++) post.advanceAfterCountedTrial();
assert.equal(post.state.phase, 'break');
post.elapse(60000);
post.resumeAfterBreak();
for (let i = 0; i < 100; i++) post.advanceAfterCountedTrial();
assert.equal(post.state.phase, 'ended');

const manual = session();
manual.startBreak('practice', false);
manual.resumeAfterBreak();
assert.equal(manual.state.phase, 'practice');
console.log('Break flow checks passed: trial counts, phase boundaries, duration, Enter key, invalid trials, final completion.');
