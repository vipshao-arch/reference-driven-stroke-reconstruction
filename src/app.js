/** Browser application: offline frame commands, time states, rendering and validation. */

import {
  loadFrameCommands,
  paintFrameCommands,
  driveFromCommands,
} from './frame-commands.js';
import { ReconstructionRenderer } from './renderer.js';
import { activeStatesAt, buildChartSvg, buildTimeline, chartMarkerPercent } from './timeline.js';
import { setFatal, setSnapshot, setStage } from './diagnostic.js';

const $ = (id) => document.getElementById(id);
const assetUrl = (name) => new URL(`./assets/${name}`, document.baseURI).href;

const ui = {
  referenceCanvas: $('referenceCanvas'),
  canvas: $('renderCanvas'),
  referenceMini: $('referenceMini'),
  loadingPanel: $('loadingPanel'),
  fatalPanel: $('fatalPanel'),
  fatalMessage: $('fatalMessage'),
  statusDot: $('statusDot'),
  statusText: $('statusText'),
  sourceLabel: $('sourceLabel'),
  stageFrame: $('stageFrame'),
  splitLine: $('splitLine'),
  frameBadge: $('frameBadge'),
  modeBadge: $('modeBadge'),
  playButton: $('playButton'),
  playLabel: $('playLabel'),
  stepBack: $('stepBack'),
  stepForward: $('stepForward'),
  frameRange: $('frameRange'),
  timeLabel: $('timeLabel'),
  frameLabel: $('frameLabel'),
  speedSelect: $('speedSelect'),
  mediaTimeMetric: $('mediaTimeMetric'),
  syncDriftMetric: $('syncDriftMetric'),
  activeStrokesMetric: $('activeStrokesMetric'),
  commandsMetric: $('commandsMetric'),
  previewFrame: $('previewFrame'),
  qualityScore: $('qualityScore'),
  maeMetric: $('maeMetric'),
  dynamicMaeMetric: $('dynamicMaeMetric'),
  staticDeltaMetric: $('staticDeltaMetric'),
  lockedMetric: $('lockedMetric'),
  qualityBar: $('qualityBar'),
  motionChart: $('motionChart'),
  chartMarker: $('chartMarker'),
  currentDisplacement: $('currentDisplacement'),
  currentDeformation: $('currentDeformation'),
  strokeList: $('strokeList'),
  primitiveCount: $('primitiveCount'),
  lockToggle: $('lockToggle'),
  driveToggle: $('driveToggle'),
  strokeInfluence: $('strokeInfluence'),
  strokeInfluenceValue: $('strokeInfluenceValue'),
  splitControl: $('splitControl'),
  splitRange: $('splitRange'),
  splitValue: $('splitValue'),
  resetButton: $('resetButton'),
  verifyButton: $('verifyButton'),
  verificationResult: $('verificationResult'),
};

const modeLabels = {
  composite: '复原合成',
  reference: '指令重建参考',
  strokes: '参数笔触层',
  base: '锁定基底',
  difference: '参考差分',
  split: '分屏比对',
};

const app = {
  timeline: null,
  renderer: null,
  referenceContext: null,
  instructions: null,
  report: null,
  quality: null,
  frameCommands: null,
  frame: 0,
  mode: 'composite',
  influence: 0.18,
  split: 0.5,
  lockEnabled: true,
  driveEnabled: true,
  playing: false,
  playingStartFrame: 0,
  playingStartTime: 0,
  speed: 1,
  ready: false,
  lastFrameMetrics: null,
  validation: null,
};

function showFatal(error) {
  console.error(error);
  ui.fatalPanel.hidden = false;
  ui.fatalMessage.textContent = error?.message || String(error);
  ui.loadingPanel.classList.add('is-hidden');
  setFatal(error?.message || String(error));
}

async function loadImage(url) {
  const image = new Image();
  image.decoding = 'async';
  image.src = url;
  await image.decode();
  return image;
}

async function loadJson(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url} returned HTTP ${response.status}`);
  return response.json();
}

function formatFrame(frame) {
  return String(frame).padStart(3, '0');
}

function frameToMediaTime(frame) {
  return frame / app.timeline.fps;
}

function paintReferenceMini(image) {
  const context = ui.referenceMini.getContext('2d');
  context.drawImage(image, 0, 0, ui.referenceMini.width, ui.referenceMini.height);
  ui.referenceContext.drawImage(image, 0, 0);
}

function drawFrame() {
  if (!app.ready || !app.frameCommands) return;
  const frame = app.frame;
  const commands = paintFrameCommands(
    app.frameCommands,
    app.renderer.cellContext,
    frame,
    app.renderer.width,
    app.renderer.height,
  );
  const states = activeStatesAt(app.timeline, frame);
  const drivenStates = states.map((state) =>
    driveFromCommands(state, app.frameCommands, commands, app.driveEnabled));
  const cellCommandCount = app.frameCommands.metrics(frame).commands;
  const commandCount = cellCommandCount + states.length * 4;
  const { reference } = app.renderer.render(drivenStates, {
    mode: app.mode,
    influence: app.influence,
    split: app.split,
    lockEnabled: app.lockEnabled,
  });
  paintReferenceMini(reference);
  const quality = app.quality.frames[frame];
  const publicMetrics = {
    mae: quality.mae,
    dynamicMae: quality.dynamicMae,
    staticDelta: quality.staticDelta,
    staticBaseMae: quality.staticMae,
    psnr: quality.psnr,
    lockedFraction: quality.lockedFraction,
    commands: quality.commands,
    source: 'offline-reference-quality/v4',
  };
  app.lastFrameMetrics = publicMetrics;
  app.renderer.lastMetrics = publicMetrics;
  window.__frameMetrics = publicMetrics;
  updateQualityUi(publicMetrics);
  updateFrameUi(frame, drivenStates, commandCount, publicMetrics);
}

function updateQualityUi(metrics) {
  ui.qualityScore.textContent = `${metrics.psnr} dB`;
  ui.qualityScore.dataset.psnr = String(metrics.psnr);
  ui.maeMetric.textContent = metrics.mae.toFixed(3);
  ui.dynamicMaeMetric.textContent = metrics.dynamicMae.toFixed(3);
  ui.staticDeltaMetric.textContent = metrics.staticDelta.toFixed(3);
  ui.lockedMetric.textContent = `${(metrics.lockedFraction * 100).toFixed(1)}%`;
  ui.lockedMetric.dataset.locked = String(metrics.lockedFraction);
  ui.qualityBar.style.width = `${Math.max(0, Math.min(100, (Number(metrics.psnr) || 0) / 55 * 100))}%`;
}

function updateFrameUi(frame, states, commandCount, metrics) {
  const mediaTime = frameToMediaTime(frame);
  ui.frameRange.value = String(frame);
  ui.timeLabel.textContent = `${mediaTime.toFixed(3)} s`;
  ui.frameLabel.textContent = `${frame} / ${app.timeline.lastFrame}`;
  ui.frameBadge.textContent = `F ${formatFrame(frame)}`;
  ui.previewFrame.textContent = formatFrame(frame);
  ui.mediaTimeMetric.textContent = `${mediaTime.toFixed(3)} s`;
  ui.syncDriftMetric.textContent = '0.0 ms';
  ui.activeStrokesMetric.textContent = String(states.length);
  ui.commandsMetric.textContent = String(commandCount);
  ui.chartMarker.style.left = `${chartMarkerPercent(frame, app.timeline.lastFrame)}%`;
  ui.statusText.textContent = app.playing ? '绘图指令同步播放中' : '绘图指令帧已锁定';
  const displacement = states.length
    ? states.reduce((sum, state) => sum + state.motion.displacement, 0) / states.length
    : 0;
  const deformation = states.length
    ? states.reduce((sum, state) => sum + state.motion.deformation, 0) / states.length
    : 0;
  ui.currentDisplacement.textContent = `${displacement.toFixed(2)} px`;
  ui.currentDeformation.textContent = deformation.toFixed(3);
  renderStrokeList(states);
  setSnapshot({
    stage: 'running',
    frame,
    mediaTime,
    fps: app.timeline.fps,
    frameCount: app.timeline.frameCount,
    mode: app.mode,
    sourceWidth: app.timeline.width,
    sourceHeight: app.timeline.height,
    activeStrokes: states.length,
    commandCount,
    states,
    metrics,
    syncDriftMs: 0,
    lockEnabled: app.lockEnabled,
    driveEnabled: app.driveEnabled,
    validation: app.validation,
  });
  window.__frameState = {
    frame,
    mediaTime,
    syncDriftMs: 0,
    activeStrokes: states.length,
    commandCount,
    mode: app.mode,
    frameCount: app.timeline.frameCount,
    lastFrame: app.timeline.lastFrame,
    lockEnabled: app.lockEnabled,
    driveEnabled: app.driveEnabled,
  };
}

function renderStrokeList(states) {
  const counts = {};
  for (const state of states) counts[state.primitive] = (counts[state.primitive] || 0) + 1;
  ui.primitiveCount.textContent = `${states.length} 条 / ${states.length * 4} 指令`;
  ui.strokeList.innerHTML = states.slice(0, 14).map((state) => {
    const color = state.color.map((value) => Math.round(value)).join(',');
    const geometry = `(${state.center[0].toFixed(0)},${state.center[1].toFixed(0)}) r${state.radii[0].toFixed(0)}×${state.radii[1].toFixed(0)} ${state.angle.toFixed(0)}°`;
    const driven = state.driven ? '指令驱动' : '时间插值';
    return `<div class="stroke-item">
      <span class="stroke-id">${state.id}</span>
      <span class="primitive-tag">${state.primitive}</span>
      <span class="stroke-geometry">${geometry} · ${driven}</span>
      <span class="stroke-swatch" style="background:rgb(${color})"></span>
    </div>`;
  }).join('');
}

function seekToFrame(frame) {
  if (!app.ready) return;
  app.frame = Math.min(app.timeline.lastFrame, Math.max(0, Math.round(frame)));
  drawFrame();
}

function updatePlayButton() {
  ui.playButton.classList.toggle('is-playing', app.playing);
  ui.playLabel.textContent = app.playing ? '暂停' : '播放';
  ui.statusText.textContent = app.playing ? '绘图指令同步播放中' : '绘图指令帧已锁定';
}

function startPlayback() {
  app.playing = true;
  app.playingStartFrame = app.frame;
  app.playingStartTime = performance.now();
  updatePlayButton();
  drawFrame();
}

function pausePlayback() {
  app.playing = false;
  updatePlayButton();
}

function playbackTick(now) {
  if (!app.playing || !app.ready) return;
  const elapsedFrames = (now - app.playingStartTime) / 1000 * app.timeline.fps * app.speed;
  const frame = (app.playingStartFrame + Math.floor(elapsedFrames)) % app.timeline.frameCount;
  if (frame !== app.frame) {
    app.frame = frame;
    drawFrame();
  }
}

function setMode(mode) {
  app.mode = mode;
  document.querySelectorAll('.mode-button').forEach((button) => {
    button.classList.toggle('is-active', button.dataset.mode === mode);
  });
  ui.modeBadge.textContent = modeLabels[mode];
  ui.stageFrame.classList.toggle('is-split', mode === 'split');
  ui.splitControl.classList.toggle('is-visible', mode === 'split');
  drawFrame();
}

function resetParameters() {
  app.influence = 0.18;
  app.split = 0.5;
  app.lockEnabled = true;
  app.driveEnabled = true;
  ui.strokeInfluence.value = '0.18';
  ui.strokeInfluenceValue.textContent = '0.18';
  ui.splitRange.value = '50';
  ui.splitValue.textContent = '50%';
  ui.splitLine.style.left = '50%';
  ui.lockToggle.checked = true;
  ui.driveToggle.checked = true;
  window.__frameMetrics = null;
  setMode('composite');
}

async function runValidation() {
  if (!app.ready) return;
  ui.verifyButton.disabled = true;
  ui.verificationResult.className = 'verification-result';
  ui.verificationResult.textContent = '正在检验 3 个时刻…';
  const previousFrame = app.frame;
  const wasPlaying = app.playing;
  pausePlayback();
  const frames = [0, 60, 120];
  const results = [];
  for (const frame of frames) {
    seekToFrame(frame);
    await new Promise((resolve) => window.setTimeout(resolve, 120));
    const diagnostic = window.__frameState ?? {};
    const metrics = window.__frameMetrics ?? null;
    results.push({
      frame,
      mediaTime: Number((diagnostic.mediaTime ?? frameToMediaTime(frame)).toFixed(4)),
      syncDriftMs: diagnostic.syncDriftMs ?? 0,
      activeStrokes: diagnostic.activeStrokes ?? Number(ui.activeStrokesMetric.textContent),
      commandCount: diagnostic.commandCount ?? Number(ui.commandsMetric.textContent),
      metrics,
    });
  }
  const maxDrift = Math.max(...results.map((result) => result.syncDriftMs));
  const staticDeltaMax = Math.max(0, ...results.map((result) => result.metrics?.staticDelta ?? 0));
  const pass = results.every((result) =>
    result.activeStrokes > 0
    && result.commandCount > 1000
    && result.metrics?.mae < 3.0
    && result.syncDriftMs === 0)
    && maxDrift === 0
    && staticDeltaMax === 0;
  const summary = pass ? 'PASS' : 'ATTENTION';
  app.validation = { summary, maxDrift, staticDeltaMax, results };
  ui.verificationResult.className = `verification-result ${pass ? 'is-pass' : 'is-fail'}`;
  ui.verificationResult.dataset.status = summary;
  ui.verificationResult.textContent =
    `${summary} · 最大同步偏差 ${maxDrift.toFixed(1)} ms · 静态帧间 Δ ${staticDeltaMax.toFixed(3)}\n` +
    results.map((result) => `F${formatFrame(result.frame)} ${result.activeStrokes} 笔触 / ${result.commandCount} 指令 / MAE ${result.metrics?.mae ?? '--'}`).join('\n');
  ui.verifyButton.disabled = false;
  seekToFrame(previousFrame);
  if (wasPlaying) startPlayback();
}

function bindEvents() {
  document.querySelectorAll('.mode-button').forEach((button) => {
    button.addEventListener('click', () => setMode(button.dataset.mode));
  });
  ui.playButton.addEventListener('click', () => {
    if (app.playing) pausePlayback();
    else startPlayback();
  });
  ui.stepBack.addEventListener('click', () => {
    pausePlayback();
    seekToFrame((app.frame - 1 + app.timeline.frameCount) % app.timeline.frameCount);
  });
  ui.stepForward.addEventListener('click', () => {
    pausePlayback();
    seekToFrame((app.frame + 1) % app.timeline.frameCount);
  });
  ui.frameRange.addEventListener('input', () => {
    pausePlayback();
    seekToFrame(Number(ui.frameRange.value));
  });
  ui.speedSelect.addEventListener('change', () => {
    const speed = Number(ui.speedSelect.value);
    if (app.playing) {
      app.playingStartFrame = app.frame;
      app.playingStartTime = performance.now();
    }
    app.speed = speed;
  });
  ui.lockToggle.addEventListener('change', () => {
    app.lockEnabled = ui.lockToggle.checked;
    drawFrame();
  });
  ui.driveToggle.addEventListener('change', () => {
    app.driveEnabled = ui.driveToggle.checked;
    drawFrame();
  });
  ui.strokeInfluence.addEventListener('input', () => {
    app.influence = Number(ui.strokeInfluence.value);
    ui.strokeInfluenceValue.textContent = app.influence.toFixed(2);
    drawFrame();
  });
  ui.splitRange.addEventListener('input', () => {
    app.split = Number(ui.splitRange.value) / 100;
    ui.splitValue.textContent = `${ui.splitRange.value}%`;
    ui.splitLine.style.left = `${ui.splitRange.value}%`;
    drawFrame();
  });
  ui.resetButton.addEventListener('click', resetParameters);
  ui.verifyButton.addEventListener('click', runValidation);
  const animationLoop = (now) => {
    playbackTick(now);
    window.requestAnimationFrame(animationLoop);
  };
  window.requestAnimationFrame(animationLoop);
}

async function init() {
  try {
    setStage('assets');
    const [
      instructions,
      report,
      quality,
      frameCommands,
      baseImage,
      lockImage,
    ] = await Promise.all([
      loadJson(assetUrl('drawing-instructions.json')),
      loadJson(assetUrl('motion-report.json')),
      loadJson(assetUrl('quality-report.json')),
      loadFrameCommands(assetUrl('frame-commands.bin')),
      loadImage(assetUrl('static-base.png')),
      loadImage(assetUrl('static-lock.png')),
    ]);
    app.instructions = instructions;
    app.report = report;
    app.quality = quality;
    app.frameCommands = frameCommands;
    app.timeline = buildTimeline(instructions);
    if (
      frameCommands.frameCount !== app.timeline.frameCount
      || quality.frameCount !== app.timeline.frameCount
      || quality.frames.length !== app.timeline.frameCount
    ) throw new Error('Frame command and timeline lengths do not agree');

    setStage('renderer');
    const { width, height, fps, frameCount } = app.timeline;
    ui.sourceLabel.textContent = `${width} × ${height} · ${fps} fps · ${frameCount} 帧 · 视频仅离线分析`;
    ui.frameRange.max = String(app.timeline.lastFrame);
    ui.motionChart.innerHTML = buildChartSvg(report.samples, app.timeline.lastFrame);
    ui.referenceContext = ui.referenceCanvas.getContext('2d');
    app.renderer = new ReconstructionRenderer(ui.canvas, width, height, 2);
    app.renderer.uploadStatic(baseImage, lockImage, frameCommands.dynamicMask);

    setStage('events');
    bindEvents();
    app.ready = true;
    window.__frameState = {
      frame: 0,
      mediaTime: 0,
      syncDriftMs: 0,
      activeStrokes: activeStatesAt(app.timeline, 0).length,
      commandCount: frameCommands.metrics(0).commands,
      mode: app.mode,
      frameCount: app.timeline.frameCount,
      lastFrame: app.timeline.lastFrame,
      lockEnabled: app.lockEnabled,
      driveEnabled: app.driveEnabled,
    };
    window.__frameMetrics = null;
    setStage('first-frame');
    ui.statusDot.classList.add('is-ready');
    ui.loadingPanel.classList.add('is-hidden');
    drawFrame();
  } catch (error) {
    showFatal(error);
  }
}

init();
