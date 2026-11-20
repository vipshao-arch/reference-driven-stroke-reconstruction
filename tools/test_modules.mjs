import { buildChartSvg, buildTimeline, activeStatesAt } from '../src/timeline.js';
import { FrameCommandSet, paintFrameCommands, driveFromCommands } from '../src/frame-commands.js';

const instructions = JSON.parse(await (await import('node:fs/promises')).readFile(new URL('../public/assets/drawing-instructions.json', import.meta.url), 'utf8'));
const report = JSON.parse(await (await import('node:fs/promises')).readFile(new URL('../public/assets/motion-report.json', import.meta.url), 'utf8'));
const timeline = buildTimeline(instructions);
const states = activeStatesAt(timeline, 60);
if (!states.length) throw new Error('No states at frame 60');
if (!states[0].points || states[0].points.length !== 12) throw new Error('Malformed state');
if (states[0].center.some((value) => !Number.isFinite(value))) throw new Error('Non-finite center');
const svg = buildChartSvg(report.samples, timeline.lastFrame);
if (!svg.includes('data-series="displacement"') || !svg.includes('data-series="deformation"')) throw new Error('Chart is missing a series');

const columns = 2;
const rows = 2;
const frameCount = 2;
const dynamicCells = 4;
const header = new Uint8Array(20);
const headerView = new DataView(header.buffer);
headerView.setUint32(0, 0x34544253, true);
headerView.setUint16(4, columns * 2, true);
headerView.setUint16(6, rows * 2, true);
headerView.setUint16(8, frameCount, true);
headerView.setUint16(10, 8, true);
headerView.setUint8(12, 8);
headerView.setUint8(13, 4);
headerView.setUint8(14, 4);
headerView.setUint8(15, 2);
headerView.setUint32(16, dynamicCells, true);
const mask = new Uint8Array([15]);
const index = new Uint8Array((frameCount + 1) * 4);
const indexView = new DataView(index.buffer);
const keyframeBytes = 1 + dynamicCells * 3;
const deltaCount = 2;
const deltaBytes = 1 + 4 + deltaCount + deltaCount * 2;
const frameBytes = [keyframeBytes, deltaBytes];
let offset = 0;
for (let frame = 0; frame <= frameCount; frame += 1) {
  indexView.setUint32(frame * 4, offset, true);
  if (frame < frameCount) offset += frameBytes[frame];
}
const payload = new Uint8Array(offset);
let frameStart = 0;
payload[frameStart] = 1;
for (let cell = 0; cell < dynamicCells; cell += 1) {
  const word = ((3 + cell) << 10) | ((2 + cell) << 5) | (1 + cell);
  new DataView(payload.buffer, frameStart + 1 + cell * 2, 2).setUint16(0, word, true);
  payload[frameStart + 1 + dynamicCells * 2 + cell] = cell % 7;
}
frameStart = frameBytes[0];
payload[frameStart] = 0;
new DataView(payload.buffer, frameStart + 1, 4).setUint32(0, deltaCount, true);
payload[frameStart + 5] = 1;
payload[frameStart + 6] = 1;
new DataView(payload.buffer, frameStart + 7, 2).setUint16(0, (20 << 10) | (10 << 5) | 5, true);
new DataView(payload.buffer, frameStart + 9, 2).setUint16(0, (25 << 10) | (15 << 5) | 8, true);
const buffer = new Uint8Array([...header, ...mask, ...index, ...payload]).buffer;
const commands = new FrameCommandSet(buffer);
if (commands.frameCount !== frameCount || commands.frame(1).keyframe) throw new Error('Frame command decode failed');
if (commands.metrics(0).commands !== dynamicCells || commands.metrics(1).commands !== deltaCount) {
  throw new Error('Frame command metrics failed');
}

const calls = [];
const pixels = new Uint8ClampedArray(4 * 4 * 4);
const mockContext = new Proxy({}, {
  get: (_target, property) => {
    if (property === 'canvas') return { width: 4, height: 4 };
    if (property === 'getImageData') return () => ({ data: pixels });
    return (...args) => calls.push([property, ...args]);
  },
  set: () => true,
});
const firstCommands = paintFrameCommands(commands, mockContext, 0, 4, 4);
if (!firstCommands.keyframe || !calls.some((call) => call[0] === 'fillRect') || !calls.some((call) => call[0] === 'stroke')) {
  throw new Error('Frame command painting is incomplete');
}
const secondCommands = paintFrameCommands(commands, mockContext, 1, 4, 4);
if (secondCommands.keyframe || commands.metrics(1).commands !== deltaCount) throw new Error('Sequential replay failed');
if (commands.paint(mockContext, 0).frame !== 0) throw new Error('Backward seek replay failed');

const testState = { ...states[0], center: [1, 1], radii: [2, 2] };
const driven = driveFromCommands(testState, commands, commands.frame(0), true);
if (!driven.driven || driven.reference.cells < 1) throw new Error('Command-driven stroke state failed');
if (driveFromCommands(testState, commands, commands.frame(0), false).driven) throw new Error('Drive toggle failed');
console.log(`Module checks passed: ${states.length} states at frame 60, ${report.samples.length} chart samples, ${commands.frameCount} command frames`);
