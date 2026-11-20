import { createHash } from 'node:crypto';
import { readdir, readFile, stat } from 'node:fs/promises';
import { resolve } from 'node:path';

const root = resolve(import.meta.dirname, '..');
const assets = resolve(root, 'public', 'assets');
const instructions = JSON.parse(await readFile(resolve(assets, 'drawing-instructions.json'), 'utf8'));
const report = JSON.parse(await readFile(resolve(assets, 'motion-report.json'), 'utf8'));
const quality = JSON.parse(await readFile(resolve(assets, 'quality-report.json'), 'utf8'));
const frameCommands = await readFile(resolve(assets, 'frame-commands.bin'));
const failures = [];

function assert(condition, message) {
  if (!condition) failures.push(message);
}

assert(instructions.schema === 'reference-driven-stroke-timeline/v3', 'unexpected instruction schema');
assert(instructions.source.fps === 24, 'unexpected source fps');
assert(instructions.source.frameCount === 124, 'unexpected source frame count');
assert(instructions.base.lockedPixelFraction > 0.2, 'locked WebGL base is too small');
assert(instructions.frameCommands.compressedBytes < 25_000_000, 'command stream is too large');
assert(instructions.frameCommands.format.startsWith('SBT4 '), 'frame commands use an unexpected format');
assert(instructions.frameCommands.commandStride === 2, 'unexpected command grid precision');
assert(instructions.base.mode.includes('RGBA'), 'static base is not alpha partitioned');
assert(instructions.tracks.length >= 10, 'too few stroke tracks');
assert(instructions.source.runtimeSource === 'assets/frame-commands.bin', 'runtime source is not the command stream');
assert(!String(instructions.source.video ?? '').endsWith('.mp4'), 'instructions point to a runtime video');
assert(instructions.frameCommands.runtimeVideo === false, 'frame commands declare a runtime video');
assert(quality.schema === 'offline-reference-quality/v4', 'unexpected quality report schema');
assert(quality.summary.meanMae < 2.1, 'high-precision reconstruction error is too high');
assert(quality.summary.maxStaticMae < 4, 'locked static pixels differ too much from reference');
assert(quality.partition.lockedFraction > 0.45, 'coherent static partition is too small');
assert(instructions.frameCommands.sha256, 'frame commands are missing a declared checksum');
assert(
  createHash('sha256').update(frameCommands).digest('hex') === instructions.frameCommands.sha256,
  'frame command checksum is invalid',
);
assert(quality.runtimeVideo === false, 'quality report declares a runtime video');
assert(quality.frames.length === instructions.source.frameCount, 'quality report frame count mismatch');
assert(quality.frames.every((frame) => frame.commands >= 0), 'invalid frame command count');
assert(quality.frames.every((frame) => frame.staticDelta === 0), 'locked base changes over time');

const allowedPrimitives = new Set(['ellipse', 'line', 'bezier', 'brush']);
let stateCount = 0;
for (const track of instructions.tracks) {
  assert(track.states.length > 0, `${track.id} has no states`);
  for (const state of track.states) {
    stateCount += 1;
    assert(allowedPrimitives.has(state.primitive), `${track.id} has unknown primitive`);
    assert(state.points.length === 12, `${track.id} has malformed contour`);
    assert(state.color.length === 3, `${track.id} has malformed color`);
    assert(state.motion && Number.isFinite(state.motion.deformation), `${track.id} has malformed motion`);
  }
}
assert(stateCount === report.states, 'report state count differs from instructions');
assert(report.samples.length === 31, 'unexpected keyframe sample count');
assert(report.samples.at(-1).frame === instructions.keyframeStride * (report.samples.length - 1), 'sample time is inconsistent');

for (const name of ['static-base.png', 'static-lock.png', 'drawing-instructions.json', 'motion-report.json', 'quality-report.json', 'static-partition.json', 'frame-commands.bin']) {
  const size = (await stat(resolve(assets, name))).size;
  assert(size > 100, `${name} is too small`);
}
const assetNames = await readdir(assets);
assert(!assetNames.some((name) => name.toLowerCase().endsWith('.mp4')), 'runtime assets contain an mp4');
const partition = JSON.parse(await readFile(resolve(assets, 'static-partition.json'), 'utf8'));
assert(partition.staticScanlineIntervals.length === instructions.source.height, 'static scanline intervals are incomplete');
assert(partition.dynamicScanlineIntervals.length === instructions.source.height, 'dynamic scanline intervals are incomplete');
assert(partition.dynamicCommandCellCount === instructions.frameCommands.dynamicCells, 'dynamic command cell count mismatch');

const html = await readFile(resolve(root, 'index.html'), 'utf8');
assert(!/<video\b/i.test(html), 'index.html contains a video element');
assert(!/\.mp4/i.test(html), 'index.html points to an mp4');
const sourceFiles = (await readdir(resolve(root, 'src'), { recursive: true, withFileTypes: true }))
  .filter((entry) => entry.isFile())
  .map((entry) => entry.name);
assert(!sourceFiles.some((name) => /\.mp4$/i.test(name)), 'browser source files contain an mp4');
for (const name of sourceFiles.filter((name) => /\.(js|css|html)$/i.test(name))) {
  const source = await readFile(resolve(root, 'src', name), 'utf8');
  assert(!/\.mp4|HTMLMediaElement|currentTime|<video\b/i.test(source), `src/${name} contains video runtime code`);
}

if (failures.length) {
  console.error(`Validation failed:\n- ${failures.join('\n- ')}`);
  process.exit(1);
}
console.log(`Assets validated: ${instructions.tracks.length} tracks, ${stateCount} states, ${quality.frameCount} command frames, ${(instructions.base.lockedPixelFraction * 100).toFixed(1)}% locked base, no runtime mp4`);
