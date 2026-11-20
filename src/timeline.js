/** Interpolate parametric drawing states on the reference frame timeline. */

const TAU = Math.PI * 2;

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function lerpAngle(a, b, t) {
  const delta = ((b - a + 90) % 180 + 180) % 180 - 90;
  return a + delta * t;
}

function lerpVector(a, b, t) {
  return a.map((value, index) => Number(lerp(value, b[index], t).toFixed(3)));
}

function lerpPoints(a, b, t) {
  return a.map((point, index) => [
    Number(lerp(point[0], b[index][0], t).toFixed(4)),
    Number(lerp(point[1], b[index][1], t).toFixed(4)),
  ]);
}

export function stateAt(track, position, stride = 4) {
  const states = track.states;
  if (!states.length) return null;
  const first = states[0];
  const last = states[states.length - 1];
  if (position < first.frame - stride || position > last.frame + stride) return null;

  let lower = 0;
  while (lower < states.length - 1 && states[lower + 1].frame <= position) lower += 1;
  const upper = Math.min(states.length - 1, lower + 1);
  const before = states[lower];
  const after = states[upper];
  const span = after.frame - before.frame;
  const t = span > 0 ? Math.min(1, Math.max(0, (position - before.frame) / span)) : 0;
  const nearest = t < 0.5 ? before : after;

  return {
    id: track.id,
    primitive: nearest.primitive,
    frame: position,
    center: lerpVector(before.center, after.center, t),
    radii: lerpVector(before.radii, after.radii, t),
    angle: Number(lerpAngle(before.angle, after.angle, t).toFixed(2)),
    points: lerpPoints(before.points, after.points, t),
    curvature: Number(lerp(before.curvature, after.curvature, t).toFixed(4)),
    color: lerpVector(before.color, after.color, t),
    width: Number(lerp(before.width, after.width, t).toFixed(2)),
    alpha: Number(lerp(before.alpha, after.alpha, t).toFixed(3)),
    area: Number(lerp(before.area, after.area, t).toFixed(1)),
    motion: {
      dx: Number(lerp(before.motion.dx, after.motion.dx, t).toFixed(2)),
      dy: Number(lerp(before.motion.dy, after.motion.dy, t).toFixed(2)),
      displacement: Number(lerp(before.motion.displacement, after.motion.displacement, t).toFixed(2)),
      flowDx: Number(lerp(before.motion.flowDx, after.motion.flowDx, t).toFixed(2)),
      flowDy: Number(lerp(before.motion.flowDy, after.motion.flowDy, t).toFixed(2)),
      deformation: Number(lerp(before.motion.deformation, after.motion.deformation, t).toFixed(4)),
    },
    birth: track.birth,
    death: track.death,
    trackLength: track.length,
    totalDisplacement: track.totalDisplacement,
    meanDeformation: track.meanDeformation,
  };
}

export function buildTimeline(instructions) {
  return {
    tracks: instructions.tracks,
    stride: instructions.keyframeStride,
    fps: instructions.source.fps,
    frameCount: instructions.source.frameCount,
    lastFrame: instructions.source.frameCount - 1,
    duration: instructions.source.duration,
    width: instructions.source.width,
    height: instructions.source.height,
  };
}

export function activeStatesAt(timeline, position) {
  return timeline.tracks
    .map((track) => stateAt(track, position, timeline.stride))
    .filter(Boolean)
    .sort((a, b) => b.area - a.area);
}

function seriesPoints(samples, key, maximum) {
  const left = 60;
  const top = 24;
  const width = 860;
  const height = 210;
  return samples
    .map((sample, index) => {
      const x = left + width * index / Math.max(samples.length - 1, 1);
      const y = top + height * (1 - Math.min(sample[key] / maximum, 1));
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(' ');
}

export function buildChartSvg(samples, lastFrame) {
  const maxDisplacement = Math.max(...samples.map((sample) => sample.meanDisplacement), 1);
  const maxDeformation = Math.max(...samples.map((sample) => sample.meanDeformation), 0.1);
  const grid = [0, 1, 2, 3, 4]
    .map((tick) => {
      const y = 24 + 210 * tick / 4;
      return `<line x1="60" y1="${y}" x2="920" y2="${y}" stroke="#2b3241"/>`;
    })
    .join('');
  return `
    <rect width="960" height="280" fill="#10131c"/>
    <rect x="60" y="24" width="860" height="210" fill="#171c28" stroke="#343b4c"/>
    ${grid}
    <polyline data-series="displacement" points="${seriesPoints(samples, 'meanDisplacement', maxDisplacement)}" fill="none" stroke="#69e2ff" stroke-width="3"/>
    <polyline data-series="deformation" points="${seriesPoints(samples, 'meanDeformation', maxDeformation)}" fill="none" stroke="#ff9bd2" stroke-width="3"/>
    <text x="60" y="16" fill="#69e2ff" font-size="12">displacement / deformation by reference keyframe</text>
    <text x="60" y="268" fill="#8d96a9" font-size="12">frame 0</text>
    <text x="870" y="268" fill="#8d96a9" font-size="12">frame ${lastFrame}</text>
  `;
}

export function chartMarkerPercent(frame, lastFrame) {
  const x = 60 + 860 * Math.min(1, Math.max(0, frame / Math.max(lastFrame, 1)));
  return (x / 960) * 100;
}
