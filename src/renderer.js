/** WebGL2 compositor for offline frame commands, parametric strokes and static textures. */

const TAU = Math.PI * 2;

function rgbCss(color, alpha = 1) {
  const rgb = color.map((value) => Math.round(Math.min(255, Math.max(0, value)))).join(',');
  return `rgba(${rgb},${Math.min(1, Math.max(0, alpha)).toFixed(3)})`;
}

const VERTEX_SHADER = `#version 300 es
in vec2 aPosition;
out vec2 vUv;
void main() {
  vUv = (aPosition + 1.0) * 0.5;
  gl_Position = vec4(aPosition, 0.0, 1.0);
}`;

const FRAGMENT_SHADER = `#version 300 es
precision highp float;
in vec2 vUv;
out vec4 outColor;
uniform sampler2D uBase;
uniform sampler2D uLock;
uniform sampler2D uCells;
uniform sampler2D uStrokes;
uniform sampler2D uDrag;
uniform sampler2D uReference;
uniform int uMode;
uniform float uSplit;
uniform float uInfluence;
uniform int uLockEnabled;

void main() {
  vec4 cell = texture(uCells, vUv);
  float locked = step(0.5, texture(uLock, vUv).r);
  float dynamicMask = uLockEnabled == 1 ? (1.0 - locked) : 1.0;
  float cellAlpha = clamp(cell.a, 0.0, 1.0) * dynamicMask;
  vec3 base = texture(uBase, vUv).rgb;
  float baseAlpha = texture(uBase, vUv).a;
  vec3 reconstructed = mix(base, cell.rgb, clamp(cellAlpha + (1.0 - baseAlpha), 0.0, 1.0));
  vec3 reference = texture(uReference, vUv).rgb;
  vec4 stroke = texture(uStrokes, vUv);
  vec4 drag = texture(uDrag, vUv);
  float ink = clamp(cell.a * 0.18 + uInfluence, 0.0, 1.0);
  vec3 stylized = mix(reconstructed, stroke.rgb, clamp(stroke.a * ink * dynamicMask, 0.0, 1.0));
  stylized = mix(stylized, drag.rgb, clamp(drag.a * uInfluence * 0.55 * dynamicMask, 0.0, 1.0));

  if (uMode == 1) {
    outColor = vec4(reference, 1.0);
  } else if (uMode == 2) {
    vec3 paper = vec3(0.035, 0.045, 0.065);
    vec3 painted = mix(paper, stroke.rgb, clamp(stroke.a * 2.1, 0.0, 1.0));
    painted = mix(painted, drag.rgb, clamp(drag.a * 1.5, 0.0, 1.0));
    painted = mix(painted, cell.rgb, clamp(cell.a * 0.55, 0.0, 1.0));
    outColor = vec4(painted, 1.0);
  } else if (uMode == 3) {
    vec3 staticOnly = mix(vec3(0.0), base, baseAlpha);
    outColor = vec4(staticOnly, 1.0);
  } else if (uMode == 4) {
    vec3 delta = abs(stylized - reference);
    outColor = vec4(min(delta * 6.0, vec3(1.0)), 1.0);
  } else if (uMode == 5) {
    outColor = vec4(vUv.x < uSplit ? stylized : reference, 1.0);
  } else {
    outColor = vec4(stylized, 1.0);
  }
}`;

function compileShader(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    throw new Error(gl.getShaderInfoLog(shader) || 'Shader compilation failed');
  }
  return shader;
}

function createTexture(gl, unit, filter) {
  const texture = gl.createTexture();
  gl.activeTexture(gl.TEXTURE0 + unit);
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([0, 0, 0, 255]));
  return texture;
}

function stateTransform(state, shiftX = 0, shiftY = 0) {
  const radians = state.angle * Math.PI / 180;
  const cos = Math.cos(radians);
  const sin = Math.sin(radians);
  const [cx, cy] = [state.center[0] + shiftX, state.center[1] + shiftY];
  const [rx, ry] = state.radii;
  return (point) => [
    cx + point[0] * rx * cos - point[1] * ry * sin,
    cy + point[0] * rx * sin + point[1] * ry * cos,
  ];
}

function setStrokeStyle(context, state, alpha, width) {
  context.strokeStyle = rgbCss(state.color, alpha);
  context.lineWidth = width;
  context.lineCap = 'round';
  context.lineJoin = 'round';
}

function bezierPath(context, points, transform) {
  const mapped = points.map(transform);
  context.beginPath();
  const firstMid = [
    (mapped[0][0] + mapped[mapped.length - 1][0]) / 2,
    (mapped[0][1] + mapped[mapped.length - 1][1]) / 2,
  ];
  context.moveTo(firstMid[0], firstMid[1]);
  for (let index = 0; index < mapped.length; index += 1) {
    const current = mapped[index];
    const next = mapped[(index + 1) % mapped.length];
    context.quadraticCurveTo(current[0], current[1], (current[0] + next[0]) / 2, (current[1] + next[1]) / 2);
  }
  context.closePath();
}

function paintCommandSet(context, state, opacity, shiftX = 0, shiftY = 0) {
  const transform = stateTransform(state, shiftX, shiftY);
  const [cx, cy] = state.center;
  const [rx, ry] = state.radii;

  // Ellipse.
  context.beginPath();
  context.ellipse(cx + shiftX, cy + shiftY, rx, ry, state.angle * Math.PI / 180, 0, TAU);
  context.fillStyle = rgbCss(state.color, opacity * 0.35);
  context.fill();

  // Line segment.
  const start = transform([-1, 0]);
  const end = transform([1, 0]);
  setStrokeStyle(context, state, opacity * 0.75, Math.max(1.2, state.width * 0.35));
  context.beginPath();
  context.moveTo(start[0], start[1]);
  context.lineTo(end[0], end[1]);
  context.stroke();

  // Quadratic bezier contour.
  const points = state.points.map((point) => [
    point[0] * (1 + state.motion.deformation * 0.06),
    point[1] * (1 - state.motion.deformation * 0.04),
  ]);
  setStrokeStyle(context, state, opacity, Math.max(1, state.width * 0.52));
  bezierPath(context, points, transform);
  context.stroke();

  // Cubic brush.
  const phase = (state.frame * 0.17 + state.curvature * 2) % TAU;
  const first = transform([-0.75 + Math.sin(phase) * 0.08, state.curvature * 0.12]);
  const control = transform([0, -state.curvature * 0.25]);
  const last = transform([0.8, Math.cos(phase) * 0.1]);
  setStrokeStyle(context, state, opacity * 1.2, Math.max(0.8, state.width * 0.2));
  context.beginPath();
  context.moveTo(first[0], first[1]);
  context.quadraticCurveTo(control[0], control[1], last[0], last[1]);
  context.stroke();
}

export class ReconstructionRenderer {
  constructor(canvas, width, height, strokeScale = 2) {
    this.canvas = canvas;
    this.width = width;
    this.height = height;
    this.strokeScale = strokeScale;
    this.commandCount = 0;
    this.lastMetrics = null;
    this.gl = canvas.getContext('webgl2', {
      alpha: false,
      antialias: false,
      depth: false,
      stencil: false,
      preserveDrawingBuffer: false,
      powerPreference: 'high-performance',
    });
    if (!this.gl) throw new Error('WebGL2 is unavailable');
    const gl = this.gl;

    const vertex = compileShader(gl, gl.VERTEX_SHADER, VERTEX_SHADER);
    const fragment = compileShader(gl, gl.FRAGMENT_SHADER, FRAGMENT_SHADER);
    this.program = gl.createProgram();
    gl.attachShader(this.program, vertex);
    gl.attachShader(this.program, fragment);
    gl.linkProgram(this.program);
    if (!gl.getProgramParameter(this.program, gl.LINK_STATUS)) {
      throw new Error(gl.getProgramInfoLog(this.program) || 'Program linking failed');
    }
    gl.deleteShader(vertex);
    gl.deleteShader(fragment);

    const vertexArray = gl.createVertexArray();
    gl.bindVertexArray(vertexArray);
    const vertices = new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]);
    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);
    const location = gl.getAttribLocation(this.program, 'aPosition');
    gl.enableVertexAttribArray(location);
    gl.vertexAttribPointer(location, 2, gl.FLOAT, false, 0, 0);

    this.textures = {
      base: createTexture(gl, 0, gl.NEAREST),
      lock: createTexture(gl, 1, gl.NEAREST),
      cells: createTexture(gl, 2, gl.NEAREST),
      strokes: createTexture(gl, 3, gl.LINEAR),
      drag: createTexture(gl, 4, gl.LINEAR),
      reference: createTexture(gl, 5, gl.NEAREST),
    };
    gl.useProgram(this.program);
    const samplers = { uBase: 0, uLock: 1, uCells: 2, uStrokes: 3, uDrag: 4, uReference: 5 };
    for (const [name, unit] of Object.entries(samplers)) {
      gl.uniform1i(gl.getUniformLocation(this.program, name), unit);
    }
    this.uniforms = {};
    for (const name of ['uMode', 'uSplit', 'uInfluence', 'uLockEnabled']) {
      this.uniforms[name] = gl.getUniformLocation(this.program, name);
    }

    this.cellCanvas = document.createElement('canvas');
    this.cellCanvas.width = width;
    this.cellCanvas.height = height;
    this.cellContext = this.cellCanvas.getContext('2d', { willReadFrequently: true });
    this.referenceCanvas = document.createElement('canvas');
    this.referenceCanvas.width = width;
    this.referenceCanvas.height = height;
    this.referenceContext = this.referenceCanvas.getContext('2d');
    this.strokeCanvas = document.createElement('canvas');
    this.strokeCanvas.width = Math.round(width / strokeScale);
    this.strokeCanvas.height = Math.round(height / strokeScale);
    this.strokeContext = this.strokeCanvas.getContext('2d');
    this.dragCanvas = document.createElement('canvas');
    this.dragCanvas.width = this.strokeCanvas.width;
    this.dragCanvas.height = this.strokeCanvas.height;
    this.dragContext = this.dragCanvas.getContext('2d');
    for (const context of [this.strokeContext, this.dragContext]) {
      context.setTransform(1 / strokeScale, 0, 0, 1 / strokeScale, 0, 0);
    }
  }

  upload(texture, unit, source) {
    const gl = this.gl;
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, source);
  }

  uploadStatic(baseImage, lockImage) {
    const gl = this.gl;
    this.baseImage = baseImage;
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
    this.upload(this.textures.base, 0, baseImage);
    this.upload(this.textures.lock, 1, lockImage);
  }

  composeReference() {
    const context = this.referenceContext;
    context.save();
    context.setTransform(1, 0, 0, 1, 0, 0);
    context.clearRect(0, 0, this.width, this.height);
    context.drawImage(this.baseImage, 0, 0, this.width, this.height);
    context.drawImage(this.cellCanvas, 0, 0);
    context.restore();
    return this.referenceCanvas;
  }

  drawStrokeLayers(states, influence, mode) {
    for (const context of [this.strokeContext, this.dragContext]) {
      context.save();
      context.setTransform(1, 0, 0, 1, 0, 0);
      context.clearRect(0, 0, this.strokeCanvas.width, this.strokeCanvas.height);
      context.restore();
    }
    this.commandCount = states.length * 4;
    const opacity = mode === 2 ? 0.85 : 0.25 + influence * 0.75;
    for (const state of states) {
      paintCommandSet(this.strokeContext, state, opacity, 0, 0);
      const trailX = -state.motion.flowDx * 0.5 - state.motion.dx * 0.08;
      const trailY = -state.motion.flowDy * 0.5 - state.motion.dy * 0.08;
      paintCommandSet(this.dragContext, state, opacity * 0.55, trailX, trailY);
    }
  }

  render(states, options) {
    const gl = this.gl;
    const modeMap = { composite: 0, reference: 1, strokes: 2, base: 3, difference: 4, split: 5 };
    this.drawStrokeLayers(states, options.influence, options.mode);
    const reference = this.composeReference();
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
    this.upload(this.textures.cells, 2, this.cellCanvas);
    this.upload(this.textures.strokes, 3, this.strokeCanvas);
    this.upload(this.textures.drag, 4, this.dragCanvas);
    this.upload(this.textures.reference, 5, reference);
    gl.viewport(0, 0, this.width, this.height);
    gl.useProgram(this.program);
    gl.uniform1i(this.uniforms.uMode, modeMap[options.mode] ?? 0);
    gl.uniform1f(this.uniforms.uSplit, options.split);
    gl.uniform1f(this.uniforms.uInfluence, options.influence);
    gl.uniform1i(this.uniforms.uLockEnabled, options.lockEnabled ? 1 : 0);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
    return { commandCount: this.commandCount, reference };
  }

  resetStaticReference() {
    // The locked base is a fixed texture and carries no temporal state.
  }
}
