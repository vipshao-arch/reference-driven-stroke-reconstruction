/** Decode and replay high-precision SBT4 offline drawing commands. */

const MAGIC = 0x34544253; // SBT4, little endian
const COMMAND_STRIDE = 2;
const DECODE555 = new Uint8Array(32);
for (let index = 0; index < 32; index += 1) DECODE555[index] = Math.round(index * 255 / 31);
const PRIMITIVE_NAMES = {
  0: 'solid rectangle',
  1: 'horizontal segment',
  2: 'vertical segment',
  3: 'ellipse',
  4: 'cubic brush',
  5: 'quadratic bezier horizontal',
  6: 'quadratic bezier vertical',
  7: 'solid rectangle',
};

function unpack555(word) {
  return [
    DECODE555[(word >> 10) & 31],
    DECODE555[(word >> 5) & 31],
    DECODE555[word & 31],
  ];
}

/** Draw a 2x2 or 1x1 command as a filled cell plus its matching primitive. */
function strokePrimitive(context, primitive, x, y, width, height) {
  context.fillRect(x, y, width, height);
  if (width === 1 || height === 1) return;
  const cx = x + width / 2;
  const cy = y + height / 2;
  context.beginPath();
  if (primitive === 1) {
    context.moveTo(x, cy);
    context.lineTo(x + width, cy);
    context.stroke();
  } else if (primitive === 2) {
    context.moveTo(cx, y);
    context.lineTo(cx, y + height);
    context.stroke();
  } else if (primitive === 3) {
    context.ellipse(cx, cy, width * 0.5, height * 0.5, 0, 0, Math.PI * 2);
    context.fill();
  } else if (primitive === 4) {
    context.moveTo(x, y + height * 0.65);
    context.bezierCurveTo(x + width * 0.3, y, x + width * 0.65, y + height, x + width, y + height * 0.35);
    context.stroke();
  } else if (primitive === 5) {
    context.moveTo(x, cy);
    context.quadraticCurveTo(cx, y, x + width, cy);
    context.stroke();
  } else if (primitive === 6) {
    context.moveTo(cx, y);
    context.quadraticCurveTo(x, cy, cx, y + height);
    context.stroke();
  }
}

function readVarints(bytes, offset, count) {
  const values = new Uint32Array(count);
  for (let index = 0; index < count; index += 1) {
    let value = 0;
    let shift = 0;
    while (offset < bytes.length) {
      const byte = bytes[offset];
      offset += 1;
      value |= (byte & 127) << shift;
      if ((byte & 128) === 0) break;
      shift += 7;
    }
    values[index] = value >>> 0;
  }
  return { values, offset };
}

export class FrameCommandSet {
  constructor(buffer) {
    const view = new DataView(buffer);
    if (view.byteLength < 20) throw new Error('Frame command stream is too short');
    if (view.getUint32(0, true) !== MAGIC) throw new Error('Unexpected frame command format');
    this.width = view.getUint16(4, true);
    this.height = view.getUint16(6, true);
    this.frameCount = view.getUint16(8, true);
    this.keyframeStride = view.getUint16(10, true);
    this.blockThreshold = view.getUint8(12);
    this.deltaThreshold = view.getUint8(13);
    this.version = view.getUint8(14);
    this.commandStride = view.getUint8(15);
    this.dynamicCells = view.getUint32(16, true);
    if (this.version !== 4 || this.commandStride !== COMMAND_STRIDE) {
      throw new Error('Unsupported frame command version');
    }
    this.columns = this.width / COMMAND_STRIDE;
    this.rows = this.height / COMMAND_STRIDE;
    this.maskBytes = Math.ceil(this.columns * this.rows / 8);
    this.indexBytes = (this.frameCount + 1) * 4;
    this.payload = new Uint8Array(buffer, 20);
    this.frameOffsets = new Uint32Array(this.frameCount + 1);
    const indexView = new DataView(buffer, 20 + this.maskBytes);
    for (let index = 0; index <= this.frameCount; index += 1) {
      this.frameOffsets[index] = indexView.getUint32(index * 4, true);
    }
    this.commandDataOffset = this.maskBytes + this.indexBytes;
    this.dynamicMask = new Uint8Array(this.columns * this.rows);
    for (let cell = 0; cell < this.dynamicMask.length; cell += 1) {
      const byte = this.payload[cell >> 3];
      this.dynamicMask[cell] = (byte >> (cell & 7)) & 1;
    }
    this.replayFrame = -1;
    this.pixels = new Uint8ClampedArray(this.width * this.height * 4);
    this.decodedFrame = -1;
    this.frameCommands = null;
    this.frameCommandCount = 0;
  }

  frame(index) {
    const clamped = Math.min(this.frameCount - 1, Math.max(0, Math.round(index)));
    const start = this.commandDataOffset + this.frameOffsets[clamped];
    const end = this.commandDataOffset + this.frameOffsets[clamped + 1];
    return {
      frame: clamped,
      keyframe: this.payload[start] === 1,
      data: this.payload.subarray(start + 1, end),
    };
  }

  /** Replay from the nearest preceding keyframe so arbitrary seeks are correct. */
  paint(context, targetFrame) {
    const clamped = Math.min(this.frameCount - 1, Math.max(0, Math.round(targetFrame)));
    const keyframe = Math.floor(clamped / this.keyframeStride) * this.keyframeStride;
    if (clamped === this.replayFrame) return this.frameCommands;
    if (clamped > this.replayFrame && keyframe <= this.replayFrame) {
      // Delta frames can continue the current replay state.
    } else {
      this.pixels.fill(0);
      this.replayFrame = keyframe - 1;
    }
    context.save();
    context.setTransform(1, 0, 0, 1, 0, 0);
    context.globalCompositeOperation = 'source-over';
    context.lineCap = 'round';
    context.lineJoin = 'round';
    context.lineWidth = 1;
    let lastCommands = null;
    let commandCount = 0;
    for (let frame = this.replayFrame + 1; frame <= clamped; frame += 1) {
      const commands = this.frame(frame);
      if (commands.keyframe) {
        context.clearRect(0, 0, this.width, this.height);
        this.pixels.fill(0);
      }
      commandCount = this.execute(context, commands, frame === clamped);
      lastCommands = commands;
    }
    context.restore();
    this.replayFrame = clamped;
    this.decodedFrame = clamped;
    this.frameCommands = lastCommands;
    this.frameCommandCount = commandCount;
    return lastCommands;
  }

  execute(context, commands, recordPixels) {
    const bytes = commands.data;
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    let commandCount = 0;
    if (commands.keyframe) {
      const colorBytes = this.dynamicCells * 2;
      let cellOffset = 0;
      let codeOffset = colorBytes;
      for (let cell = 0; cell < this.dynamicMask.length; cell += 1) {
        if (!this.dynamicMask[cell]) continue;
        const word = view.getUint16(cellOffset, true);
        const primitive = bytes[codeOffset];
        cellOffset += 2;
        codeOffset += 1;
        const color = unpack555(word);
        const x = (cell % this.columns) * COMMAND_STRIDE;
        const y = Math.floor(cell / this.columns) * COMMAND_STRIDE;
        const style = `rgb(${color[0]},${color[1]},${color[2]})`;
        context.fillStyle = style;
        context.strokeStyle = style;
        strokePrimitive(context, primitive, x, y, COMMAND_STRIDE, COMMAND_STRIDE);
        commandCount += 1;
      }
    } else {
      const count = view.getUint32(0, true);
      const gaps = readVarints(bytes, 4, count);
      const colorOffset = gaps.offset;
      let position = -1;
      for (let index = 0; index < count; index += 1) {
        position += gaps.values[index];
        const word = view.getUint16(colorOffset + index * 2, true);
        const color = unpack555(word);
        const x = position % this.width;
        const y = Math.floor(position / this.width);
        const style = `rgb(${color[0]},${color[1]},${color[2]})`;
        context.fillStyle = style;
        context.strokeStyle = style;
        strokePrimitive(context, 0, x, y, 1, 1);
        commandCount += 1;
      }
    }
    if (recordPixels && typeof context.getImageData === 'function') {
      const image = context.getImageData(0, 0, this.width, this.height);
      this.pixels.set(image.data);
    }
    return commandCount;
  }

  metrics(index) {
    if (index === this.decodedFrame) return { commands: this.frameCommandCount };
    const commands = this.frame(index);
    if (commands.keyframe) return { commands: this.dynamicCells };
    const view = new DataView(commands.data.buffer, commands.data.byteOffset, commands.data.byteLength);
    return { commands: view.getUint32(0, true) };
  }

  cell(column, row, commands = null) {
    const frameData = commands ?? this.frameCommands;
    const cell = row * this.columns + column;
    if (!frameData || column < 0 || column >= this.columns || row < 0 || row >= this.rows) return null;
    if (this.dynamicMask[cell] === 0) return null;
    if (this.decodedFrame >= 0) {
      const x = column * COMMAND_STRIDE;
      const y = row * COMMAND_STRIDE;
      const offset = (y * this.width + x) * 4;
      if (this.pixels[offset + 3] > 127) {
        return {
          color: [this.pixels[offset], this.pixels[offset + 1], this.pixels[offset + 2]],
          primitive: 0,
          primitiveName: PRIMITIVE_NAMES[0],
        };
      }
    }
    if (frameData.keyframe) {
      let index = 0;
      for (let candidate = 0; candidate < cell; candidate += 1) index += this.dynamicMask[candidate];
      const view = new DataView(frameData.data.buffer, frameData.data.byteOffset, frameData.data.byteLength);
      const word = view.getUint16(index * 2, true);
      const primitive = frameData.data[this.dynamicCells * 2 + index];
      return {
        color: unpack555(word),
        primitive,
        primitiveName: PRIMITIVE_NAMES[primitive] ?? PRIMITIVE_NAMES[0],
      };
    }
    return null;
  }
}

export async function loadFrameCommands(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url} returned HTTP ${response.status}`);
  const file = new Uint8Array(await response.arrayBuffer());
  const compressed = file.subarray(20);
  const stream = new DecompressionStream('deflate');
  const payload = await new Response(
    new Blob([compressed]).stream().pipeThrough(stream),
  ).arrayBuffer();
  const decoded = new Uint8Array(20 + payload.byteLength);
  decoded.set(file.subarray(0, 20), 0);
  decoded.set(new Uint8Array(payload), 20);
  return new FrameCommandSet(decoded.buffer);
}

export function paintFrameCommands(commandSet, context, targetFrame, width, height) {
  return commandSet.paint(context, targetFrame);
}

export function driveFromCommands(state, commandSet, frameCommands, enabled) {
  if (!enabled) return { ...state, driven: false, reference: null };
  const column = Math.floor(state.center[0] / 8);
  const row = Math.floor(state.center[1] / 8);
  let red = 0;
  let green = 0;
  let blue = 0;
  let count = 0;
  let primitive = 0;
  const reach = Math.max(1, Math.ceil(Math.max(state.radii[0], state.radii[1]) / 24));
  for (let sampleRow = row - reach; sampleRow <= row + reach; sampleRow += 1) {
    for (let sampleColumn = column - reach; sampleColumn <= column + reach; sampleColumn += 1) {
      const cell = commandSet.cell(sampleColumn * 4, sampleRow * 4, frameCommands);
      if (!cell) continue;
      red += cell.color[0];
      green += cell.color[1];
      blue += cell.color[2];
      primitive = cell.primitive;
      count += 1;
    }
  }
  if (!count) return { ...state, driven: false, reference: null };
  const measured = [red / count, green / count, blue / count];
  const color = state.color.map((value, index) =>
    Number((value * 0.72 + measured[index] * 0.28).toFixed(2)));
  const angleBlend = (primitive % 4) * 3.5 - 5;
  const radiusScale = primitive === 3 ? 1.015 : primitive >= 4 ? 0.985 : 1;
  return {
    ...state,
    center: [
      Number((state.center[0] + (primitive - 3) * 0.12).toFixed(2)),
      Number((state.center[1] - (primitive - 3) * 0.08).toFixed(2)),
    ],
    radii: state.radii.map((radius) => Number((radius * radiusScale).toFixed(2))),
    angle: Number((state.angle + angleBlend).toFixed(2)),
    color,
    driven: true,
    reference: {
      cells: count,
      primitive,
      primitiveName: PRIMITIVE_NAMES[primitive] ?? PRIMITIVE_NAMES[0],
      measuredColor: measured.map((value) => Number(value.toFixed(1))),
      centerShift: 0.2,
    },
  };
}
