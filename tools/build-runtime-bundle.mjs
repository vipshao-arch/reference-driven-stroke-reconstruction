/** Create a package containing exactly the files loaded by the browser app. */

import { createHash } from 'node:crypto';
import { copyFile, mkdir, readdir, readFile, rm, writeFile } from 'node:fs/promises';
import { spawn } from 'node:child_process';
import { dirname, relative, resolve, sep } from 'node:path';

const root = resolve(import.meta.dirname, '..');
const dist = resolve(root, 'dist');
const release = resolve(root, 'release');
const bundle = resolve(release, 'runtime-bundle');
const manifestPath = resolve(release, 'runtime-bundle-manifest.json');
const archivePath = resolve(release, 'runtime-bundle.zip');

function assertNoRuntimeVideoText(name, text) {
  if (!/\.(?:html|js|css)$/i.test(name)) return;
  if (/<video\b|HTMLMediaElement|currentTime|\.mp4/i.test(text)) {
    throw new Error(`Runtime bundle contains video code: ${name}`);
  }
}

async function assertZipHasNoVideo(zipPath) {
  const data = await readFile(zipPath);
  const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
  let endOffset = -1;
  for (let offset = data.length - 22; offset >= 0 && offset >= data.length - 65558; offset -= 1) {
    if (view.getUint32(offset, true) === 0x06054b50) {
      endOffset = offset;
      break;
    }
  }
  if (endOffset < 0) throw new Error('Runtime ZIP has no central directory');
  const entries = view.getUint16(endOffset + 10, true);
  let offset = view.getUint32(endOffset + 16, true);
  for (let index = 0; index < entries; index += 1) {
    if (view.getUint32(offset, true) !== 0x02014b50) throw new Error('Runtime ZIP has a damaged central directory');
    const nameLength = view.getUint16(offset + 28, true);
    const extraLength = view.getUint16(offset + 30, true);
    const commentLength = view.getUint16(offset + 32, true);
    const name = data.subarray(offset + 46, offset + 46 + nameLength).toString('utf8');
    if (/\.mp4$/i.test(name)) throw new Error(`Runtime ZIP contains a video: ${name}`);
    offset += 46 + nameLength + extraLength + commentLength;
  }
}

for (const target of [dist, release, bundle, manifestPath, archivePath]) {
  const relation = relative(root, target);
  if (relation.startsWith('..') || relation.startsWith(sep)) {
    throw new Error(`Refusing to write outside the workspace: ${target}`);
  }
}

const runtimeFiles = [
  'index.html',
  'assets/static-base.png',
  'assets/static-lock.png',
  'assets/drawing-instructions.json',
  'assets/motion-report.json',
  'assets/quality-report.json',
  'assets/static-partition.json',
  'assets/frame-commands.bin',
  'src/app.js',
  'src/diagnostic.js',
  'src/frame-commands.js',
  'src/renderer.js',
  'src/styles.css',
  'src/timeline.js',
];

if (runtimeFiles.some((name) => /\.mp4$/i.test(name))) {
  throw new Error('Runtime bundle list contains a video');
}

await rm(bundle, { recursive: true, force: true });
await mkdir(bundle, { recursive: true });

let totalBytes = 0;
const files = [];
for (const name of runtimeFiles) {
  const source = resolve(dist, name);
  const destination = resolve(bundle, name);
  if (!source.startsWith(dist + sep)) throw new Error(`Invalid source: ${source}`);
  if (!destination.startsWith(bundle + sep)) throw new Error(`Invalid destination: ${destination}`);
  await mkdir(dirname(destination), { recursive: true });
  const data = await readFile(source);
  assertNoRuntimeVideoText(name, data.toString('utf8'));
  await copyFile(source, destination);
  totalBytes += data.length;
  files.push({
    path: name,
    bytes: data.length,
    sha256: createHash('sha256').update(data).digest('hex'),
  });
}

const manifest = {
  schema: 'runtime-browser-assets/v3',
  description: 'Only files loaded by index.html and its imported browser modules; source video is offline-only',
  fileCount: files.length,
  totalBytes,
  files,
};

await rm(archivePath, { force: true });
if (process.platform === 'win32') {
  await new Promise((done, reject) => {
    const compress = spawn(
      'powershell.exe',
      ['-NoProfile', '-Command', 'Compress-Archive', '-LiteralPath', bundle, '-DestinationPath', archivePath, '-CompressionLevel', 'Optimal'],
      { stdio: 'inherit' },
    );
    compress.on('error', reject);
    compress.on('close', (code) => code === 0 ? done() : reject(new Error(`Compress-Archive exited with ${code}`)));
  });
} else {
  await new Promise((done, reject) => {
    const zip = spawn('zip', ['-q', '-r', archivePath, '.'], { cwd: bundle, stdio: 'inherit' });
    zip.on('error', reject);
    zip.on('close', (code) => code === 0 ? done() : reject(new Error(`zip exited with ${code}`)));
  });
}
const archiveBytes = (await readFile(archivePath)).length;
const archiveSha256 = createHash('sha256').update(await readFile(archivePath)).digest('hex');
await assertZipHasNoVideo(archivePath);
const bundleEntries = await readdir(bundle, { recursive: true, withFileTypes: true });
if (bundleEntries.some((entry) => entry.isFile() && /\.mp4$/i.test(entry.name))) {
  throw new Error('Runtime bundle contains a video');
}
manifest.archive = {
  path: 'release/runtime-bundle.zip',
  bytes: archiveBytes,
  sha256: archiveSha256,
  runtimeVideo: false,
  verifiedNoVideoEntries: true,
};
await writeFile(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`, 'utf8');
console.log(`runtime files: ${files.length}, total bytes: ${totalBytes}`);
console.log(`bundle: ${bundle}`);
console.log(`archive: ${archivePath} (${archiveBytes} bytes, sha256 ${archiveSha256})`);
console.log(`manifest: ${manifestPath}`);
