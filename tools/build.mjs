import { cp, mkdir, readdir, readFile, rm, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';

const root = resolve(import.meta.dirname, '..');
const dist = resolve(root, 'dist');

await rm(dist, { recursive: true, force: true });
await mkdir(dist, { recursive: true });
for (const directory of ['public', 'src']) {
  const files = await readdir(resolve(root, directory), { recursive: true, withFileTypes: true });
  if (files.some((entry) => entry.isFile() && /\.mp4$/i.test(entry.name))) {
    throw new Error(`Refusing to build: ${directory}/ contains an mp4 runtime asset`);
  }
}
await cp(resolve(root, 'public'), dist, { recursive: true });
await cp(resolve(root, 'src'), resolve(dist, 'src'), { recursive: true });

let html = await readFile(resolve(root, 'index.html'), 'utf8');
html = html.replaceAll(' href="/src/', ' href="./src/')
  .replaceAll(' src="/src/', ' src="./src/')
  .replaceAll(' src="/assets/', ' src="./assets/')
  .replaceAll(' href="/assets/', ' href="./assets/');
await writeFile(resolve(dist, 'index.html'), html, 'utf8');
console.log(`Static build written to ${dist}`);
