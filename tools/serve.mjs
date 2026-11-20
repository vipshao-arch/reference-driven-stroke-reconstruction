import { createServer } from 'node:http';
import { readFile, stat } from 'node:fs/promises';
import { extname, join, normalize, resolve, sep } from 'node:path';

const rootArgument = process.argv[2] ?? '.';
const root = resolve(process.cwd(), rootArgument);
const publicRoot = resolve(root, 'public');
const port = Number(process.env.PORT ?? 4173);

const contentTypes = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon',
};

async function safePath(urlPath) {
  const decoded = decodeURIComponent(urlPath.split('?')[0]);
  const relative = decoded === '/' ? 'index.html' : decoded.replace(/^\/+/, '');
  const candidate = normalize(join(root, relative));
  const publicCandidate = normalize(join(publicRoot, relative));
  const resolved = (await stat(candidate).catch(() => null)) ? candidate : publicCandidate;
  if (resolved !== root && !resolved.startsWith(root + sep)) return null;
  return resolved;
}

const server = createServer(async (request, response) => {
  try {
    let filePath = await safePath(request.url ?? '/');
    if (!filePath) {
      response.writeHead(403).end('Forbidden');
      return;
    }
    const info = await stat(filePath).catch(() => null);
    if (info?.isDirectory()) filePath = join(filePath, 'index.html');
    const data = await readFile(filePath);
    const type = contentTypes[extname(filePath).toLowerCase()] ?? 'application/octet-stream';
    const headers = { 'Content-Type': type, 'Cache-Control': 'no-store' };
    response.writeHead(200, { ...headers, 'Content-Length': data.length });
    response.end(data);
  } catch {
    response.writeHead(404, { 'Content-Type': 'text/plain; charset=utf-8' }).end('Not found');
  }
});

server.listen(port, '127.0.0.1', () => {
  console.log(`Serving ${root} at http://127.0.0.1:${port}`);
});
