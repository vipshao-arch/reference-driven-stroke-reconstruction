import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';

const app = await readFile(resolve(import.meta.dirname, '../src/app.js'), 'utf8');
const errors = [];
if (!app.includes('window.__frameState = {')) errors.push('missing frame-state diagnostic export');
if (!app.includes('window.__frameMetrics = publicMetrics')) errors.push('missing metrics diagnostic export');
if (!app.includes('setSnapshot({')) errors.push('missing snapshot export');
if (!app.includes('ui.qualityScore.dataset.psnr = String(metrics.psnr)')) errors.push('missing psnr data attribute');
if (errors.length) {
  console.error(errors.join('\n'));
  process.exit(1);
}
console.log('Application diagnostic exports are present');
