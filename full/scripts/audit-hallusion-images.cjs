const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const [dataset, imageRoot] = process.argv.slice(2);
if (!dataset || !imageRoot) throw new Error('Usage: node audit-hallusion-images.cjs DATA.json IMAGE_ROOT');
const raw = fs.readFileSync(dataset);
const rows = JSON.parse(raw);
if (!Array.isArray(rows) || rows.length !== 1129) throw new Error('Expected 1129 questions');
const root = fs.realpathSync(imageRoot);
const images = [...new Set(rows.map(row => row.filename).filter(Boolean))];
const missing = [];
const records = [];
for (const filename of images) {
  const target = path.resolve(root, filename);
  if (!target.startsWith(root + path.sep)) throw new Error('Image outside root');
  if (!fs.existsSync(target) || !fs.statSync(target).isFile() || fs.statSync(target).size === 0) {
    missing.push(filename);
    continue;
  }
  const resolved = fs.realpathSync(target);
  if (!resolved.startsWith(root + path.sep)) throw new Error('Image symlink outside root');
  records.push({filename, sha256: crypto.createHash('sha256').update(fs.readFileSync(target)).digest('hex')});
}
console.log(JSON.stringify({status: missing.length ? 'missing_images' : 'passed', samples: rows.length,
  unique_images: images.length, dataset_sha256: crypto.createHash('sha256').update(raw).digest('hex'),
  missing, images: records}, null, 2));
if (missing.length || images.length !== 346) process.exitCode = 1;
