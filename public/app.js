const HIGH_RISK = ["retinal detachment", "glaucoma", "diabetic retinopathy"];
const $ = s => document.querySelector(s);
const pct = n => `${(n * 100).toFixed(1)}%`;
const nice = s => s.replaceAll('_', ' ');
let imageFile;

function setReady() {
  $('#model-status').textContent = 'Ready - analysis runs in the cloud using the same preprocessing as training';
  $('#analyze').disabled = false;
}

async function toJpegBlob(file, maxSide = 1600) {
  const bmp = await createImageBitmap(file);
  const scale = Math.min(1, maxSide / Math.max(bmp.width, bmp.height));
  const canvas = document.createElement('canvas');
  canvas.width = Math.round(bmp.width * scale);
  canvas.height = Math.round(bmp.height * scale);
  canvas.getContext('2d').drawImage(bmp, 0, 0, canvas.width, canvas.height);
  return new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.9));
}

async function analyze() {
  if (!imageFile) return;
  const btn = $('#analyze');
  btn.disabled = true;
  btn.textContent = 'Analyzing...';
  try {
    const blob = await toJpegBlob(imageFile);
    const res = await fetch('/api/predict', { method: 'POST', headers: { 'Content-Type': 'image/jpeg' }, body: blob });
    const out = await res.json();
    if (!res.ok) throw new Error(out.error || 'Request failed');
    const names = out.classes.map(nice);
    const ranked = out.probabilities.map((p, i) => ({ p, i })).sort((a, b) => b.p - a.p).slice(0, 3);
    $('#empty-result').classList.add('hidden');
    $('#result').classList.remove('hidden');
    $('#top-class').textContent = names[ranked[0].i];
    $('#top-score').textContent = pct(ranked[0].p);
    $('#predictions').innerHTML = ranked.map(r =>
      `<div class="prediction"><div class="prediction-top"><span>${names[r.i]}</span><b>${pct(r.p)}</b></div><div class="track"><div class="fill" style="width:${r.p * 100}%"></div></div></div>`).join('');
    $('#quality-value').textContent = out.quality.passed ? 'Pass' : 'Check';
    $('#quality-note').textContent = out.quality.note;
    const needsReferral = ranked[0].p < 0.60 || HIGH_RISK.includes(names[ranked[0].i]) || !out.quality.passed;
    $('#referral').textContent = needsReferral ? 'Clinical review recommended: confidence, condition or image-quality safeguard triggered.' : '';
    $('#referral').classList.toggle('hidden', !needsReferral);
  } catch (err) {
    alert(err.message);
  } finally {
    btn.disabled = false;
    btn.innerHTML = 'Analyze image <span>&rarr;</span>';
  }
}

/* ---------- evidence dashboard ---------- */
function plotLines(selector, series, boundary, zeroToOne) {
  const svg = $(selector);
  const n = series[0].values.length;
  if (!n) { svg.innerHTML = ''; return; }
  const all = series.flatMap(s => s.values);
  let min = zeroToOne ? 0 : Math.min(...all), max = zeroToOne ? 1 : Math.max(...all);
  if (max - min < 1e-6) max = min + 1;
  const w = 600, h = 225, left = 44, top = 10, bottom = 26;
  const x = i => left + i * (w - left) / Math.max(n - 1, 1);
  const y = v => top + (max - v) * (h - top - bottom) / (max - min);
  let svgHtml = '';
  for (let t = 0; t <= 4; t++) {
    const v = min + (max - min) * t / 4, yy = y(v);
    svgHtml += `<line class="gridline" x1="${left}" x2="${w}" y1="${yy}" y2="${yy}"/><text class="axis-label" x="0" y="${yy + 3}">${v.toFixed(2)}</text>`;
  }
  if (boundary > 0 && boundary < n) {
    const bx = x(boundary - 0.5);
    svgHtml += `<line class="gridline" x1="${bx}" x2="${bx}" y1="${top}" y2="${h - bottom}" stroke-dasharray="4 4"/><text class="axis-label" x="${bx + 4}" y="${top + 10}">fine-tuning starts</text>`;
  }
  svgHtml += `<text class="axis-label" x="${left}" y="${h - 6}">epoch 1</text><text class="axis-label" x="${w - 46}" y="${h - 6}">epoch ${n}</text>`;
  svgHtml += series.map(s => `<polyline class="${s.cls}" fill="none" points="${s.values.map((v, i) => `${x(i)},${y(v)}`).join(' ')}"/>`).join('');
  svg.innerHTML = svgHtml;
}

function curvesUI(history) {
  if (!history || !history.length) return;
  const boundary = history.findIndex(r => r.phase === 'finetune');
  plotLines('#curves', [
    { values: history.map(r => r.train_f1), cls: 'trainline' },
    { values: history.map(r => r.val_f1), cls: 'valline' }], boundary, true);
  if (history[0].train_loss !== undefined) {
    plotLines('#loss-curves', [
      { values: history.map(r => r.train_loss), cls: 'trainline' },
      { values: history.map(r => r.val_loss), cls: 'valline' }], boundary, false);
  }
}

function splitUI(sc) {
  if (!sc) return;
  const classes = Object.keys(sc.train);
  const total = c => sc.train[c] + sc.val[c] + sc.test[c];
  const max = Math.max(...classes.map(total));
  $('#split-bars').innerHTML = classes.map(c => {
    const seg = (k, cls) => `<div class="${cls}" style="width:${sc[k][c] / max * 100}%" title="${k}: ${sc[k][c]}"></div>`;
    return `<div class="split-row"><span>${nice(c)}</span><div class="split-track">${seg('train', 'seg-train')}${seg('val', 'seg-val')}${seg('test', 'seg-test')}</div><b>${total(c)}</b></div>`;
  }).join('');
  const sum = k => classes.reduce((a, c) => a + sc[k][c], 0);
  const tr = sum('train'), va = sum('val'), te = sum('test'), all = tr + va + te;
  $('#m-dataset').textContent = `${all.toLocaleString()} original images`;
  $('#m-split').textContent = `${tr.toLocaleString()} / ${va.toLocaleString()} / ${te.toLocaleString()} (train / val / test)`;
}

function tableUI(rows) {
  if (!rows || rows[0].precision === undefined) return;
  const head = '<tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>AUC</th><th>Test images</th></tr>';
  const body = rows.map(r => {
    const small = r.support < 30 ? ' <span class="flag" title="Very few test images: treat this row as indicative only">small n</span>' : '';
    return `<tr><td>${nice(r.class)}${small}</td><td>${r.precision.toFixed(2)}</td><td>${r.recall.toFixed(2)}</td><td>${r.f1.toFixed(2)}</td><td>${r.auc === null ? '-' : r.auc.toFixed(3)}</td><td>${r.support}</td></tr>`;
  }).join('');
  $('#class-table').innerHTML = head + body;
}

function confusionUI(cm) {
  if (!cm) return;
  const labels = cm.labels.map(nice);
  let html = '<tr><th></th>' + labels.map((_, i) => `<th title="${labels[i]}">${i + 1}</th>`).join('') + '</tr>';
  cm.counts.forEach((row, r) => {
    const total = row.reduce((a, b) => a + b, 0) || 1;
    html += `<tr><th class="rowhead">${r + 1}. ${labels[r]}</th>` + row.map(v => {
      const ratio = v / total;
      const bg = `rgba(22,118,111,${(ratio * 0.9).toFixed(2)})`;
      return `<td style="background:${bg};color:${ratio > 0.5 ? '#fff' : 'inherit'}" title="${v} of ${total}">${v}</td>`;
    }).join('') + '</tr>';
  });
  $('#cm-table').innerHTML = html;
  $('#cm-legend').textContent = 'Columns: ' + labels.map((l, i) => `${i + 1} = ${l}`).join(' | ');
}

function evidenceUI(d) {
  $('#macro-f1').textContent = pct(d.metrics.macro_f1);
  $('#weighted-f1').textContent = d.metrics.weighted_f1 !== undefined ? pct(d.metrics.weighted_f1) : '-';
  $('#accuracy').textContent = pct(d.metrics.accuracy);
  $('#test-total').textContent = d.metrics.test_samples.toLocaleString();
  $('#f1-bars').innerHTML = d.per_class.map(x =>
    `<div class="f1-row"><span>${nice(x.class)}</span><div class="track"><div class="fill" style="width:${x.f1 * 100}%"></div></div><b>${x.f1.toFixed(2)}</b></div>`).join('');
  curvesUI(d.history);
  splitUI(d.split_counts);
  tableUI(d.per_class);
  confusionUI(d.confusion_matrix);
  if (d.model) {
    const head = (d.history || []).filter(r => r.phase === 'head').length;
    const ft = (d.history || []).filter(r => r.phase === 'finetune').length;
    $('#m-model').textContent = d.model.architecture;
    $('#m-model-note').textContent = `ImageNet-pretrained, ${d.model.image_size} x ${d.model.image_size} px, border crop + CLAHE preprocessing, class-weighted cross-entropy with label smoothing, mild augmentation. ${head} head-only + ${ft} fine-tuning epochs run. Best checkpoint chosen on validation macro-F1.`;
  }
}

function selectImage(file) {
  imageFile = file;
  if (!imageFile) return;
  if (!['image/jpeg', 'image/png'].includes(imageFile.type)) { alert('Please choose a JPEG or PNG image.'); imageFile = null; return; }
  if (imageFile.size > 25 * 1024 * 1024) { alert('Please choose an image under 25 MB.'); imageFile = null; return; }
  $('#preview').src = URL.createObjectURL(imageFile);
  $('#preview-wrap').classList.remove('hidden');
  $('#dropzone').classList.add('hidden');
  $('#model-status').textContent = 'Image ready. Select Analyze image when you are ready.';
}

$('#image-input').addEventListener('change', e => selectImage(e.target.files[0]));
['dragenter', 'dragover'].forEach(event => $('#dropzone').addEventListener(event, e => {
  e.preventDefault();
  $('#dropzone').classList.add('dragover');
}));
['dragleave', 'drop'].forEach(event => $('#dropzone').addEventListener(event, e => {
  e.preventDefault();
  $('#dropzone').classList.remove('dragover');
}));
$('#dropzone').addEventListener('drop', e => selectImage(e.dataTransfer.files[0]));
$('#clear-image').onclick = () => {
  $('#image-input').value = '';
  imageFile = null;
  $('#preview-wrap').classList.add('hidden');
  $('#dropzone').classList.remove('hidden');
  $('#model-status').textContent = 'Ready - choose a JPEG or PNG fundus image to begin.';
};
$('#analyze').onclick = analyze;
setReady();
fetch('/metrics.json').then(r => r.json()).then(evidenceUI).catch(console.error);
