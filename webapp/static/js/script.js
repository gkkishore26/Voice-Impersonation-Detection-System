/* ============ Voice Impersonation Detection - Frontend Logic ============ */

// ---------- element refs ----------
const $ = (id) => document.getElementById(id);

const tabUpload = $('tabUpload'), tabRecord = $('tabRecord');
const panelUpload = $('panelUpload'), panelRecord = $('panelRecord');
const fileInput = $('fileInput'), dropzone = $('dropzone');
const dzFile = $('dzFile'), dzFileName = $('dzFileName');
const btnAnalyze = $('btnAnalyze'), statusEl = $('status');

const btnRec = $('btnRec'), btnStop = $('btnStop'), btnUse = $('btnUse'), btnClear = $('btnClear');
const recTime = $('recTime'), recPlayback = $('recPlayback'), recHint = $('recHint');

const inputCard = $('inputCard'), resultCard = $('resultCard'), errorCard = $('errorCard');
const errMsg = $('errMsg'), btnErrBack = $('btnErrBack');

let selectedFile = null;   // File from upload OR recorded blob
let mediaRecorder = null, chunks = [], recBlob = null;
let recTimer = null, recSeconds = 0, recordedFile = null;

// ---------- tabs ----------
function setTab(upload) {
  tabUpload.classList.toggle('active', upload);
  tabRecord.classList.toggle('active', !upload);
  panelUpload.hidden = !upload;
  panelRecord.hidden = upload;
  refreshAnalyze();
}
tabUpload.onclick = () => setTab(true);
tabRecord.onclick = () => setTab(false);

// ---------- upload ----------
function pickFile(file) {
  if (!file) return;
  selectedFile = file;
  dzFileName.textContent = `${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`;
  dzFile.hidden = false;
  recordedFile = null;
  refreshAnalyze();
}
fileInput.onchange = () => pickFile(fileInput.files[0]);

['dragover', 'dragenter'].forEach(ev =>
  dropzone.addEventListener(ev, e => { e.preventDefault(); dropzone.classList.add('drag'); }));
['dragleave', 'drop'].forEach(ev =>
  dropzone.addEventListener(ev, e => { e.preventDefault(); dropzone.classList.remove('drag'); }));
dropzone.addEventListener('drop', e => pickFile(e.dataTransfer.files[0]));

// ---------- recording ----------
btnRec.onclick = async () => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    chunks = [];
    mediaRecorder = new MediaRecorder(stream);
    mediaRecorder.ondataavailable = e => { if (e.data.size) chunks.push(e.data); };
    mediaRecorder.onstop = () => {
      stream.getTracks().forEach(t => t.stop());
      recBlob = new Blob(chunks, { type: mediaRecorder.mimeType || 'audio/webm' });
      recPlayback.src = URL.createObjectURL(recBlob);
      recPlayback.hidden = false;
      btnUse.disabled = false;
      btnClear.disabled = false;
      recHint.textContent = 'Recording ready! Click Use Recording to select it';
    };
    mediaRecorder.start();
    recSeconds = 0;
    recTime.textContent = '00:00';
    recTimer = setInterval(() => {
      recSeconds++;
      const m = String(Math.floor(recSeconds / 60)).padStart(2, '0');
      const s = String(recSeconds % 60).padStart(2, '0');
      recTime.textContent = `${m}:${s}`;
      if (recSeconds >= 60) btnStop.onclick();   // auto-stop at 60s
    }, 1000);
    btnRec.disabled = true; btnRec.classList.add('recording');
    btnStop.disabled = false; btnUse.disabled = true; btnClear.disabled = true;
    recHint.textContent = '🔴 Recording... (max 60 seconds)';
  } catch (e) {
    recHint.textContent = '❌ Microphone access denied: ' + e.message;
  }
};

btnStop.onclick = () => {
  if (mediaRecorder && mediaRecorder.state !== 'inactive') mediaRecorder.stop();
  clearInterval(recTimer);
  btnRec.disabled = false; btnRec.classList.remove('recording');
  btnStop.disabled = true;
};

btnUse.onclick = () => {
  const ext = (recBlob.type.includes('mp4') || recBlob.type.includes('m4a')) ? 'm4a' : 'webm';
  recordedFile = new File([recBlob], `live_recording.${ext}`, { type: recBlob.type });
  selectedFile = recordedFile;
  refreshAnalyze();
  recHint.textContent = `✔ Recording selected (${(recordedFile.size / 1024).toFixed(0)} KB) — ready to analyze`;
};

btnClear.onclick = () => {
  recBlob = null; recordedFile = null;
  recPlayback.hidden = true; recPlayback.src = '';
  btnUse.disabled = true; btnClear.disabled = true;
  recTime.textContent = '00:00';
  recHint.textContent = 'Cleared. Record again.';
  if (!panelUpload.hidden === false) { /* keep */ }
  selectedFile = fileInput.files[0] || null;
  refreshAnalyze();
};

// ---------- analyze ----------
function refreshAnalyze() {
  btnAnalyze.disabled = !selectedFile;
}
function showStatus(msg, cls) {
  statusEl.textContent = msg;
  statusEl.className = 'status' + (cls ? ' ' + cls : '');
}

btnAnalyze.onclick = async () => {
  if (!selectedFile) return;

  // UI: loading
  btnAnalyze.disabled = true;
  errorCard.hidden = true;

  // ---- streamed analysis: live backend progress, fallback to /predict ----
  if (typeof EventSource !== 'undefined') {
    try {
      await analyzeStreamed();
      return;
    } catch (e) {
      showStatus('stream failed - retrying direct... ' + e.message);
    }
  }
  await analyzeDirect();
};

async function analyzeStreamed() {
  // step 1: upload file, get reference
  showStatus('\u25B8 uploading file...');
  const fd = new FormData();
  fd.append('audio', selectedFile, selectedFile.name);
  const up = await fetch('/upload', { method: 'POST', body: fd });
  const uj = await up.json();
  if (!up.ok || uj.error) { showError(uj.error || 'Upload failed'); btnAnalyze.disabled = false; return; }

  // step 2: stream backend progress
  await new Promise((resolve, reject) => {
    const es = new EventSource('/predict_stream?name=' + encodeURIComponent(uj.name));
    let finished = false;
    es.onmessage = (ev) => {
      let m;
      try { m = JSON.parse(ev.data); } catch (e) { return; }
      if (m.type === 'step') {
        showStatus('\u25B8 ' + m.msg);
      } else if (m.type === 'result') {
        finished = true;
        es.close();
        showResult(m.result);
        resolve();
      } else if (m.type === 'error') {
        finished = true;
        es.close();
        showError(m.error);
        resolve();
      }
    };
    es.onerror = () => {
      es.close();
      if (!finished) reject(new Error('stream connection lost'));
      else resolve();
    };
  });
  btnAnalyze.disabled = false;
}

async function analyzeDirect() {
  showStatus('\u23F3 Analyzing voice patterns...');

  const fd = new FormData();
  fd.append('audio', selectedFile, selectedFile.name);

  try {
    const res = await fetch('/predict', { method: 'POST', body: fd });
    const data = await res.json();

    if (!res.ok || data.error) {
      showError(data.error || 'Server error');
      return;
    }
    showResult(data);
  } catch (e) {
    showError('Network error: ' + e.message);
  } finally {
    btnAnalyze.disabled = false;
    if (statusEl.textContent.startsWith('⏳')) showStatus('');
  }
};

function showError(msg) {
  showStatus('❌ ' + msg, 'err');
  inputCard.hidden = false;
  resultCard.hidden = true;
  errorCard.hidden = false;
  errMsg.textContent = msg;
}

function showResult(d) {
  showStatus('');
  inputCard.hidden = true;
  errorCard.hidden = true;
  resultCard.hidden = false;

  const isReal = d.is_real === true;
  const unc = d.uncertain === true;
  const verdict = $('verdict');
  verdict.className = 'verdict ' + (unc ? 'uncertain' : (isReal ? 'real' : 'fake'));
  $('verdictIcon').textContent = unc ? '?' : (isReal ? '🧑' : '🤖');
  $('verdictText').textContent = d.label;
  $('verdictConf').innerHTML = `<b>${d.confidence}%</b><span>CONFIDENCE</span>`;

  // animated bars
  requestAnimationFrame(() => {
    $('barReal').style.width = d.real_probability + '%';
    $('barFake').style.width = d.fake_probability + '%';
  });
  $('valReal').textContent = d.real_probability + '%';
  $('valFake').textContent = d.fake_probability + '%';

  // details
  const spec = $('specImg');
  if (d.spectrogram) { spec.src = d.spectrogram + '?t=' + Date.now(); spec.hidden = false; }
  else spec.hidden = true;

  $('dDur').textContent = d.duration ? (d.duration >= 60 ? (d.duration / 60).toFixed(1) + ' min' : d.duration + ' s') : '—';
  $('dMode').textContent = d.mode === 'segmented'
    ? `Segmented (${d.n_segments} × 3s blocks)`
    : 'Single (3s window)';
  $('dTime').textContent = (d.elapsed + ' s')
    + (d.infer_ms !== undefined ? ` (GPU infer ${d.infer_ms} ms)` : '');
  $('dModel').textContent = d.model_epoch
    ? `CNN (epoch ${d.model_epoch}${d.model_val_acc ? ', val acc ' + d.model_val_acc + '%' : ''})`
    : 'Not trained yet';
  $('dDev').textContent = d.device === 'cuda' ? 'GPU' : 'CPU';
  $('dTs').textContent = new Date().toLocaleString();

  // segmented timeline
  const segSection = $('segSection'), segGrid = $('segGrid');
  if (d.mode === 'segmented' && Array.isArray(d.segments)) {
    segSection.hidden = false;
    $('segTitle').textContent =
      `Segment Analysis — ${d.real_votes}/${d.n_segments} blocks judged Real Human`;
    segGrid.innerHTML = d.segments.map(s => {
      const cls = s.is_real === null ? 'unknown' : (s.is_real ? 'real' : 'fake');
      const t = fmtTime(s.start);
      return `<div class="seg-block ${cls}" title="${t} — ${s.label} (${s.confidence}%)">${t}</div>`;
    }).join('');
  } else {
    segSection.hidden = true;
  }

  if (d.warning) showStatus('⚠️ ' + d.warning, 'err');

  // acoustic evidence block
  const evS = $('evSection');
  if (d.findings && d.findings.length) {
    evS.hidden = false;
    $('evExplain').textContent = d.explanation || '';
    $('evExplain').hidden = !d.explanation;
    $('evList').innerHTML = d.findings.map(f =>
      `<li class="ev-${f.direction}"><span>${f.tag}</span>${f.text}</li>`).join('');
    const lim = d.limitations || [];
    $('evLimits').textContent = lim.length ? ('Limits: ' + lim.join(' · ')) : '';
    $('evLimits').hidden = !lim.length;
  } else {
    evS.hidden = true;
  }

  resultCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ---------- result actions ----------
$('btnNew').onclick = () => {
  resultCard.hidden = true;
  inputCard.hidden = false;
  window.scrollTo({ top: 0, behavior: 'smooth' });
};
$('btnPrint').onclick = () => window.print();
btnErrBack.onclick = () => {
  errorCard.hidden = true;
  showStatus('');
};

// ---------- helpers ----------
function fmtTime(sec) {
  const m = String(Math.floor(sec / 60)).padStart(2, '0');
  const s = String(Math.floor(sec % 60)).padStart(2, '0');
  return `${m}:${s}`;
}

// ---------- health check on load ----------
fetch('/health').then(r => r.json()).then(h => {
  const hm = $('healthModel'), hd = $('healthDevice');
  if (h.model_loaded) {
    if (hm) hm.textContent = h.epoch ? `PROFESSOR K / EPOCH ${h.epoch}` : 'PROFESSOR K';
    if (hd) hd.textContent = 'GPU';
  } else {
    if (hm) hm.textContent = 'NOT TRAINED';
    if (hd) hd.textContent = 'GPU';
    showStatus('⚠️ Model not trained yet — predictions will be unreliable', 'err');
  }
}).catch(() => {
  const hm = $('healthModel');
  if (hm) hm.textContent = 'OFFLINE';
});

setTab(true);
