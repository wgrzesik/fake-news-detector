import { LABELS } from './config.js';

export function getElements(doc = document) {
  return {
    statusBox: doc.getElementById('statusBox'),
    analyzeBtn: doc.getElementById('analyzeBtn'),
    resultArea: doc.getElementById('resultArea'),
    errorMsg: doc.getElementById('errorMsg'),
    label: doc.getElementById('predictionLabel'),
    confidence: doc.getElementById('confidenceScore'),
    progressBar: doc.getElementById('confidenceBar'),
    modelName: doc.getElementById('modelName'),
    icon: doc.getElementById('resultIcon'),
  };
}

export function setStatus(els, text) {
  els.statusBox.textContent = text;
}

export function showError(els, text) {
  els.errorMsg.textContent = text;
  els.errorMsg.classList.remove('hidden');
}

export function hideTransientPanels(els) {
  els.resultArea.classList.add('hidden');
  els.errorMsg.classList.add('hidden');
}

function classifyLabel(rawLabel) {
  if (rawLabel === LABELS.REAL) return 'real';
  if (rawLabel === LABELS.FAKE) return 'fake';
  return 'uncertain';
}

export function renderResult(els, data) {
  const variant = classifyLabel(data.label);
  els.label.textContent = data.label ?? '--';
  els.label.className = `label ${variant}`;

  els.progressBar.className = `progress-fill progress-fill--${variant}`;

  const score = Number.isFinite(data.score) ? data.score : 0;
  const percentage = `${(score * 100).toFixed(1)}%`;
  els.confidence.textContent = percentage;
  els.progressBar.style.width = percentage;

  const meta = data.meta ?? {};
  const modelLabel = meta.used_model ?? 'unknown';
  const datasetLabel = meta.used_dataset ? ` | ${meta.used_dataset}` : '';
  els.modelName.textContent = `${modelLabel}${datasetLabel}`;

  els.resultArea.classList.remove('hidden');
}