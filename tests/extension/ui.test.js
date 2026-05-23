import { getElements, renderResult, setStatus, showError, hideTransientPanels } from '../../extension/ui.js';

function setupDom() {
  document.body.innerHTML = `
    <div id="statusBox"></div>
    <button id="analyzeBtn"></button>
    <div id="resultArea" class="hidden"></div>
    <div id="errorMsg" class="hidden"></div>
    <span id="predictionLabel"></span>
    <span id="confidenceScore"></span>
    <div id="confidenceBar"></div>
    <span id="modelName"></span>
    <div id="resultIcon"></div>
  `;
  return getElements();
}

describe('ui', () => {
  it('renders REAL result with green variant', () => {
    const els = setupDom();
    renderResult(els, { label: 'REAL', score: 0.91, meta: { used_model: 'rf', used_dataset: 'ISOT' } });

    expect(els.label.textContent).toBe('REAL');
    expect(els.label.className).toContain('real');
    expect(els.progressBar.className).toContain('progress-fill--real');
    expect(els.confidence.textContent).toBe('91.0%');
    expect(els.progressBar.style.width).toBe('91.0%');
    expect(els.modelName.textContent).toBe('rf | ISOT');
    expect(els.resultArea.classList.contains('hidden')).toBe(false);
  });

  it('treats NEUTRAL as uncertain', () => {
    const els = setupDom();
    renderResult(els, { label: 'NEUTRAL', score: 0, meta: {} });
    expect(els.label.className).toContain('uncertain');
    expect(els.progressBar.className).toContain('progress-fill--uncertain');
    expect(els.modelName.textContent).toBe('unknown');
  });

  it('shows/hides transient panels and status text', () => {
    const els = setupDom();
    setStatus(els, 'hello');
    expect(els.statusBox.textContent).toBe('hello');

    showError(els, 'bad');
    expect(els.errorMsg.classList.contains('hidden')).toBe(false);
    expect(els.errorMsg.textContent).toBe('bad');

    hideTransientPanels(els);
    expect(els.errorMsg.classList.contains('hidden')).toBe(true);
    expect(els.resultArea.classList.contains('hidden')).toBe(true);
  });
});
