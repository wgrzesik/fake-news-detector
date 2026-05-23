import { handleAnalyzeClick } from '../../extension/popup.js';
import { getElements } from '../../extension/ui.js';
import { MESSAGES } from '../../extension/config.js';

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

describe('handleAnalyzeClick', () => {
  it('rejects too-short selections without calling the API', async () => {
    const els = setupDom();
    const analyzeText = jest.fn();
    await handleAnalyzeClick(els, { readSelectedText: async () => 'hi', analyzeText });

    expect(analyzeText).not.toHaveBeenCalled();
    expect(els.statusBox.textContent).toBe(MESSAGES.TOO_SHORT);
  });

  it('renders API result on success', async () => {
    const els = setupDom();
    const analyzeText = jest.fn().mockResolvedValue({
      label: 'FAKE',
      score: 0.42,
      meta: { used_model: 'xgb', used_dataset: 'WELFake' },
    });
    await handleAnalyzeClick(els, {
      readSelectedText: async () => 'this is a long enough selection',
      analyzeText,
    });

    expect(analyzeText).toHaveBeenCalled();
    expect(els.label.textContent).toBe('FAKE');
    expect(els.statusBox.textContent).toBe(MESSAGES.DONE);
  });

  it('shows API_DOWN error on network failure', async () => {
    const els = setupDom();
    const analyzeText = jest.fn().mockRejectedValue(Object.assign(new Error('network'), { name: 'ApiError' }));
    await handleAnalyzeClick(els, {
      readSelectedText: async () => 'this is a long enough selection',
      analyzeText,
    });

    expect(els.errorMsg.textContent).toBe(MESSAGES.API_DOWN);
    expect(els.errorMsg.classList.contains('hidden')).toBe(false);
    expect(els.statusBox.textContent).toBe(MESSAGES.ERROR);
  });
});