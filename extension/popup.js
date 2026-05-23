import { CONFIG, MESSAGES } from './config.js';
import { analyzeText, truncateText, ApiError } from './api.js';
import {
  getElements,
  setStatus,
  showError,
  hideTransientPanels,
  renderResult,
} from './ui.js';

function getSelectionText() {
  return window.getSelection().toString();
}

export async function readSelectedText(chromeApi = chrome) {
  const [tab] = await chromeApi.tabs.query({ active: true, currentWindow: true });
  const results = await chromeApi.scripting.executeScript({
    target: { tabId: tab.id },
    func: getSelectionText,
  });
  return results?.[0]?.result ?? '';
}

export async function handleAnalyzeClick(els, deps = {}) {
  const reader = deps.readSelectedText ?? readSelectedText;
  const analyzer = deps.analyzeText ?? analyzeText;

  hideTransientPanels(els);
  setStatus(els, MESSAGES.FETCHING);

  let selected;
  try {
    selected = truncateText(await reader());
  } catch (e) {
    showError(els, MESSAGES.API_DOWN);
    setStatus(els, MESSAGES.ERROR);
    return;
  }

  if (selected.length < CONFIG.MIN_TEXT_LENGTH) {
    setStatus(els, MESSAGES.TOO_SHORT);
    return;
  }

  setStatus(els, MESSAGES.ANALYZING);

  try {
    const data = await analyzer(selected);
    renderResult(els, data);
    setStatus(els, MESSAGES.DONE);
  } catch (error) {
    const text = error instanceof ApiError && error.message === 'timeout'
      ? MESSAGES.TIMEOUT
      : MESSAGES.API_DOWN;
    showError(els, text);
    setStatus(els, MESSAGES.ERROR);
  }
}

document.addEventListener('DOMContentLoaded', () => {
  const els = getElements();
  els.analyzeBtn.addEventListener('click', () => handleAnalyzeClick(els));
});