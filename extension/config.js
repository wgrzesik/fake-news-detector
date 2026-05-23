export const CONFIG = Object.freeze({
  API_URL: 'http://127.0.0.1:8000/predict',
  MIN_TEXT_LENGTH: 10,
  MAX_TEXT_LENGTH: 5000,
  REQUEST_TIMEOUT_MS: 15000,
});

export const LABELS = Object.freeze({
  REAL: 'REAL',
  FAKE: 'FAKE',
  NEUTRAL: 'NEUTRAL',
});

export const MESSAGES = Object.freeze({
  IDLE: 'Select text on any webpage to analyze its credibility.',
  FETCHING: 'Fetching text...',
  ANALYZING: 'Analyzing...',
  DONE: 'Analysis complete.',
  ERROR: 'Error.',
  TOO_SHORT: `Please select a longer text fragment (min. ${10} characters).`,
  API_DOWN: "Could not connect to the API. Did you start 'api.py'?",
  TIMEOUT: 'Request timed out. Please try again.',
});