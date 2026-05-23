import { CONFIG } from './config.js';

export class ApiError extends Error {
  constructor(message, { status, cause } = {}) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.cause = cause;
  }
}

export async function analyzeText(text, { fetchImpl = fetch, signal } = {}) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), CONFIG.REQUEST_TIMEOUT_MS);
  const composedSignal = signal ?? controller.signal;

  try {
    const response = await fetchImpl(CONFIG.API_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
      signal: composedSignal,
    });

    if (!response.ok) {
      throw new ApiError(`Server error: ${response.status}`, { status: response.status });
    }
    return await response.json();
  } catch (error) {
    if (error.name === 'AbortError') {
      throw new ApiError('timeout', { cause: error });
    }
    if (error instanceof ApiError) throw error;
    throw new ApiError('network', { cause: error });
  } finally {
    clearTimeout(timeoutId);
  }
}

export function truncateText(text, maxLength = CONFIG.MAX_TEXT_LENGTH) {
  const trimmed = (text ?? '').trim();
  return trimmed.length > maxLength ? trimmed.slice(0, maxLength) : trimmed;
}