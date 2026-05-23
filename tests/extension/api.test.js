import { analyzeText, truncateText, ApiError } from '../../extension/api.js';
import { CONFIG } from '../../extension/config.js';

describe('truncateText', () => {
  it('trims whitespace and respects max length', () => {
    expect(truncateText('   hello   ')).toBe('hello');
    expect(truncateText('a'.repeat(CONFIG.MAX_TEXT_LENGTH + 50))).toHaveLength(CONFIG.MAX_TEXT_LENGTH);
  });

  it('handles null/undefined', () => {
    expect(truncateText(null)).toBe('');
    expect(truncateText(undefined)).toBe('');
  });
});

describe('analyzeText', () => {
  const ok = (body) => ({
    ok: true,
    status: 200,
    json: async () => body,
  });

  it('POSTs JSON and returns parsed body', async () => {
    const fetchImpl = jest.fn().mockResolvedValue(ok({ label: 'REAL', score: 0.9 }));
    const data = await analyzeText('hello world', { fetchImpl });

    expect(fetchImpl).toHaveBeenCalledTimes(1);
    const [url, init] = fetchImpl.mock.calls[0];
    expect(url).toBe(CONFIG.API_URL);
    expect(init.method).toBe('POST');
    expect(JSON.parse(init.body)).toEqual({ text: 'hello world' });
    expect(init.headers['Content-Type']).toBe('application/json');
    expect(data).toEqual({ label: 'REAL', score: 0.9 });
  });

  it('throws ApiError on non-2xx', async () => {
    const fetchImpl = jest.fn().mockResolvedValue({ ok: false, status: 500, json: async () => ({}) });
    await expect(analyzeText('x', { fetchImpl })).rejects.toMatchObject({
      name: 'ApiError',
      status: 500,
    });
  });

  it('wraps network errors as ApiError("network")', async () => {
    const fetchImpl = jest.fn().mockRejectedValue(new Error('boom'));
    await expect(analyzeText('x', { fetchImpl })).rejects.toMatchObject({
      name: 'ApiError',
      message: 'network',
    });
  });

  it('maps abort to ApiError("timeout")', async () => {
    const fetchImpl = jest.fn().mockImplementation(() => {
      const err = new Error('aborted');
      err.name = 'AbortError';
      return Promise.reject(err);
    });
    await expect(analyzeText('x', { fetchImpl })).rejects.toMatchObject({
      name: 'ApiError',
      message: 'timeout',
    });
  });
});