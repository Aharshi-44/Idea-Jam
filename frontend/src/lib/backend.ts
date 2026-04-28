const DEFAULT_BACKEND_BASE_URL = "http://127.0.0.1:8000";

function backendBaseUrl() {
  return process.env.BACKEND_BASE_URL ?? DEFAULT_BACKEND_BASE_URL;
}

export function buildBackendUrl(path: string) {
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  return `${backendBaseUrl()}${normalizedPath}`;
}

export async function parseBackendError(response: Response) {
  try {
    const payload = (await response.json()) as { detail?: string };
    if (payload?.detail) {
      return payload.detail;
    }
  } catch {
    // no-op: fall back to status text
  }
  return `Backend request failed (${response.status})`;
}
