/**
 * Central place for all calls to the backend.
 *
 * The backend currently returns mock data (see /backend/routers/translate.py).
 * When the real Thai2ThSL / ThSL2Thai model is ready, only the backend needs
 * to change — these function signatures and return shapes are the "contract"
 * the frontend already expects, so no frontend changes should be needed.
 */

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

export interface ThaiToThSLResult {
  /** The recognized/normalized Thai input echoed back */
  inputText: string;
  /** The gloss sequence the model produced (for debugging / captions) */
  glosses: string[];
  /** URL of the generated (or mock) ThSL video to embed */
  videoUrl: string;
}

export interface ThSLToThaiResult {
  /** The predicted Thai sentence */
  text: string;
  /** Confidence score 0-1, if the model provides one */
  confidence?: number;
}

class ApiError extends Error {
  constructor(message: string, public status?: number) {
    super(message);
    this.name = "ApiError";
  }
}

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new ApiError(
      `Request failed (${res.status}): ${body || res.statusText}`,
      res.status
    );
  }
  return res.json() as Promise<T>;
}

/** Thai text (or transcribed speech) -> ThSL video */
export async function translateThaiToThSL(
  text: string
): Promise<ThaiToThSLResult> {
  const res = await fetch(`${API_BASE_URL}/api/translate/thai-to-thsl`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  return handleResponse<ThaiToThSLResult>(res);
}

/** ThSL video (uploaded file or recorded blob) -> Thai text */
export async function translateThSLToThai(
  file: File | Blob
): Promise<ThSLToThaiResult> {
  const formData = new FormData();
  formData.append("video", file, "input.webm");

  const res = await fetch(`${API_BASE_URL}/api/translate/thsl-to-thai`, {
    method: "POST",
    body: formData,
  });
  return handleResponse<ThSLToThaiResult>(res);
}

export { ApiError };
