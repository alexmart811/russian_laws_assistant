const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export interface Source {
  article_id: number;
  article_title: string;
  article_text: string;
  codex: string;
  score: number;
}

export interface GenerateResponse {
  query: string;
  answer: string;
  sources: Source[];
}

export async function generateAnswer(query: string): Promise<GenerateResponse> {
  const response = await fetch(`${API_URL}/generate`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      query,
      limit: 5,
      score_threshold: 0.5,
    }),
  });

  if (!response.ok) {
    throw new Error(`API error: ${response.status} ${response.statusText}`);
  }

  return response.json();
}
