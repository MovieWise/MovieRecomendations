import type { AuthResponse, FeedResponse, MovieInfo, Reaction, RecommendationResponse, ProfileResponse } from "./types";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000/api/v1";

async function request<T>(path: string, token: string | null, options: RequestInit = {}): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...options.headers
    }
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(typeof payload.detail === "string" ? payload.detail : JSON.stringify(payload.detail ?? payload));
  }
  return response.json() as Promise<T>;
}

export function authenticate(initData: string): Promise<AuthResponse> {
  return request<AuthResponse>("/auth/telegram", null, {
    method: "POST",
    body: JSON.stringify({ init_data: initData })
  });
}

export function loadFeed(token: string, limit = 20): Promise<FeedResponse> {
  return request<FeedResponse>(`/movies/feed?limit=${limit}`, token);
}

export function searchMovies(token: string, query: string, limit = 20): Promise<FeedResponse> {
  const params = new URLSearchParams({ q: query, limit: String(limit) });
  return request<FeedResponse>(`/movies/search?${params.toString()}`, token);
}

export function saveReaction(token: string, movieId: number, reaction: Reaction): Promise<void> {
  return request<void>("/reactions", token, {
    method: "POST",
    body: JSON.stringify({ movie_id: movieId, reaction, source: "telegram-mini-app" })
  });
}

export function deleteReaction(token: string, movieId: number): Promise<void> {
  return request<void>(`/reactions/${movieId}`, token, {
    method: "DELETE"
  });
}

export function loadProfile(token: string): Promise<ProfileResponse> {
  return request<ProfileResponse>("/profile/ratings", token);
}

export function generateRecommendations(token: string, topN = 10): Promise<RecommendationResponse> {
  return request<RecommendationResponse>("/recommendations/generate", token, {
    method: "POST",
    body: JSON.stringify({ top_n: topN, candidate_top_k: 100 })
  });
}

export function loadMovie(token: string, movieId: number): Promise<MovieInfo> {
  return request<MovieInfo>(`/movies/${movieId}`, token);
}
