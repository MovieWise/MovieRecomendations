export type Reaction = "like" | "dislike";

export type MovieInfo = {
  movie_id: number;
  imdb_id?: string | null;
  title?: string | null;
  year?: string | null;
  genre?: string | null;
  plot?: string | null;
  poster?: string | null;
  rating?: string | null;
  runtime?: string | null;
  director?: string | null;
  actors?: string | null;
  imdb_url?: string | null;
};

export type AuthResponse = {
  access_token: string;
  token_type: "bearer";
  user: {
    id: number;
    telegram_id: number;
    username?: string | null;
    first_name?: string | null;
    last_name?: string | null;
  };
};

export type FeedResponse = {
  movies: MovieInfo[];
  rated_count: number;
};

export type ProfileResponse = {
  ratings: Array<{ movie_id: number; reaction: Reaction; updated_at: string; movie?: MovieInfo | null }>;
  liked_count: number;
  disliked_count: number;
};

export type RecommendationResponse = {
  success: boolean;
  recommendations: MovieInfo[];
  processing_time: number;
  model: string;
};

declare global {
  interface Window {
    Telegram?: {
      WebApp?: {
        initData: string;
        ready: () => void;
        expand: () => void;
        colorScheme?: "light" | "dark";
      };
    };
  }
}
