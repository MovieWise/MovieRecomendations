from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


ReactionValue = Literal["like", "dislike"]


class TelegramAuthRequest(BaseModel):
    init_data: str = Field(min_length=1)


class UserResponse(BaseModel):
    id: int
    telegram_id: int
    username: str | None = None
    first_name: str | None = None
    last_name: str | None = None


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


class MovieInfo(BaseModel):
    movie_id: int
    imdb_id: str | None = None
    title: str | None = None
    year: str | None = None
    genre: str | None = None
    plot: str | None = None
    poster: str | None = None
    rating: str | None = None
    runtime: str | None = None
    director: str | None = None
    actors: str | None = None
    imdb_url: str | None = None


class MovieFeedResponse(BaseModel):
    movies: list[MovieInfo]
    rated_count: int


class ReactionRequest(BaseModel):
    movie_id: int
    reaction: ReactionValue
    source: str | None = "telegram-mini-app"
    session_id: str | None = None
    metadata: dict[str, Any] | None = None


class ReactionResponse(BaseModel):
    movie_id: int
    reaction: ReactionValue
    updated_at: datetime


class ProfileReaction(BaseModel):
    movie_id: int
    reaction: ReactionValue
    updated_at: datetime
    movie: MovieInfo | None = None


class ProfileResponse(BaseModel):
    user: UserResponse
    ratings: list[ProfileReaction]
    liked_count: int
    disliked_count: int


class RecommendationRequest(BaseModel):
    top_n: int = Field(default=10, ge=1, le=50)
    candidate_top_k: int = Field(default=100, ge=10, le=500)


class RecommendationResponse(BaseModel):
    success: bool
    recommendations: list[MovieInfo]
    processing_time: float
    model: str = "ease_lgbm"


class HealthResponse(BaseModel):
    status: str
    database: str
    omdb_configured: bool
    model_available: bool
    model_missing_artifacts: list[str]


class ErrorResponse(BaseModel):
    detail: str
