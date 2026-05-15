import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  authenticate,
  deleteReaction,
  generateRecommendations,
  loadFeed,
  loadProfile,
  saveReaction,
  searchMovies
} from "./api";
import type { MovieInfo, ProfileResponse, Reaction } from "./types";
import "./styles.css";

type AppState = "boot" | "onboarding" | "catalog" | "error";
type ViewMode = "catalog" | "ratings";

function normalizeError(error: unknown) {
  return error instanceof Error ? error.message : "Не удалось выполнить действие";
}

function MovieTile({
  movie,
  onReact,
  busy
}: {
  movie: MovieInfo;
  onReact: (movieId: number, reaction: Reaction) => void;
  busy: boolean;
}) {
  return (
    <article className="movie-tile">
      <div className="poster">
        {movie.poster ? <img src={movie.poster} alt="" /> : <div className="poster-empty">{movie.title?.slice(0, 1) ?? "M"}</div>}
      </div>
      <div className="tile-body">
        {movie.rating && movie.rating !== "N/A" && (
          <div className="imdb-rating" aria-label={`IMDb ${movie.rating}`}>
            <span className="rating-star filled">★</span>
            <strong>{movie.rating}</strong>
          </div>
        )}
        <h3>{movie.title ?? "Без названия"}</h3>
        <p>{[movie.year, movie.genre?.split(",")[0]].filter(Boolean).join(" · ")}</p>
      </div>
      <div className="tile-actions">
        <button className="icon negative" onClick={() => onReact(movie.movie_id, "dislike")} disabled={busy} aria-label="Не понравилось">
          −
        </button>
        <button className="icon positive" onClick={() => onReact(movie.movie_id, "like")} disabled={busy} aria-label="Понравилось">
          +
        </button>
      </div>
    </article>
  );
}

function RecommendationRow({ movie }: { movie: MovieInfo }) {
  return (
    <article className="recommendation-row">
      <div className="row-poster">
        {movie.poster ? <img src={movie.poster} alt="" /> : <span>{movie.title?.slice(0, 1) ?? "M"}</span>}
      </div>
      <div>
        <h3>{movie.title ?? "Без названия"}</h3>
        {movie.rating && movie.rating !== "N/A" && (
          <div className="imdb-rating row-rating" aria-label={`IMDb ${movie.rating}`}>
            <span className="rating-star filled">★</span>
            <strong>{movie.rating}</strong>
          </div>
        )}
        <p>{[movie.year, movie.runtime, movie.genre].filter(Boolean).join(" · ")}</p>
        {movie.plot && <p className="plot">{movie.plot}</p>}
      </div>
    </article>
  );
}

function RatingRow({
  rating,
  onReact,
  onDelete,
  busy
}: {
  rating: ProfileResponse["ratings"][number];
  onReact: (movieId: number, reaction: Reaction) => void;
  onDelete: (movieId: number) => void;
  busy: boolean;
}) {
  const movie = rating.movie;
  const liked = rating.reaction === "like";
  return (
    <article className="rating-row">
      <div className="rating-poster">
        {movie?.poster ? <img src={movie.poster} alt="" /> : <span>{movie?.title?.slice(0, 1) ?? "M"}</span>}
      </div>
      <div className="rating-copy">
        <h3>{movie?.title ?? `Movie #${rating.movie_id}`}</h3>
        {movie?.rating && movie.rating !== "N/A" && (
          <div className="imdb-rating row-rating" aria-label={`IMDb ${movie.rating}`}>
            <span className="rating-star filled">★</span>
            <strong>{movie.rating}</strong>
          </div>
        )}
        <p>{[movie?.year, movie?.genre?.split(",")[0]].filter(Boolean).join(" · ")}</p>
        <span className={liked ? "reaction-badge liked" : "reaction-badge disliked"}>
          {liked ? "Понравилось" : "Не моё"}
        </span>
      </div>
      <div className="rating-actions">
        <button
          className={liked ? "small-action negative" : "small-action positive"}
          onClick={() => onReact(rating.movie_id, liked ? "dislike" : "like")}
          disabled={busy}
        >
          Изменить
        </button>
        <button className="small-action neutral" onClick={() => onDelete(rating.movie_id)} disabled={busy}>
          Удалить
        </button>
      </div>
    </article>
  );
}

function App() {
  const [token, setToken] = useState<string | null>(null);
  const [state, setState] = useState<AppState>("boot");
  const [movies, setMovies] = useState<MovieInfo[]>([]);
  const [profile, setProfile] = useState<ProfileResponse | null>(null);
  const [recommendations, setRecommendations] = useState<MovieInfo[]>([]);
  const [query, setQuery] = useState("");
  const [viewMode, setViewMode] = useState<ViewMode>("catalog");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const ratedCount = useMemo(() => (profile ? profile.liked_count + profile.disliked_count : 0), [profile]);

  useEffect(() => {
    const webApp = window.Telegram?.WebApp;
    webApp?.ready();
    webApp?.expand();
    const initData = webApp?.initData || import.meta.env.VITE_TELEGRAM_INIT_DATA;
    if (!initData) {
      setState("onboarding");
      return;
    }
    authenticate(initData)
      .then(async (auth) => {
        setToken(auth.access_token);
        const [feed, nextProfile] = await Promise.all([loadFeed(auth.access_token, 24), loadProfile(auth.access_token)]);
        setMovies(feed.movies);
        setProfile(nextProfile);
        setState("catalog");
      })
      .catch((err) => {
        setError(normalizeError(err));
        setState("error");
      });
  }, []);

  useEffect(() => {
    if (!token || state !== "catalog") return;
    const timeout = window.setTimeout(async () => {
      setBusy(true);
      try {
        const result = query.trim() ? await searchMovies(token, query, 24) : await loadFeed(token, 24);
        setMovies(result.movies);
      } catch (err) {
        setError(normalizeError(err));
      } finally {
        setBusy(false);
      }
    }, 280);
    return () => window.clearTimeout(timeout);
  }, [query, token, state]);

  async function react(movieId: number, reaction: Reaction) {
    if (!token) return;
    setBusy(true);
    try {
      await saveReaction(token, movieId, reaction);
      const [nextProfile, nextMovies] = await Promise.all([
        loadProfile(token),
        query.trim() ? searchMovies(token, query, 24) : loadFeed(token, 24)
      ]);
      setProfile(nextProfile);
      setMovies(nextMovies.movies);
    } catch (err) {
      setError(normalizeError(err));
      setState("error");
    } finally {
      setBusy(false);
    }
  }

  async function removeRating(movieId: number) {
    if (!token) return;
    setBusy(true);
    try {
      await deleteReaction(token, movieId);
      const [nextProfile, nextMovies] = await Promise.all([
        loadProfile(token),
        query.trim() ? searchMovies(token, query, 24) : loadFeed(token, 24)
      ]);
      setProfile(nextProfile);
      setMovies(nextMovies.movies);
    } catch (err) {
      setError(normalizeError(err));
      setState("error");
    } finally {
      setBusy(false);
    }
  }

  async function generate() {
    if (!token) return;
    setBusy(true);
    try {
      const result = await generateRecommendations(token, 10);
      setRecommendations(result.recommendations);
    } catch (err) {
      setError(normalizeError(err));
      setState("error");
    } finally {
      setBusy(false);
    }
  }

  if (state === "boot") {
    return (
      <main className="app-shell center">
        <div className="loader">Загружаем подборку</div>
      </main>
    );
  }

  if (state === "onboarding") {
    return (
      <main className="app-shell center">
        <section className="welcome">
          <span className="brand-mark">MR</span>
          <h1>MovieRecs</h1>
          <p>Откройте Mini App из Telegram или передайте локальный `VITE_TELEGRAM_INIT_DATA`.</p>
        </section>
      </main>
    );
  }

  if (state === "error") {
    return (
      <main className="app-shell center">
        <section className="notice">
          <h1>Не получилось</h1>
          <p>{error ?? "Попробуйте обновить приложение."}</p>
          <button className="button primary" onClick={() => window.location.reload()}>Обновить</button>
        </section>
      </main>
    );
  }

  return (
    <main className="app-shell">
      <header className="header">
        <div>
          <span className="brand">MovieRecs</span>
          <h1>Что вы уже смотрели?</h1>
        </div>
        <button className="button primary top-cta" onClick={generate} disabled={busy || ratedCount === 0}>
          Рекомендации
        </button>
      </header>

      <section className="search-panel">
        <div className="tabs" role="tablist" aria-label="Разделы">
          <button className={viewMode === "catalog" ? "tab active" : "tab"} onClick={() => setViewMode("catalog")}>
            Каталог
          </button>
          <button className={viewMode === "ratings" ? "tab active" : "tab"} onClick={() => setViewMode("ratings")}>
            Мои оценки
          </button>
        </div>
        <input
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Найти фильм"
          aria-label="Найти фильм"
          disabled={viewMode === "ratings"}
        />
        <div className="profile-line">
          <span>{profile?.liked_count ?? 0} понравилось</span>
          <span>{profile?.disliked_count ?? 0} не моё</span>
          <span>{ratedCount} всего</span>
        </div>
      </section>

      {viewMode === "catalog" ? (
        <section className="catalog-grid" aria-busy={busy}>
          {movies.map((movie) => (
            <MovieTile key={movie.movie_id} movie={movie} onReact={react} busy={busy} />
          ))}
        </section>
      ) : (
        <section className="ratings-list">
          {(profile?.ratings ?? []).map((rating) => (
            <RatingRow key={rating.movie_id} rating={rating} onReact={react} onDelete={removeRating} busy={busy} />
          ))}
        </section>
      )}

      {viewMode === "catalog" && movies.length === 0 && (
        <section className="notice inline">
          <h2>Ничего не найдено</h2>
          <p>Попробуйте другое название или очистите поиск.</p>
        </section>
      )}

      {viewMode === "ratings" && (profile?.ratings.length ?? 0) === 0 && (
        <section className="notice inline">
          <h2>Оценок пока нет</h2>
          <p>Найдите фильмы в каталоге и отметьте, понравились они вам или нет.</p>
        </section>
      )}

      <section className="recommendation-panel">
        <div className="panel-head">
          <div>
            <span className="section-label">EASE + LightGBM</span>
            <h2>Персональные рекомендации</h2>
          </div>
        </div>
        {recommendations.length > 0 ? (
          <div className="recommendation-list">
            {recommendations.map((movie) => <RecommendationRow key={movie.movie_id} movie={movie} />)}
          </div>
        ) : (
          <p className="panel-empty">Оцените несколько фильмов и нажмите “Получить”.</p>
        )}
      </section>
    </main>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
