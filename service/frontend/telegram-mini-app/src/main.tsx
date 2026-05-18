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
  onOpen,
  busy
}: {
  movie: MovieInfo;
  onReact: (movieId: number, reaction: Reaction) => void;
  onOpen: (movie: MovieInfo) => void;
  busy: boolean;
}) {
  return (
    <article className="movie-tile">
      <button className="poster poster-button" onClick={() => onOpen(movie)} type="button">
        {movie.poster ? <img src={movie.poster} alt="" /> : <div className="poster-empty">{movie.title?.slice(0, 1) ?? "M"}</div>}
      </button>
      <div className="tile-body">
        {movie.rating && movie.rating !== "N/A" && (
          <div className="imdb-rating" aria-label={`IMDb ${movie.rating}`}>
            <span className="rating-star filled">★</span>
            <strong>{movie.rating}</strong>
          </div>
        )}
        <button className="title-button" onClick={() => onOpen(movie)} type="button">
          <h3>{movie.title ?? "Без названия"}</h3>
        </button>
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

function MovieDetailModal({ movie, onClose }: { movie: MovieInfo; onClose: () => void }) {
  return (
    <div className="modal-backdrop" role="dialog" aria-modal="true" aria-label={movie.title ?? "Фильм"}>
      <section className="movie-modal">
        <button className="modal-close" onClick={onClose} aria-label="Закрыть" type="button">×</button>
        <div className="modal-poster">
          {movie.poster ? <img src={movie.poster} alt="" /> : <div className="poster-empty">{movie.title?.slice(0, 1) ?? "M"}</div>}
        </div>
        <div className="modal-body">
          {movie.rating && movie.rating !== "N/A" && (
            <div className="imdb-rating modal-rating" aria-label={`IMDb ${movie.rating}`}>
              <span className="rating-star filled">★</span>
              <strong>{movie.rating}</strong>
            </div>
          )}
          <h2>{movie.title ?? "Без названия"}</h2>
          <p className="modal-meta">{[movie.year, movie.runtime, movie.genre].filter(Boolean).join(" · ")}</p>
          {movie.plot && <p className="modal-plot">{movie.plot}</p>}
          <dl className="modal-facts">
            {movie.director && (
              <>
                <dt>Режиссёр</dt>
                <dd>{movie.director}</dd>
              </>
            )}
            {movie.actors && (
              <>
                <dt>В ролях</dt>
                <dd>{movie.actors}</dd>
              </>
            )}
          </dl>
          {movie.imdb_url && <a className="imdb-link" href={movie.imdb_url}>Открыть IMDb</a>}
        </div>
      </section>
    </div>
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
  const [selectedMovie, setSelectedMovie] = useState<MovieInfo | null>(null);
  const [query, setQuery] = useState("");
  const [viewMode, setViewMode] = useState<ViewMode>("catalog");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const ratedCount = useMemo(() => (profile ? profile.liked_count + profile.disliked_count : 0), [profile]);
  const queryText = query.trim();
  const feedTitle = queryText ? "Результаты поиска" : ratedCount > 0 ? "Рекомендации для вас" : "Популярные фильмы";
  const feedSubtitle = queryText
    ? "Выберите фильмы, которые уже смотрели."
    : ratedCount > 0
      ? "Лента пересчитывается по вашим оценкам."
      : "Холодный старт: начните с популярных фильмов.";

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
        const nextProfile = await loadProfile(auth.access_token);
        await loadMainFeed(auth.access_token, nextProfile, "");
        setProfile(nextProfile);
        setState("catalog");
      })
      .catch((err) => {
        setError(normalizeError(err));
        setState("error");
      });
  }, []);

  async function loadMainFeed(accessToken: string, nextProfile: ProfileResponse | null, currentQuery = query) {
    const normalizedQuery = currentQuery.trim();
    if (normalizedQuery) {
      const result = await searchMovies(accessToken, normalizedQuery, 24);
      setMovies(result.movies);
      return;
    }
    const nextRatedCount = nextProfile ? nextProfile.liked_count + nextProfile.disliked_count : 0;
    if (nextRatedCount > 0) {
      const result = await generateRecommendations(accessToken, 24);
      setMovies(result.recommendations);
      return;
    }
    const result = await loadFeed(accessToken, 24);
    setMovies(result.movies);
  }

  useEffect(() => {
    if (!token || state !== "catalog" || viewMode !== "catalog") return;
    const timeout = window.setTimeout(async () => {
      setBusy(true);
      try {
        await loadMainFeed(token, profile, query);
      } catch (err) {
        setError(normalizeError(err));
      } finally {
        setBusy(false);
      }
    }, 280);
    return () => window.clearTimeout(timeout);
  }, [query, token, state, viewMode]);

  async function react(movieId: number, reaction: Reaction) {
    if (!token) return;
    setBusy(true);
    try {
      await saveReaction(token, movieId, reaction);
      const nextProfile = await loadProfile(token);
      setProfile(nextProfile);
      await loadMainFeed(token, nextProfile, query);
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
      const nextProfile = await loadProfile(token);
      setProfile(nextProfile);
      await loadMainFeed(token, nextProfile, query);
    } catch (err) {
      setError(normalizeError(err));
      setState("error");
    } finally {
      setBusy(false);
    }
  }

  async function refreshFeed() {
    if (!token || !profile) return;
    setBusy(true);
    try {
      await loadMainFeed(token, profile, query);
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
          <h1>{feedTitle}</h1>
        </div>
        <button className="button primary top-cta" onClick={refreshFeed} disabled={busy}>
          Обновить
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

      <section className="feed-summary">
        <span>{ratedCount > 0 && !queryText ? "EASE + LightGBM" : "Каталог"}</span>
        <p>{busy ? "Обновляем ленту..." : feedSubtitle}</p>
      </section>

      {viewMode === "catalog" ? (
        <section className="catalog-grid" aria-busy={busy}>
          {movies.map((movie) => (
            <MovieTile key={movie.movie_id} movie={movie} onReact={react} onOpen={setSelectedMovie} busy={busy} />
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
      {selectedMovie && <MovieDetailModal movie={selectedMovie} onClose={() => setSelectedMovie(null)} />}
    </main>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
