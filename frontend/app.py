import streamlit as st
import requests
import pandas as pd

# --- НАСТРОЙКИ ---
st.set_page_config(page_title="Movie RecSys", page_icon="🍿", layout="wide")

BACKEND_URL = st.secrets.get("BACKEND_URL", "http://localhost:8000")
OMDB_API_KEY = st.secrets.get("OMDB_API_KEY", "335f100c")
LINKS_PATH = st.secrets.get("LINKS_PATH", "../fastapi_recsys/recommendation_service/data/links.csv")

# --- ФУНКЦИИ ---
@st.cache_data
def load_base_data():
    try:
        df = pd.read_csv(LINKS_PATH)
        return (df, 
                df.set_index('movieId')['imdbId'].to_dict(), 
                df['title'].unique().tolist(), 
                dict(zip(df['title'], df['movieId'])))
    except Exception as e:
        st.error(f"Ошибка загрузки данных: {e}")
        return pd.DataFrame(), {}, [], {}

links, movie_to_imdb, all_titles, movie_to_id = load_base_data()

@st.cache_data(show_spinner=False)
def get_movie_info(movie_id):
    imdb_id = movie_to_imdb.get(movie_id)
    if not imdb_id: return None
    url = f"http://www.omdbapi.com/?i=tt{str(imdb_id).zfill(7)}&apikey={OMDB_API_KEY}"
    try:
        resp = requests.get(url, timeout=5).json()
        return resp if resp.get("Response") == "True" else None
    except: return None

def render_full_card(movie_data, pos=None):
    with st.container(border=True):
        col1, col2 = st.columns([1, 2.3])
        with col1:
            poster = movie_data.get("Poster")
            st.image(poster if poster != "N/A" else "https://via.placeholder.com/300x450", use_container_width=True)
        with col2:
            if pos: st.markdown(f"**#{pos} Рекомендация**")
            title = movie_data.get('Title', 'N/A')
            st.subheader(f"{title[:30]}..." if len(title) > 33 else title)
            st.caption(f"🎭 {movie_data.get('Genre')}  •  ⏱️ {movie_data.get('Runtime')}")
            
            plot = movie_data.get('Plot', 'N/A')
            st.write(f"{plot[:160]}..." if plot != "N/A" else "Описание отсутствует")
            
            m1, m2, m3 = st.columns(3)
            m1.metric("⭐ IMDb", movie_data.get("imdbRating"))
            m2.metric("🗳️ Голоса", movie_data.get("imdbVotes"))
            m3.metric("🏆 Score", movie_data.get("Metascore"))
            
            if movie_data.get("imdbID"):
                st.link_button("Открыть на IMDb", f"https://www.imdb.com/title/{movie_data['imdbID']}", use_container_width=True)

# --- ИНТЕРФЕЙС ---
st.title("🍿 Персональный подбор кино")

with st.sidebar:
    st.header("⚙️ Настройки")
    selected_model = st.selectbox(
        "Алгоритм рекомендации:",
        options=["puresvd", "ease"],
        help="PureSVD — классика матричного разложения. EASE — современный и мощный линейный алгоритм."
    )
    top_n = st.slider("Количество рекомендаций:", 1, 15, 6)

# Используем форму, чтобы избежать перезагрузки при первом клике
with st.form("main_search_form", border=False):
    col_left, col_right = st.columns([3, 2])
    
    with col_left:
        selected_titles = st.multiselect(
            "Выберите фильмы, которые вам нравятся:", 
            options=all_titles, 
            placeholder="Введите названия..."
        )
    
    with col_right:
        st.markdown("<p style='margin-bottom: 28px;'></p>", unsafe_allow_html=True)
        submit_button = st.form_submit_button("🚀 Сгенерировать рекомендации", type="primary", use_container_width=True)

current_ids = [movie_to_id[t] for t in selected_titles]

# --- ЛОГИКА ОТОБРАЖЕНИЯ ---
if not current_ids:
    st.subheader("🔥 Популярно сейчас")
    pop_ids = []
    try:
        resp = requests.post(f"{BACKEND_URL}/forward", json={"user_id": 28, "model": "mostpop", "top_n": 20}, timeout=2)
        if resp.status_code == 200:
            pop_ids = resp.json().get("recommendations", [])
    except: pass
    
    if not pop_ids and not links.empty:
        pop_ids = links['movieId'].head(12).tolist()

    if pop_ids:
        with st.container(height=580, border=False):
            cols_count = 4
            for i in range(0, len(pop_ids), cols_count):
                cols = st.columns(cols_count)
                chunk = pop_ids[i:i+cols_count]
                for j, m_id in enumerate(chunk):
                    with cols[j]:
                        info = get_movie_info(m_id)
                        if info:
                            with st.container(border=True):
                                st.image(info["Poster"] if info["Poster"] != "N/A" else "https://via.placeholder.com/150", use_container_width=True)
                                t = info['Title']
                                st.markdown(f"**{t[:18]}...**" if len(t) > 20 else f"**{t}**")
                                st.caption(f"⭐ {info['imdbRating']} | {info['Year']}")
                                st.link_button("IMDb", f"https://www.imdb.com/title/{info['imdbID']}", use_container_width=True)
else:
    if submit_button:
        st.divider()
        with st.spinner("Смешиваем ингредиенты для ваших рекомендаций..."):
            try:
                payload = {"selected_movie_ids": current_ids, "model": selected_model, "top_n": top_n}
                response = requests.post(f"{BACKEND_URL}/predict_raw", json=payload, timeout=20)
                
                if response.status_code == 200:
                    recs = response.json().get("recommendations", [])
                    if recs:
                        st.subheader("🎯 Специально для вас:")
                        cl, cr = st.columns(2)
                        for idx, m_id in enumerate(recs):
                            movie_data = get_movie_info(m_id)
                            if movie_data:
                                target = cl if idx % 2 == 0 else cr
                                with target: render_full_card(movie_data, pos=idx+1)
                    else:
                        st.warning("Ничего не нашли. Попробуйте выбрать другие фильмы.")
            except Exception as e:
                st.error(f"Ошибка бэкенда: {e}")
    else:
        st.info("Нажмите кнопку выше, чтобы запустить поиск рекомендаций.")