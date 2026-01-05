import streamlit as st
import requests
import pandas as pd

# Настройки страницы
st.set_page_config(
    page_title="Movie Recommender System",
    page_icon="🍿",
    layout="wide"
)

# Константы
BACKEND_URL = "http://localhost:8000"

# Movie imbd poster key
# Here is your key: 335f100c
# Please append it to all of your API requests,
# OMDb API: http://www.omdbapi.com/?i=tt3896198&apikey=335f100c

links = pd.read_csv("../fastapi_recsys/recommendation_service/data/links.csv")
movie_to_imdb = links.set_index('movieId')['imdbId'].to_dict()

# Стиль страницы
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# Заголовок страницы
st.title("🍿 Персональный подбор кино")
st.markdown("Выберите модель и укажите фильмы, которые вам нравятся, чтобы получить рекомендации.")

# Панель настроек
with st.sidebar:
    st.header("⚙️ Настройки")
    
    # 1. Выбор модели
    selected_model = st.selectbox(
        "Алгоритм рекомендации:",
        options=["puresvd", "ease"],
        help="PureSVD — классика матричного разложения, EASE — современный линейный подход."
    )
    
    # 2. Количество рекомендаций
    top_n = st.slider("Количество фильмов:", 1, 10, 5)
    
# Основной интерфейс

# Инициализируем session_state, если его еще нет
if "final_ids" not in st.session_state:
    st.session_state.final_ids = set()

st.subheader("📝 Ваши предпочтения")

@st.cache_data
def get_data():
    # Замените путь на ваш актуальный
    links = pd.read_csv("../fastapi_recsys/recommendation_service/data/links.csv")
    return links, links['title'].unique().tolist(), dict(zip(links['title'], links['movieId']))

links, all_titles, movie_to_id = get_data()

with st.form("movie_selection_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        movie_input = st.text_input("Введите ID (через запятую):", placeholder="1, 10, 500")
    
    with col2:
        selected_titles = st.multiselect("Выберите названия:", options=all_titles, max_selections=20)
    
    submit_button = st.form_submit_button("Подтвердить выбор")

# Если нажата кнопка в форме — сохраняем результат в session_state
if submit_button:
    current_ids = set()
    
    if movie_input:
        ids = [int(i.strip()) for i in movie_input.split(",") if i.strip().isdigit()]
        current_ids.update(ids)
    
    if selected_titles:
        current_ids.update([movie_to_id[t] for t in selected_titles])
    
    # Сохраняем в состояние сессии
    st.session_state.final_ids = current_ids

# Показываем таблицу, если в session_state есть ID
if st.session_state.final_ids:
    st.write(f"Выбрано ID: {len(st.session_state.final_ids)}")
    selected_df = links[links['movieId'].isin(st.session_state.final_ids)][['movieId', 'title']]
    st.table(selected_df)

# Запрос к бэку
if st.button("Сгенерировать рекомендации"):
    if not st.session_state.final_ids:
        st.warning("⚠️ Пожалуйста, введите хотя бы один ID фильма.")
    else:
        try:
            # Превращаем строку в список чисел
            user_movie_ids = list(st.session_state.final_ids)
            
            # Подготовка данных для отправки
            payload = {
                "selected_movie_ids": user_movie_ids,
                "model": selected_model,
                "top_n": top_n
            }
            
            with st.spinner(f"Модель {selected_model} анализирует ваши вкусы..."):
                response = requests.post(f"{BACKEND_URL}/predict_raw", json=payload)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    st.subheader("Мы рекомендуем:")
                    
                    recs = result.get("recommendations", [])
                    if recs:
                        recs = {
                            "Место": range(1, len(recs) + 1),
                            "ID фильма": recs,
                            "IMdBid": [movie_to_imdb.get(r) for r in recs] # Безопасное получение id
                        }

                        movie_details = []

                        for imdb_id in recs["IMdBid"]:
                            # Формируем ссылку. Важно: zfill(7) добавит нули, если ID короткий (напр. tt0111161)
                            url = f"http://www.omdbapi.com/?i=tt{str(imdb_id).zfill(7)}&apikey=335f100c"
                            
                            try:
                                response = requests.get(url)
                                response.raise_for_status() # Проверка на ошибки (404, 500 и т.д.)
                                
                                data = response.json()
                                
                                if data.get("Response") == "True":
                                    movie_details.append(data)
                                else:
                                    print(f"Ошибка OMDB для ID {imdb_id}: {data.get('Error')}")
                                    
                            except Exception as e:
                                print(f"Ошибка сети: {e}")

                    else:
                        st.write("Ничего не найдено :(")
                    
                    left, center, right = st.columns([0.5, 2, 0.5])

                    with center:
                        for movie in movie_details:
                            # Используем контейнер с рамкой
                            with st.container(border=True):
                                # Разделяем на колонку для постера и колонку для текста
                                col1, col2 = st.columns([1, 2.5])
                                
                                with col1:
                                    # Проверяем наличие постера (OMDb иногда возвращает "N/A")
                                    poster_url = movie["Poster"] if movie["Poster"] != "N/A" else "https://via.placeholder.com/300x450?text=No+Poster"
                                    st.image(poster_url, use_container_width=True)
                                    
                                with col2:
                                    st.subheader(f"{movie['Title']} ({movie['Year']})")
                                    
                                    # Жанры в виде "тегов"
                                    st.caption(f"🎭 {movie['Genre']}  •  ⏱️ {movie['Runtime']}")
                                    
                                    # Краткое описание
                                    if movie.get("Plot") != "N/A":
                                        st.write(f"_{movie['Plot']}_")
                                    
                                    # Метрики
                                    m1, m2, m3 = st.columns(3)
                                    m1.metric("Рейтинг", f"{movie['imdbRating']}")
                                    m2.metric("Голоса", movie['imdbVotes'])
                                    m3.metric("Metascore", movie.get('Metascore', 'N/A'))
                                    
                                    # Кнопка-ссылка на IMDb
                                    imdb_id = movie.get('imdbID')
                                    if imdb_id:
                                        st.link_button("Открыть на IMDb", f"https://www.imdb.com/title/{imdb_id}")
        except ValueError:
            st.error("❌ Ошибка: Вводите только числа через запятую.")
        except Exception as e:
            st.error(f"❌ Не удалось подключиться к API: {e}")
