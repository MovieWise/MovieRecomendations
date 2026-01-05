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

# --- СТИЛИЗАЦИЯ ---
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

# --- ЗАГОЛОВОК ---
st.title("🍿 Персональный подбор кино")
st.markdown("Выберите модель и укажите фильмы, которые вам нравятся, чтобы получить рекомендации.")

# --- БОКОВАЯ ПАНЕЛЬ (НАСТРОЙКИ) ---
with st.sidebar:
    st.header("⚙️ Настройки")
    
    # 1. Выбор модели
    selected_model = st.selectbox(
        "Алгоритм рекомендации:",
        options=["puresvd", "ease"],
        help="PureSVD — классика матричного разложения, EASE — современный линейный подход."
    )
    
    # 2. Количество рекомендаций
    top_n = st.slider("Количество фильмов:", 1, 20, 10)
    
# --- ОСНОВНОЙ ИНТЕРФЕЙС ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 Ваши предпочтения")
    
    # В реальном приложении здесь лучше загрузить список фильмов из БД/CSV
    # Для теста введем ID вручную или через удобный ввод
    movie_input = st.text_input(
        "Введите ID фильмов через запятую:",
        placeholder="Например: 1, 10, 550",
        help="Введите ID из вашей базы данных (MovieLens или другой)"
    )
    
    # Если хочешь сделать красиво, можно добавить multiselect, 
    # если предварительно подгрузить список названий
    # movies_list = [1, 2, 3, 10, 20...] 
    # selected_ids = st.multiselect("Или выберите из списка:", movies_list)

# --- ЛОГИКА ЗАПРОСА ---
if st.button("Сгенерировать рекомендации 🚀"):
    if not movie_input:
        st.warning("⚠️ Пожалуйста, введите хотя бы один ID фильма.")
    else:
        try:
            # Превращаем строку в список чисел
            user_movie_ids = [int(x.strip()) for x in movie_input.split(",")]
            
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
                    
                    # Переходим во вторую колонку для вывода
                    with col2:
                        st.subheader("🎯 Мы рекомендуем:")
                        
                        # Создаем таблицу для наглядности
                        recs = result.get("recommendations", [])
                        if recs:
                            df = pd.DataFrame({
                                "Место": range(1, len(recs) + 1),
                                "ID фильма": recs
                            })
                            st.table(df)
                            
                            st.success(f"⏱ Время обработки: {result.get('processing_time', 0):.4f} сек.")
                        else:
                            st.write("Ничего не найдено :(")
                else:
                    st.error(f"Ошибка сервера: {response.status_code}")
                    st.json(response.json())
                    
        except ValueError:
            st.error("❌ Ошибка: Вводите только числа через запятую.")
        except Exception as e:
            st.error(f"❌ Не удалось подключиться к API: {e}")
