import pickle
import os
from scipy.sparse import load_npz
import numpy as np
import joblib

class ModelManager:
    def __init__(self):
        self.user_encoder = None
        self.item_encoder = None
        self.most_popular_items = None

        # PureSVD
        self.U = None
        self.S = None
        self.Vt = None
        self.R_ratings = None
        self.n_factors = None

        # EASE
        self.ease_weights = None
        self.ease_item_encoder = None # Отдельный энкодер для EASE, если он отличается
        self.ease_user_encoder = None
        self.ease_interactions = None
        
    def load_mostpop(self):
        """Загружает данные для MostPop модели"""
        # Путь к файлу с данными
        data_path = os.path.join('data', 'mostpop_data.pkl')
        
        # Загружаем данные
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        # Сохраняем в атрибуты класса
        self.user_encoder = data['user_encoder']
        self.item_encoder = data['item_encoder']
        self.most_popular_items = data['most_popular_items']
        
        print(f"✅ MostPop загружен. Популярных фильмов: {len(self.most_popular_items)}")
        return self
    
    def predict_mostpop(self, user_id, top_n=10):
        """
        Предсказание для пользователя.
        user_id - исходный ID пользователя (не encoded)
        """
        # 1. Проверяем, что пользователь существует
        if user_id not in self.user_encoder.classes_:
            raise ValueError(f"Пользователь {user_id} не найден в системе")
        
        # 2. Берем топ-N популярных фильмов (в encoded формате)
        recommendations_enc = self.most_popular_items[:top_n]
        
        # 3. Преобразуем обратно в исходные movieId
        recommendations = self.item_encoder.inverse_transform(recommendations_enc)
        
        return recommendations.tolist()
    
    def get_user_stats(self):
        """Информация о загруженных данных"""
        if self.user_encoder is None:
            return "Модель не загружена"
        
        return {
            'users_total': len(self.user_encoder.classes_),
            'items_total': len(self.item_encoder.classes_),
            'most_popular_items': len(self.most_popular_items)
        }
    
    def load_puresvd(self):
        data_path = os.path.join('data', 'puresvd_data.pkl')
        with open(data_path, 'rb') as f:
            puresvd_data = pickle.load(f)
        
        self.U = puresvd_data['U']
        self.S = puresvd_data['S']
        self.Vt = puresvd_data['Vt']
        self.R_ratings = puresvd_data['R_ratings']
        self.n_factors = puresvd_data['n_factors']

        print(f"✅ PureSVD загружен. Факторов: {self.n_factors}")
        print(f"   Пользователей: {self.U.shape[0]}, Фильмов: {self.Vt.shape[1]}")
        return self

    def predict_puresvd(self, user_id, top_n = 10):
        """
        Рекомендации PureSVD для пользователя.
        user_id - исходный ID пользователя (не encoded)
        """
        # 1. Проверяем, что пользователь существует
        if user_id not in self.user_encoder.classes_:
            raise ValueError(f"Пользователь {user_id} не найден в системе")
        
        # 2. Преобразуем в encoded
        user_id_enc = self.user_encoder.transform([user_id])[0]

        # 3. Получаем предсказания:
        V_scaled = self.S @ self.Vt

        # 4. Предсказания для одного пользователя
        user_factors = self.U[user_id_enc]  # вектор факторов пользователя
        scores = user_factors @ V_scaled    # предсказания для всех фильмов

        # 5. Исключаем уже просмотренные
        # Получаем индексы фильмов, которые пользователь уже оценивал
        rated_indices = self.R_ratings[user_id_enc].nonzero()[1]
        scores[rated_indices] = -np.inf  # исключаем

        # 6. Берем топ-N
        top_indices = np.argsort(scores)[::-1][:top_n]
        recommendations = self.item_encoder.inverse_transform(top_indices)
        
        return recommendations.tolist()

    def load_ease(self):
        """Загрузка всех компонентов именно для EASE"""
        data_dir = 'data'
        
        # Загружаем матрицу весов B
        self.ease_weights = np.load(os.path.join(data_dir, 'ease_weights_f16.npy'))
        
        # Загружаем матрицу взаимодействий X
        self.ease_interactions = load_npz(os.path.join(data_dir, 'ease_interaction_matrix.npz'))
        
        # Загружаем энкодеры
        self.ease_user_encoder = joblib.load(os.path.join(data_dir, 'ease_user_encoder.joblib'))
        self.ease_item_encoder = joblib.load(os.path.join(data_dir, 'item_encoder.joblib'))

        print(f"✅ EASE компоненты загружены успешно")
        return self

    def predict_ease(self, user_id, top_n=10):
        # 1. Проверка пользователя
        if user_id not in self.ease_user_encoder.classes_:
            raise ValueError(f"User {user_id} not found for EASE")

        # 2. Кодируем ID
        user_idx = self.ease_user_encoder.transform([user_id])[0]

        # 3. Извлекаем строку взаимодействий пользователя (X_u)
        user_row = self.ease_interactions[user_idx].toarray().flatten()

        # 4. Вычисляем скоры: dot(X_u, B)
        scores = user_row.astype(np.float32) @ self.ease_weights.astype(np.float32)

        # 5. Маскируем уже просмотренные айтемы
        seen_indices = user_row.nonzero()[0]
        scores[seen_indices] = -np.inf

        # 6. Сортируем и декодируем
        top_indices = np.argsort(scores)[::-1][:top_n]
        recommendations = self.ease_item_encoder.inverse_transform(top_indices)

        return recommendations.tolist()