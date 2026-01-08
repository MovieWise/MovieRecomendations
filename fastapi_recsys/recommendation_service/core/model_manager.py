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
        self.ease_item_encoder = None 
        self.ease_user_encoder = None
        self.ease_interactions = None
        
    def load_mostpop(self):
        """Load data for mostpop model"""
        # Path to data file
        data_path = os.path.join('data', 'mostpop_data.pkl')
        
        # Load data
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        # Save to class attributes
        self.user_encoder = data['user_encoder']
        self.item_encoder = data['item_encoder']
        self.most_popular_items = data['most_popular_items']
        
        print(f" MostPop загружен. Популярных фильмов: {len(self.most_popular_items)}")
        return self
    
    def predict_mostpop(self, user_id, top_n=10):
        """
        Mostpop user recs.
        user_id - origin ID of user
        """
        # Check that the user exists
        if user_id not in self.user_encoder.classes_:
            raise ValueError(f"Bad Request: Пользователь {user_id} не найден в системе")
        
        # Take top n popular films
        recommendations_enc = self.most_popular_items[:top_n]
        
        # Convert back to the original movieId
        recommendations = self.item_encoder.inverse_transform(recommendations_enc)
        
        return recommendations.tolist()
    
    def get_user_stats(self):
        """Info about downloaded data"""
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

        print(f" PureSVD загружен. Факторов: {self.n_factors}")
        print(f"   Пользователей: {self.U.shape[0]}, Фильмов: {self.Vt.shape[1]}")
        return self

    def predict_puresvd(self, user_id, top_n = 10):
        """
        PureSVD user recs.
        user_id - origin ID of user
        """
        # Check that the user exists
        if user_id not in self.user_encoder.classes_:
            raise ValueError(f"Bad Request: Пользователь {user_id} не найден в системе")
        
        # Transform to encoded
        user_id_enc = self.user_encoder.transform([user_id])[0]

        # Get prediction
        V_scaled = self.S @ self.Vt

        # Preds for one user
        user_factors = self.U[user_id_enc]  # user features vector
        scores = user_factors @ V_scaled    # prediction for all movies

        # Exclude seen
        # Get ids of seen movies
        rated_indices = self.R_ratings[user_id_enc].nonzero()[1]
        scores[rated_indices] = -np.inf 

        # Take top n
        top_indices = np.argsort(scores)[::-1][:top_n]
        recommendations = self.item_encoder.inverse_transform(top_indices)
        
        return recommendations.tolist()

    def load_ease(self):
        """Load data for EASE"""
        data_dir = 'data'
        
        # Load weights matrix B
        self.ease_weights = np.load(os.path.join(data_dir, 'ease_weights_f16.npy'))
        
        # Load matrix of interactions X
        self.ease_interactions = load_npz(os.path.join(data_dir, 'ease_interaction_matrix.npz'))
        
        # Load encoders
        self.ease_user_encoder = joblib.load(os.path.join(data_dir, 'ease_user_encoder.joblib'))
        self.ease_item_encoder = joblib.load(os.path.join(data_dir, 'item_encoder.joblib'))

        print(f" EASE компоненты загружены успешно")
        return self

    def predict_ease(self, user_id, top_n=10):
        # Check the user
        if user_id not in self.ease_user_encoder.classes_:
            raise ValueError(f"Bad Request: Пользователь {user_id} не найден в системе")

        # Encode id
        user_idx = self.ease_user_encoder.transform([user_id])[0]

        # Extract the user interaction string (X_u)
        user_row = self.ease_interactions[user_idx].toarray().flatten()

        # Calculate scores: dot(X_u, B)
        scores = user_row.astype(np.float32) @ self.ease_weights.astype(np.float32)

        # Exclude seen items
        seen_indices = user_row.nonzero()[0]
        scores[seen_indices] = -np.inf

        # Sort and decode
        top_indices = np.argsort(scores)[::-1][:top_n]
        recommendations = self.ease_item_encoder.inverse_transform(top_indices)

        return recommendations.tolist()

    def predict_for_new_user(self, item_ids, model_name="puresvd", top_n=10):
        # 1. Превращаем внешние ID фильмов во внутренние индексы
        # Используем твой существующий item_encoder
        if model_name == "puresvd":
            item_indices = self.item_encoder.transform(item_ids)
        elif model_name == "ease":
            item_indices = self.ease_item_encoder.transform(item_ids)
        # 2. Создаем вектор предпочтений (zeros)
        # Размер вектора = количеству всех фильмов в системе
        if model_name == "puresvd":
            num_items = len(self.item_encoder.classes_)
        else:
            num_items = 10000
        user_vector = np.zeros(num_items)
        user_vector[item_indices] = 1  # Ставим 1 там, где пользователю понравилось
        
        if model_name == "puresvd":
            # Логика PureSVD для нового вектора:
            # scores = (user_vector @ Vt.T) @ Vt
            # Но у тебя есть Vt и S, можно сделать проще через проекцию:
            V = self.Vt.T
            scores = user_vector @ V @ V.T
            
        elif model_name == "ease":
            # Логика EASE: scores = X_u @ B
            scores = user_vector.astype(np.float32) @ self.ease_weights.astype(np.float32)
        
        # 3. Исключаем те фильмы, которые пользователь уже выбрал
        scores[item_indices] = -np.inf
        
        # 4. Берем топ и декодируем обратно в ID фильмов
        top_indices = np.argsort(scores)[::-1][:top_n]
        recommendations = self.item_encoder.inverse_transform(top_indices)
        
        return recommendations.tolist()
