import pickle
import os

class ModelManager:
    def __init__(self):
        self.user_encoder = None
        self.item_encoder = None
        self.most_popular_items = None
        
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