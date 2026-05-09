from core.model_manager import ModelManager

# Создаем и загружаем модель
model = ModelManager().load_mostpop()

# Проверяем статистику
print("📊 Статистика:", model.get_user_stats())

# Тестируем предсказание для реального пользователя из датасета
test_user_id = 28  # Возьми любого пользователя из train_df['userId']
try:
    recommendations = model.predict_mostpop(test_user_id, top_n=5)
    print(f"🎯 Рекомендации для пользователя {test_user_id}: {recommendations}")
except ValueError as e:
    print(f"❌ Ошибка: {e}")
    print("Попробуй другой user_id из train_df['userId']")