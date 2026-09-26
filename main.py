from src.processing import DataProcessor


def main():
    processor = DataProcessor(verbose=True)

    processor.load_data()  # загружаем данные из БД
    processor.preprocess_data() # выполняем предобработку
    processor.create_user_profiles() # создание профилей пользователей

    # Анализ по шагам
    processor.analyze_return_rate_by_segments() # доли возвратов по сегментам
    processor.test_hypothesis_event_type() # проверка гипотезы 1
    processor.test_hypothesis_region() # проверка гипотезы 2
    processor.analyze_weekday() # анализ влияния дня недели первого заказа на возврат
    processor.analyze_revenue_by_group() # сравнение чека у пользователей с 1 и более заказами
    processor.run_correlation_analysis() # корреляционный анализ с числом заказов 

    # Итоговый отчет
    processor.generate_report() # генерирует итоговый отчет с выводами

if __name__ == "__main__":
    main()