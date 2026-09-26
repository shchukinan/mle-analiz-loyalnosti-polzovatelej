from src.processing import DataProcessor
from src.visualization import DataVisualizer


def main(save_figures: bool = True):
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

    # Итоговый отчет в консоль и в файл
    processor.generate_report() # генерирует итоговый отчет с выводами

    
    # Визуализации
    if save_figures:
        viz = DataVisualizer()

        # 1. Распределение пользователей по device / event_type / region
        viz.plot_users_by_segments(processor.user_profile)

        # 2. Доля возвратов по device / event_type / region
        viz.plot_retention_by_segments(processor.user_profile)

        # 3. Средняя выручка: 1 заказ vs 2+ заказа (KDE)
        viz.plot_revenue_distribution_by_group(processor.user_profile)

        # 4. Retention по дню недели
        viz.plot_weekday_retention(processor.user_profile)

        # 5. Корреляция: тепловая карта phi_k + топ-признаки
        viz.plot_correlation_summary(
            processor.phik_overview,
            target='total_orders',
            top_n=10,
        )

        # 6. Гипотеза 1: спорт vs концерты
        viz.plot_segment_comparison(
            processor.user_profile,
            seg_a='спорт',
            seg_b='концерты',
        )

        # 7. Гипотеза 2: размер региона vs retention
        viz.plot_region_scatter_bubble(processor.user_profile)

        # 8. Распределение выручки (гистограмма + лог + boxplot)
        viz.plot_revenue_distribution(processor.df)
        
        #9. Распределение числа билетов в заказе
        viz.plot_tickets_distribution(processor.df)

        print('\nВсе визуализации сохранены в reports/figures/')


if __name__ == "__main__":
    main(save_figures=True)