import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class DataVisualizer:
    """
    Отвечает за построение и сохранение графиков по результатам анализа.
    Все изображения сохраняются в reports/figures/.

    Метод plot_correlation_summary генерирует сразу две картинки:
      - тепловую карту матрицы phi_k,
      - ранжирование признаков по силе связи с целевой переменной.
    """

    WEEKDAY_LABELS = ['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс']

    def __init__(self, output_dir: str = 'reports/figures'):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        sns.set_theme(style='whitegrid')

    # Метод для сохранения изображений
    def _save(self, name: str) -> None:
        path = os.path.join(self.output_dir, name)
        plt.tight_layout()
        plt.savefig(path, dpi=120, bbox_inches='tight')
        plt.close()
        print(f'Сохранено: {path}')

    
    # 1. Распределение пользователей по трём признакам (device, event_type, region) 
    #  три столбчатых диаграммы
    def plot_users_by_segments(self, profile: pd.DataFrame,
                               event_col: str = 'first_event_type',
                               device_col: str = 'first_device',
                               region_col: str = 'first_region',
                               top_n_regions: int = 10) -> None:
        """
        Распределение пользователей по устройству, типу мероприятия и региону.
        Для региона — топ-N + агрегат «Остальные».
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 7))

        # --- 1. Тип устройства
        dev = profile[device_col].value_counts()
        dev.sort_values().plot(kind='barh', ax=axes[0], color='steelblue')
        axes[0].set_title('Пользователи по признаку\n«Тип устройства»',
                          fontsize=12)
        axes[0].set_xlabel('Количество пользователей')
        axes[0].set_xlim(0, dev.max() * 1.2)
        axes[0].grid(True, alpha=0.3, axis='x')
        for i, v in enumerate(dev.sort_values().values):
            axes[0].text(v + dev.max() * 0.01, i, f'{v:,}',
                         va='center', fontsize=10)

        # --- 2. Тип мероприятия
        ev = profile[event_col].value_counts()
        ev.sort_values().plot(kind='barh', ax=axes[1], color='steelblue')
        axes[1].set_title('Пользователи по признаку\n«Тип мероприятия»',
                          fontsize=12)
        axes[1].set_xlabel('Количество пользователей')
        axes[1].set_xlim(0, ev.max() * 1.2)
        axes[1].grid(True, alpha=0.3, axis='x')
        for i, v in enumerate(ev.sort_values().values):
            axes[1].text(v + ev.max() * 0.01, i, f'{v:,}',
                         va='center', fontsize=10)

        # --- 3. Регион (топ-N + «Остальные»)
        reg = profile[region_col].value_counts()
        top = reg.head(top_n_regions).copy()
        other_sum = reg.iloc[top_n_regions:].sum()
        top['Остальные'] = other_sum
        top.sort_values().plot(kind='barh', ax=axes[2], color='steelblue')
        axes[2].set_title(f'Пользователи по признаку\n'
                          f'«Регион» (топ-{top_n_regions} + Остальные)',
                          fontsize=12)
        axes[2].set_xlabel('Количество пользователей')
        axes[2].set_xlim(0, top.max() * 1.2)
        axes[2].grid(True, alpha=0.3, axis='x')
        for i, v in enumerate(top.sort_values().values):
            axes[2].text(v + top.max() * 0.01, i, f'{v:,}',
                         va='center', fontsize=10)

        self._save('01_users_by_segments.png')

    # 2. Доля возвратов по трём признакам(device, event_type, region)
    def plot_retention_by_segments(self, profile: pd.DataFrame,
                                   event_col: str = 'first_event_type',
                                   device_col: str = 'first_device',
                                   region_col: str = 'first_region',
                                   top_n_regions: int = 10) -> None:
        """
        Доля возвратов по устройству, типу мероприятия и региону.
        Линия среднего по выборке добавляется на каждый график.
        """
        overall = profile['is_two'].mean()

        fig, axes = plt.subplots(1, 3, figsize=(18, 9))

        # Собираем статистику по каждому признаку
        def _agg(col):
            return (
                profile
                .groupby(col, observed=True)
                .agg(users=('user_id', 'count'),
                     return_rate=('is_two', 'mean'))
                .sort_values('return_rate')
            )

        datasets = [
            (device_col, _agg(device_col), 'Тип устройства'),
            (event_col, _agg(event_col), 'Тип мероприятия'),
            (region_col, _agg(region_col).nlargest(top_n_regions, 'users')
                .sort_values('return_rate'),
             f'Топ-{top_n_regions} регионов'),
        ]

        for ax, (col, stats, title) in zip(axes, datasets):
            stats['return_rate'].plot(kind='barh', ax=ax, color='steelblue')
            ax.axvline(overall, color='black', linestyle='--',
                       linewidth=1.5, label=f'Среднее: {overall:.1%}')
            ax.set_title(f'Доля возвратов по признаку\n«{title}»', fontsize=12)
            ax.set_xlabel('Доля пользователей с 2+ заказами')
            ax.set_xlim(0, stats['return_rate'].max() * 1.35)
            ax.grid(True, alpha=0.3, axis='x')
            ax.legend(loc='lower right', fontsize=9)
            for i, (rate, users) in enumerate(
                    zip(stats['return_rate'], stats['users'])):
                ax.text(rate + 0.005, i, f'{rate:.1%} (n={users:,})',
                        va='center', fontsize=9)

        self._save('02_retention_by_segments.png')

    # 3. Распределение средней выручки: 1 заказ vs 2+ заказа
    def plot_revenue_distribution_by_group(
            self, profile: pd.DataFrame,
            col: str = 'avg_revenue_per_order',
            group_col: str = 'is_two') -> None:
        """Сравнительные гистограммы плотности по группам."""
        one_order = profile[profile[group_col] == 0]
        returned = profile[profile[group_col] == 1]

        fig, ax = plt.subplots(figsize=(12, 6))

        sns.histplot(one_order[col], stat='density', binwidth=50,
                     kde=True, alpha=0.5, color='red',
                     label=f'1 заказ (n={len(one_order)})', ax=ax)
        sns.histplot(returned[col], stat='density', binwidth=50,
                     kde=True, alpha=0.5, color='blue',
                     label=f'2+ заказа (n={len(returned)})', ax=ax)

        ax.set_title('Распределение средней выручки с заказа: '
                     'одна покупка vs возврат', fontsize=13)
        ax.set_xlabel('Средняя выручка с заказа, руб.')
        ax.set_ylabel('Плотность')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        self._save('03_revenue_distribution_by_group.png')

    # 4. Retention по дню недели
    def plot_weekday_retention(self, profile: pd.DataFrame) -> None:
        """Число пользователей и доля возвратов по дню недели первого заказа."""
        profile = profile.copy()
        profile['weekday'] = profile['first_order_dt'].dt.dayofweek

        stats = (
            profile
            .groupby('weekday')
            .agg(users=('user_id', 'count'),
                 return_rate=('is_two', 'mean'))
            .reindex(range(7))
        )
        overall = profile['is_two'].mean()

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        axes[0].bar(self.WEEKDAY_LABELS, stats['users'], color='steelblue')
        axes[0].set_title('Число пользователей по дню недели\n'
                          'первого заказа', fontsize=12)
        axes[0].set_xlabel('День недели')
        axes[0].set_ylabel('Пользователей')
        axes[0].grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(stats['users']):
            axes[0].text(i, v + stats['users'].max() * 0.01,
                         f'{v:,}', ha='center', fontsize=10)

        axes[1].bar(self.WEEKDAY_LABELS, stats['return_rate'], color='steelblue')
        axes[1].axhline(overall, color='black', linestyle='--',
                        linewidth=1.5, label=f'Среднее: {overall:.1%}')
        axes[1].set_title('Доля возвратов по дню недели\n'
                          'первого заказа', fontsize=12)
        axes[1].set_xlabel('День недели')
        axes[1].set_ylabel('Доля пользователей с 2+ заказами')
        axes[1].legend(loc='lower right')
        axes[1].grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(stats['return_rate']):
            axes[1].text(i, v + 0.005, f'{v:.1%}', ha='center', fontsize=10)

        self._save('04_weekday_retention.png')

    # 5. Корреляционный анализ: тепловая карта phi_k + топ-признаки
    def plot_correlation_summary(self, phik_matrix: pd.DataFrame,
                                 target: str = 'total_orders',
                                 top_n: int = 10,
                                 save_heatmap: bool = True,
                                 save_top_bar: bool = True) -> None:
        """
        Генерирует сразу два графика по корреляционной матрице:
          1) тепловую карту всей матрицы phi_k,
          2) ранжирование признаков по силе связи с целевой переменной.

        Параметры save_heatmap / save_top_bar позволяют отключить
        любой из двух графиков, если нужен только один.
        """
        if save_heatmap:
            fig, ax = plt.subplots(figsize=(14, 10))
            sns.heatmap(
                phik_matrix,
                annot=True, fmt='.2f',
                cmap='coolwarm', vmin=0, vmax=1,
                linewidths=0.5, ax=ax,
            )
            ax.set_title('Тепловая карта корреляционной матрицы $\\phi_k$',
                         fontsize=14)
            self._save('05_phik_heatmap.png')

        if save_top_bar:
            top = (
                phik_matrix
                .loc[phik_matrix.index != target, [target]]
                .sort_values(target, ascending=True)
                .tail(top_n)
            )
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(top.index, top[target], color='steelblue')
            for i, v in enumerate(top[target]):
                ax.text(v + 0.005, i, f'{v:.2f}', va='center', fontsize=10)
            ax.set_title(f'Топ-{top_n} признаков по связи с «{target}»',
                         fontsize=13)
            ax.set_xlabel('Коэффициент $\\phi_k$')
            ax.set_xlim(0, top[target].max() * 1.15)
            ax.grid(True, alpha=0.3, axis='x')
            self._save('06_top_phik_features.png')

    # 6. Гипотеза 1: спорт vs концерты (violin)
    def plot_segment_violin(self, profile: pd.DataFrame,
                            col: str = 'first_event_type',
                            seg_a: str = 'спорт',
                            seg_b: str = 'концерты') -> None:
        """Violin-plot для сравнения retention двух сегментов."""
        df_plot = profile[profile[col].isin([seg_a, seg_b])]

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.violinplot(data=df_plot, x=col, y='is_two',
                       order=[seg_a, seg_b], ax=ax)
        ax.set_title(f'Сравнение Retention:\n«{seg_a}» vs «{seg_b}»',
                     fontsize=13)
        ax.set_xlabel('Тип мероприятия первого заказа')
        ax.set_ylabel('Доля возвратов (is_two)')
        self._save('07_segment_violin.png')

    # 7. Гипотеза 2: размер региона vs retention (scatter)

    def plot_region_scatter(self, profile: pd.DataFrame,
                            col: str = 'first_region',
                            min_users: int = 100) -> None:
        """Скаттер: число пользователей в регионе vs доля возвратов."""
        stats = (
            profile
            .groupby(col, observed=True)
            .agg(users=('user_id', 'count'),
                 return_rate=('is_two', 'mean'))
            .query('users >= @min_users')
        )
        overall = profile['is_two'].mean()

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(stats['users'], stats['return_rate'],
                   s=60, alpha=0.6, color='teal', edgecolor='black')

        # Линейный тренд
        z = np.polyfit(stats['users'], stats['return_rate'], 1)
        x_range = np.linspace(stats['users'].min(),
                              stats['users'].max(), 100)
        ax.plot(x_range, np.polyval(z, x_range),
                color='gray', linestyle=':', label='Линейный тренд')

        ax.axhline(overall, color='red', linestyle='--',
                   label=f'Среднее: {overall:.1%}')
        ax.set_xscale('log')
        ax.set_title(f'Размер региона vs доля возвратов\n'
                     f'(регионы ≥ {min_users} пользователей)', fontsize=13)
        ax.set_xlabel('Число пользователей в регионе (лог-шкала)')
        ax.set_ylabel('Доля возвратов')
        ax.legend()
        ax.grid(True, alpha=0.3)
        self._save('08_region_scatter.png')

    # 8. Распределение выручки (гистограмма + лог + boxplot)
    
    def plot_revenue_distribution(self, df: pd.DataFrame,
                                  col: str = 'revenue_rub') -> None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        sns.histplot(df[col].dropna(), bins=50, ax=axes[0])
        axes[0].set_title('Распределение выручки')
        axes[0].set_xlabel('Выручка, руб.')

        sns.histplot(np.log1p(df[col].dropna()), bins=50, ax=axes[1])
        axes[1].set_title('log(выручка + 1)')
        axes[1].set_xlabel('log(выручка + 1)')

        sns.boxplot(x=df[col].dropna(), ax=axes[2])
        axes[2].set_title('Boxplot выручки')

        self._save('09_revenue_distribution.png')

    # 9. Распределение числа билетов в заказе
    def plot_tickets_distribution(self, df: pd.DataFrame,
                                  col: str = 'tickets_count') -> None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        sns.histplot(df[col], binwidth=1, ax=axes[0])
        axes[0].set_title('Число билетов в заказе')
        sns.boxplot(x=df[col], ax=axes[1])
        axes[1].set_title('Boxplot числа билетов')
        self._save('10_tickets_distribution.png')