import os
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

from scipy.stats import chi2_contingency, spearmanr
from statsmodels.stats.proportion import proportions_ztest
import phik  

from .reporters import DataFrameReporter


class DataProcessor:
    """
    Отвечает за полный цикл работы с данными:
    загрузку из БД, предобработку, построение профилей пользователей,
    сегментный анализ, проверку гипотез, корреляционный анализ
    и формирование итогового отчета.
    """

    # Дни недели для читаемых подписей
    WEEKDAY_MAP = {0: 'Пн', 1: 'Вт', 2: 'Ср', 3: 'Чт', 4: 'Пт', 5: 'Сб', 6: 'Вс'}

    def __init__(self, verbose: bool = True):
        load_dotenv()
        self.verbose = verbose

        self.db_config = {
            'user': os.getenv('DB_USER'),
            'pwd': os.getenv('DB_PASSWORD'),
            'host': os.getenv('DB_HOST'),
            'port': os.getenv('DB_PORT'),
            'db': os.getenv('DB_NAME'),
        }
        self.engine = self._create_db_connection()

        self.df = None                 # сырые данные + предобработанные
        self.user_profile = None       # профили пользователей
        self.segment_stats = {}        # доли возвратов по сегментам

        self.reporter = DataFrameReporter(
            float_format='0.04f',
            percent_format='0.02%',
            include_all=True,
        )

        # Накопительная статистика для итогового отчета
        self.stats = {
            'initial_rows': None,
            'rows_after_duplicates': None,
            'rows_after_outliers': None,
            'initial_users': None,
            'users_after_outlier_filter': None,
            'overall_return_rate': None,
            'hypothesis_event_type': None,
            'hypothesis_region': None,
            'weekday_chi2': None,
            'revenue_by_group': None,
        }

    
    # Подключение к БД
    def _create_db_connection(self):
        """Формирует строку подключения и создает engine SQLAlchemy."""
        connection_string = 'postgresql://{}:{}@{}:{}/{}'.format(
            self.db_config['user'],
            self.db_config['pwd'],
            self.db_config['host'],
            self.db_config['port'],
            self.db_config['db'],
        )
        return create_engine(connection_string)

    
    # Загрузка данных
    def load_data(self) -> pd.DataFrame:
        """Загружает данные из SQL базы и сохраняет в self.df."""
        if self.verbose:
            print("Загрузка данных из БД...")

        query = '''
        WITH set_config_precode AS (
          SELECT set_config('synchronize_seqscans', 'off', true)
        )
        SELECT
            p.user_id,
            p.device_type_canonical,
            p.order_id,
            p.created_dt_msk AS order_dt,
            p.created_ts_msk AS order_ts,
            p.currency_code,
            p.revenue,
            p.tickets_count,
            EXTRACT(DAY FROM (
                p.created_dt_msk - LAG(p.created_dt_msk) OVER (
                    PARTITION BY p.user_id
                    ORDER BY p.created_dt_msk
                )
            ))::int AS days_since_prev,
            p.event_id,
            e.event_name_code AS event_name,
            e.event_type_main,
            p.service_name,
            r.region_name,
            c.city_name
        FROM afisha.purchases AS p
        JOIN afisha.events AS e
            ON p.event_id = e.event_id
        JOIN afisha.city AS c
            ON e.city_id = c.city_id
        JOIN afisha.regions AS r
            ON c.region_id = r.region_id
        WHERE p.device_type_canonical IN ('mobile', 'desktop')
          AND e.event_type_main <> 'фильм'
        ORDER BY p.user_id;
        '''
        self.df = pd.read_sql_query(query, con=self.engine)
        self.stats['initial_rows'] = len(self.df)

        if self.verbose:
            self.reporter.show_report(self.df, 'ОТЧЕТ: Сырые данные из БД')

        return self.df

    
    # Предобработка
    def preprocess_data(self) -> pd.DataFrame:
        """
        Полная предобработка:
        - конвертация валют в рубли;
        - оптимизация типов;
        - удаление неявных дубликатов заказов;
        - фильтрация выбросов по выручке.
        """
        if self.df is None:
            raise ValueError("Данные не загружены. Сначала вызовите load_data().")

        if self.verbose:
            print("\nНачата предобработка данных...")

        # 1. Загрузка данных о курсе тенге и конвертация валют
        tenge_url = 'https://code.s3.yandex.net/datasets/final_tickets_tenge_df.csv'
        tenge_df = pd.read_csv(tenge_url)
        tenge_df['data'] = pd.to_datetime(tenge_df['data'])

        self.df = self.df.merge(
            tenge_df, left_on='order_dt', right_on='data', how='left'
        ).drop(columns=['data', 'cdx'])

        self.df['revenue_rub'] = np.where(
            self.df['currency_code'] == 'kzt',
            self.df['revenue'] * self.df['curs'] / self.df['nominal'],
            self.df['revenue'],
        )

        # 2. Оптимизация типов
        self.df['days_since_prev'] = self.df['days_since_prev'].astype('Int64')
        for col in self.df.select_dtypes(include='object').columns:
            self.df[col] = self.df[col].astype('category')
        for col in ['order_id', 'tickets_count', 'event_id', 'nominal']:
            self.df[col] = pd.to_numeric(self.df[col], downcast='integer')
        for col in ['revenue', 'curs', 'revenue_rub']:
            self.df[col] = pd.to_numeric(self.df[col], downcast='float')

        # 3. Удаление неявных дубликатов заказов
        #    (одни и те же заказы, но с разными order_id)
        subset_for_duplicates = [
            'user_id', 'device_type_canonical', 'order_ts', 'currency_code',
            'revenue', 'tickets_count', 'event_id', 'event_name',
            'event_type_main', 'service_name', 'region_name', 'city_name',
        ]
        self.df.drop_duplicates(subset=subset_for_duplicates, inplace=True)
        self.stats['rows_after_duplicates'] = len(self.df)

        if self.verbose:
            print(f"Удалено неявных дубликатов: "
                  f"{self.stats['initial_rows'] - self.stats['rows_after_duplicates']}")

        # 4. Фильтрация выбросов по выручке
        q99_revenue = self.df['revenue_rub'].quantile(0.99)
        self.df = self.df[
            (self.df['revenue_rub'] <= q99_revenue) & (self.df['revenue_rub'] >= 0)
        ]
        self.stats['rows_after_outliers'] = len(self.df)

        if self.verbose:
            removed = self.stats['rows_after_duplicates'] - self.stats['rows_after_outliers']
            print(f"Удалено выбросов по revenue_rub: {removed} "
                  f"({removed / self.stats['rows_after_duplicates']:.2%})")
            self.reporter.show_report(self.df, 'ОТЧЕТ: Данные после предобработки')

        return self.df


    # Шаг 3. Профили пользователей
    def create_user_profiles(self) -> pd.DataFrame:
        """
        Строит агрегированный профиль по каждому пользователю.
        Добавляет is_two и отбрасывает аномально активных пользователей
        (выше 99-го перцентиля по orders_count).
        """
        if self.df is None:
            raise ValueError("Данные не загружены/не обработаны.")

        if self.verbose:
            print("\nСоздание профилей пользователей...")

        df_sorted = self.df.sort_values('order_ts').reset_index(drop=True)

        self.user_profile = (
            df_sorted
            .groupby('user_id', as_index=False, observed=True)
            .agg(
                first_order_dt=('order_dt', 'min'),
                last_order_dt=('order_dt', 'max'),
                first_device=('device_type_canonical', 'first'),
                first_region=('region_name', 'first'),
                first_event_type=('event_type_main', 'first'),
                orders_count=('order_id', 'count'),
                avg_revenue_per_order=('revenue_rub', 'mean'),
                avg_days_between_orders=('days_since_prev', 'mean'),
            )
        )
        self.user_profile['is_two'] = (
            self.user_profile['orders_count'] >= 2
        ).astype('int8')

        self.stats['initial_users'] = len(self.user_profile)

        # Фильтрация аномально активных пользователей
        q99_orders = self.user_profile['orders_count'].quantile(0.99)
        self.user_profile = self.user_profile.query('orders_count <= @q99_orders')

        self.stats['users_after_outlier_filter'] = len(self.user_profile)
        self.stats['overall_return_rate'] = self.user_profile['is_two'].mean()

        if self.verbose:
            print(f"Всего пользователей: {self.stats['initial_users']}")
            print(f"После фильтрации по 99-му перцентилю: "
                  f"{self.stats['users_after_outlier_filter']}")
            self.reporter.show_report(self.user_profile, 'ОТЧЕТ: Профили пользователей')

        return self.user_profile

    
    # Считаем доли возвратов по сегментам
    def analyze_return_rate_by_segments(self) -> dict:
        """Считает долю возвратов по device / event_type / region."""
        if self.user_profile is None:
            raise ValueError("Сначала создайте профили пользователей.")

        if self.verbose:
            print("\nРасчет доли возвратов по сегментам...")

        for col in ['first_device', 'first_event_type', 'first_region']:
            stats = (
                self.user_profile
                .groupby(col, observed=True)
                .agg(
                    users=('user_id', 'count'),
                    return_rate=('is_two', 'mean'),
                )
                .sort_values('users', ascending=False)
            )
            self.segment_stats[col] = stats

        return self.segment_stats


    # Проверка гипотез
    def test_hypothesis_event_type(self) -> dict:
        """Гипотеза 1: Тип мероприятия влияет на вероятность возврата на Яндекс Афишу:
        пользователи, которые совершили первый заказ на спортивные мероприятия, 
        совершают повторный заказ чаще, чем пользователи, 
        оформившие свой первый заказ на концерты"""
        sport = self.user_profile[self.user_profile['first_event_type'] == 'спорт']
        concert = self.user_profile[self.user_profile['first_event_type'] == 'концерты']

        _, pval = proportions_ztest(
            [sport['is_two'].sum(), concert['is_two'].sum()],
            [len(sport), len(concert)],
        )

        result = {
            'sport_rate': sport['is_two'].mean(),
            'concert_rate': concert['is_two'].mean(),
            'p_value': pval,
            'rejected': pval < 0.05,
        }
        self.stats['hypothesis_event_type'] = result

        if self.verbose:
            print("\nГипотеза 1 (тип мероприятия влияет на вероятность возврата):")
            print(f"  Доля возвратов — спорт:    {result['sport_rate']:.2%}")
            print(f"  Доля возвратов — концерты: {result['concert_rate']:.2%}")
            print(f"  p-value: {result['p_value']:.4f} → "
                  f"{'отвергаем H₀: различие долей статистически значимо' if result['rejected'] else 'не отвергаем H₀: значимых различий не обнаружено'}")

        return result

    def test_hypothesis_region(self) -> dict:
        """Гипотеза 2: В регионах, где больше всего пользователей посещают мероприятия, 
        выше доля повторных заказов, чем в менее активных регионах."""
        region_stats = (
            self.user_profile
            .groupby('first_region', observed=True)
            .agg(
                users=('user_id', 'count'),
                return_rate=('is_two', 'mean'),
            )
            .query('users >= 100')
        )
        corr, pval = spearmanr(region_stats['users'], region_stats['return_rate'])

        result = {
            'spearman_corr': corr,
            'p_value': pval,
            'rejected': pval < 0.05,
        }
        self.stats['hypothesis_region'] = result

        if self.verbose:
            print("\nГипотеза 2 (связь числа пользователей региона и доли возвратов):")
            print(f"  Корреляция Спирмена: {result['spearman_corr']:.3f}")
            print(f"  p-value: {result['p_value']:.4f} → "
                  f"{'связь значима' if result['rejected'] else 'значимой связи нет'}")

        return result

    def analyze_weekday(self) -> dict:
        """Проверка связи дня недели первого заказа с возвратом (χ²-тест)."""
        self.user_profile['first_order_weekday'] = (
            self.user_profile['first_order_dt'].dt.dayofweek
        )

        weekday_stats = (
            self.user_profile
            .groupby('first_order_weekday', observed=True)
            .agg(
                users=('user_id', 'count'),
                return_rate=('is_two', 'mean'),
            )
        )

        contingency = pd.crosstab(
            self.user_profile['first_order_weekday'],
            self.user_profile['is_two'],
        )
        chi2, pval, dof, _ = chi2_contingency(contingency)

        result = {
            'chi2': chi2,
            'p_value': pval,
            'spread': weekday_stats['return_rate'].max() - weekday_stats['return_rate'].min(),
        }
        self.stats['weekday_chi2'] = result

        if self.verbose:
            print("\nДень недели первого заказа:")
            print(f"  χ²: {result['chi2']:.2f}, p-value: {result['p_value']:.4f}")
            print(f"  Разброс долей возвратов между днями: {result['spread']:.2%}")
            print(f"  Разброс долей возвратов между днями: {result['spread']:.2%}, при такой большой выборке эффект практически не применим")

        return result

    
    # Поведенческие особенности
    def analyze_revenue_by_group(self) -> dict:
        """Сравнение чека у пользователей с 1 заказом и 2+ заказа."""
        one_order = self.user_profile[self.user_profile['is_two'] == 0]
        returned = self.user_profile[self.user_profile['is_two'] == 1]

        result = {
            'mean_one_order': one_order['avg_revenue_per_order'].mean(),
            'median_one_order': one_order['avg_revenue_per_order'].median(),
            'mean_returned': returned['avg_revenue_per_order'].mean(),
            'median_returned': returned['avg_revenue_per_order'].median(),
        }
        self.stats['revenue_by_group'] = result

        if self.verbose:
            print("\nСравнение чека между группами:")
            print(f"  1 заказ:      mean = {result['mean_one_order']:.2f}, "
                  f"median = {result['median_one_order']:.2f}")
            print(f"  2+ заказов:   mean = {result['mean_returned']:.2f}, "
                  f"median = {result['median_returned']:.2f}")

        return result

    
    # Корреляционный анализ phi_k
    def run_correlation_analysis(self) -> pd.DataFrame:
        """Считает матрицу phi_k и печатает топ-признаки по связи с total_orders."""
        if self.verbose:
            print("\nКорреляционный анализ phi_k...")

        df_corr = self.user_profile.copy().rename(
            columns={'orders_count': 'total_orders'}
        )

        # Производные признаки
        df_corr['first_order_month'] = df_corr['first_order_dt'].dt.month
        df_corr['last_order_month'] = df_corr['last_order_dt'].dt.month
        df_corr['lifetime_days'] = (
            df_corr['last_order_dt'] - df_corr['first_order_dt']
        ).dt.days

        # Категоризация интервалов между заказами
        def interval_bin(x):
            if pd.isna(x) or x < 0:
                return 'один заказ'
            elif x < 7:
                return 'менее недели'
            elif x < 30:
                return '1–4 недели'
            return 'более месяца'

        df_corr['days_interval_cat'] = (
            df_corr['avg_days_between_orders'].apply(interval_bin)
        )
        df_corr = df_corr.drop(
            columns=['user_id', 'first_order_weekday_name',
                     'avg_days_between_orders'],
            errors='ignore',
        )

        interval_cols = ['total_orders', 'avg_revenue_per_order', 'lifetime_days']

        phik_overview = df_corr.phik_matrix(
            interval_cols=interval_cols
        ).round(3)

        self.phik_overview = phik_overview

        if self.verbose:
            top = (
                phik_overview
                .loc[phik_overview.index != 'total_orders', ['total_orders']]
                .sort_values('total_orders', ascending=False)
            )
            print("Топ-10 признаков по связи с total_orders:")
            print(top.head(10))

        return phik_overview

    
    # Шаг 8. Итоговый отчет
    def generate_report(self, save_path: str = 'reports/final_report.txt') -> str:
        """Собирает итоговый отчет по проекту."""
        sep = '=' * 40
        lines = []

        lines.append(sep)
        lines.append('ИТОГОВЫЙ ОТЧЕТ ПО ПРОЕКТУ')
        lines.append('Анализ лояльности пользователей Яндекс Афиши')
        lines.append(sep)

        # 1. Объем данных и очистка ---
        init_rows = self.stats['initial_rows']
        after_dup = self.stats['rows_after_duplicates']
        after_out = self.stats['rows_after_outliers']
        init_users = self.stats['initial_users']
        final_users = self.stats['users_after_outlier_filter']

        lines.append('\n1. ОБЪЕМ ДАННЫХ И ЭТАПЫ ОЧИСТКИ')
        lines.append(f'  Исходный датасет:            {init_rows:,} заказов')
        lines.append(f'  Удалено дубликатов заказов:  {init_rows - after_dup:,} '
                     f'({(init_rows - after_dup) / init_rows:.2%})')
        lines.append(f'  Удалено выбросов по выручке: {after_dup - after_out:,} '
                     f'({(after_dup - after_out) / after_dup:.2%})')
        lines.append(f'  Итоговый объем заказов:      {after_out:,} '
                     f'({after_out / init_rows:.2%} от исходного)')
        lines.append('')
        lines.append(f'  Пользователей до фильтрации:    {init_users:,}')
        lines.append(f'  Пользователей после фильтрации: {final_users:,} '
                     f'(-{init_users - final_users:,})')

        # 2. Основные метрики ---
        lines.append('\n2. КЛЮЧЕВЫЕ МЕТРИКИ')
        lines.append(f'  Средняя доля возвратов:     '
                     f'{self.stats["overall_return_rate"]:.2%}')
        lines.append(f'  Средняя выручка с заказа:   '
                     f'{self.user_profile["avg_revenue_per_order"].mean():.2f} руб.')
        lines.append(f'  Медианная выручка с заказа: '
                     f'{self.user_profile["avg_revenue_per_order"].median():.2f} руб.')

        # 3. Проверка гипотез ---
        lines.append('\n3. РЕЗУЛЬТАТЫ ПРОВЕРКИ ГИПОТЕЗ')

        h1 = self.stats.get('hypothesis_event_type')
        if h1:
            lines.append('  Гипотеза 1. Тип первого мероприятия влияет на возврат клиента:')
            lines.append(f'    Спорт:    {h1["sport_rate"]:.2%}')
            lines.append(f'    Концерты: {h1["concert_rate"]:.2%}')
            lines.append(f'    p-value:  {h1["p_value"]:.4f}')
            verdict = ('различие статистически значимо (H₀ отвергается)'
                       if h1['rejected'] else 'различие незначимо')
            lines.append(f'    Вывод:    {verdict}.')
            lines.append('    Интерпретация: спорт удерживает хуже, чем концерты; '
                         'первая гипотеза опровергается.')

        h2 = self.stats.get('hypothesis_region')
        if h2:
            lines.append('  Гипотеза 2. Размер региона по числу пользователей влияет на возврат клиента:')
            lines.append(f'    Корреляция Спирмена: {h2["spearman_corr"]:.3f}')
            lines.append(f'    p-value:             {h2["p_value"]:.4f}')
            verdict = ('связь значима' if h2['rejected'] else 'значимой связи нет')
            lines.append(f'    Вывод: {verdict}.')
            lines.append('    Интерпретация: регион с большим числом пользователей '
                         'не демонстрирует систематически более высокий возврат.')

        wk = self.stats.get('weekday_chi2')
        if wk:
            lines.append('  День недели первого заказа:')
            lines.append(f'    χ²: {wk["chi2"]:.2f}, p-value: {wk["p_value"]:.4f}')
            lines.append(f'    Разброс долей между днями: {wk["spread"]:.2%}')
            lines.append('    Интерпретация: формальная значимость есть, '
                         'но практический эффект мал (< 4 п.п.).')

        # 4. Поведенческие особенности
        rb = self.stats.get('revenue_by_group')
        if rb:
            delta = (rb['median_returned'] - rb['median_one_order']) / rb['median_one_order']
            lines.append('\n4. ПОВЕДЕНЧЕСКИЕ ОСОБЕННОСТИ')
            lines.append('  Средняя/медианная выручка с заказа:')
            lines.append(f'    Пользователи с 1 заказом: {rb["mean_one_order"]:.2f} / '
                         f'{rb["median_one_order"]:.2f} руб.')
            lines.append(f'    Вернувшиеся (2+):         {rb["mean_returned"]:.2f} / '
                         f'{rb["median_returned"]:.2f} руб.')
            lines.append(f'    Медиана у вернувшихся выше на {delta:.1%}.')
            lines.append('    Интерпретация: средние почти совпадают, но медиана '
                         'у вернувшихся выше. Они реже делают «пробные» '
                         'микропокупки в диапазоне 0–200 руб.')

        # Рекомендации 
        lines.append('\n5. РЕКОМЕНДАЦИИ ДЛЯ БИЗНЕСА')
        lines.append('  1. Стимулировать более «серьёзную» первую покупку '
                     '(300–900 руб.) — это связано с более высоким retention.')
        lines.append('  2. Учитывать тип первого мероприятия при таргетинге: '
                     'концерты и театр дают более лояльных клиентов, чем спорт.')
        lines.append('  3. День недели и регион первого заказа — слабые предикторы, '
                     'не стоит тратить на них маркетинговый бюджет.')
        lines.append('  4. Для прогноза retention строить ML-модель на '
                     'поведенческих признаках (частота, средний чек, '
                     'время между заказами), а не только на сегментации.')

        lines.append('\n' + sep)
        report = '\n'.join(lines)
        print(report)

        # Сохранение отчета в файл
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report)

        return report