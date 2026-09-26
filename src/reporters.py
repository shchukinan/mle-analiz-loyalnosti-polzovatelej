import pandas as pd


class DataFrameReporter:
    """
    Генерирует краткий отчет о состоянии датафрейма:
    размер, дубликаты, пропуски, описательные статистики.
    """
    def __init__(self, float_format='0.05f', percent_format='0.02%', include_all=False):
        self.float_format = float_format
        self.percent_format = percent_format
        self.include_all = include_all

    def show_report(self, df: pd.DataFrame, title: str = None) -> None:
        if title:
            print(f"\n{'=' * 60}")
            print(title)
            print('=' * 60)

        rows, cols = df.shape
        print(f'Количество столбцов: {cols}')
        print(f'Количество строк:    {rows}')

        duplicates = df.duplicated().sum()
        print(f'Количество явных дубликатов: {duplicates}')
        print(f'Доля явных дубликатов:       {duplicates / rows:{self.percent_format}}')

        total_missing = df.isna().sum().sum()
        missing_ratio = df.isna().mean(axis=None)
        print(f'Количество пропусков: {total_missing}')
        print(f'Доля пропусков:       {missing_ratio:{self.float_format}}')

        print('\nОписательные статистики:')
        print(df.describe(include='all' if self.include_all else None))