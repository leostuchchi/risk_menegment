import warnings
warnings.filterwarnings('ignore')
import os
import glob
from datetime import datetime
import pandas as pd
import numpy as np
import shap
import xgboost
from sklearn.model_selection import train_test_split

from modules.take_parquet import take_data
from modules.logging_info import setup_logging

"""
Функции модуля:
def new_features - Создание новых признаков 
def high_corr - Удаление признаков с низкой корреляцией
def important - Удаление признаков с нулевой значимостью
def duplicates - Удаление дубликатов большего класса
def preparations - Запуск обработки данных
def aggregator - Функция агрегирования данных запроса для предсказания
def preparations_predict - Подготовка данных для предсказания
def fileout - Создание имени файла для записи предсказания
def take_data_to_predict - Подготовка тестового файла для предсказаний

"""

# Настраиваем логирование
logger = setup_logging()

# абсолютный путь к  проекту
path = os.environ.get('PROJECT_PATH', os.path.join(os.path.expanduser('~'), 'scoring_credit_default'))

current_path = os.environ.get('PROJECT_PATH', os.path.join(os.path.expanduser('~'), 'scoring_credit_default'))
data_path = os.path.join(current_path, 'data')



# Feature engineering

def new_features(df):
    """
    Создание новых признаков
    """

    # 1. Отношение планового количества дней до закрытия кредита к количеству брошенных платежей
    df['planned_to_zero_loans_530'] = df['pre_pterm'] * df['is_zero_loans530']

    # 2. Сумма брошенных платежей от 30 до 60 дней и от 60 до 90 дней
    df['suspended_loans_3060_6090'] = df['is_zero_loans3060'] + df['is_zero_loans6090']

    # 3. Процент использования кредита при отсутствии просрочек
    df['utilization_no_overdue'] = df['pre_util'] * (1 - (df['is_zero_loans530'] + df['is_zero_loans3060'] + df['is_zero_loans6090'] + df['is_zero_loans90']))

    # 4. Сумма просрочек по каждому типу кредита
    df['overdue_by_credit_type'] = df['pre_over2limit'] * (df['enc_loans_credit_type'].astype('category').cat.codes)

    # 5. Отношение максимальной просрочки к использованию кредита
    df['max_overdue_to_utilization'] = df['pre_maxover2limit'] / (df['pre_util'] + 1)

    # 6. Сумма кодов платежей 2, 3, 4, 15, 16, 23
    df['enc_payment_sum'] = df[['enc_paym_2', 'enc_paym_3', 'enc_paym_4', 'enc_paym_15', 'enc_paym_16', 'enc_paym_23']].sum(axis=1)

    # Проверка новых признаков
    print(df[['planned_to_zero_loans_530', 'suspended_loans_3060_6090', 'utilization_no_overdue',
           'overdue_by_credit_type', 'max_overdue_to_utilization', 'enc_payment_sum']].columns)

    logger.info('preprocessing. new_features. created')

    return df

# Data cleaning


def high_corr(df):
    """
    Удаление признаков с низкой корреляцией
    """
    min_corr = []
    for i in df.columns:
        j = df[i]
        corr = df['target'].corr(j)
        if -0.01 < corr and corr < 0.01:
            min_corr.append(i)

    print(min_corr)
    logger.info('preprocessing, high_corr, drop low correlated: %s', min_corr)
    df.drop(min_corr, axis=1, inplace=True, errors='ignore')

    return df




def important(df):
    """
    Удаление признаков с нулевой значимостью
    """
    # Определение данных и разделение на обучающую и тестовую выборки
    X = df.drop(['target'], axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Обучение модели
    params = {
        'objective': 'binary:logistic',
        'max_depth': 3,
        'learning_rate': 0.1
    }
    model = xgboost.train(params, xgboost.DMatrix(X_train, y_train), num_boost_round=21)
    # Создание объекта Explainer и получение SHAP значений
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    # Преобразуем shap_values в DataFrame для удобной работы
    shap_df = pd.DataFrame(shap_values, columns=X.columns)
    # Создание переменной no_good для признаков с нулевой значимостью
    no_good = shap_df.mean().index[shap_df.mean() == 0].tolist()
    # Выводим список признаков с нулевой значимостью
    print("Признаки с нулевой значимостью:")
    print('0_importances', no_good)
    logger.info('preprocessing, important, drop 0-important: %s', no_good)
    df.drop(no_good, axis=1, inplace=True, errors='ignore')

    return df



def duplicates(df):
    """
    Удаление дубликатов большего класса
    """
    # контроль целевой переменной
    print('до', df['target'].value_counts())
    # создаем маску для дубликатов, где target == 1
    mask = df.duplicated(keep=False) & (df['target'] == 1)
    # Оставляем дубликаты, где target == 1
    df_keep_target_1 = df[mask]
    # Удаляем все дубликаты, кроме тех, которые соответствуют маске
    df_unique = df[~mask].drop_duplicates()
    # Объединяем оба DataFrame
    df = pd.concat([df_unique, df_keep_target_1]).sort_index()
    print('после', df['target'].value_counts())
    print('Размер DataFrame:', df.shape)
    logger.info('preprocessing, duplicates removed, target values: %s', df['target'].value_counts())
    logger.info('preprocessing, duplicates removed, shape: %s', df.shape)

    return df



def preparations():
    """
    Запуск обработки данных
    """
    # Получение данных
    df = take_data()
    #df = pd.read_parquet(f'{path}/data/train/to_pipeline.parquet')
    # Сразу удаляем id
    df.drop(['id'], axis=1, inplace=True, errors='ignore')
    # Заполняем пропуски нулями
    df.fillna(0, inplace=True)
    df = new_features(df)
    df = high_corr(df)
    df = important(df)
    df = duplicates(df)
    # Записываем файл для возможности проверки данных
    df.to_csv(os.path.join(path, 'data', 'train', 'prepared_df.csv'), index=False)
    logger.info('preprocessing, preparations, final_df taken, shape, : %s', df.shape)

    return df





def aggregator(df: pd.DataFrame) -> pd.DataFrame:
    """
    Функция агрегирования данных запроса для предсказания
    """
    # Создаем словарь для агрегирования
    agg_d = {col: 'mean' for col in df.columns if col not in ['id', 'rn']}
    agg_d['rn'] = 'count'  # Для 'rn' считаем количество строк
    # Группировка и агрегация данных
    prepared_df = df.groupby('id').agg(agg_d).reset_index(drop=False)
    # Проверяем количества строк для каждого id
    single_row_ids = df['id'].value_counts()[df['id'].value_counts() == 1].index.tolist()
    # Фильтруем строки, где id имеет единственную запись
    single_rows_df = df[df['id'].isin(single_row_ids)]
    # Объединяем агрегированные данные и строки с единственным id
    final_df = pd.concat([prepared_df, single_rows_df], ignore_index=True).drop_duplicates('id', keep='first')
    # Приводим все колонки к типу int, кроме 'rn'
    final_df = final_df.astype({col: int for col in final_df.columns if col != 'rn'})
    final_df['rn'] = final_df['rn'].astype(int)  # Указываем 'rn' тоже как int
    logger.info('predictions, request to predict was aggregated')

    return final_df




def preparations_predict(df=None):
    """
    Подготовка данных для предсказания
    """
    # считываем имена признаков
    data = pd.read_csv(os.path.join(path, 'data', 'train', 'prepared_df.csv'), nrows=0)
    features = data.columns.tolist()
    # Удаляем 'target' из списка признаков, если он существует
    if 'target' in features:
        features.remove('target')
    # Если df не передан, считываем из файла запроса
    if df is None:
        all_files = glob.glob(f'{path}/data/to_predict/*.csv')
        df = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True) if len(all_files) > 0 else None
        print('df_to_predict.shape:', df.shape)
        total_missing = df.isnull().sum().sum()
        print(f'Общее количество пропущенных значений: {total_missing}')

    id_df = df['id']
    # если данные неагрегированы производится агрегация согласно начальным условиям получения данных
    df_agregated = aggregator(df)
    # создаем признаки
    df_features = new_features(df_agregated)
    # фильтруем признаки Создаем новый DataFrame, оставив только 'id' и признаки из features
    predict_file = df_features[['id'] + features]
    logger.info('predictions, preparations_predict was taken')

    return predict_file, id_df


def fileout():
    """
    Создание имени файла для предсказания с добавлением даты и времени.
    """
    # Убедитесь, что 'path' определена где-то в вашей программе
    directory = f'{path}/data/models'  # известная директория
    extension = '.pkl'  # известное расширение
    today_date = datetime.now().strftime('%Y-%m-%d')  # Получение текущей даты в формате YYYY-MM-DD
    current_time = datetime.now().strftime('%H-%M-%S')  # Получение текущего времени в формате HH-MM-SS

    for filename in os.listdir(directory):
        if filename.endswith(extension):
            file_name = filename.replace(extension, '')  # Убираем расширение
            file_name_with_date_time = f"{file_name}_{today_date}_{current_time}"  # Добавляем дату и время к имени файла
            print(file_name_with_date_time)  # Печатаем новое имя файла с датой и временем
            return file_name_with_date_time  # Возвращаем новое имя файла



def take_data_to_predict():
    """
    Подготовка тестового файла для предсказаний
    """
    df = pd.read_parquet(os.path.join(path, 'data', 'train', 'to_pipeline.parquet'))

    # Выбираем 10 случайных id с target = 0 и 1
    ids_0 = df[df['target'] == 0]['id'].drop_duplicates().sample(n=10, random_state=1)
    ids_1 = df[df['target'] == 1]['id'].drop_duplicates().sample(n=10, random_state=1)

    # Объединяем уникальные id
    selected_ids = pd.concat([ids_0, ids_1])

    result_df = df[df['id'].isin(selected_ids)]

    # Записываем файл для предсказаний
    result_df.to_csv(os.path.join(data_path, 'to_predict', 'to_predict.csv'), index=False)
    print('Файл для предсказаний записан:', os.path.join(data_path, 'to_predict', 'to_predict.csv'))
    logger.info('preprocessing, take_data_to_predict, taken, shape, : %s', result_df.shape)

    return result_df



if __name__ == '__main__':
    preparations()
