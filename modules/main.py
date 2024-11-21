import os
import glob
import joblib
import numpy as np
import pandas as pd
import optuna
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score, StratifiedKFold
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from datetime import datetime

from modules.preprocessing import preparations, fileout
from modules.preprocessing import preparations_predict
import logging
from modules.logging_info import setup_logging


"""
Функции модуля:
def compute_class_weights - Функция для вычисления весов классов
def objective - Функция подбора гиперпараметров и модели
def pipeline - Основная функция для подготовки данных и обучения лучшей модели
def predictor - Функция предсказаний
"""


# Настраиваем логирование
logger = setup_logging()

# абсолютный путь к  проекту
path = os.environ.get('PROJECT_PATH', os.path.join(os.path.expanduser('~'), 'scoring_credit_default'))



def compute_class_weights(y_train):
    """
    Функция для вычисления весов классов
    """
    if len(np.unique(y_train)) == 2:
        weight0 = len(y_train) / (2 * np.sum(y_train == 0)) if np.sum(y_train == 0) > 0 else 1
        weight1 = len(y_train) / (2 * np.sum(y_train == 1)) if np.sum(y_train == 1) > 0 else 1
        return {0: weight0, 1: weight1}
    return {0: 1, 1: 1}




def objective(trial, X, y, preprocessor):
    """
    функция подбора гиперпараметров и лучшей модели
    """
    logger.info("modeling, start tuning")
    class_weights = compute_class_weights(y)
    model_name = trial.suggest_categorical('model', ['xgboost', 'catboost', 'lightgbm'])
    model_name = trial.suggest_categorical('model', ['xgboost'])
    if model_name == 'xgboost':
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 10)
        }
        model = XGBClassifier(**params)

    elif model_name == 'catboost':
        params = {
            'depth': trial.suggest_int('depth', 2, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'iterations': trial.suggest_int('iterations', 50, 300),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 10)
        }
        model = CatBoostClassifier(**params, silent=True)

    else:  # lightgbm
        params = {
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'scale_pos_weight': class_weights[1] / class_weights[0]
        }
        model = LGBMClassifier(**params)

    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    score = cross_val_score(pipe, X, y, cv=cv, scoring='roc_auc')

    return score.mean()




def pipeline():
    """
    основная функция для подготовки данных и обучения лучшей модели
    """
    # Получаем обработанные данные
    df = pd.read_csv(os.path.join(path, 'data', 'train', 'prepared_df.csv'))
    #df = preparations()
    # Разделяем фичи и целевую переменную
    X = df.drop('target', axis=1)
    y = df['target']

    # разделение числовых и категориальных признаков
    numerical_features = [col for col in X.columns if len(X[col].value_counts()) > 9]
    categorical_features = [col for col in X.columns if len(X[col].value_counts()) <= 9]

    # создаем pipeline
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # кодируем и стандартизируем признаки
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('numerical', numerical_transformer, numerical_features),
        ('categorical', categorical_transformer, categorical_features)
    ])


    # запускаем подбор гиперпараметров в optuna
    study = optuna.create_study(direction='maximize')
    # при использовании текущих данных 20 итераций study более чем достаточно
    study.optimize(lambda trial: objective(trial, X, y, preprocessor), n_trials=1)

    best_params = study.best_params
    print(f'Best parameters: {best_params}')
    logger.info(f'Best parameters: {best_params}')

    # Выбор модели на основе лучших параметров
    if best_params['model'] == 'xgboost':
        model = XGBClassifier(**best_params)
    elif best_params['model'] == 'catboost':
        # Удаляем параметр 'model', так как он вызывает ошибку
        best_params.pop('model')
        model = CatBoostClassifier(**best_params, silent=True)
    else:
        best_params.pop('model')  # Удаляем 'model' для lightgbm
        model = LGBMClassifier(**best_params)


    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    # Обучаем Pipeline
    pipe.fit(X, y)

    # Обучаем финальную модель
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    score = cross_val_score(pipe, X, y, cv=cv, scoring='roc_auc').mean()
    print(f'Final model ROC AUC: {score:.4f}')

    # проверяем метрики, с непроходной метрикой модель не записываем
    if score < 0.75:
        print('Метрика модели ниже требуемой, модель не записана.')
        logging.warning('Metric is lower than required, the model is not saved')
        return

    # добавляем метаданные
    metadata = {
        "name": "Scoring_default",
        "author": "leostuchchi",
        "version": 2,
        "date": datetime.now(),
        "roc_auc": score
    }

    """
    Сохранение модели
    """
    # удаляем предыдущие модели
    for file in glob.glob(os.path.join(path, f'data/models/*.pkl')):
        os.remove(file)
    # Запись лучшей модели
    model_filename = os.path.join(path, f'data/models/default_{round(score, 4)}_roc_auc.pkl')
    joblib.dump({'model': model, 'metadata': metadata}, model_filename)
    logging.info(f'Model is saved as {model_filename}')

    """
    Сохранение pipeline
    """
    # Запись pipeline
    pipe_filename = os.path.join(path, f'data/pipe/pipe.pkl')
    joblib.dump(pipe, pipe_filename)
    logging.info(f'pipeline is saved as {pipe_filename}')




def predictor(df=None):
    """
    Функция предсказания
    """
    if df is None:
        predict_file, id_df = preparations_predict()
    else:
        predict_file, id_df = preparations_predict(df)

    # Загрузка обученного pipeline
    pipe = joblib.load(os.path.join(path, f'data/pipe/pipe.pkl'))

    # Предсказание
    predictions = pipe.predict_proba(predict_file)
    # Извлекаем только вероятности для класса [1]
    probabilities_class_1 = predictions[:, 1] * 100  # Умножаем на 100 для процентов
    results = []
    # Округляем и переводим в целое число
    results.extend(zip(id_df, probabilities_class_1.astype(int)))
    results = pd.DataFrame(results)
    results = results.rename(columns={0: 'id', 1: 'percent_default'})

    file_name = fileout()
    results.to_csv(os.path.join(path, 'data', 'predictions', file_name), index=False)
    print(results)
    return results



if __name__ == '__main__':
    #pipeline()
    predictor()
