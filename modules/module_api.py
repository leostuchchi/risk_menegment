import os
import glob
import joblib
import json
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, create_model
from modules.main import predictor
import warnings
warnings.filterwarnings('ignore')


path = os.environ.get('PROJECT_PATH', os.path.join(os.path.expanduser('~'), 'scoring_credit_default'))

app = FastAPI()

model_file = glob.glob(f'{path}/data/models/*.pkl')

with open(*model_file, 'rb') as file:
    model = joblib.load(file)
print(model['metadata'])




# Функция для создания динамического класса
def create_dynamic_model():
    df = pd.read_parquet(f'{path}/data/train/to_pipeline.parquet')
    df.drop(['target'], axis=1, inplace=True)
    # Словарь для хранения аннотаций типов
    fields = {}

    # Проходим через все столбцы DataFrame и получаем их типы
    for column in df.columns:
        if pd.api.types.is_integer_dtype(df[column]):
            field_type = int
        elif pd.api.types.is_float_dtype(df[column]):
            field_type = float
        elif pd.api.types.is_string_dtype(df[column]):
            field_type = str
        else:
            field_type = str  # Или другой тип по умолчанию

        # Добавляем поле в словарь
        fields[column] = (field_type, ...)

    # Создаем модель с помощью create_model из Pydantic
    Form = create_model('Form', **fields)

    return Form


# Создаем динамический класс
DynamicForm = create_dynamic_model()
data_to_input = DynamicForm.schema()
print(DynamicForm.schema())



class Prediction(BaseModel):
    id: int
    predict: int



@app.post('/predict_api', response_model=Prediction)
def predict_api(form: DynamicForm):
    # преобразование json в pd.DataFrame
    data_dict = form.dict()
    df = pd.DataFrame([data_dict])  # Оборачиваем в список для создания DataFrame
    # передаем запрос на предсказание
    print(df)
    # агрегация, обработка, кодирование и нормализация запроса производится в predictions
    result = predictor(df)
    print(result)
    # Получаем идентификатор (id) и предсказание (predict)
    prediction_id = result.iloc[0][0]
    prediction_value = result.iloc[0][1]

    return Prediction(id=prediction_id, predict=prediction_value)




@app.get('/status')
def status():
    return 'I am Fine'


@app.get('/version')
def version():
    return model['metadata']


@app.get('/info')
def info():
    answer = ('модель для предсказания дефолта: предсказывает процентную вероятность дефолта',
              'на вход принимает данные:', data_to_input)
    return answer



def check_predict():
    result = predictor()
    print(result)

    return result

if __name__ == '__main__':
    check_predict()

