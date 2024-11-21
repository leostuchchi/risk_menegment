import os
import pandas as pd
import tqdm

from modules.logging_info import setup_logging
# Настраиваем логирование
logger = setup_logging()

# Установка путей проекта

# абсолютный путь к  проекту
current_path = os.environ.get('PROJECT_PATH', os.path.join(os.path.expanduser('~'), 'scoring_credit_default'))
# расположение этого модуля - (os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(current_path, 'data')  # Обновите путь к папке с данными
path = os.path.join(data_path, 'parquet')  # Путь к файлам с данными
path_to_target = os.path.join(data_path, 'target', 'train_target.csv')
path_to_data = os.path.join(data_path, 'train')

def read_parquet_dataset_from_local(path_to_dataset: str, start_from: int = 0,
                                     num_parts_to_read: int = 2, columns=None, verbose=False) -> pd.DataFrame:
    """
    Reads num_parts_to_read partitions, converts them to pd.DataFrame and returns.
    """
    dataset_paths = sorted([os.path.join(path_to_dataset, filename) for filename in os.listdir(path_to_dataset)
                             if filename.startswith('train')])

    start_from = max(0, start_from)
    chunks = dataset_paths[start_from: start_from + num_parts_to_read]

    if not chunks:
        return pd.DataFrame()  # Return empty DataFrame if no chunks.

    if verbose:
        print('Reading chunks:\n')
        for chunk in chunks:
            print(chunk)

    dataframes = []
    for chunk_path in tqdm.tqdm(chunks, desc="Reading dataset with pandas"):
        df = pd.read_parquet(chunk_path, columns=columns)
        dataframes.append(df)

    # Concatenate only if there are dataframes to concatenate
    return pd.concat(dataframes, ignore_index=True) if dataframes else pd.DataFrame()

def prepare_transactions_dataset(path_to_dataset: str, num_parts_to_preprocess_at_once: int = 1,
                                 num_parts_total: int = 50, save_to_path=None, verbose: bool = False):
    """
    Prepares and aggregates transactions data into a ready DataFrame.
    """
    # значения целевой переменной
    targets = pd.read_csv(path_to_target)
    # Переименуем целевую переменную - target
    targets.rename(columns={'flag': 'target'}, inplace=True)

    final_df = pd.DataFrame()  # For final aggregation

    for step in tqdm.tqdm(range(0, num_parts_total, num_parts_to_preprocess_at_once),
                          desc="Transforming transactions data"):
        transactions_frame = read_parquet_dataset_from_local(path_to_dataset, start_from=step,
                                                             num_parts_to_read=num_parts_to_preprocess_at_once,
                                                             verbose=verbose)

        if transactions_frame.empty:
            continue  # Skip to the next step if no data

        # Агрегация по mean, int
        # Create aggregation dictionary
        agg_d = {col: 'mean' for col in transactions_frame.columns if col not in ['id', 'rn']}
        agg_d['rn'] = 'count'  # Count 'rn'

        # Grouping and aggregating data
        prepared_df = transactions_frame.groupby('id').agg(agg_d).reset_index(drop=False).astype(int)

        # Append to final DataFrame
        final_df = pd.concat([final_df, prepared_df], ignore_index=True)

    # Объединяем с целевыми значениями
    final_df = final_df.merge(targets, on='id', how='inner')

    if save_to_path:
        final_df.to_parquet(os.path.join(save_to_path, 'to_pipeline.parquet'), index=False)

    return final_df  # Return the final aggregated DataFrame


# Вызываем функцию для подготовки набора данных
def take_data():
    logger.info("take_data, receiving new data")

    data = prepare_transactions_dataset(path, num_parts_to_preprocess_at_once=2, num_parts_total=50,
                                      save_to_path=os.path.join(path_to_data))
    print('Размерность агрегированных по id данных с target:', data.shape)
    print('Значения целевой переменной:', data['target'].value_counts())
    print('Данные сохранены:', path_to_data)

    logger.info(f'take_data, Dimension of Aggregated Data: {data.shape}')
    logger.info(f'take_data, Target variable values: {data["target"].value_counts()}')
    logger.info(f'take_data, Data saved in: {path_to_data}')

    return data




if __name__ == '__main__':
    take_data()