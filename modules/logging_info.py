import os
import logging
from datetime import datetime

# Укажем путь к файлам корневая директория: os.path.path.join(os.path.expanduser('~'))
path = os.environ.get('PROJECT_PATH', os.path.join(os.path.expanduser('~'), 'scoring_credit_default'))


def setup_logging():
    """
    Настраиваем логирование. Создает лог файл в data/logs.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Получаем текущую дату и время для названия лог файла
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = os.path.join(path, f'data/logs/app_{timestamp}.log')

    # Создаем имена файлов для логов
    #log_filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'modules.log')
    #log_filename = os.path.join(path, f'data/logs/modules.log')

    # Создаем обработчик файлового лога
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.DEBUG)

    # Создаем формат логов
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Добавляем обработчик к логгеру
    logger.addHandler(file_handler)

    return logger
