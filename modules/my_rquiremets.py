import os
import subprocess

# абсолютный путь к  проекту
path = os.environ.get('PROJECT_PATH', os.path.join(os.path.expanduser('~'), 'scoring_credit_default'))



def save_requirements_to_file():
    # Определяем путь к текущему скрипту, чтобы сохранить файл в ту же директорию
    current_directory = os.path.dirname(os.path.abspath(__file__))
    #requirements_file_path = os.path.join(current_directory, 'requirements.txt')
    requirements_file_path = os.path.join(path, 'requirements.txt')
    # Используем subprocess для выполнения pip freeze и записи зависимостей в файл
    with open(requirements_file_path, 'w') as f:
        subprocess.run(['pip', 'freeze'], stdout=f)

    print(f'Requirements have been saved to {requirements_file_path}')



if __name__ == "__main__":
    save_requirements_to_file()


