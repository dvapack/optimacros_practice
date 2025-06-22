import pandas as pd
import json


class Validation():
    """
    Класс для валидации гиперпараметров
    """
    def __init__(self):
        self.algorithm = None
        self.hyperparams = None

    def __load_hyperparams(self, filepath: str):
        """
        Метод для загрузки гиперпараметров, полученных от OptiMacros
        :param filepath: Путь до файла csv с гиперпараметрами;
        """
        try:
            self.hyperparams = pd.read_csv(filepath)
            # переводим сразу из json в python-словари
            self.hyperparams.iloc[:, 1] = self.hyperparams.iloc[:, 1].apply(json.loads)
        except FileNotFoundError:
            print(f"Такого файла не существует")

    def __validate_hyperparam(self, param: dict):
        """
        Метод для валидации одного конкретного гиперпараметра
        """


    def validate_hyperparams(self, filepath: str):
        """
        Метод для валидации списка гиперпараметров
        :param filepath: Путь до файла csv с гиперпараметрами;
        """
        self.__load_hyperparams(filepath)





 