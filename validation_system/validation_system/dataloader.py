import pandas as pd
import json
import os

import joblib

from sklearn.linear_model import ElasticNet, HuberRegressor, Lasso, \
                                RANSACRegressor, Ridge, TheilSenRegressor
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from statsmodels.tsa.api import Holt, SimpleExpSmoothing, ExponentialSmoothing, SARIMAX


class DataLoader():
    """
    Класс для работы с загрузкой и выгрузкой гиперпараметров моделей и самих экземпляров обученных моделей.
    """
    def __init__(self, filepath: str):
        self.filepath = filepath
        pass

    def load_hyperparams_from_optimacros(self) -> tuple[list, list]:
        """
        Метод для загрузки гиперпараметров, полученных от OptiMacros
        :return: Список моделей и список гиперпараметров
        """
        try:
            data = pd.read_csv(self.filepath)
            # переводим сразу из json в python-словари
            data.loc[:, 'Params'] = data.loc[:, 'Params'].apply(json.loads)
            models = data.iloc[:, 2].to_list()
            hyperparams = []
            for model in models:
                model_info = data.loc[(data.Model == model), 'Params'].iloc[0]
                hyperparams.append(model_info[model])
            return models, hyperparams
        except FileNotFoundError:
            print(f"Такого файла не существует")

    def save_hyperparams_from_python(self, models: list, hyperparams: list) -> str:
        """
        Метод для сохранения полученных оптимальных гиперпараметров от Python.
        :param models: Список моделей;
        :param hyperparams: Список гиперпараметров.
        :return: Путь до файла с сохранёнными гиперпараметрами
        """
        data = {"Models": models, "Best_params": hyperparams}
        df = pd.DataFrame(data)
        df.loc[:, 'Best_params'] = df.loc[:, 'Best_params'].apply(json.dumps)
        filepath = f"best_params_{self.filepath}"
        df.to_csv(filepath)
        return filepath

    def backup_hyperparams(self, models: list, hyperparams: list) -> str:
        """
        Метод для резервного копирования гиперпараметров.
        :param models: Список моделей;
        :param hyperparams: Список гиперпараметрой.
        :return: Путь до файла с резервной копией.
        """
        data = {"Models": models, "Params": hyperparams}
        df = pd.DataFrame(data)
        df.loc[:, 'Params'] = df.loc[:, 'Params'].apply(json.dumps)
        filepath = f"back_up_{self.filepath}"
        df.to_csv(filepath)
        return filepath
    
    def __save_custom_model(self, model: str, params: list):
        """
        Метод для для сохранения гиперпараметров самописных моделей.
        :param model: Название модели;
        :paran params: Гипепараметры модели.
        """
        new_data = pd.DataFrame({'Models': model, 'Params': params})
        # если файл существует и не пуст - читаем и добавляем данные
        if os.path.exists(self.filepath) and os.path.getsize(self.filepath) > 0:
            old_data = pd.read_csv(f"{self.filepath}.csv")
            combined_data = pd.concat([old_data, new_data], ignore_index=True)
            combined_data.to_csv(f"{self.filepath}.csv", index=False)
        else:  # если файла нет или он пуст - просто сохраняем новые данные
            new_data.to_csv(f"{self.filepath}.csv", index=False)

    def save_trained_model(self, model, filepath: str):
        """
        Метод для сохранения обученной модели. В зависимости от модели 
        сохраняет либо саму модель, либо только её гиперпараметры. Если передаются гиперпараметры модели, необходимо
        передавать словарь вида {"model_name": [hyperparams]}. Если передаётся сама модель, то необходимо передавать сам объект.
        :param model: Обученная модель;
        :param filepath: Путь до файла. Возможна дозапись в конец файла. Не указывать расширение файла - оно будет выбрано автоматически.
        """
        self.filepath = filepath
        custom_models = ["croston_tsb", "rol_mean", "const"]
        if isinstance(model, dict):
            model_name = list(model.keys())[0] # т.к возвращается итерируемый объект
            params = list(model.values())[0]
            if model_name in custom_models:
                self.__save_custom_model(model_name, params)
        elif isinstance(model, RandomForestRegressor):
            joblib.dump(model, f"{self.filepath}.joblib")
        elif isinstance(model, ElasticNet):
            joblib.dump(model, f"{self.filepath}.joblib")
        elif isinstance(model, HuberRegressor):
            joblib.dump(model, f"{self.filepath}.joblib")
        elif isinstance(model, Lasso):
            joblib.dump(model, f"{self.filepath}.joblib")
        elif isinstance(model, RANSACRegressor):
            joblib.dump(model, f"{self.filepath}.joblib")
        elif isinstance(model, Ridge):
            joblib.dump(model, f"{self.filepath}.joblib")
        elif isinstance(model, TheilSenRegressor):
            joblib.dump(model, f"{self.filepath}.joblib")
        elif isinstance(model, CatBoostRegressor):
            model.save_model(self.filepath, foramt="cbm")
        elif isinstance(model, Holt):
            joblib.dump(model, f"{self.filepath}.joblib")
        elif isinstance(model, SimpleExpSmoothing):
            joblib.dump(model, f"{self.filepath}.joblib")
        elif isinstance(model, ExponentialSmoothing):
            joblib.dump(model, f"{self.filepath}.joblib")
                

            