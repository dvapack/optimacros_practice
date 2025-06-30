import pandas as pd
import json
import os
from datetime import datetime

import joblib

from sklearn.linear_model import ElasticNet, HuberRegressor, Lasso, \
                                RANSACRegressor, Ridge, TheilSenRegressor
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
import statsmodels
from statsmodels.tsa.api import Holt, SimpleExpSmoothing, ExponentialSmoothing, SARIMAX
from prophet import Prophet
from prophet.serialize import model_to_json

from .validation import Validation


class DataLoader():
    """
    Класс для работы с загрузкой и выгрузкой гиперпараметров моделей и самих экземпляров обученных моделей.
    """
    def __init__(self, filepath: str):
        """
        :param filepath: Путь до файла
        """
        self.filepath = filepath

    def load_hyperparams_from_optimacros(self) -> tuple[list, list]:
        """
        Метод для загрузки гиперпараметров, полученных от OptiMacros

        :return: Список моделей и список гиперпараметров
        """
        try:
            data = pd.read_csv(self.filepath)
            # переводим сразу из json в python-словари
            data.loc[:, 'Params'] = data.loc[:, 'Params'].apply(json.loads)
            models = data.loc[:, 'Models'].to_list()
            hyperparams = []
            for model in models:
                model_info = data.loc[(data.Models == model), 'Params'].iloc[0]
                hyperparams.append(model_info[model])
            return models, hyperparams
        except FileNotFoundError:
            print(f"Такого файла не существует")

    def load_default_hyperparams(self, filepath: str) -> tuple[list, list]:
        """
        Метод для загрузки стандартных значений гиперпараметров.

        :param filepath: Путь до json файла со стандартными значениями гиперпараметров
        :return: Кортеж из моделей и значений гиперпараметров.
        """
        try:
            with open(filepath, 'r') as file:
                data = json.load(file)
            models = list(data.keys())
            hyperparams = [data[model] for model in models]
            return models, hyperparams
        except FileNotFoundError:
            print(f"Такого файла не существует")


    def backup_hyperparams(self, models: list, hyperparams: list) -> str:
        """
        Метод для резервного копирования гиперпараметров.

        :param models: Список моделей;
        :param hyperparams: Список гиперпараметрой.
        :return: Путь до файла с резервной копией.
        """
        dir_path = os.path.dirname(self.filepath)
        file_name = os.path.basename(self.filepath)
        current_datetime = datetime.now().strftime("%Y_%m_%d_%H_%M")
        versions = [current_datetime] * len(models)
        data = {"Models": models, "Params": hyperparams, "Versions": versions}
        df = pd.DataFrame(data)
        df.loc[:, 'Params'] = df.loc[:, 'Params'].apply(json.dumps)
        filepath = os.path.join(dir_path, f"{current_datetime}_back_up_{file_name}")
        df.to_csv(filepath, index=False)
        return filepath
    

    
    def __save_custom_model(self, model: str, hyperparams: list):
        """
        Метод для для сохранения гиперпараметров самописных моделей.

        :param model: Название модели;
        :param hyperparams: Гипепараметры модели.
        """
        current_datetime = datetime.now().strftime("%Y_%m_%d_%H_%M")
        new_data = pd.DataFrame({'Models': model, 'Params': hyperparams, 'Version': current_datetime})
        # если файл существует и не пуст - читаем и добавляем данные
        if os.path.exists(self.filepath) and os.path.getsize(self.filepath) > 0:
            old_data = pd.read_csv(f"{self.filepath}.csv")
            combined_data = pd.concat([old_data, new_data], ignore_index=True)
            combined_data.to_csv(f"{self.filepath}.csv", index=False)
        else:  # если файла нет или он пуст - просто сохраняем новые данные
            new_data.to_csv(f"{self.filepath}.csv", index=False)

    def __get_params_from_sklearn_model(self, model_name: str, model) -> dict:
        """
        Метод для извлечения параметров из Sklearn моделей.

        :param model_name: Название модели;
        :param model: Экземпляр модели.
        :return: Словарь с гиперпараметрами.
        """
        params = model.get_params()
        return {"Model": model_name,
                    "Params": params}
    
    def __get_params_from_statsmodels_model(self, model_name: str, model) -> dict:
        """
        Метод для извлечения параметров из Statsmodels моделей.

        :param model_name: Название модели;
        :param model: Экземпляр модели.
        :return: Словарь с гиперпараметрами.
        """
        params = model.params
        return {"Model": model_name,
                    "Params": params}    
    
    def __get_params_from_prophet_model(self, model_name: str, model) -> dict:
        """
        Метод для извлечения параметров из Prophet модели.

        :param model_name: Название модели;
        :param model: Экземпляр модели.
        :return: Словарь с гиперпараметрами.
        """
        params = model.params
        return {"Model": model_name,
                    "Params": params}   

    def get_params(self, model, model_fit = None) -> dict:
        """
        Метод для извлечения параметров из обученных экземпляров моделей.
        Поддерживаются следующие модели:

        "RandomForestRegressor", "ElasticNet", "HuberRegressor", "Lasso", "RANSACRegressor", 
        "Ridge", "TheilSenRegressor", "CatBoostRegressor", "Holt", "SimpleExpSmoothing", 
        "ExponentialSmoothing", "Prophet". 
        
        При использовании Sklearn моделей или Prophet необходимо передавать сам экземпляр модели без model_fit
        
        При использовании statsmodels необходимо передавать model и результат model.fit(), т.е results = model.fit() - необходимо передать results.
        
        :param model: Экземпляр модели.
        :param model_fit: Результат fit() для statsmodels моделей. Необязательный параметр.
        :return: Словарь с гиперпараметрами.
        """
        if isinstance(model, RandomForestRegressor):
            return self.__get_params_from_sklearn_model("random_forest", model)
        elif isinstance(model, ElasticNet):
            return self.__get_params_from_sklearn_model("elastic_net", model)
        elif isinstance(model, HuberRegressor):
            return self.__get_params_from_sklearn_model("huber", model)
        elif isinstance(model, Lasso):
            return self.__get_params_from_sklearn_model("lasso", model)
        elif isinstance(model, RANSACRegressor):
            return self.__get_params_from_sklearn_model("ransac", model)
        elif isinstance(model, Ridge):
            return self.__get_params_from_sklearn_model("ridge", model)
        elif isinstance(model, TheilSenRegressor):
            return self.__get_params_from_sklearn_model("theil_sen", model)
        elif isinstance(model, CatBoostRegressor):
            return self.__get_params_from_sklearn_model("catboost", model) # в catboost такой же метод для извлечения гиперпараметров
        elif isinstance(model, Holt) and isinstance(model_fit, statsmodels.tsa.holtwinters.results.HoltWintersResultsWrapper):
            return self.__get_params_from_statsmodels_model("holt", model_fit)
        elif isinstance(model, SimpleExpSmoothing) and isinstance(model_fit, statsmodels.tsa.holtwinters.results.HoltWintersResultsWrapper):
            return self.__get_params_from_statsmodels_model("exp_smoothing", model_fit)
        elif isinstance(model, ExponentialSmoothing) and isinstance(model_fit, statsmodels.tsa.holtwinters.results.HoltWintersResultsWrapper):
            return self.__get_params_from_statsmodels_model("holt_winters", model_fit)
        elif isinstance(model, SARIMAX) and isinstance(model_fit, statsmodels.tsa.statespace.sarimax.SARIMAXResultsWrapper):
            return self.__get_params_from_statsmodels_model("sarima", model_fit)
        elif isinstance(model, Prophet):
            return self.__get_params_from_prophet_model("prophet", model)
        else:
            raise ValueError("Неподдерживаемая модель")


    def save_trained_model(self, model, filepath: str):
        """
        Метод для сохранения обученной модели. В зависимости от модели 
        сохраняет либо саму модель, либо только её гиперпараметры. Если передаются гиперпараметры модели, необходимо
        передавать словарь вида {"model_name": [hyperparams]}. Если передаётся сама модель, то необходимо передавать сам объект.
        :param model: Обученная модель;
        :param filepath: Путь до файла. Возможна дозапись в конец файла. Не указывать расширение файла - оно будет выбрано автоматически.
        """
        self.filepath = filepath
        custom_models = ["croston_tsb", "rol_mean", "const", "symfit_fourier_fft", "random_forest", 
                         "elastic_net", "huber", "lasso", "ransac", 
                         "ridge", "theil_sen", "catboost", 
                         "holt", "exp_smoothing", "holt_winters", 
                         "sarima", "prophet", "polynomial"]
        if isinstance(model, dict):
            model_name = list(model.keys())[0] # т.к возвращается итерируемый объект
            params = list(model.values())[0]
            if model_name in custom_models:
                self.__save_custom_model(model_name, params)
            else:
                raise ValueError("Неподдерживаемая модель")
        elif isinstance(model, RandomForestRegressor):
            joblib.dump(model, f"RandomForestRegressor_{self.filepath}.joblib")
        elif isinstance(model, ElasticNet):
            joblib.dump(model, f"ElasticNet_{self.filepath}.joblib")
        elif isinstance(model, HuberRegressor):
            joblib.dump(model, f"HuberRegressor_{self.filepath}.joblib")
        elif isinstance(model, Lasso):
            joblib.dump(model, f"Lasso_{self.filepath}.joblib")
        elif isinstance(model, RANSACRegressor):
            joblib.dump(model, f"RANSACRegressor_{self.filepath}.joblib")
        elif isinstance(model, Ridge):
            joblib.dump(model, f"Ridge_{self.filepath}.joblib")
        elif isinstance(model, TheilSenRegressor):
            joblib.dump(model, f"TheilSenRegressor_{self.filepath}.joblib")
        elif isinstance(model, CatBoostRegressor):
            model.save_model(f"CatBoostRegressor_{self.filepath}", format="cbm")
        elif isinstance(model, Holt):
            joblib.dump(model, f"Holt_{self.filepath}.joblib")
        elif isinstance(model, SimpleExpSmoothing):
            joblib.dump(model, f"SimpleExpSmoothing_{self.filepath}.joblib")
        elif isinstance(model, ExponentialSmoothing):
            joblib.dump(model, f"ExponentialSmoothing_{self.filepath}.joblib")
        elif isinstance(model, SARIMAX):
            joblib.dump(model, f"SARIMAX_{self.filepath}.joblib")
        elif isinstance(model, Prophet):
            with open(f'Prophet_{self.filepath}.json', 'w') as fout:
                fout.write(model_to_json(model))
        else:
            raise ValueError("Неподдерживаемая модель")
                
def main():
    dataloader = DataLoader("/home/flowers/Uni/Practice/optimacros_practice/validation_system/test/om.csv")
    models, hyperparams = dataloader.load_hyperparams_from_optimacros()
    validator = Validation(models, hyperparams)
    validated_data = validator.get_validated_hyperparams()
    dataloader.backup_hyperparams(validated_data[0], validated_data[1])
    #list_a = ["new"]
    #print(list(list_a))

if __name__ == "__main__":
    main()
