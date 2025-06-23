import pandas as pd
import json


import pandas as pd
import json


class DataLoader():
    """
    Класс для загрузки данных из csv файла.
    Методы в нём должны переопределяться под конкретный формат хранения данных в csv файле.
    """
    def __init__(self, filepath: str):
        self.filepath = filepath

    def load_hyperparams(self) -> tuple[list, list]:
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
    
class Validation():
    """
    Класс для валидации гиперпараметров
    """
    def __init__(self, models: list, hyperparams: list):
        """
        :param models: Список моделей для валидации;
        :param hyperparams: Список соответствующих гиперпараметров для валидации
        """
        self.models = models
        self.hyperparams = hyperparams

    def __check_lists_equal(self, list_a: list, list_b: list):
        """
        Метод для проверки эквивалентности двух списков (порядок не важен)
        :param list_a: Первый список для сравнения;
        :param list_b: Второй список для сравнения.
        """
        if set(list_a) != set(list_b):
            raise ValueError("Элементы в списках не совпадают")
        
    def __strong_check_value(self, number, left_border, right_border):
        """
        Метод для проверки, что число строго находится в допустимом диапазоне, т.е > или <
        :param number: Число для проверки;
        :param left_boarder: Левая граница проверки;
        :param right_border: Правая граница проверки;
        """
        if not (number > left_border and number < right_border):
            raise ValueError("Число находятся вне разрешенного диапазона")
        
    def __check_value(self, number, left_border, right_border):
        """
        Метод для проверки, что число находится в допустимом диапазоне, т.е >= или <=.
        :param number: Число для проверки;
        :param left_boarder: Левая граница проверки;
        :param right_border: Правая граница проверки;
        """
        if not (number >= left_border and number <= right_border):
            raise ValueError("Число находятся вне разрешенного диапазона")
    
    def __strong_check_values(self, list: list, left_border, right_border, default_value):
        """
        Метод для проверки, что числа в списке строго находятся в допустимом диапазоне (> и <). Если это не так,
        число заменяется на стандартное значение
        :param list: Список чисел для проверки;
        :param left_boarder: Левая граница проверки;
        :param right_border: Правая граница проверки;
        :param default_value: Стандартное значение для замены некорректных данных.
        """
        for number in list:
            try:
                number = self.__check_value(number, left_border, right_border)
            except ValueError:
                number = default_value
    
    def __check_values(self, list: list, left_border, right_border, default_value):
        """
        Метод для проверки, что числа в списке находятся в допустимом диапазоне (>= и <=). Если это не так,
        число заменяется на стандартное значение
        :param list: Список чисел для проверки;
        :param left_boarder: Левая граница проверки;
        :param right_border: Правая граница проверки;
        :param default_value: Стандартное значение для замены некорректных данных.
        """
        for number in list:
            try:
                number = self.__strong_check_value(number, left_border, right_border)
            except ValueError:
                number = default_value

    def __croston_tsb(self, data: dict):
        """
        Метод для проверки гиперпараметров модели Croston TSB
        :param data: Гиперпараметры модели для валидации 
        """
        default_params = ["croston_tsb_min_alpha", "croston_tsb_max_alpha", 
                          "croston_tsb_min_beta", "croston_tsb_max_beta",
                          "croston_tsb_step"]
        provided_params = list(data.keys())
        self.__check_lists_equal(default_params, provided_params)
        # задаём стандартные значения
        default_min_alpha = 0
        default_max_alpha = 1
        default_min_beta = 0
        default_max_beta = 1
        default_step = 0.1
        # проверка alpha
        min_alpha = 0
        max_alpha = 1
        provided_min_alpha = [data.get("croston_tsb_min_alpha")]
        self.__check_values(provided_min_alpha, min_alpha, max_alpha, default_min_alpha)
        provided_max_alpha = [data.get("croston_tsb_max_alpha")]
        # проверка, чтобы max_alpha был > min_alpha
        self.__check_values(provided_max_alpha, provided_min_alpha, max_alpha, default_max_alpha)
        # проверка beta
        min_beta = 0
        max_beta = 1
        provided_min_beta = [data.get("croston_tsb_min_beta")]
        self.__check_values(provided_min_beta, min_beta, max_beta, default_min_beta)
        provided_max_beta = [data.get("croston_tsb_max_beta")]
        # проверка, чтобы max_beta был > min_beta
        self.__check_values(provided_max_beta, provided_min_beta, max_beta, default_max_beta)
        # проверка step
        min_step = 0
        max_step = 1
        provided_step = [data.get("croston_tsb_step")]
        self.__strong_check_values(provided_step, min_step, max_step, default_step)       
    
    def __elastic_net(self, data: dict):
        """
        Метод для проверки гиперпараметров модели Elastic Net
        :param data: Гиперпараметры модели для валидации 
        """
        default_params = ["elastic_net_min_alpha", "elastic_net_max_alpha", 
                          "elastic_net_min_l1", "elastic_net_max_l1",
                          "elastic_net_step"]
        provided_params = list(data.keys())
        self.__check_lists_equal(default_params, provided_params)
        # задаём стандартные значения
        default_min_alpha = 0
        default_max_alpha = 1
        default_min_l1 = 0
        default_max_l1 = 1
        default_step = 0.05
        # проверка alpha
        min_alpha = 0
        max_alpha = 5 # в документации sklearn до бесконечности
        provided_min_alpha = [data.get("elastic_net_min_alpha")]
        self.__check_values(provided_min_alpha, min_alpha, max_alpha, default_min_alpha)
        provided_max_alpha = [data.get("elastic_net_max_alpha")]
        # проверка, чтобы max_alpha был > min_alpha
        self.__check_values(provided_max_alpha, provided_min_alpha, max_alpha, default_max_alpha)
        # проверка l1
        min_l1 = 0
        max_l1 = 1
        provided_min_l1 = [data.get("elastic_net_min_l1")]
        self.__check_values(provided_min_l1, min_l1, max_l1, default_min_l1)
        provided_max_l1 = [data.get("elastic_net_max_l1")]
        # проверка, чтобы max_l1 был > min_l1
        self.__check_values(provided_max_l1, provided_min_l1, max_l1, default_max_l1)
        # проверка step
        min_step = 0
        max_step = 1
        provided_step = [data.get("croston_tsb_step")]
        self.__check_values(provided_step, min_step, max_step, default_step)         

    def __exp_smoothing(self, data: dict):
        """
        Метод для проверки гиперпараметров модели Expontetial Smoothing
        :param data: Гиперпараметры модели для валидации 
        """
        default_params = ["exp_smoothing_min_alpha", "exp_smoothing_max_alpha", 
                          "exp_smoothing_step"]
        provided_params = list(data.keys())
        self.__check_lists_equal(default_params, provided_params)
        # задаём стандартные значения
        default_min_alpha = 0
        default_max_alpha = 1
        default_step = 0.1
        # проверка alpha
        min_alpha = 0
        max_alpha = 1
        provided_min_alpha = [data.get("exp_smoothing_min_alpha")]
        self.__check_values(provided_min_alpha, min_alpha, max_alpha, default_min_alpha)
        provided_max_alpha = [data.get("exp_smoothing_max_alpha")]
        # проверка, чтобы max_alpha был > min_alpha
        self.__check_values(provided_max_alpha, provided_min_alpha, max_alpha, default_max_alpha)
        # проверка step
        min_step = 0
        max_step = 1
        provided_step = [data.get("exp_smoothing_step")]
        self.__check_values(provided_step, min_step, max_step, default_step) 

    def __holt(self, data: dict):
        """
        Метод для проверки гиперпараметров модели Holt
        :param data: Гиперпараметры модели для валидации 
        """
        default_params = ["holt_min_alpha", "holt_max_alpha",
                          "holt_min_beta", "holt_max_beta",
                          "holt_step"]
        provided_params = list(data.keys())
        self.__check_lists_equal(default_params, provided_params)
        # задаём стандартные значения
        default_min_alpha = 0
        default_max_alpha = 1
        default_min_beta = 0
        default_max_beta = 1
        default_step = 0.1
        # проверка alpha
        min_alpha = 0
        max_alpha = 1
        provided_min_alpha = [data.get("holt_min_alpha")]
        self.__check_values(provided_min_alpha, min_alpha, max_alpha, default_min_alpha)
        provided_max_alpha = [data.get("holt_max_alpha")]
        # проверка, чтобы max_alpha был > min_alpha
        self.__check_values(provided_max_alpha, provided_min_alpha, max_alpha, default_max_alpha)
        # проверка beta
        min_beta = 0
        max_beta = 1
        provided_min_beta = [data.get("holt_min_beta")]
        self.__check_values(provided_min_beta, min_beta, max_beta, default_min_beta)
        provided_max_beta = [data.get("holt_max_beta")]
        # проверка, чтобы max_beta был > min_beta
        self.__check_values(provided_max_beta, provided_min_beta, max_beta, default_max_beta)
        # проверка step
        min_step = 0
        max_step = 1
        provided_step = [data.get("holt_step")]
        self.__check_values(provided_step, min_step, max_step, default_step) 

    def __holt_winters(self, data: dict):
        """
        Метод для проверки гиперпараметров модели Holt Winters
        :param data: Гиперпараметры модели для валидации 
        """
        default_params = ["holt_winters_min_seasonality", "holt_winters_max_seasonality",
                          "holt_winters_trend_types", "holt_winters_seasonal_types"]
        provided_params = list(data.keys())
        self.__check_lists_equal(default_params, provided_params)
        # задаём стандартные значения
        default_trend_types = ["add", "mul", None]
        default_seasonal_types = ["add", "mul", None]
        default_sesonalities = {
            "Year": [4, 6, 12, 52, 365],
            "Week": [7, 14],
            "Daily": [24, 48]
        }

    def __huber(self, data: dict):
        """
        Метод для проверки гиперпараметров модели Huber
        :param data: Гиперпараметры модели для валидации 
        """
        default_params = ["huber_min_degrees", "huber_max_degrees"]
        provided_params = list(data.keys())
        self.__check_lists_equal(default_params, provided_params)
        # задаём стандартные значения
        default_min_degrees = 1
        default_max_degrees = 1.35 # в доке sklearn это стандартное значение
        # проверка alpha
        min_degrees = 1
        max_degrees = 100
        provided_min_degrees = [data.get("huber_min_degrees")]
        self.__check_values(provided_min_degrees, min_degrees, max_degrees, default_min_degrees)
        provided_max_degrees = [data.get("huber_max_degrees")]
        # проверка, чтобы max_degrees был > min_degrees
        self.__check_values(provided_max_degrees, provided_min_degrees, max_degrees, default_max_degrees)

    def __lasso(self, data: dict):
        """
        Метод для проверки гиперпараметров модели Lasso
        :param data: Гиперпараметры модели для валидации 
        """
        default_params = ["lasso_min_alpha", "lasso_max_alpha", 
                          "lasso_step"]
        provided_params = list(data.keys())
        self.__check_lists_equal(default_params, provided_params)
        # задаём стандартные значения
        default_min_alpha = 0
        default_max_alpha = 1
        default_step = 0.05
        # проверка alpha
        min_alpha = 0
        max_alpha = 5 # в документации sklearn до бесконечности 
        provided_min_alpha = [data.get("lasso_min_alpha")]
        self.__check_values(provided_min_alpha, min_alpha, max_alpha, default_min_alpha)
        provided_max_alpha = [data.get("lasso_max_alpha")]
        # проверка, чтобы max_alpha был > min_alpha
        self.__check_values(provided_max_alpha, provided_min_alpha, max_alpha, default_max_alpha)
        # проверка step
        min_step = 0
        max_step = 1
        provided_step = [data.get("lasso_step")]
        self.__check_values(provided_step, min_step, max_step, default_step) 

    def __polynomial(self, data: dict):
        """
        Метод для проверки гиперпараметров модели Polynomial
        :param data: Гиперпараметры модели для валидации 
        """
        default_params = ["polynomial_min_degrees", "polynomial_max_degrees"]
        provided_params = list(data.keys())
        self.__check_lists_equal(default_params, provided_params)
        # задаём стандартные значения
        default_min_degrees = 0
        default_max_degrees = 2 # в доке sklearn это стандартное значение
        # проверка alpha
        min_degrees = 0
        max_degrees = 5
        provided_min_degrees = [data.get("polynomial_min_degrees")]
        self.__check_values(provided_min_degrees, min_degrees, max_degrees, default_min_degrees)
        provided_max_degrees = [data.get("polynomial_max_degrees")]
        # проверка, чтобы max_degrees был > min_degrees
        self.__check_values(provided_max_degrees, provided_min_degrees, max_degrees, default_max_degrees)

    def __ransac(self, data: dict):
        """
        Метод для проверки гиперпараметров модели Ransac
        :param data: Гиперпараметры модели для валидации 
        """
        default_params = ["ransac_min_degrees", "ransac_max_degrees"]
        provided_params = list(data.keys())
        self.__check_lists_equal(default_params, provided_params)
        # задаём стандартные значения
        default_min_degrees = 1
        default_max_degrees = 2
        # проверка alpha
        min_degrees = 1
        max_degrees = 5
        provided_min_degrees = [data.get("ransac_min_degrees")]
        self.__check_values(provided_min_degrees, min_degrees, max_degrees, default_min_degrees)
        provided_max_degrees = [data.get("ransac_max_degrees")]
        # проверка, чтобы max_degrees был > min_degrees
        self.__check_values(provided_max_degrees, provided_min_degrees, max_degrees, default_max_degrees)

    def __ridge(self, data: dict):
        """
        Метод для проверки гиперпараметров модели Ridge
        :param data: Гиперпараметры модели для валидации 
        """
        default_params = ["ridge_min_alpha", "ridge_max_alpha", 
                          "ridge_step"]
        provided_params = list(data.keys())
        self.__check_lists_equal(default_params, provided_params)
        # задаём стандартные значения
        default_min_alpha = 0
        default_max_alpha = 1
        default_step = 0.05
        # проверка alpha
        min_alpha = 0
        max_alpha = 5 # в документации sklearn до бесконечности 
        provided_min_alpha = [data.get("ridge_min_alpha")]
        self.__check_values(provided_min_alpha, min_alpha, max_alpha, default_min_alpha)
        provided_max_alpha = [data.get("ridge_max_alpha")]
        # проверка, чтобы max_alpha был > min_alpha
        self.__check_values(provided_max_alpha, provided_min_alpha, max_alpha, default_max_alpha)
        # проверка step
        min_step = 0
        max_step = 1
        provided_step = [data.get("ridge_step")]
        self.__check_values(provided_step, min_step, max_step, default_step) 
  
    def __catboost(self, data: dict):
        """
        Метод для проверки гиперпараметров модели Catboost
        :param data: Гиперпараметры модели для валидации 
        """
        default_params = ["learning_rate", "n_estimators", "depth"]
        provided_params = list(data.keys())
        self.__check_lists_equal(default_params, provided_params)
        # задаём стандартные значения из документации
        default_learning_rate = 0.03
        default_n_estimators = 15 # в документации 1000, 
        # но по материалам, которые мне скинули - больше 30 не используется. Поэтому поставил среднее в 15
        default_depth = 6
        # проверка learning_rate
        min_learning_rate = 0
        max_learning_rate = 1
        provided_learning_rate = data.get("learning_rate")
        self.__check_values(provided_learning_rate, min_learning_rate, max_learning_rate, default_learning_rate)
        # проверка n_estimators
        min_n_estimators = 1
        max_n_estimators = 50
        provided_n_estimators = data.get("n_estimators")
        self.__check_values(provided_n_estimators, min_n_estimators, max_n_estimators, default_n_estimators)
        # проверка depth
        min_depth = 0
        max_depth = 16
        provided_depth = data.get("depth")
        self.__check_values(provided_depth, min_depth, max_depth, default_depth)


    def __validate_hyperparam(self, model: str, param: dict):
        """
        Метод для валидации одного конкретного гиперпараметра
        :param param: Словарь из одного элемента с данными конкретного запуска модели
        """
        if model == 'croston_tsb':
            self.__croston_tsb(param)
        elif model == 'elastic_net':
            self.__elastic_net(param)
        elif model == 'exp_smoothing':
            self.__exp_smoothing(param)
        elif model == 'holt':
            self.__holt(param)
        elif model == 'holt_winters':
            self.__holt_winters(param)
        elif model == 'huber':
            self.__huber(param)
        elif model == 'lasso':
            self.__lasso(param)
        elif model == 'polynomial':
            self.__polynomial(param)
        elif model == 'ransac':
            self.__ransac(param)
        elif model == 'ridge':
            self.__ridge(param)
        elif model == 'rol_mean':
            self.__rol_mean(param)
        elif model == 'theil_sen':
            self.__theil_sen(param)
        elif model == 'const':
            self.__const(param)
        elif model == 'catboost':
            self.__catboost(param)
        elif model == 'sarima':
            self.__sarima(param)
        elif model == 'prophet':
            self.__prophet(param)
        elif model == 'random_forest':
            self.__random_forest(param)
        elif model == 'symfit_fourier_fft':
            self.__symfit_fourier_fft(param)


    def validate_hyperparams(self):
        """
        Метод для валидации списка гиперпараметров
        """
        for model, params in self.models, self.hyperparams:
            self.__validate_hyperparam(model, params)





 