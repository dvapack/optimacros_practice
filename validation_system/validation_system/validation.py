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
    
    def __check_list_is_subset(self, list_a: list, list_b: list):
        """
        Метод для проверки, что один список является подмножеством другого списка
        :param list_a: Исходный список;
        :param list_b: Список для проверки, является ли он подмножеством.
        """
        if not set(list_b).issubset(list_a):
            raise ValueError("В списке находятся недопустимые значения")
        
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

    def __rol_mean(self, data: dict):
        """
        Метод для проверки гиперпараметров модели Rol Mean
        :param data: Гиперпараметры модели для валидации 
        """
        default_params = ["rol_mean_min_window_size", "rol_mean_max_window_size", 
                          "rol_mean_weights_type", "rol_mean_weights_coeffs"]
        provided_params = list(data.keys())
        self.__check_lists_equal(default_params, provided_params)
        # задаём стандартные значения
        default_min_window_size = 1
        default_max_window_size = 10
        default_weights_coeffs = 2
        # проверка window size
        min_window_size = 1
        max_windows_size = 10 
        provided_min_window_size = [data.get("rol_mean_min_window_size")]
        self.__check_values(provided_min_window_size, min_window_size, max_windows_size, default_min_window_size)
        provided_max_window_size = [data.get("ridge_max_alpha")]
        # проверка, чтобы max_window_size был > min_window_size
        self.__check_values(provided_max_window_size, provided_min_window_size, max_windows_size, default_max_window_size)
        # проверка weight_coeffs
        min_weight_coeffs = 0
        max_weight_coeffs = 10 # поменять значение после ресерча
        provided_weight_coeffs = [data.get("rol_mean_weights_coeffs")]
        self.__check_values(provided_weight_coeffs, min_weight_coeffs, max_weight_coeffs, default_weights_coeffs) 
        # проверка weights_type
        default_weights_type = ["new"]
        provided_weigths_type = [data.get("rol_mean_weights_type")]
        self.__check_list_is_subset(default_weights_type, provided_weigths_type)

    def __theil_sen(self, data: dict):
        """
        Метод для проверки гиперпараметров модели Theil Sen
        :param data: Гиперпараметры модели для валидации 
        """
        default_params = ["theil_sen_min_degrees", "theil_sen_max_degrees"]
        provided_params = list(data.keys())
        self.__check_lists_equal(default_params, provided_params)
        # задаём стандартные значения
        default_min_degrees = 1
        default_max_degrees = 3
        # проверка alpha
        min_degrees = 1
        max_degrees = 5
        provided_min_degrees = [data.get("theil_sen_min_degrees")]
        self.__check_values(provided_min_degrees, min_degrees, max_degrees, default_min_degrees)
        provided_max_degrees = [data.get("theil_sen_max_degrees")]
        # проверка, чтобы max_degrees был > min_degrees
        self.__check_values(provided_max_degrees, provided_min_degrees, max_degrees, default_max_degrees)

    def __const(self, data: dict):
        """
        Метод для проверки гиперпараметров модели Const
        :param data: Гиперпараметры модели для валидации 
        """
        default_params = ["type"]
        provided_params = list(data.keys())
        self.__check_lists_equal(default_params, provided_params)
        # проверка weights_type
        default_type = ["median"]
        provided_type = [data.get("type")]
        self.__check_list_is_subset(default_type, provided_type)
  
    def __sarima(self, data: dict):
        """
        Метод для проверки гиперпараметров модели Sarima
        :param data: Гиперпараметры модели для валидации 
        """
        default_params = ["min_p", "max_p", "min_d", "max_d", "min_q",
                          "max_q", "min_P", "max_P", "min_D", "max_D",
                          "min_Q", "max_Q"]
        provided_params = list(data.keys())
        self.__check_lists_equal(default_params, provided_params)
        # задаём стандартные значения
        default_min_p = 0
        default_max_p = 1
        default_min_d = 0
        default_max_d = 1
        default_min_q = 0
        default_max_q = 1
        default_min_P = 0
        default_max_P = 1
        default_min_D = 0
        default_max_D = 1
        default_min_Q = 0
        default_max_Q = 1
        # проверка p
        min_p = 0
        max_p = 1
        provided_min_p = [data.get("min_p")]
        self.__check_values(provided_min_p, min_p, max_p, default_min_p)
        provided_max_p = [data.get("max_p")]
        # проверка, чтобы max_p был > min_p
        self.__check_values(provided_max_p, provided_min_p, max_p, default_max_p)
        # проверка d
        min_d = 0
        max_d = 1
        provided_min_d = [data.get("min_d")]
        self.__check_values(provided_min_d, min_d, max_d, default_min_d)
        provided_max_d = [data.get("max_d")]
        # проверка, чтобы max_d был > min_d
        self.__check_values(provided_max_d, provided_min_d, max_d, default_max_d)
        # проверка q
        min_q = 0
        max_q = 1
        provided_min_q = [data.get("min_q")]
        self.__check_values(provided_min_q, min_q, max_q, default_min_q)
        provided_max_q = [data.get("max_q")]
        # проверка, чтобы max_q был > min_q
        self.__check_values(provided_max_q, provided_min_q, max_q, default_max_q)
        # проверка P
        min_P = 0
        max_P = 1
        provided_min_P = [data.get("min_P")]
        self.__check_values(provided_min_P, min_P, max_P, default_min_P)
        provided_max_P = [data.get("max_P")]
        # проверка, чтобы max_P был > min_P
        self.__check_values(provided_max_P, provided_min_P, max_P, default_max_P)
        # проверка D
        min_D = 0
        max_D = 1
        provided_min_D = [data.get("min_D")]
        self.__check_values(provided_min_D, min_D, max_D, default_min_D)
        provided_max_D = [data.get("max_D")]
        # проверка, чтобы max_D был > min_D
        self.__check_values(provided_max_D, provided_min_D, max_D, default_max_D)
        # проверка Q
        min_Q = 0
        max_Q = 1
        provided_min_Q = [data.get("min_Q")]
        self.__check_values(provided_min_Q, min_Q, max_Q, default_min_Q)
        provided_max_Q = [data.get("max_Q")]
        # проверка, чтобы max_Q был > min_Q
        self.__check_values(provided_max_Q, provided_min_Q, max_Q, default_max_Q)

    def __prophet(self, data: dict):
        """
        Метод для проверки гиперпараметров модели Prophet
        :param data: Гиперпараметры модели для валидации 
        """
        default_params = ["seasonality_mode", "changepoint_prior_scale", 
                          "seasonality_prior_scale"]
        provided_params = list(data.keys())
        self.__check_lists_equal(default_params, provided_params)
        # задаём стандартные значения
        default_changepoint_prior_scale = 0.05
        default_seasonality_prior_scale = 5
        # проверка changepoint_prior_scale
        min_changepoint_prior_scale = 0.000001
        max_changepoint_prior_scale = 0.7
        provided_changepoint_prior_scale = [data.get("changepoint_prior_scale")]
        self.__check_values(provided_changepoint_prior_scale, min_changepoint_prior_scale, max_changepoint_prior_scale, default_changepoint_prior_scale)
        # проверка seasonality_prior_scale
        min_seasonality_prior_scale = 0.000001
        max_seasonality_prior_scale = 100
        provided_seasonality_prior_scale = [data.get("seasonality_prior_scale")]
        self.__check_values(provided_seasonality_prior_scale, min_seasonality_prior_scale, max_seasonality_prior_scale, default_seasonality_prior_scale) 
        # проверка seasonality_mode
        default_seasonality_mode = ["additive","multiplicative"]
        provided_seasonality_mode = [data.get("seasonality_mode")]
        self.__check_list_is_subset(default_seasonality_mode, provided_seasonality_mode)

    def __is_valid_max_features(self, max_features):
        """
        Метод для проверки гиперпараметра max_features в модели Random Forest
        :param max_features: Значение гиперпараметра.
        """
        if isinstance(max_features, int):
            if not 1 <= max_features:
                raise ValueError("Некорректное значение max_features")
        elif isinstance(max_features, float):
            if not 0.0 < max_features <= 1.0:
                raise ValueError("Некорректное значение max_features")
        elif max_features not in ["sqrt", "log2", None]:
            raise ValueError("Некорректное значение max_features")
        else:
            raise ValueError("Некорректное значение max_features")
        
    def __check_max_features_values(self, list: list, default_value):
        """
        Метод для проверки, что элементы в списке доступны для использования в качестве
        значения гиперпараметра max_features в модели Random Forest.
        :param list: Список чисел для проверки;
        :param default_value: Стандартное значение для замены некорректных данных.
        """
        for element in list:
            try:
                element = self.__is_valid_max_features(element)
            except ValueError:
                element = default_value

    def __random_forest(self, data: dict):
        """
        Метод для проверки гиперпараметров модели Random Forest
        :param data: Гиперпараметры модели для валидации 
        """
        default_params = ["max_features", "n_estimators", "max_depth",
                          "min_samples_split", "min_samples_leaf"]
        provided_params = list(data.keys())
        self.__check_lists_equal(default_params, provided_params)
        # задаём стандартные значения из документации
        default_max_features = 1.0 # из документации sklearn
        default_n_estimators = 100 # из документации sklearn
        default_max_depth = None # из документации sklearn
        default_min_samples_split = 2 # из документации sklearn
        default_min_samples_leaf = 1 # из документации sklearn
        # проверка max_features
        # возможные принимаемые значения {“sqrt”, “log2”, None}, int or float, default=1.0
        provided_max_features = data.get("max_features")
        self.__check_max_features_values(provided_max_features, default_max_features)
        # проверка n_estimators
        min_n_estimators = 1
        max_n_estimators = 100
        provided_n_estimators = data.get("n_estimators")
        self.__strong_check_values(provided_n_estimators, min_n_estimators, max_n_estimators, default_n_estimators)
        # проверка max_depth
        min_max_depth = 1
        max_max_depth = 1000
        provided_depth = data.get("max_depth")
        self.__check_values(provided_depth, min_max_depth, max_max_depth, default_max_depth)
        # проверка min_samples_split
        min_min_samples_split = 1
        max_min_samples_split = 100
        provided_min_samples_split = data.get("min_samples_split")
        self.__check_values(provided_min_samples_split, min_min_samples_split, max_min_samples_split, default_min_samples_split)
        # проверка min_samples_leaf
        min_min_samples_leaf = 1
        max_min_samples_leaf = 100
        provided_min_samples_leaf = data.get("min_samples_leaf")
        self.__check_values(provided_min_samples_leaf, min_min_samples_leaf, max_min_samples_leaf, default_min_samples_leaf)


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
        self.__strong_check_values(provided_learning_rate, min_learning_rate, max_learning_rate, default_learning_rate)
        # проверка n_estimators
        min_n_estimators = 1
        max_n_estimators = 50
        provided_n_estimators = data.get("n_estimators")
        self.__strong_check_values(provided_n_estimators, min_n_estimators, max_n_estimators, default_n_estimators)
        # проверка depth
        min_depth = 0
        max_depth = 16
        provided_depth = data.get("depth")
        self.__strong_check_values(provided_depth, min_depth, max_depth, default_depth)

    def __symfit_fourier_fft(self, data: dict):
        """
        Метод для проверки гиперпараметров модели Symfit Fourier FFT
        :param data: Гиперпараметры модели для валидации 
        """
        default_params = ["min_components", "max_components"]
        provided_params = list(data.keys())
        self.__check_lists_equal(default_params, provided_params)
        # задаём стандартные значения
        default_min_components = 1
        default_max_components = 10
        # проверка components
        min_components = 1
        max_components = 20
        provided_min_components = [data.get("min_components")]
        self.__check_values(provided_min_components, min_components, max_components, default_min_components)
        provided_max_components = [data.get("max_components")]
        # проверка, чтобы max_comppnents был > min_components
        self.__check_values(provided_max_components, provided_min_components, max_components, default_max_components)


    def __validate_hyperparam(self, model: str, param: dict):
        """
        Метод для валидации одного конкретного гиперпараметра
        :param param: Словарь из одного элемента с данными конкретного запуска модели
        """
        match model:
            case 'croston_tsb':
                self.__croston_tsb(param)
            case 'elastic_net':
                self.__elastic_net(param)
            case 'exp_smoothing':
                self.__exp_smoothing(param)
            case 'holt':
                self.__holt(param)
            case 'holt_winters':
                self.__holt_winters(param)
            case 'huber':
                self.__huber(param)
            case 'lasso':
                self.__lasso(param)
            case 'polynomial':
                self.__polynomial(param)
            case 'ransac':
                self.__ransac(param)
            case 'ridge':
                self.__ridge(param)
            case 'rol_mean':
                self.__rol_mean(param)
            case 'theil_sen':
                self.__theil_sen(param)
            case 'const':
                self.__const(param)
            case 'catboost':
                self.__catboost(param)
            case 'sarima':
                self.__sarima(param)
            case 'prophet':
                self.__prophet(param)
            case 'random_forest':
                self.__random_forest(param)
            case 'symfit_fourier_fft':
                self.__symfit_fourier_fft(param)
            case _:
                raise ValueError("Не поддерживаемая модель")


    def validate_hyperparams(self):
        """
        Метод для валидации списка гиперпараметров
        """
        for model, params in self.models, self.hyperparams:
            try:
                self.__validate_hyperparam(model, params)
            except ValueError as e:
                print(e)





 