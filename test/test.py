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
        "return: Список моделей и список гиперпараметров
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
        
    def __check_value(self, number, left_border, right_border):
        """
        Метод для проверки, что число находится в допустимом диапазоне
        :param number: Число для проверки;
        :param left_boarder: Левая граница проверки;
        :param right_border: Правая граница проверки;
        """
        if not (number > left_border and number < right_border):
            raise ValueError("Число находятся вне разрешенного диапазона")
    
    def __check_values(self, list: list, left_border, right_border, default_value):
        """
        Метод для проверки, что числа в списке находятся в допустимом диапазоне. Если это не так,
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

    def __croston_tsb(self, data: dict):
        """
        Метод для проверки гиперпараметров модели Croston_tsb
        :param data: Гиперпараметры модели для валидации 
        """
        default_params = ["croston_tsb_min_alpha", "croston_tsb_max_alpha", 
                          "croston_tsb_min_beta", "croston_tsb_max_beta"]
        provided_params = list(data.keys())
        self.__check_lists_equal(default_params, provided_params)
        # задаём стандартные значения из документации
        default_min_alpha = 0
        default_max_alpha = 1
        default_min_beta = 0
        default_max_beta = 1
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
    
    def __elastic_net(self, data: dict):
        """
        Метод для проверки гиперпараметров модели Elastic_net
        :param data: Гиперпараметры модели для валидации 
        """
    
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


def main():
    filepath = '/home/flowers/Uni/Practice/optimacros_practice/test/om_models.csv'
    hyperparams = pd.read_csv(filepath)
    # переводим сразу из json в python-словари
    #print(hyperparams.loc[0, 'Params'])
    #json.loads(hyperparams.loc[2, 'Params'])
    hyperparams.loc[:, 'Params'] = hyperparams.loc[:, 'Params'].apply(json.loads)
    #hyperparams.loc[(hyperparams.Model == model), 'Params']
    #params = hyperparams.loc[:, 'Params'].to_list()
    models = hyperparams.loc[:, 'Model'].to_list()
    params = hyperparams.loc[(hyperparams.Model == models[0]), 'Params'].iloc[0][models[0]]
    print(params)
    #print(hyperparams.loc[(hyperparams.iloc[:, 2] == models[1]), "Параметры"].iloc[0][models[1]][0][0][models[1]])
    #print(hyperparams.loc[models[0]].iloc[1])
    #print(hyperparams.iloc[:, 2].to_list())
    

if __name__ == "__main__":
    main()
