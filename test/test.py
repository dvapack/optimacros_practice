import pandas as pd
import json


class DataLoader():
    """
    Класс для загрузки данных из csv файла.
    Методы в нём должны переопределяться под конкретный формат хранения данных в csv файле.
    """
    def __init__(self, filepath: str):
        self.filepath = filepath

    def __load_hyperparams(self, filepath: str):
        """
        Метод для загрузки гиперпараметров, полученных от OptiMacros
        :param filepath: Путь до файла csv с гиперпараметрами;
        """
        try:
            self.hyperparams = pd.read_csv(filepath)
            # переводим сразу из json в python-словари
            self.hyperparams.loc[:, "Параметры"] = self.hyperparams.iloc[:, "Параметры"].apply(json.loads)
            self.models = self.hyperparams.iloc[:, 2].to_list()
        except FileNotFoundError:
            print(f"Такого файла не существует")

    

class Validation():
    """
    Класс для валидации гиперпараметров
    """
    def __init__(self):
        self.models = None
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
            self.models = self.hyperparams.iloc[:, 2].to_list()
        except FileNotFoundError:
            print(f"Такого файла не существует")

    def __croston_tsb(self, data: dict):
        return
    
    def __elastic_net(self, data: dict):
        return
    
    def __catboost(self, data: dict):
        """
        Метод для проверки гиперпараметров модели Catboost
        """



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


    def __get_hyperparams(self, model: str):
        """
        Метод для извлечения гиперпараметров из json файла.
        Должен предопределяться программистом под определенный формат хранения данных в csv файле.
        :param model: Название модели
        """


    def validate_hyperparams(self, filepath: str):
        """
        Метод для валидации списка гиперпараметров
        :param filepath: Путь до файла csv с гиперпараметрами;
        """
        self.__load_hyperparams(filepath)
        for model in self.models:
            model_info = self.hyperparams.loc[(
                self.hyperparams.iloc[:, 2] == model), "Параметры"].iloc[0] # Извлекаем данные конкретной модели
            params = model_info[model][0][0][model] # Извлекаем именно гиперпараметры модели
            self.__validate_hyperparam(model, params)


def main():
    filepath = '/home/flowers/Uni/Practice/optimacros_practice/test/om_models.csv'
    hyperparams = pd.read_csv(filepath, encoding='cp1251')
    # переводим сразу из json в python-словари
    #print(hyperparams.loc[0, 'Params'])
    #json.loads(hyperparams.loc[2, 'Params'])
    hyperparams.loc[:, 'Params'] = hyperparams.loc[:, 'Params'].apply(json.loads)
    models = hyperparams.loc[:, 'Model'].to_list()
    print(hyperparams.head())
    #print(hyperparams.loc[(hyperparams.iloc[:, 2] == models[1]), "Параметры"].iloc[0][models[1]][0][0][models[1]])
    #print(hyperparams.loc[models[0]].iloc[1])
    #print(hyperparams.iloc[:, 2].to_list())
    

if __name__ == "__main__":
    main()
