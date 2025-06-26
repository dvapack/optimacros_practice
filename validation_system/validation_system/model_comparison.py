import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from typing import Optional

class ModelComparator:
    """
    Класс для сравнения:
    - Гиперпараметров моделей с эталонными значениями
    - Гиперпараметров разных версий моделей
    - Метрик качества (WAPE, MAPE)
    - Визуализации результатов
    """
    
    def __init__(self):
        self.results = pd.DataFrame(columns=['Test', 't-stat', 'p-value', 'Result'])
        
    def compare_with_baseline(self, current_params: dict, baseline_params: dict, alpha: float = 0.05):
        """
        Сравнение параметров текущей модели с эталонными значениями (t-тест)
        
        :param current_params: Текущие параметры модели {'param1': value1, ...}
        :param baseline_params: Эталонные параметры {'param1': value1, ...}
        :param alpha: Уровень значимости
        """
        for param in baseline_params:
            if param in current_params:
                current_vals = current_params[param]
                baseline_vals = baseline_params[param]
                
                t_stat, p_val = ttest_ind(current_vals, baseline_vals)
                result = "Разные" if p_val < alpha else "Одинаковые"
                
                self.results.loc[len(self.results)] = [f"Сравнение с эталонным {param}", t_stat, p_val, result]
    
    def compare_versions(self, version1_params: dict, version2_params: dict, alpha: float = 0.05):
        """
        Сравнение параметров двух версий модели
        
        :param version1_params: Параметры первой версии
        :param version2_params: Параметры второй версии
        :param alpha: Уровень значимости
        """
        common_params = set(version1_params) & set(version2_params)
        for param in common_params:
            v1_vals = version1_params[param]
            v2_vals = version2_params[param]
            
            t_stat, p_val = ttest_ind(v1_vals, v2_vals)
            result = "Разные" if p_val < alpha else "Одинаковые"
            
            self.results.loc[len(self.results)] = [f"Сравнение версий {param}", t_stat, p_val, result]
    
    def compare_metrics(self, metrics_v1: dict, metrics_v2: dict, alpha: float = 0.05):
        """
        Сравнение метрик (WAPE, MAPE) между двумя версиями
        
        :param metrics_v1: Метрики первой версии {'WAPE': [vals], 'MAPE': [vals]}
        :param metrics_v2: Метрики второй версии
        :param alpha: Уровень значимости
        """
        for metric in metrics_v1:
            if metric in metrics_v2:
                t_stat, p_val = ttest_ind(metrics_v1[metric], metrics_v2[metric])
                result = "Разные" if p_val < alpha else "Одинаковые"
                
                self.results.loc[len(self.results)] = [f"Метрика {metric}", t_stat, p_val, result]
    
    def visualize_comparison(self, save_path: Optional[str] = None) -> None:
        """
        Визуализация результатов сравнения
        
        :param save_path: Путь для сохранения графиков (если None - показ в окне)
        """
        plt.figure(figsize=(12, 6))
        
        # p-values
        plt.subplot(1, 2, 1)
        significant = self.results['p-value'] < 0.05
        colors = ['red' if s else 'green' for s in significant]
        
        plt.barh(self.results['Test'], self.results['p-value'], color=colors)
        plt.axvline(0.05, color='black', linestyle='--')
        plt.title('Statistical significance (p-values)')
        plt.xlabel('p-value')
        plt.ylabel('Test')
        
        # метрики
        if 'Metric WAPE' in self.results['Test'].values:
            plt.subplot(1, 2, 2)
            wape_row = self.results[self.results['Test'] == 'Metric WAPE'].iloc[0]
            mape_row = self.results[self.results['Test'] == 'Metric MAPE'].iloc[0]
            
            plt.bar(['WAPE', 'MAPE'], 
                    [wape_row['Statistic'], mape_row['Statistic']],
                    color=['skyblue', 'orange'])
            plt.title('Metric comparison (t-statistics)')
            plt.ylabel('t-statistic')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def get_results(self) -> pd.DataFrame:
        """Возвращает DataFrame с результатами сравнений"""
        return self.results

