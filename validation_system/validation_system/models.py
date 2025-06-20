import pandas as pd
import numpy as np

def croston_tsb_fit(timestamps: pd.Series, ts: pd.Series, alpha: float, beta: float, n_predict: int,
                    time_step: str, sample: pd.DataFrame) -> (pd.Series, dict):
    """
    Прогноз - метод Кростона.
    :param timestamps: Временные метки;
    :param ts: Временной ряд;
    :param alpha: Параметр сглаживания для уровня;
    :param beta: Параметр сглаживания для вероятности;
    :param n_predict: Количество предсказаний;
    :param time_step: Шаг прогноза;
    :param sample: Датафрейм с признаками (не используется);
    :return: Фрейм с предсказанием
    """
    d = np.array(ts)
    cols = len(d)
    d = np.append(d, [np.nan] * n_predict)

    # Уровень(a), Вероятность(p), Прогноз(f)
    a, p, f = np.full((3, cols + n_predict + 1), np.nan)

    # Инициализация
    first_occurrence = np.argmax(d[:cols] > 0)
    a[0] = d[first_occurrence]
    p[0] = 1 / (1 + first_occurrence)
    f[0] = p[0] * a[0]

    # Заполнение матриц уровня и вероятности, прогноз
    for t in range(cols):
        if d[t] > 0:
            a[t + 1] = alpha * d[t] + (1 - alpha) * a[t]
            p[t + 1] = beta * 1 + (1 - beta) * p[t]
        else:
            a[t + 1] = a[t]
            p[t + 1] = (1 - beta) * p[t]
        f[t + 1] = p[t + 1] * a[t + 1]
    a[cols + 1:cols + n_predict + 1] = a[cols]
    p[cols + 1:cols + n_predict + 1] = p[cols]
    f[cols + 1:cols + n_predict + 1] = f[cols]

    ts_res = pd.Series(index=range(cols + n_predict), dtype='float64')
    ts_res.loc[ts_res.index] = f[1:]
    timestamps_res = pd.date_range(start=timestamps.iloc[0], freq=time_step, periods=len(ts_res))
    dict_seasonality, marks_res = seasonal_coefficients(ts, timestamps, time_step, timestamps_res)
    df = pd.DataFrame({'y_pred': ts_res, 'indexes': marks_res})
    df.loc[cols + 1:cols + n_predict + 1, 'y_pred'] = df.iloc[cols + 1:cols + n_predict + 1].apply(
        lambda x: x.y_pred * dict_seasonality[x.indexes], axis=1)
    return df.y_pred.reset_index(drop=True)  # , {'alpha': alpha, 'beta': beta}

def rol_mean_fit(timestamps: pd.Series, time_step: str, ts: pd.Series, n_predict: int, window_size: int,
                 weights_coeffs: list, weights_type: str, sample: pd.DataFrame):
    """
    Создает скользящее среднее, формирует список компонентов

    :param ts: Временной ряд
    :param n_predict: Количество периодов для прогнозирования
    :param window_size: Размер окна
    :param weights_coeffs: Весовые коэффциенты
    :param weights_type: Тип весов
    :param sample: Датафрейм с признаками (не используется);
    :return: Прогноз
    """
    weights_coeffs = calculate_weights_coeffs(window_size, weights_type, weights_coeffs)
    ts_res = ts.copy()
    ts_base = None
    for i in range(n_predict):
        ts_res[len(ts_res.index)] = np.nan
        rol = ts_res.fillna(0).rolling(window_size)
        if i == 0:
            ts_base = rol.apply(lambda x: weighted_mean(x, weights_coeffs)).shift(1)[:-1]
            ts_base[:window_size] = ts[:window_size].values
        ts_res = ts_res.where(pd.notna, other=rol.apply(lambda x: weighted_mean(x, weights_coeffs)).shift(1))
    ts_res.loc[ts_base.index] = ts_base.values
    return ts_res

def const(ts: pd.Series, n_predict: int, type: str) -> (pd.Series, float):
    """
    Прогноз - константа
    :param ts: Временной ряд;
    :param n_predict: Количество предсказаний;
    :param type: Тип константы;
    :return: Фрейм с предсказанием
    """
    n = len(ts.index)
    value = None
    match (type):
        case 'Moda':
            value = ts.mode()[0]
        case 'Mean':
            value = ts.mean()
        case 'Min':
            value = ts.min()
        case 'Max':
            value = ts.max()
        case 'Median':
            value = ts.median()

    ts_res = pd.Series(value, index=range(n + n_predict), dtype='float64')
    return ts_res, value
