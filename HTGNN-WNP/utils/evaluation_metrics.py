# import time
#
# import numpy as np
#
# def data_np(y_true, y_pred):
#    y_true =  [float(item[0]) for item in y_true]
#    np_true = np.array(y_true)
#
#    y_pred = [float(item[0].detach().numpy()) for item in y_pred]
#    np_pred = np.array(y_pred)
#    return np_true, np_pred
#
# def RMSE(y_true, y_pred):
#    return np.sqrt(((y_pred - y_true) ** 2).mean())
#
# def MAE(y_true, y_pred):
#    return np.mean(np.abs(y_pred - y_true))
#
#
# def MAPE(y_true, y_pred):
#    return np.mean(np.abs((y_pred - y_true) / y_true)) * 100


import numpy as np



def MAPE(y_true, y_pred):
    return round(np.mean(np.abs((y_pred - y_true) / y_true)) * 100, 2)

