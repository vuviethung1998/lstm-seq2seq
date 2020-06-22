import logging
import os
import csv
import random
import pandas as pd
import pickle
import sys
import numpy as np
import tensorflow as tf
import scipy.sparse as sp
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.sparse import linalg
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime


def get_logger(log_dir, name, log_filename='info.log', level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Add file handler and stdout handler
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
    file_handler.setFormatter(formatter)
    # Add console handler.
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    # Add google cloud log handler
    logger.info('Log directory: %s', log_dir)
    return logger


def prepare_train_valid_test_2d(data, test_size, valid_size):
    train_len = int(data.shape[0] * (1 - test_size - valid_size))
    valid_len = int(data.shape[0] * valid_size)

    train_set = data[0:train_len]
    valid_set = data[train_len: train_len + valid_len]
    test_set = data[train_len + valid_len:]

    return train_set, valid_set, test_set


def create_data(data, seq_len, input_dim, output_dim, horizon):

    T = data.shape[0]
    # K = data.shape[1]

    # bm = binary_matrix(r, T, K)
    _data = data.copy()
    # _std = np.std(data)
    # _data[bm == 0] = np.random.uniform(_data[bm == 0] - _std, _data[bm == 0] + _std)
    en_x = np.zeros(shape=((T - seq_len - horizon), seq_len, input_dim))  # (T - seqlen - horizon) : so bo = T / (seqlen + horizon) - 1
    de_x = np.zeros(shape=((T - seq_len - horizon), horizon, output_dim))
    de_y = np.zeros(shape=((T - seq_len - horizon), horizon, output_dim))

    # lấy điện năng,
    load = _data[:, -1].copy()
    load = load.reshape(load.shape[0], 1)

    for i in range(T - seq_len - horizon):
        for j in range(input_dim):
            en_x[i, :, j] = _data[i: i + seq_len, j]

        de_x[i, :, :] = load[i + seq_len - 1:i + seq_len + horizon - 1]
        de_x[i, 0, :] = 0
        de_y[i, :, :] = load[i + seq_len:i + seq_len + horizon]

    return en_x, de_x, de_y


def load_dataset(seq_len, horizon, input_dim, output_dim, dataset, r, test_size, valid_size, type='power', **kwargs):
    data_raw = pd.read_csv(dataset)

    data_raw['timestamp'] = data_raw['date'].apply(getTimeStamp)
    data_raw['month'] = data_raw['date'].apply(getMonth)
    data_raw['dayOfYear'] = data_raw['date'].apply(getDayOfYear)
    data_raw['weekNum'] = data_raw['date'].apply(getWeekNum)

    _raw_data = data_raw.copy()
    if type=='power':
        raw_data = _raw_data['power']
    elif type == 'power_temp':
        raw_data = _raw_data[['temp', 'power']]
    elif type == 'power_holiday':
        raw_data = _raw_data[['holiday', 'power']]
    elif type == 'power_increase':
        raw_data = _raw_data[['increase', 'power']]
    elif type == 'power_timestamp':
        raw_data = _raw_data[['timestamp','power']]
    elif type == 'power_week':
        raw_data = _raw_data[['weekNum','power']]
    elif type == 'power_month':
        raw_data = _raw_data[['month','power']]
    elif type == 'power_month_holiday':
        raw_data = _raw_data[['month', 'holiday', 'power']]
    elif type == 'power_holiday_increase':
        raw_data = _raw_data[['increase', 'holiday','power']]
    elif type == 'temp_month':
        raw_data = _raw_data[['month', 'temp']]
    elif type =='power_temp_increase':
        raw_data = _raw_data[['increase', 'temp', 'power']]
    elif type == 'power_temp_holiday_increase':
        raw_data = _raw_data[['increase', 'holiday', 'temp', 'power']]
    elif type == 'power_month_holiday_increase':
        raw_data = _raw_data[['increase', 'holiday', 'month', 'power']]
    elif type == 'power_month_holiday_increase_temp':    
        raw_data = _raw_data[['increase', 'holiday', 'month', 'temp', 'power']]
    print('|--- Splitting train-test set.')

    train_data2d, valid_data2d, test_data2d = prepare_train_valid_test_2d(data=raw_data, test_size=test_size,
                                                                          valid_size=valid_size)

    # check shape of train_data2d if 1d -> reshape to 2d
    if len(train_data2d.shape) == 1:
        # print("train_data2d shape: " + str(len(train_data2d[])) )
        # print(train_data2d)
        train_data2d = np.array(train_data2d).reshape(len(train_data2d), 1)
        valid_data2d = np.array(valid_data2d).reshape(len(valid_data2d), 1)
        test_data2d  = np.array(test_data2d).reshape(len(test_data2d), 1)

    print('|--- Normalizing the train set.')
    data = {}
    scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
    scaler.fit(train_data2d)
    train_data2d_norm = scaler.transform(train_data2d)
    valid_data2d_norm = scaler.transform(valid_data2d)
    test_data2d_norm = scaler.transform(test_data2d)

    data['test_data_norm'] = test_data2d_norm.copy()
    data['valid_data_norm'] = valid_data2d_norm.copy()

    encoder_input_train, decoder_input_train, decoder_target_train = create_data(train_data2d_norm,
                                                                                 seq_len=seq_len,
                                                                                 input_dim=input_dim,
                                                                                 output_dim=output_dim,
                                                                                 horizon=horizon)
    encoder_input_val, decoder_input_val, decoder_target_val = create_data(valid_data2d_norm,
                                                                           seq_len=seq_len,
                                                                           input_dim=input_dim,
                                                                           output_dim=output_dim,
                                                                           horizon=horizon)
    encoder_input_eval, decoder_input_eval, decoder_target_eval = create_data(test_data2d_norm,
                                                                              seq_len=seq_len,
                                                                              input_dim=input_dim,
                                                                              output_dim=output_dim,
                                                                              horizon=horizon)

    for cat in ["train", "val", "eval"]:
        e_x, d_x, d_y = locals()["encoder_input_" + cat], locals()[
            "decoder_input_" + cat], locals()["decoder_target_" + cat]
        print(cat, "e_x: ", e_x.shape, "d_x: ", d_x.shape, "d_y: ", d_y.shape)
        data["encoder_input_" + cat] = e_x
        data["decoder_input_" + cat] = d_x
        data["decoder_target_" + cat] = d_y
    data['scaler'] = scaler

    return data


def cal_error(test_arr, prediction_arr, log_dir, alg):
    with np.errstate(divide='ignore', invalid='ignore'):
        # cal mse
        error_mae = mean_absolute_error(test_arr, prediction_arr)
        print('MAE: %.3f' % error_mae)

        # cal rmse
        error_mse = mean_squared_error(test_arr, prediction_arr)
        error_rmse = np.sqrt(error_mse)
        print('RMSE: %.3f' % error_rmse)

        # cal mape
        y_true, y_pred = np.array(test_arr), np.array(prediction_arr)
        error_mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        print('MAPE: %.3f' % error_mape)
        error_list = [error_mae, error_rmse, error_mape]
        with open(log_dir + alg + "_result.txt", 'a') as f:
            f.write("error_mae: " + str(error_mae) + "\n")
            f.write("error_rmse: " + str(error_rmse) + "\n")
            f.write("error_mape: " + str(error_mape) + "\n")
        return error_list


def binary_matrix(r, row, col):
    tf = np.array([1, 0])
    bm = np.random.choice(tf, size=(row, col), p=[r, 1.0 - r])
    return bm


def save_metrics(error_list, log_dir, alg):
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    error_list.insert(0, dt_string)
    with open(log_dir + alg + "_metrics.csv", 'a') as file:
        writer = csv.writer(file)
        writer.writerow(error_list)

def getMonth(_str):
    return _str.split("-")[1]

def getDayOfYear(_str):
    from datetime import datetime
    date = datetime.strptime(_str, '%Y-%m-%d')
    return date.timetuple().tm_yday

def getTimeStamp(_str):
    from datetime import datetime
    date = datetime.strptime(_str, '%Y-%m-%d')
    return int(datetime.timestamp(date)/1000)

def getWeekNum(_str):
    from datetime import datetime
    date = datetime.strptime(_str, '%Y-%m-%d')
    return datetime.date(date).isocalendar()[1]