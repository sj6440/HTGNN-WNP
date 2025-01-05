import csv
import torch
import time

def read_param_data(data_path):

    with open(data_path,encoding='utf-8') as f:
        data_csv = csv.DictReader(f)
        data_list = []
        for row in data_csv:
            numerical_data = [float(value) for value in row.values() if value.replace('.', '', 1).isdigit()]
            data_list.append(numerical_data)
    tensor_data = torch.tensor(data_list, dtype=torch.float32)
    return tensor_data

# def read_yearlab_data(data_path):
#     with open(data_path,encoding='utf-8') as f:
#         data_csv = csv.reader(f)
#         data_list = []
#
#         for row in data_csv:
#             numerical_data = [float(value) for value in row if value.replace('.', '', 1).isdigit()]
#             data_list.append(numerical_data)
#     print(data_list)
#     tensor_data = torch.tensor(data_list, dtype=torch.float32)
#     print(tensor_data)
#     time.sleep(666)
#     return tensor_data

# def read_yearlab_data(data_path):
#     print(data_path)
#     with open(data_path,encoding='utf-8') as f:
#         data_csv = csv.DictReader(f)
#         data_list = []
#         for row in data_csv:
#             numerical_data = [float(value) for value in row.values() if value.replace('.', '', 1).isdigit()]
#             data_list.append(numerical_data)
#         print(data_list)
#     tensor_data = torch.tensor(data_list, dtype=torch.float32)
#     print(tensor_data)
#     time.sleep(666)
#     return tensor_data

def read_yearlab_data(data_path):
    with open(data_path, encoding='utf-8') as f:
        data_csv = csv.reader(f)
        data_list = []
        for row in data_csv:
            numerical_data = []
            for value in row:
                try:
                    numerical_data.append(float(value))
                except ValueError:
                    numerical_data.append(0.0)
            data_list.append(numerical_data)
    tensor_data = torch.tensor(data_list, dtype=torch.float32)

    return tensor_data