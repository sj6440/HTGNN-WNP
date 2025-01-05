import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from net.HTGNN_WNP import HTGNN_WNP
from utils.dataloader import waterdataset

import argparse
import csv
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def get_arguments():
    parser = argparse.ArgumentParser(description="This is a Waternet test model")
    parser.add_argument("--learning-rate", type=float, default=2e-3, help="Base learning rate for training.")
    parser.add_argument("--loss", type=str, default="L1", choices=["L1", "mse"], help="Waternet loss function.")
    parser.add_argument("--optimizer", type=str, default="Adam", help="Waternet optimizer")
    parser.add_argument("--dataset", type=str, default='waterdataset', help="our dataset framework")
    parser.add_argument("--cuda", default=True, help="Run on CPU or GPU")
    parser.add_argument("--gpus", type=str, default="0", help="choose gpu device. If run on CPU this arg can be ignored")
    parser.add_argument("--epoch", type=int, default=500, help="Number of epochs to train.")
    parser.add_argument("--random_seed", type=int, default=202408102059, help="Random seed.")
    return parser.parse_args()

def configure_dataset_model(args
):
    if args.dataset == 'waterdataset':
        args.batch_size = 100
        args.data_dir = 'Add the dataset path here.'
        args.restore_from = ''
        args.snapshot_dir = './weights'
    else:
        raise ValueError("Dataset error: Unsupported dataset specified.")

def mape(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def save_to_csv(lab_test, pre_test, output_dir='./result'):
    os.makedirs(output_dir, exist_ok=True)
    lab_test_path = os.path.join(output_dir, 'lab_test.csv')
    pre_test_path = os.path.join(output_dir, 'pre_test.csv')
    
    # Save lab_test
    with open(lab_test_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Index', 'Label'])
        for index, item in enumerate(lab_test):
            writer.writerow([index, item])
    
    # Save pre_test
    with open(pre_test_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Index', 'Prediction'])
        for index, item in enumerate(pre_test):
            writer.writerow([index, item])

def main():
    print("=====> Evaluate starting......")
    args = get_arguments()
    configure_dataset_model(args)

    device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")

    if args.dataset == 'waterdataset':
        test_dataset = waterdataset(args.data_dir, "param", "lab", "global_year", train=False)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)

    net = HTGNN_WNP().to(device)
    weights_file_path = "./weights/Add model weights."
    net.load_state_dict(torch.load(weights_file_path))
    net.eval()

    all_outputs = []
    all_targets = []

    with torch.no_grad():
        for data_n, data_r, data_g, targets in test_dataloader:
            outputs, _ = net(data_r.to(device), data_g.to(device), data_n.to(device))
            all_outputs.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    # Concatenate all batches
    all_outputs = np.concatenate(all_outputs, axis=0).flatten()
    all_targets = np.concatenate(all_targets, axis=0).flatten()
    


    # Calculate metrics
    r2 = r2_score(all_targets, all_outputs)
    rmse = np.sqrt(mean_squared_error(all_targets, all_outputs))
    mae = mean_absolute_error(all_targets, all_outputs)
    mape_value = mape(all_targets, all_outputs)

    print(f"R-squared: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MAPE: {mape_value:.4f}%")
    
    # Optionally, save results to CSV
    save_to_csv(all_targets, all_outputs)

if __name__ == '__main__':
    main()

