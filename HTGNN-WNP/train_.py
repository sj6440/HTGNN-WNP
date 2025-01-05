import argparse
import os
import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from net.HTGNN_WNP import HTGNN_WNP
from torch.utils.data import DataLoader
from utils.dataloader import waterdataset
import csv
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def get_arguments():
    parser = argparse.ArgumentParser(description="This is a Waternet test model")
    parser.add_argument("--learning-rate", type=float, default=0.1, help="Base learning rate for training.")
    parser.add_argument("--loss", type=str, default="mse", choices=["L1", "mse", "SL1"], help="Waternet loss function.")
    parser.add_argument("--optimizer", type=str, default="Adadelta", choices=["SGD", "Adam", "RMSprop", "Adagrad", "Adadelta"], help="Waternet optimizer")
    parser.add_argument("--dataset", type=str, default='waterdataset', help="our dataset framework")
    parser.add_argument("--cuda", default=True, help="Run on CPU or GPU")
    parser.add_argument("--gpus", type=str, default="1",
                        help="choose gpu device. If run on CPU this arg can be ignored")
    parser.add_argument("--epoch", type=int, default=5000, help="Number of epochs to train.")
    parser.add_argument("--random_seed", type=int, default=3213165, help="Random seed.")
    return parser.parse_args()


def set_optimizer_lr(optimizer, Init_lr, epoch):
    decay_factor = 0.99 ** (epoch // 20)
    lr = Init_lr * decay_factor
    print("=======>> Current learning rate: '{}'".format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def configure_dataset_model(args):
    if args.dataset == 'waterdataset':
        args.batch_size = 100
        args.data_dir = 'Add the dataset path here.'
        args.restore_from = ''
        args.snapshot_dir = './weights'
    else:
        print("dataset error")


def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def main():
    args = get_arguments()
    print("=====> Configure dataset and model")
    configure_dataset_model(args)
    print(args)

    if args.cuda:
        print("=====> Set GPU for training")
        print("====> Use gpu id: '{}'".format(args.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
        print("=====> Set CPU for training")

    print("=====> Random Seed: ", args.random_seed)
    torch.manual_seed(args.random_seed)
    if args.cuda:
        torch.cuda.manual_seed(args.random_seed)

    if args.dataset == 'waterdataset':
        param_path = r"param"
        concentration_path = "lab"
        global_variant = "global_year"
        waterdata_train = waterdataset(args.data_dir, param_path, concentration_path, global_variant, train=True)
        waterdata_test = waterdataset(args.data_dir, param_path, concentration_path, global_variant, train=False)
        train_dataloader = DataLoader(waterdata_train, batch_size=args.batch_size, shuffle=False, num_workers=1)
        test_dataloader = DataLoader(waterdata_test, batch_size=args.batch_size, shuffle=False, num_workers=1)
    else:
        print("dataset error")

    net = HTGNN_WNP().to(device)
    # weights_file_path = "./weights/best_model_R_2:0.50.pth"
    # net.load_state_dict(torch.load(weights_file_path))

    if args.loss == 'L1':
        loss_fn = nn.L1Loss()
        print("=====> Using L1 loss function")
    elif args.loss == 'mse':
        loss_fn = nn.MSELoss()
        print("=====> Using MSE loss function")
    elif args.loss == 'SL1':
        loss_fn = nn.SmoothL1Loss()
        print("=====> Using SmoothL1Loss function")

    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=0.0)
        print("=====> Using SGD optimizer")
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate, betas=(0.9, 0.990), weight_decay=1e-5)
        print("=====> Using Adam optimizer")
    elif args.optimizer == 'RMSprop':
        optimizer = torch.optim.RMSprop(net.parameters(), lr=args.learning_rate, weight_decay=0.0)
        print("=====> Using RMSprop optimizer")
    elif args.optimizer == 'Adagrad':
        optimizer = torch.optim.Adagrad(net.parameters(), lr=args.learning_rate, lr_decay=0.0001, weight_decay=1e-5)
        print("=====> Using Adagrad optimizer")
    elif args.optimizer == 'Adadelta':
        optimizer = torch.optim.Adadelta(net.parameters(), lr=args.learning_rate, rho=0.9, eps=1e-8, weight_decay=0.0001)
        print("=====> Using Adadelta optimizer")



    epoch = args.epoch
    writer = SummaryWriter(log_dir="{}/logs".format(args.snapshot_dir))

    # Record the training results.
    training_results = []
    testing__results = []
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    result_dir = os.path.join(os.getcwd(), "result")

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    result_csv = os.path.join(result_dir, f"training_{current_time}.csv")
    with open(result_csv, mode='w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(["train_step", "loss", "mae", "mape", "rmse", "R_squared"])

    test_result = os.path.join(result_dir, f"testing_{current_time}.csv")
    with open(test_result, mode='w', newline='') as file:
        test_writer = csv.writer(file)
        test_writer.writerow(["epoch", "loss", "mae", "mape", "rmse", "R_squared"])


    best_R_squared = -100
    best_model_path = None

    for i in range(epoch):
        print("-------epoch  {} -------".format(i + 1))

        net.train()
        epoch_out = []
        epoch_targets = []
        total_training_loss = 0
        step_number = 0
        for step, [data_n, data_r, data_g, targets] in enumerate(train_dataloader):
            outputs, _ = net(data_r.to(device), data_g.to(device), data_n.to(device))
            targets = targets.to(device)
            loss = loss_fn(outputs, targets)
            epoch_out.append(outputs.cpu().detach().numpy())
            epoch_targets.append(targets.cpu().numpy())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_training_loss += loss.item()
            step_number += 1

        training_Loss = round(total_training_loss / step_number, 4)
        print("training  Loss: {:.4f}".format(training_Loss))
        lab = np.concatenate(epoch_targets, axis=0).flatten()
        pre = np.concatenate(epoch_out, axis=0).flatten()

        R_2 = r2_score(lab, pre)
        rmse = np.sqrt(mean_squared_error(lab, pre))
        mae = mean_absolute_error(lab, pre)
        mape_value = mape(lab, pre)
        print("R_squared：{:.2f}, RMSE：{:.2f}, MAE: {:.2f}, MAPE:{:.2f}".format(R_2, rmse, mae, mape_value))
        writer.add_scalar("train_mae", mae.item(), i)
        writer.add_scalar("train_mape", mape_value.item(), i)
        writer.add_scalar("train_rmse", rmse.item(), i)
        writer.add_scalar("trian_R_squared", R_2.item(), i)

        training_results.append([training_Loss, mae.item(), mape_value.item(), rmse.item(), R_2.item()])

        with open(result_csv, mode='a', newline='') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow([i, f"{loss.item():.4f}", mae.item(), mape_value.item(), rmse.item(), R_2.item()])

        #set_optimizer_lr(optimizer, args.learning_rate, i)

        print("====================Test starting=======================")
        net.eval()
        total_test_loss = 0
        epoch_num = 0
        epoch_out_test = []
        epoch_targets_test = []
        with torch.no_grad():
            for data_n, data_r, data_g, targets in test_dataloader:
                outputs, _ = net(data_r.to(device), data_g.to(device), data_n.to(device))
                targets = targets.to(device)

                epoch_out_test.append(outputs.cpu().numpy())
                epoch_targets_test.append(targets.cpu().numpy())

                loss = loss_fn(outputs, targets)
                total_test_loss += loss.item()
                epoch_num += 1

        lab_test = np.concatenate(epoch_targets_test, axis=0).flatten()
        pre_test = np.concatenate(epoch_out_test, axis=0).flatten()


        R_2_t = r2_score(lab_test, pre_test)
        rmse_t = np.sqrt(mean_squared_error(lab_test, pre_test))
        mae_t = mean_absolute_error(lab_test, pre_test)
        mape_value = mape(lab_test, pre_test)

        print("test set Loss: {:.4f}".format(total_test_loss / epoch_num))
        print("R_squared：{:.2f}, RMSE：{:.2f}, MAE: {:.2f}, MAPE:{:.2f}".format(R_2_t, rmse_t, mae_t, mape_value))

        writer.add_scalar("test_loss", total_test_loss / epoch_num, i)
        writer.add_scalar("test_mae", mae_t.item(), i)
        writer.add_scalar("test_mape", mape_value.item(), i)
        writer.add_scalar("test_rmse", rmse_t.item(), i)
        writer.add_scalar("R_squared", R_2_t.item(), i)

        testing__results.append(
            [(total_test_loss / epoch_num), mae_t.item(), mape_value.item(), rmse_t.item(), R_2_t.item()])

        with open(test_result, mode='a', newline='') as file:
            test_writer = csv.writer(file)
            test_writer.writerow(
                [i, f"{(total_test_loss / epoch_num):.4f}", mae_t.item(), mape_value.item(), rmse_t.item(),
                 R_2_t.item()])


        print("====================Test end============================")

        last_model_path = "{}/last_model.pth".format(args.snapshot_dir)
        torch.save(net.state_dict(), last_model_path)
        print("Last model saved.")

        if R_2_t > best_R_squared:
            best_R_squared = R_2_t
            best_model_path = "{}/best_model_R_2:{:.2f}.pth".format(args.snapshot_dir, best_R_squared)
            torch.save(net.state_dict(), best_model_path)
            print("New best model saved with R_squared: {:.2f}".format(best_R_squared))

        print("====================Save end============================")
        print("  ")
        print("  ")
        print("  ")
    writer.close()


if __name__ == '__main__':
    main()

