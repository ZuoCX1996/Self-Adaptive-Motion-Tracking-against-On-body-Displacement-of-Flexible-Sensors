import torch.nn.functional as F
import torch.nn as nn
import torch
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd
from utils import *
from model.SeqModel import model_eval
from model.SeqModel import LSTM_train
from Aplus.data import BaseDataset


### 进行多轮域自适应实验 ###

# 模型训练


DATA_ROOT_PATH = './data'


# 实验设置
BATCH_SIZE = 128
ENCODE_DEPTH = 1
ffe = False
split_rate=0.5
Y_FRAME = 1
csv_path = "./data/singlesj_clean.csv"
test_round = 20

# 选择数据的传感器数据
feat_col=['r1', 'r6']
# feat_col=['r1', 'r2', 'r5', 'r6']
# feat_col=['r1', 'r2', 'r3', 'r4', 'r5', 'r6']

if __name__ == "__main__":
    np.random.seed(42)

    # 记录每组数据自适应的结果
    result_all = []

    # 创建输实验结果记录文件
    result_column = ['数据段', 'mae', 'std', 'p25', 'p50', 'p75',
                     'mae_2', 'std_2', 'p25_2', 'p50_2', 'p75_2']
    result_df = pd.DataFrame(columns=result_column)

    # 进行n轮实验
    for round_idx in range(test_round):

        torch.manual_seed(42)

        print(f'{round_idx + 1} | {test_round}')

        # seed取0-10000之间的随机数 用于数据及随机划分
        seed = np.random.randint(10000)
        print(f'seed {seed}')
        dataset = Smartpad_Data(csv_path=csv_path, split_rate=split_rate, seed=seed, feat_col=feat_col, y_len=Y_FRAME)
        X, Y = dataset.get_train_data()
        X_test, Y_test = dataset.get_test_data()


        # 初始模型训练
        model = LSTM_train(X=X, Y=Y, X_TEST=X_test, Y_TEST=Y_test, Y_FRAME=Y_FRAME, num_epoch=30, restore=False, ffe=ffe,
                           filename='lstm',
                           lr=1e-3, use_bn=False, verbose=False)


        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # 遍历每组测试集数据
        for test_group_idx in range(dataset.test_size):
            result_group = [dataset.test_group[test_group_idx]]
            print(f'{test_group_idx+1} | {dataset.test_size}')

            X_test, Y_test = dataset.get_test_data(test_group_idx)
            val = MyDataset(X_test, Y_test)
            val_loader = DataLoader(val, batch_size=128, shuffle=False)


            #读取初始模型参数
            model.restore()
            # model.encode_layer.print_value()

            # 记录初始模型在本组数据上的误差
            print('适应前: ')
            pred, real, result_1 = model_eval(model=model, data_loader=val_loader, device=device)
            result_group += result_1

            # 各版本的域自适应


            model.transfer_DAN(x_source=X, y_source=Y, x_target=X_test, lr=1e-3, batch_size=128, epoch=50)
            # model.transfer_DANN(x_source=X, y_source=Y, x_target=X_test, lr=1e-3, batch_size=128, epoch=50)
            # model.transfer_DSAN(x_source=X, y_source=Y, x_target=X_test, lr=1e-3, batch_size=128, epoch=50)


            # data_source = MyDataset(X, Y)
            # model.transfer_CORAL(data_source=data_source, data_target=val, x_target=X_test, lr=1e-3, batch_size=128, epoch=50)
            # model.transfer_AdvSKM(data_source=data_source, data_target=val, lr=1e-3, batch_size=128, epoch=50)
            # model.transfer_HoMM(data_source=data_source, data_target=val, lr=1e-3, batch_size=128, epoch=20)


            # 记录自适应后的预测误差
            print('适应后: ')
            pred, real, result_2 = model_eval(model=model, data_loader=val_loader, device=device)
            result_group += result_2
            model.affine_layer.print_value()
            # model.encode_layer.print_value()
            print('————————————————————————————————————————————————')

            result_all.append(result_group)

    # 记录实验结果至excel表格
    result_df[result_column[0]] = np.array(result_all)[:, 0]
    result_df[result_column[1:]] = np.array(result_all)[:, 1:].astype(np.float32)
    # 记录自适应后的mae下降值
    result_df['mae_reduction'] = result_df['mae'] - result_df['mae_2']

    err_record = np.array(result_df['mae_2'])
    err = err_record.mean()
    std = err_record.std()

    print(f'mae:{err}, std:{std}')

    result_df.to_excel('./result.xlsx')
#
