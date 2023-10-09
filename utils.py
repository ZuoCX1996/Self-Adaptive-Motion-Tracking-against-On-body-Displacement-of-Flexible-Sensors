import numpy as np
import pandas as pd
import time
from scipy.signal import butter, lfilter
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from torch import nn
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import math

# ----------各类误差指标----------
def mae(pred, real, max_cut=160):
    """
    计算平均误差

    :param pred: list, 预测序列
    :param real: list, 真实序列

    """

    pred = np.array(pred).reshape(-1)
    real = np.array(real).reshape(-1)

    # if max_cut is not None:
    #     bool_index = real < max_cut
    #     pred = pred[bool_index]
    #     real = real[bool_index]


    err = np.array(real).reshape(-1) - np.array(pred).reshape(-1)
    err_abs = np.abs(err)
    return err_abs.mean()

def corr(pred, real):
    data_mat = [np.array(real).reshape(-1), np.array(pred).reshape(-1)]
    return np.corrcoef(data_mat)[0, 1]

def mae_each(pred, real):
    """
    计算平均误差

    :param pred: list, 预测序列
    :param real: list, 真实序列

    """
    pred = np.array(pred).reshape(-1)
    real = np.array(real).reshape(-1)

    # max_cut = 160
    #
    # if max_cut is not None:
    #     bool_index = real < max_cut
    #     pred = pred[bool_index]
    #     real = real[bool_index]

    err = np.array(real).reshape(-1) - np.array(pred).reshape(-1)
    err_abs = np.abs(err)

    return err_abs



# ----------信号处理/分析相关-----------

def position_encoding(x, noise=None):
    # x = x * np.pi / 60
    # cos_encode = np.cos(x)
    # sin_encode = np.sin(x)
    # encode = np.concatenate([cos_encode, sin_encode], axis=2)

    idf = 2
    x = x * np.pi / 60
    if noise is not None:
        x += (np.random.rand(x.shape[0], 1, x.shape[2]) - 0.5) * noise
    cos_encode = np.concatenate([np.cos(x*2**i) for i in range(idf)], axis=2)
    sin_encode = np.concatenate([np.sin(x*2**i) for i in range(idf)], axis=2)
    encode = np.concatenate([cos_encode, sin_encode], axis=2)
    return encode

def butter_lowpass_filter(data, cutoff, fs, order=5, delay_fix=2):
    # 参数说明：
    # data：输入序列，为1维numpy
    #
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    data_0 = data[0]
    y = lfilter(b, a, data - data_0) + data_0
    y = np.concatenate([y[delay_fix:], data[-delay_fix:]], axis=0)
    return y

def bidirectional_lowpass_filter(data, cutoff, fs, order=5):
    # 参数说明：
    # data：输入序列，为1维numpy
    #
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    data_0 = data[0]
    y = lfilter(b, a, data - data_0) + data_0

    data = data[::-1]
    data_0 = data[0]
    y_reverse = lfilter(b, a, data - data_0) + data_0

    y = (y + y_reverse[::-1]) / 2

    # print(y.shape)

    # y = np.concatenate([y[delay_fix:], data[-delay_fix:]], axis=0)
    return y

def bidirectional_filter_for_tensor(tensor_X, cutoff=3, fs=50, order=3):
    out_list = []
    tensor_X = tensor_X.detach().cpu()
    for channel in range(tensor_X.shape[1]):
        seq = np.array(tensor_X[:, channel])
        # plt.plot(seq)
        # seq = bidirectional_lowpass_filter(data=seq, cutoff=cutoff, fs=fs, order=order)
        seq = butter_lowpass_filter(data=seq, cutoff=cutoff, fs=fs, order=order,delay_fix=2)
        # plt.plot(seq)
        # plt.show()
        out_list.append(torch.tensor(seq, dtype=torch.float32).unsqueeze(dim=1))
    out = torch.cat(out_list, dim=1)
    return out

# 计算大于cutoff的信号功率
def getSignalPower(data, sampling_rate, fft_size=100, cutoff=3):
    for i in range(len(data) // fft_size):
        if i == 0:
            xf = np.fft.rfft(data[fft_size * i:fft_size * (i + 1)])  # 进行FFT
        else:
            xf += np.fft.rfft(data[fft_size * i:fft_size * (i + 1)])  # 进行FFT
    xf = xf / ((i + 1) * fft_size)
    freqs = np.linspace(0, sampling_rate / 2, int(fft_size / 2 + 1)) #频率

    selected_freqs = freqs.copy()
    selected_freqs[selected_freqs <= cutoff] = 0
    selected_freqs[selected_freqs > cutoff] = 1
    # print(selected_freqs)

    xfp = abs(xf)  # 幅度值
    xfp = xfp**2

    xfp = xfp*selected_freqs

    return np.sum(xfp)

def get_spectrum(data, sampling_rate=50, fft_size=400):
    data=np.array(data).reshape(-1)
    for i in range(len(data) // fft_size):
        if i == 0:
            xf = np.fft.rfft(data[fft_size * i:fft_size * (i + 1)])  # 进行FFT
        else:
            xf += np.fft.rfft(data[fft_size * i:fft_size * (i + 1)])  # 进行FFT
    xf = xf / ((i + 1) * fft_size)
    xf = abs(xf)
    freqs = np.linspace(0, sampling_rate / 2, int(fft_size / 2 + 1)) #频率
    return freqs, xf


def getSignalSNR(data, sampling_rate, fft_size=256, cutoff=3, p_out=False):
    for i in range(len(data) // fft_size):
        if i == 0:
            xf = np.fft.rfft(data[fft_size * i:fft_size * (i + 1)])  # 进行FFT
        else:
            xf += np.fft.rfft(data[fft_size * i:fft_size * (i + 1)])  # 进行FFT
    xf = xf / ((i + 1) * fft_size)
    freqs = np.linspace(0, sampling_rate / 2, int(fft_size / 2 + 1)) #频率

    noise_freqs = freqs.copy()
    noise_freqs[noise_freqs <= cutoff] = 0
    noise_freqs[noise_freqs > cutoff] = 1

    signal_freqs = freqs.copy()
    signal_freqs[signal_freqs <= cutoff] = 1
    signal_freqs[signal_freqs > cutoff] = 0
    signal_freqs[0] = 0  #直流分量不记录在内


    xfp = abs(xf)  # 幅度值
    xfp = xfp**2   # 功率谱

    if p_out:
        return np.sum(xfp*signal_freqs), np.sum(xfp*noise_freqs)
    snr = np.sum(xfp*signal_freqs) / np.sum(xfp*noise_freqs)

    return 10 * np.log10(snr)

# 对np array进行线性插值
def np_linear_interpolation(x, n_interp):
    data_len = len(x)
    x_t = x[:-1]
    x_t_plus1 = x[1:]

    interp_x = []
    for i in range(n_interp + 1):
        ratio  = 1 - i * (1/(n_interp + 1))
        temp = ratio*x_t + (1-ratio)*x_t_plus1
        interp_x.append(temp)
        # print(temp)
    interp_x = np.array(interp_x)
    interp_x = interp_x.swapaxes(0, 1)
    # print(interp_x)

    result = None
    for i in range(interp_x.shape[0]):
        if i == 0:
            result = interp_x[i]
        else:
            result = np.concatenate([result, interp_x[i]], axis=0)

    result = np.concatenate([result, x[-1:]], axis=0)
    # print(result)
    return result

# 对np array进行分段线性插值
def np_linear_interpolation_in_seg(x, n_interp, seg_len):
    result = None
    for i in range(len(x)//seg_len):
        seg_interp = np_linear_interpolation(x[i*seg_len:(i+1)*seg_len], n_interp=n_interp)
        if i == 0:
            result = seg_interp
        else:
            result = np.concatenate([result, seg_interp], axis=0)

    result = np.concatenate([result, x[-1:]], axis=0)
    # print(result)
    return result



# ----------数据读取相关-----------
def idexer(idx):
    idx = idx
    group_idx = idx // 6
    inner_idx = (idx) % 6
    np.random.seed(42)
    for i in range(group_idx+1):
        seed = np.random.randint(10000)
        print(seed)

    return seed, inner_idx


# 生成随机索引值（不重复）
def random_index(data_len, sample_rate=1.0, seed=None):
    index = [i for i in range(data_len)]
    df_index = pd.DataFrame({'index': index})
    # print(df_index)
    if seed is not None:
        rand_select = np.array(df_index.sample(frac=sample_rate, random_state=seed)['index']).tolist()
    else:
        rand_select = np.array(df_index.sample(frac=sample_rate, random_state=int(time.time()))['index']).tolist()

    # print(rand_select)

    return rand_select

def np_downsample(data, label, sample_rate=0.5, seq_len=None, seed=114514, return_res=False):
    data_len = len(label)

    if seq_len is None:
        rand_select = random_index(data_len=data_len, sample_rate=sample_rate, seed=seed)
    else:
        rand_select_begin_index = random_index(data_len=data_len//seq_len, sample_rate=sample_rate, seed=seed)
        # rand_select_begin_index = np.random.randint(0, data_len//seq_len, [int(data_len*sample_rate/seq_len)])
        # print(rand_select_begin_index)
        rand_select = []
        for i in rand_select_begin_index:
            rand_select += [i*seq_len + j for j in range(seq_len)]

    # print(rand_select)
    X = data[rand_select]
    Y = label[rand_select]

    if return_res == True:
        set_index_all = set([i for i in range(data_len)])
        set_rand_select = set(rand_select)
        rand_not_select = list(set_index_all - set_rand_select)


        X_res = data[rand_not_select]
        Y_res = label[rand_not_select]

        # print(len(X_res))
        return X, X_res, Y, Y_res

    return X, Y

def read_data(x_path, y_path, pe=False, noise=None):
    x_train = np.load(
        x_path,
        allow_pickle=True)
    y_train = np.load(
        y_path,
        allow_pickle=True)

    #简单标准化，待修改
    if pe:
        # data = position_encoding(x_train, noise=noise)
        data =x_train
    else:
        # data = (x_train - 640) / 120
        data = x_train
    # data=data[:,:,:,np.newaxis]

    label=y_train
    # label=np.reshape(label,[label.shape[0],1])


    # label=np.transpose(label)

    return data, label

def multi_group_data_split(angle_list):
    DATA_ROOT_PATH = 'D:\Dataset\SmartPad数据/regression'

    X, Y = None, None
    X_TEST, Y_TEST = None, None
    for angle in angle_list:
        X_PATH = DATA_ROOT_PATH + f"/multimt_x_{angle}.npy"
        Y_PATH = DATA_ROOT_PATH + f"/multimt_y_{angle}.npy"
        x, y = read_data(X_PATH, Y_PATH)

def seq2batch(X, Y, sample_n=10, lowpass=False, y_len=1, label_fix=False):
    X = np.array(X)
    Y = np.array(Y)

    if lowpass:
        for i in range(X.shape[1]):
            X[:, i] = butter_lowpass_filter(data=X[:, i], cutoff=3, fs=50, order=3, delay_fix=5)
            # X[:, i] = bidirectional_lowpass_filter(data=X[:, i], cutoff=3, fs=50, order=3)
    # plt.plot(Y[:, 0])

    if label_fix:
        Y_filter = butter_lowpass_filter(data=Y[:, 0], cutoff=5, fs=50, order=3, delay_fix=3)
        for i in range(len(Y)):
            if Y[i, 0] >= 125:
                if Y[i, 0] < 145:
                    weight = (Y[i, 0] - 125) / 20
                    Y[i, 0] = weight * Y_filter[i] + (1-weight) * Y[i, 0]
                else:
                    Y[i, 0] = Y_filter[i]

    # plt.plot(Y[:, 0])
    # # plt.plot(Y_filter)
    # plt.show()
    X = X[:-5, :]
    Y = Y[:-5]


    X = X.tolist()
    Y = Y.tolist()

    sample_total = len(X)

    X_batch = []
    Y_batch = []

    for i in range(sample_total - sample_n + 1):
        X_batch.append(X[i:i+sample_n])
        Y_batch.append(Y[i+sample_n-y_len:i+sample_n])

    return X_batch, Y_batch

class MyDataset(Dataset):
    def __init__(self, X, y=None):
        # 数据转换为tensor

        self.data = torch.FloatTensor(X)

        # label数据归一化
        self.label = torch.FloatTensor(y) / 180


    def __getitem__(self, idx):
        if self.label is not None:
            return self.data[idx], self.label[idx]
        else:
            return self.data[idx]

    def __len__(self):
        return len(self.data)


class Smartpad_Data():
    def __init__(self, csv_path="./data/singlesj.csv", split_rate=0.2, seed=42, lowpass=True,
                 feat_col=['r1', 'r6'], y_len=10, x_len=10):
        FEAT_COL = feat_col
        CAT_NAME = 'rawname'
        LABEL_COL = ['angle']

        self.FEAT_COL = FEAT_COL
        self.CAT_NAME = CAT_NAME
        self.LABEL_COL = LABEL_COL
        self.lowpass = lowpass
        self.data_path = csv_path
        self.x_len = x_len
        self.y_len = y_len

        # 序列开头存在异常 可舍去
        self.head_cut = 50

        self.cut_group = ['CZP_71_0']

        data = pd.read_csv(csv_path)
        data = data.loc[(data['lateral'] >= -2) & ((data['lateral'] <= 2))]
        data = data.loc[(data['circular'] >= 355) | ((data['circular'] <= 5))]
        group_list = data[CAT_NAME].unique()

        self.data = data

        # 去除噪声较大的数据
        # ————————————————————————————————————————————————
        # clean_group = []
        for group_name in group_list:
            group_data = self.data.loc[(self.data[self.CAT_NAME] == group_name)]

            # 噪声分析
            data = np.array(group_data[self.FEAT_COL])
            noise_power_each = 0
            SNR_each = []
            for j in range(2):
                noise_power_each += getSignalPower(data[:, j], sampling_rate=50, fft_size=128, cutoff=3)
                SNR_each.append(getSignalSNR(data[:, j], sampling_rate=50, fft_size=128, cutoff=3))
            print(group_name, noise_power_each)

        self.fix_group = []

        group_list = np.array(group_list)
        # ——————————————————————————————————————————————————


        # shuffle
        group_list = group_list[random_index(len(group_list), sample_rate=1, seed=seed)]

        train_size = int(len(group_list) * split_rate)

        self.train_group = group_list[:train_size]
        self.test_group = group_list[train_size:]

        X = []
        Y = []

        for group_name in self.train_group:
        # for group_name in drop_group:
            group_data = self.data.loc[(self.data[CAT_NAME] == group_name)]
            print(f'{group_name}: angle:{np.array(group_data["angle"]).min()}-{np.array(group_data["angle"]).max()}')

            label_fix = False


            # sample_n:每个样本包含的帧数 y_len:每个label包含的帧数, 设置为1时为当前帧，设置为10时为当前帧+过去9帧
            X_group, Y_group = seq2batch(group_data[FEAT_COL], group_data[LABEL_COL], sample_n=x_len, lowpass=self.lowpass,
                                         y_len=self.y_len, label_fix=label_fix)

            if group_name in self.cut_group:
                X += X_group[self.head_cut:]
                Y += Y_group[self.head_cut:]
            else:
                X += X_group
                Y += Y_group

        X = np.array(X)
        # Y = np.array(Y).clip(min=0, max=179.99)
        Y = np.squeeze(Y, axis=2)


        self.train_x = X
        self.train_y = Y
        self.test_size = len(self.test_group)

    def get_train_data(self):
        print(f"训练集: {self.train_group}")
        return self.train_x, self.train_y

    def get_test_data(self, group_idx=None):
        X = []
        Y = []

        # select_group = [self.train_group[np.random.randint(0, len(self.train_group))]]

        if group_idx is not None:
            select_group = [self.test_group[group_idx]]
        else:
            select_group = self.test_group

        print(f"测试集: {select_group}")
        for group_name in select_group:
            group_data = self.data.loc[(self.data[self.CAT_NAME] == group_name)]
            # sample_n:每个样本包含的帧数 y_len:每个label包含的帧数, 设置为1时为当前帧，设置为10时为当前帧+过去9帧
            label_fix = False
            if group_name in self.fix_group:
                label_fix = True
            X_group, Y_group = seq2batch(group_data[self.FEAT_COL], group_data[self.LABEL_COL], sample_n=self.x_len, lowpass=self.lowpass,
                                         y_len=self.y_len, label_fix=label_fix)
            if group_name in self.cut_group:
                X += X_group[self.head_cut:]
                Y += Y_group[self.head_cut:]
            else:
                X += X_group
                Y += Y_group

        X = np.array(X)
        # Y = np.array(Y).clip(min=0, max=179.99)
        Y = np.squeeze(Y, axis=2)

        return X, Y

    def get_data_by_gname(self, group_name_list=None):
        X = []
        Y = []

        # select_group = [self.train_group[np.random.randint(0, len(self.train_group))]]

        select_group = group_name_list

        print(f"获取{select_group}数据")
        for group_name in select_group:
            group_data = self.data.loc[(self.data[self.CAT_NAME] == group_name)]
            # sample_n:每个样本包含的帧数 y_len:每个label包含的帧数, 设置为1时为当前帧，设置为10时为当前帧+过去9帧
            label_fix = False
            if group_name in self.fix_group:
                label_fix = True
            X_group, Y_group = seq2batch(group_data[self.FEAT_COL], group_data[self.LABEL_COL], sample_n=self.x_len, lowpass=self.lowpass,
                                         y_len=self.y_len, label_fix=label_fix)
            if group_name in self.cut_group:
                X += X_group[self.head_cut:]
                Y += Y_group[self.head_cut:]
            else:
                X += X_group
                Y += Y_group

            # 噪声分析
            data = np.array(group_data[self.FEAT_COL])
            noise_power_each = 0
            SNR_each = []
            for j in range(data.shape[1]):
                noise_power_each += getSignalPower(data[:, j], sampling_rate=50, fft_size=128, cutoff=3)
                SNR_each.append(getSignalSNR(data[:, j], sampling_rate=50, fft_size=128, cutoff=3))
            print(f'group:{group_name} noise:{noise_power_each / 6} SNR:{np.array(SNR_each)}')

        X = np.array(X)
        # Y = np.array(Y).clip(min=0, max=179.99)
        Y = np.squeeze(Y, axis=2)

        return X, Y


def KF(Z, over, Q, R):
    # 定义超参数
    over = over
    Q = Q  # 4e-4
    R = R  # 0.25
    # 定义尺寸函数
    cc = [over, 1]
    # 定义迭代的初始参数
    X_bar = np.zeros(cc)
    Xbar = np.zeros(cc)
    K = np.zeros(cc)
    P_ = np.zeros(cc)
    P = np.zeros(cc)
    P[0] = 1
    Xbar[0] = Z[0]
    # 循环迭代
    for n in range(1, over):
        # 时间更新
        X_bar[n] = Xbar[n - 1]
        P_[n] = P[n - 1] + Q
        # 状态更新
        K[n] = P_[n] / (P_[n] + R)
        Xbar[n] = X_bar[n] + K[n] * (Z[n] - X_bar[n])
        P[n] = (1 - K[n]) * P_[n]
    return Xbar



