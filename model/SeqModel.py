from torch.nn.parameter import Parameter
from utils import *
# from utils import bidirectional_filter_for_tensor
import time
from tqdm import tqdm
from torch.autograd import Variable
import torch.nn.functional as F
from myFunctions.tensor_functions import MmdLoss
from myFunctions.tensor_functions import *
import random
from torch.autograd import Function
from typing import Any, Optional, Tuple
from Aplus.models import *
from loss_funcs import SD_loss
from sklearn.model_selection import train_test_split

from loss_funcs import *

def get_index(num_domain=2):
    index = []
    for i in range(num_domain):
        for j in range(i+1, num_domain+1):
            index.append((i, j))
    return index

def HoMM(xs, xt):
    xs = xs - torch.mean(xs, axis=0)
    xt = xt - torch.mean(xt, axis=0)
    xs = torch.unsqueeze(xs, axis=-1)
    xs = torch.unsqueeze(xs, axis=-1)
    xt = torch.unsqueeze(xt, axis=-1)
    xt = torch.unsqueeze(xt, axis=-1)
    xs_1 = xs.permute(0, 2, 1, 3)
    xs_2 = xs.permute(0, 2, 3, 1)
    xt_1 = xt.permute(0, 2, 1, 3)
    xt_2 = xt.permute(0, 2, 3, 1)
    HR_Xs = xs * xs_1 * xs_2  # dim: b*L*L*L
    HR_Xs = torch.mean(HR_Xs, axis=0)  # dim: L*L*L
    HR_Xt = xt * xt_1 * xt_2
    HR_Xt = torch.mean(HR_Xt, axis=0)
    return torch.mean((HR_Xs - HR_Xt) ** 2)

# 打印网络参数数量
def print_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(F'Parameters: Total: {total_num} Trainable:{trainable_num}')

# 激活函数
def Activation_Layer(act_name):
    """
    构建激活函数层

    Args:
        act_name: str , 激活函数名称
    Return:
        act_layer: 激活函数对象（nn.Module）
    """
    if isinstance(act_name, str):
        if act_name.lower() == 'sigmoid':
            act_layer = nn.Sigmoid()
        elif act_name.lower() == 'relu':
            act_layer = nn.ReLU(inplace=True)
        elif act_name.lower() == 'prelu':
            act_layer = nn.PReLU()
        elif act_name.lower() == 'leakyrelu':
            act_layer = nn.LeakyReLU(negative_slope=1e-2)
        elif act_name.lower() == 'elu':
            act_layer = nn.ELU()
        elif act_name.lower() == 'tanh':
            act_layer = nn.Tanh()
    elif issubclass(act_name, nn.Module):
        act_layer = act_name()
    else:
        raise NotImplementedError

    return act_layer


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


class GradientReverseFunction(Function):
    """
    重写自定义的梯度计算方式
    """
    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None
class Mk_Layer(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.fc_1 = nn.Linear(feat_dim, feat_dim//2)
        self.fc_2 = nn.Linear(feat_dim, feat_dim//2)
        self.bn = nn.BatchNorm1d(feat_dim//2)
        self.relu = nn.ReLU()
        self.cos_w = 1/int(np.sqrt(feat_dim/2))
        self.relu_w = self.cos_w / 2

    def forward(self,x):
        out_1 = self.fc_1(x)
        out_2 = self.fc_2(x)

        out_1 = self.bn(out_1)
        out_2 = self.bn(out_2)

        out_1 = self.relu(out_1)
        out_2 = torch.cos(out_2)

        return torch.cat([out_1, out_2], dim=1)

class AdvSKM_Layer(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.mk_layer_1 = Mk_Layer(feat_dim)
        self.mk_layer_2 = Mk_Layer(feat_dim)
    def forward(self, x):
        out_1 = self.mk_layer_1(x)
        out_2 = self.mk_layer_2(x)
        GradientReverseFunction.apply(out_2)
        out = out_1 + out_2
        return out

class Adv_DomainClassifier(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.domain_classier = nn.Sequential()
        # self.domain_classier.add_module('c_reverse_layer', ReverseLayerF())
        self.domain_classier.add_module('c_fc1', nn.Linear(feat_dim, 128))
        self.domain_classier.add_module('c_bn', nn.BatchNorm1d(128))
        self.domain_classier.add_module('c_ReLU', nn.ReLU())
        self.domain_classier.add_module('c_fc2', nn.Linear(128, 2))
        # self.domain_classier.add_module('c_softmax', nn.LogSoftmax(dim=1))
    def forward(self, x):
        GradientReverseFunction.apply(x)
        x = self.domain_classier(x)
        return x

class Affine_layer(nn.Module):
    # 该版本仅含有bias与scaling(scale)参数
    """
    data_range：数据的最大变化范围
    dim：输入数据维度
    depth：编码深度
    pe：是否使用傅立叶特征编码
    """

    def __init__(self, data_range=240, offset=640, dim=6, depth=2, ffe=True):
        """
        data_range：数据的最大变化范围
        dim：输入数据维度
        depth：编码深度
        pe：是否使用傅立叶特征编码
        """
        super(Affine_layer, self).__init__()
        self.data_range = data_range
        self.offset = offset
        self.depth = depth

        #  缩放系数初始化
        self.scale_value = Parameter(torch.Tensor(torch.randn(dim)*0.0))

        # 相位偏移初始化
        self.bias = Parameter(torch.Tensor(torch.randn(dim) * np.pi * 0 / 24))

        self.scale_value.requires_grad = False
        self.bias.requires_grad = False
        self.ffe = ffe
        print(f'ffe:{ffe}')
        self.state = 0
        self.grad = False
        self.remap = True

        if ffe:
            self.output_dim = dim * depth * 2
        else:
            self.output_dim = dim

    def print_value(self):
        print('scale value:')
        print(self.scale_value)
        print('bias')
        print(self.bias)

    # 参数解冻
    def unfreeze(self):
        self.grad=True
        self.bias.requires_grad = True
        self.scale_value.requires_grad = True

    # 参数冻结
    def freeze(self):
        self.grad = False
        self.bias.requires_grad = False
        self.scale_value.requires_grad = False

    # 手动设置bias
    def manual_bias(self, bias):
        for index, value in enumerate(bias):
            self.bias[index] = value
            # self.scale_value[index] = value

    def forward(self, x):
        if self.ffe:
            if self.grad:
                # print(self.bias.requires_grad)
                # 用于交替训练bias与scaling weight
                # print(self.state)
                # if self.state%2 == 0:
                #     self.bias.requires_grad = True
                #     self.scale_value.requires_grad = False
                #     # self.state += 1
                # else:
                #     self.bias.requires_grad = False
                #     self.scale_value.requires_grad = True
                #     # self.state += 1

                self.bias.requires_grad = True
                self.scale_value.requires_grad = True

            x = x - self.offset
            x = (2 * np.pi * x) / self.data_range

            if self.remap:
                x = x * (self.scale_value+1)
                x = x + self.bias * np.pi

        else:
            x = (x - self.offset) / self.data_range
            if self.remap:
                x = x * (self.scale_value+1)
                x = x + self.bias * np.pi

        if self.ffe:
            encode_list = []
            for i in range(self.depth):
                encode_list.append(torch.cat([torch.cos(x * (i + 1)), torch.sin(x * (i + 1))], dim=2))
            x = torch.cat(encode_list, dim=2)

        return x

# LSTM模型
class MyLSTM(nn.Module):
    def __init__(self, input_dim, output_dim, layer=4, input_encode_depth=None, encode_trainable=False, ffe=True,
                 use_bn=False, data_range=240, data_offset=680):
        super().__init__()
        self.affine_layer = None
        if input_encode_depth is not None:
            self.affine_layer = Affine_layer(data_range=data_range, offset=data_offset, dim=input_dim, ffe=ffe, depth=input_encode_depth)
            if encode_trainable == False:
                self.affine_layer.freeze()
            else:
                self.affine_layer.unfreeze()
        self.use_bn = use_bn
        self.bn = nn.BatchNorm1d(self.affine_layer.output_dim, affine=False)
        self.relu = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.fc_in = nn.Linear(self.affine_layer.output_dim, 128)
        self.lstm = nn.LSTM(input_size=128, hidden_size=256,
                            num_layers=layer, batch_first=True, dropout=0, bidirectional=False)

        self.data_range = data_range
        self.data_offset = data_offset


        self.fc_out_2 = nn.Linear(layer * 256 * 2, 128)
        self.fc_out = nn.Linear(128, output_dim)

        # DANN使用
        self.domain_classier = nn.Sequential()
        self.domain_classier.add_module('c_fc1', nn.Linear(128, 64))
        self.domain_classier.add_module('c_bn', nn.BatchNorm1d(64))
        self.domain_classier.add_module('c_ReLU', nn.ReLU(True))
        self.domain_classier.add_module('c_fc2', nn.Linear(64, 2))
        self.domain_classier.add_module('c_softmax', nn.LogSoftmax(dim=1))

        self.adv_layer = None
        self.optim = None


    def forward(self, x, add_noise=False, hidden_out=False, domain_classify=False):
        # x.shape: [batch_size, seq_len, input_size]

        if self.affine_layer is not None:
            x = self.affine_layer(x)

            if self.use_bn:
                x = x.permute(0, 2, 1)
                x = self.bn(x)
                x = x.permute(0, 2, 1)
        else:
            x = self.bn(x)

        x = self.fc_in(x)
        x = self.relu(x)
        output, (h_n, c_n) = self.lstm(x)
        # output: 128x10x256
        hc = torch.cat((h_n, c_n), dim=0)
        hc = hc.permute(1, 0, 2)
        hc = hc.reshape(hc.shape[0], -1)

        hc = self.fc_out_2(hc)

        if add_noise:
            noise_tensor = 0.05 * torch.randn(hc.shape[0], hc.shape[1]).float().to('cuda')
            hc += noise_tensor

        # hc = self.fc_out(hc)

        if hidden_out:
            hidden = hc
            return hidden

        if domain_classify:
            hc = GradientReverseFunction.apply(hc)
            classifier_out = self.domain_classier(hc)
            # regressor_out = self.fc_out(hc)
            return classifier_out

        hc = self.relu2(hc)
        hc = self.fc_out(hc)

        return hc

    def transfer_SD(self, x_target, lr=5e-4, batch_size=512, epoch=5, affine_only=True, alpha=0.5):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        learning_rate = lr

        batch_num = len(x_target) // batch_size
        x_target = x_target[0:batch_num*batch_size]

        train = torch.FloatTensor(x_target).to(device)

        model = self.train().to(device)

        if affine_only:
            model.affine_layer.unfreeze()
            optimizer = torch.optim.Adam(self.affine_layer.parameters(), lr=learning_rate, weight_decay=1e-4)
        else:
            model.affine_layer.freeze()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        for epoch_idx in range(epoch):
            # print(f'transfer epoch: {epoch_idx} | {epoch}')
            model.train()
            # 交替训练bias与scale
            model.affine_layer.unfreeze()
            for i in range(batch_num):
                # 生成label
                inputs = train[i*batch_size:(i+1)*batch_size, :, :]
                # n x 10
                outputs = model(inputs)
                align_outputs_list = []
                # 输出裁剪&对齐
                channel_num = outputs.shape[1]
                for j in range(channel_num):
                    align_outputs_list.append(outputs[channel_num-j-1: batch_size-j, j].unsqueeze(dim=1))

                align_outputs = torch.cat(align_outputs_list, dim=1)
                # print(align_outputs.shape)

                optimizer.zero_grad()
                batch_loss = SD_loss(align_outputs.unsqueeze(-1), alpha=alpha)
                batch_loss.backward()
                optimizer.step()

    def transfer_CORAL(self, data_source, data_target, lr=5e-4, batch_size=128, epoch=5, domain_loss='coral',
                       domain_loss_weight=1):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        data_loader_target = DataLoader(dataset=data_target, batch_size=batch_size, shuffle=True,
                                        drop_last=True)
        model = self.train().to(device)
        # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        if self.optim is None:
            self.optim = torch.optim.Adam(model.parameters(), lr=lr)
        optimizer = self.optim
        criterion_source = nn.MSELoss()
        if domain_loss == 'coral':
            criterion_target = CORAL
        elif domain_loss == 'mmd':
            criterion_target = MMDLoss()
        elif domain_loss == 'HoMM':
            criterion_target = HoMM

        for epoch_idx in range(epoch):
            # print(f'transfer epoch: {epoch_idx} | {epoch}')
            model.train()
            # 交替训练bias与scale
            # model.affine_layer.unfreeze(mode=epoch_idx % 5 + 1)
            for i, data in enumerate(data_loader_target):
                data_loader_source = DataLoader(dataset=data_source, batch_size=batch_size, shuffle=True,
                                                drop_last=True)
                for source_data in data_loader_source:
                    x_s, y_s = source_data
                    x_s = x_s.to('cuda:0')
                    y_s = y_s.to('cuda:0')
                    break

                x_t, _ = data
                x_t = x_t.to('cuda:0')
                y_t_pred, y_t_hidden = model(x_t), model(x_t, hidden_out=True)
                y_s_pred, y_s_hidden = model(x_s), model(x_s, hidden_out=True)

                optimizer.zero_grad()
                source_loss = criterion_source(y_s_pred, y_s)
                domain_loss = criterion_target(y_s_hidden, y_t_hidden) * domain_loss_weight
                loss_all = source_loss + domain_loss
                loss_all.backward()
                optimizer.step()

    def transfer_DAN(self, x_source=None, y_source=None, x_target=None, batch_size=512, lr=5e-4, epoch=5):
        # 数据集划分/直接读取训练与测试数据

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        learning_rate = lr
        criterion_source = nn.MSELoss()
        # criterion_target = CORAL
        criterion_target = MmdLoss

        batch_num_source = len(x_source) // batch_size
        x_source = x_source[0:batch_num_source * batch_size]
        y_source = y_source[0:batch_num_source * batch_size] / 180

        batch_num_target = len(x_target) // batch_size
        x_target = x_target[0:batch_num_target * batch_size]

        x_source = torch.FloatTensor(x_source).to(device)
        y_source = torch.FloatTensor(y_source).to(device)

        x_target = torch.FloatTensor(x_target).to(device)

        model = self.train().to(device)


        optimizers = []
        optimizers.append(torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0))

        # optimizers.append(torch.optim.Adam([{'params': self.affine_layer.bias}], lr=learning_rate, weight_decay=0.001))
        # optimizers.append(
        #     torch.optim.Adam([{'params': self.affine_layer.scale_value}], lr=learning_rate, weight_decay=0.001))
        for epoch_idx in range(epoch):
            # print(f'transfer epoch: {epoch_idx} | {epoch}')
            model.train()
            # 交替训练bias与scale
            # model.affine_layer.unfreeze(mode=epoch_idx % 5 + 1)
            for i in range(batch_num_target):

                x_target_batch = x_target[i * batch_size:(i + 1) * batch_size, :, :]

                ramdom_index = random.randint(0,batch_num_source-1)

                x_source_batch = x_source[ramdom_index * batch_size:(ramdom_index + 1) * batch_size, :, :]
                y_source_batch = y_source[ramdom_index * batch_size:(ramdom_index + 1) * batch_size, :]

                # n x 10
                pred = model(x_source_batch)

                hidden_source = model(x_source_batch, hidden_out=True)
                hidden_target = model(x_target_batch, hidden_out=True)

                for optim in optimizers:
                    optim.zero_grad()

                loss_mse = criterion_source(pred, y_source_batch)
                loss_domain = criterion_target(hidden_source, hidden_target)

                loss = loss_mse + loss_domain
                loss.backward()

                for optim in optimizers:
                    optim.step()


    def transfer_DANN(self, x_source=None, y_source=None, x_target=None, batch_size=512, lr=5e-4, epoch=5):
        # 数据集划分/直接读取训练与测试数据

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        learning_rate = lr
        criterion_source = nn.MSELoss()
        criterion_domain_classify = nn.CrossEntropyLoss()

        batch_num_source = len(x_source) // batch_size
        x_source = x_source[0:batch_num_source * batch_size]
        y_source = y_source[0:batch_num_source * batch_size] / 180

        batch_num_target = len(x_target) // batch_size
        x_target = x_target[0:batch_num_target * batch_size]

        x_source = torch.FloatTensor(x_source).to(device)
        y_source = torch.FloatTensor(y_source).to(device)

        x_target = torch.FloatTensor(x_target).to(device)


        model = self.train().to(device)


        optimizers = []
        optimizers.append(torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0))

        # optimizers.append(torch.optim.Adam([{'params': self.affine_layer.bias}], lr=learning_rate, weight_decay=0.001))
        # optimizers.append(
        #     torch.optim.Adam([{'params': self.affine_layer.scale_value}], lr=learning_rate, weight_decay=0.001))
        for epoch_idx in range(epoch):
            # print(f'transfer epoch: {epoch_idx} | {epoch}')
            model.train()
            # 交替训练bias与scale
            # model.affine_layer.unfreeze(mode=epoch_idx % 5 + 1)
            for i in range(batch_num_target):

                x_target_batch = x_target[i * batch_size:(i + 1) * batch_size, :, :]

                ramdom_index = random.randint(0,batch_num_source-1)

                x_source_batch = x_source[ramdom_index * batch_size:(ramdom_index + 1) * batch_size, :, :]
                y_source_batch = y_source[ramdom_index * batch_size:(ramdom_index + 1) * batch_size, :]

                x_source_label = (torch.ones([batch_size]) * 0).long().to(device)
                x_target_label = (torch.ones([batch_size]) * 1).long().to(device)
                # n x 10
                pred = model(x_source_batch)

                x_source_target = torch.cat([x_source_batch, x_target_batch], dim=0)
                x_source_target_label = torch.cat([x_source_label, x_target_label], dim=0)
                label_pred = model(x_source_target, domain_classify=True)

                for optim in optimizers:
                    optim.zero_grad()

                loss_mse = criterion_source(pred, y_source_batch)
                loss_domain = criterion_domain_classify(label_pred, x_source_target_label)

                loss = loss_mse + loss_domain
                loss.backward()

                for optim in optimizers:
                    optim.step()


    def transfer_DSAN(self,x_source=None, y_source=None, x_target=None, batch_size=512, lr=5e-4, epoch=5):
        # 数据集划分/直接读取训练与测试数据
        transfer_loss_args = {
            "loss_type": lmmd,
            "num_class": 128,
            "max_inter":1000
        }
        # 数据集划分/直接读取训练与测试数据
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        learning_rate = lr
        criterion_source = nn.MSELoss()
        # criterion_target = CORAL
        criterion_target = MmdLoss

        batch_num_source = len(x_source) // batch_size
        x_source = x_source[0:batch_num_source * batch_size]
        y_source = y_source[0:batch_num_source * batch_size] / 180

        batch_num_target = len(x_target) // batch_size
        x_target = x_target[0:batch_num_target * batch_size]

        x_source = torch.FloatTensor(x_source).to(device)
        y_source = torch.FloatTensor(y_source).to(device)
        # 修改这里
        y_source = y_source

        x_target = torch.FloatTensor(x_target).to(device)

        model = self.train().to(device)


        optimizers = []
        optimizers.append(torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0))

        # optimizers.append(torch.optim.Adam([{'params': self.affine_layer.bias}], lr=learning_rate, weight_decay=0.001))
        # optimizers.append(
        #     torch.optim.Adam([{'params': self.affine_layer.scale_value}], lr=learning_rate, weight_decay=0.001))
        for epoch_idx in range(epoch):
            # print(f'transfer epoch: {epoch_idx} | {epoch}')
            model.train()
            # 交替训练bias与scale
            # model.affine_layer.unfreeze(mode=epoch_idx % 5 + 1)
            for i in range(batch_num_target):

                x_target_batch = x_target[i * batch_size:(i + 1) * batch_size, :, :]

                ramdom_index = random.randint(0,batch_num_source-1)

                x_source_batch = x_source[ramdom_index * batch_size:(ramdom_index + 1) * batch_size, :, :]
                y_source_batch = y_source[ramdom_index * batch_size:(ramdom_index + 1) * batch_size, :]

                # n x 10
                pred = model(x_source_batch)

                hidden_source = model(x_source_batch, hidden_out=True)
                hidden_target = model(x_target_batch, hidden_out=True)

                for optim in optimizers:
                    optim.zero_grad()

                loss_mse = criterion_source(pred, y_source_batch)
                # 编写DSAN_loss
                kwargs = {}
                kwargs['source_label'] = y_source_batch
                kwargs['target_logits'] = torch.nn.functional.softmax(hidden_target, dim=1)
                transfer_loss = LMMDLoss(**transfer_loss_args)
                loss_domain = transfer_loss(hidden_source, hidden_target, **kwargs)
                loss = loss_mse + loss_domain
                loss.backward()
                for optim in optimizers:
                    optim.step()

    def transfer_HoMM(self, data_source, data_target, lr=5e-4, batch_size=128, epoch=5, domain_loss_weight=1):
        self.transfer_CORAL(data_source=data_source, data_target=data_target, batch_size=batch_size, lr=lr,
                            epoch=epoch, domain_loss='HoMM', domain_loss_weight=domain_loss_weight)

    def transfer_AdvSKM(self, data_source, data_target, lr=5e-4, batch_size=128, epoch=5, domain_loss_weight=1):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        adv_layer = AdvSKM_Layer(feat_dim=128).to('cuda:0')

        data_loader_target = DataLoader(dataset=data_target, batch_size=batch_size, shuffle=True,
                                        drop_last=True)
        model = self.train().to(device)
        # optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        self.optim = torch.optim.Adam(model.parameters(), lr=lr)
        optimizer = self.optim
        criterion_source = nn.MSELoss()

        # 原论文是用MMD 但MMD效果不太稳定 改了CORAL
        # criterion_target = CORAL
        criterion_target = MMDLoss()

        for epoch_idx in range(epoch):
            # print(f'transfer epoch: {epoch_idx} | {epoch}')
            model.train()
            # 交替训练bias与scale
            # model.affine_layer.unfreeze(mode=epoch_idx % 5 + 1)
            for i, data in enumerate(data_loader_target):
                data_loader_source = DataLoader(dataset=data_source, batch_size=batch_size, shuffle=True,
                                                drop_last=True)
                for source_data in data_loader_source:
                    x_s, y_s = source_data
                    x_s = x_s.to('cuda:0')
                    y_s = y_s.to('cuda:0')
                    break

                x_t, _ = data
                x_t = x_t.to('cuda:0')
                y_t_pred, y_t_hidden = model(x_t), model(x_t, hidden_out=True)
                y_s_pred, y_s_hidden = model(x_s), model(x_s, hidden_out=True)

                y_t_hidden = adv_layer(y_t_hidden)
                y_s_hidden = adv_layer(y_s_hidden)

                optimizer.zero_grad()
                source_loss = criterion_source(y_s_pred, y_s)
                domain_loss = criterion_target(y_s_hidden, y_t_hidden) * domain_loss_weight
                loss_all = source_loss + domain_loss
                loss_all.backward()
                optimizer.step()

    def restore(self, checkpoint_path='./' + 'LSTM.pth'):
        try:
            self.load_state_dict(torch.load(checkpoint_path))
            print("Check Point Loaded")
        except FileNotFoundError as r:
            print("Check Point doesn't exist")


# 监督自适应
def LSTM_adapt(model=None, X=None, Y=None, X_TEST=None, Y_TEST=None, split_rate=0.2, num_epoch=30, pe=False,
               filename=None, restore=False, lr=2e-4, encode_only=False, save=True, verbose=True, threshold=None):
    X_train, y_train = X, Y
    X_test, y_test = X_TEST, Y_TEST

    print(len(X_train))
    print(len(X_test))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    seed = 42
    BATCH_SIZE = 128
    learning_rate = lr
    criterion = nn.MSELoss()

    # print(y_train)

    train = MyDataset(X_train, y_train, pe=pe)
    val = MyDataset(X_test, y_test, pe=pe)
    train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val, batch_size=BATCH_SIZE, shuffle=True)

    optimizers = []



    if model.affine_layer is not None:
        if encode_only:
            model.affine_layer.unfreeze()
            optimizers.append(torch.optim.Adam(model.affine_layer.parameters(), lr=learning_rate))
            # optimizers.append(
            #     torch.optim.Adam([{'params': model.affine_layer.bias}], lr=learning_rate))
        else:
            model.affine_layer.freeze()
            # optimizers.append(torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0))
            # optimizers.append(torch.optim.Adam(model.encoder.parameters(), lr=learning_rate, weight_decay=0))
            # optimizers.append(torch.optim.Adam(model.lstm.parameters(), lr=learning_rate, weight_decay=learning_rate))
            optimizers.append(torch.optim.Adam(model.fc_out.parameters(), lr=learning_rate, weight_decay=0))
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)

    if lr <= 0:
        return model

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    epoch_mae_list = []
    epoch_list = []
    epoch_err_df = pd.DataFrame(columns=['epoch', 'mae'])
    for epoch in range(num_epoch):
        model.train()
        # model.affine_layer.print_value()
        train_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            for optimizer in optimizers:
                optimizer.zero_grad()
            # optimizer_lstm.zero_grad()
            outputs = model(inputs)
            # outputs = model(inputs, add_noise=True)
            batch_loss = criterion(outputs[:, -1], labels[:, -1])
            batch_loss.backward()

            for optimizer in optimizers:
                optimizer.step()
            # optimizer_lstm.step()
            train_loss += (batch_loss)
            # print(train_loss)
            # model.affine_layer.state += 1

        model.eval()  # set the model to evaluation mode
        val_loss = 0
        batch_mae = []
        batch_mae_each = None
        batch_real_each = None
        batch_pred_each = None

        model.affine_layer.state += 1

        with torch.no_grad():
            for i, data in enumerate(val_loader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                if threshold is not None:
                    labels = torch.where(labels > threshold, outputs.detach(), labels).detach()
                batch_loss = criterion(outputs, labels)
                val_loss += batch_loss.item()

                # preds是概率最大对应的独热编码的索引，label.data是独热编码，将独热编码转化为索引，看这两个索引的tensor是否相同
                pred = np.array(outputs[:, -1].cpu()).reshape(-1) * 180
                real = np.array(labels[:, -1].cpu()).reshape(-1) * 180
                batch_mae.append(mae(pred=pred, real=real))

                if batch_mae_each is None:
                    batch_mae_each = mae_each(pred=pred, real=real)
                    batch_real_each = real
                    batch_pred_each = pred
                else:
                    batch_mae_each = np.concatenate([batch_mae_each, mae_each(pred=pred, real=real)], axis=0)
                    batch_real_each = np.concatenate([batch_real_each, real], axis=0)
                    batch_pred_each = np.concatenate([batch_pred_each, pred], axis=0)

        err_df = pd.DataFrame(columns=['pred', 'real', 'mae'])
        err_df['mae'] = batch_mae_each
        err_df['real'] = batch_real_each
        err_df['pred'] = batch_pred_each

        # 记录每epoch的误差
        epoch_mae = np.array(batch_mae_each).mean()
        epoch_mae_list.append(epoch_mae)
        epoch_list.append(epoch)

        if verbose:
            print('[{:03d}/{:03d}] Train Loss: {:3.6f} | Val  loss: {:3.6f}'.format(
                epoch + 1, num_epoch, train_loss / len(train_loader), val_loss / len(val_loader)
            ))

            print('mae: {:3.4f} std: {:3.4f} corr: {:3.4f}'.format(epoch_mae, np.std(batch_mae_each),
                                                                   err_df['real'].corr(err_df['pred'])))
            print('percentiles: {:3.4f} {:3.4f} {:3.4f}'.format(np.percentile(batch_mae_each, 25),
                                                                np.percentile(batch_mae_each, 50),
                                                                np.percentile(batch_mae_each, 75)))

    epoch_err_df['epoch'] = epoch_list
    epoch_err_df['mae'] = epoch_mae_list
    # epoch_err_df.to_excel(f'./record/{filename}.xlsx', index=False)
    if save:
        torch.save(model.state_dict(), './LSTM.pth')

    # err_df.to_excel(f'./record/{filename}.xlsx')



    return model

def model_eval(model, data_loader, device, verbose=True, bias=None, kf=False):
    model.eval()  # set the model to evaluation mode
    criterion = nn.MSELoss()
    val_loss = 0
    batch_mae = []
    batch_mae_each = None
    batch_real_each = None
    batch_pred_each = None
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            inputs, labels = data
            if len(labels) <= 2:
                print('pass')
                continue

            if bias is not None:
                inputs = inputs + bias
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            batch_loss = criterion(outputs, labels)
            val_loss += batch_loss.item()

            # preds是概率最大对应的独热编码的索引，label.data是独热编码，将独热编码转化为索引，看这两个索引的tensor是否相同

            pred = np.array(outputs.cpu()) * 180
            real = np.array(labels[:, -1].cpu()) * 180
            batch_mae.append(mae(pred=pred[:, -1], real=real))

            if kf:
                over = len(pred[:, -1])
                Z = pred[:, -1].reshape(over, 1)
                X = KF(Z, over, 0.1, 0.5)
                pred = X
                # print(X.shape)
                # plt.plot(Z)
                # plt.plot(X)
                # plt.show()

            if batch_mae_each is None:
                batch_mae_each = mae_each(pred=pred[:, -1], real=real)
                batch_real_each = real
                batch_pred_each = pred[:, -1]

                batch_pred_each_all = pred
            else:
                batch_mae_each = np.concatenate([batch_mae_each, mae_each(pred=pred[:, -1], real=real)], axis=0)
                batch_real_each = np.concatenate([batch_real_each, real], axis=0)
                batch_pred_each = np.concatenate([batch_pred_each, pred[:, -1]], axis=0)
                batch_pred_each_all = np.concatenate([batch_pred_each_all, pred], axis=0)

    err_df = pd.DataFrame(columns=['pred', 'real', 'mae'])
    err_df['mae'] = batch_mae_each
    err_df['real'] = batch_real_each
    err_df['pred'] = batch_pred_each

    # 记录每epoch的误差
    # epoch_mae = np.array(batch_mae_each).mean()

    # 不计算大于阈值的lmae
    # batch_mae_each = np.array(batch_mae_each)
    # bool_index = batch_real_each < 160
    # batch_mae_each = batch_mae_each[bool_index]
    epoch_mae = np.array(batch_mae_each).mean()

    std = np.std(batch_mae_each)
    corr = err_df['real'].corr(err_df['pred'])
    percentile_25 = np.percentile(batch_mae_each, 25)
    percentile_50 = np.percentile(batch_mae_each, 50)
    percentile_75 = np.percentile(batch_mae_each, 75)
    if verbose:
        print('mae: {:3.4f} std: {:3.4f} corr: {:3.4f}'.format(epoch_mae, std,
                                                               corr))
        print('percentiles: {:3.4f} {:3.4f} {:3.4f}'.format(percentile_25,
                                                            percentile_50,
                                                            percentile_75))

    result = [epoch_mae, std, percentile_25, percentile_50, percentile_75]


    return batch_pred_each_all, batch_real_each, result

def model_hidden_out(model, data_loader, device):
    model.eval()  # set the model to evaluation mode

    hidden_outputs = None
    label_out = None
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs, hidden_out=True)
            outputs = np.array(outputs.detach().cpu())
            labels = np.array(labels.detach().cpu())
            # print(outputs)
            if hidden_outputs is None:
                hidden_outputs = outputs
                label_out = labels
            else:
                hidden_outputs = np.concatenate([hidden_outputs, outputs], axis=0)
                label_out = np.concatenate([label_out, labels], axis=0)

    return hidden_outputs, label_out*180


def LSTM_train(model=None, X=None, Y=None, X_TEST=None, Y_TEST=None, Y_FRAME=1, split_rate=0.2, num_epoch=30, ffe=False,
               filename=None, restore=False, lr=2e-4, encode_only=False, save=True, verbose=True, use_bn=False):

    # 数据集划分/直接读取训练与测试数据
    if X_TEST is None:
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=split_rate)
    else:
        X_train, y_train = X, Y
        X_test, y_test = X_TEST, Y_TEST

    print(len(X_train))
    print(len(X_test))


    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    seed = 42
    BATCH_SIZE = 128
    learning_rate = lr
    criterion = nn.MSELoss()


    train = MyDataset(X_train, y_train)
    val = MyDataset(X_test, y_test)
    train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val, batch_size=BATCH_SIZE, shuffle=True)


    input_encode_depth = 1


    if model is None:
        model = MyLSTM(input_dim=X.shape[2], output_dim=Y_FRAME, layer=3, input_encode_depth=input_encode_depth,
                                  ffe=ffe, encode_trainable=False, use_bn=use_bn).to(device)
        if restore:
            file_name = 'LSTM.pth'
            model.restore(checkpoint_path='./' + file_name)

    optimizers = []

    if model.affine_layer is not None:
        # 只训练编码层
        if encode_only:
            model.affine_layer.unfreeze()
            optimizers.append(torch.optim.Adam(model.affine_layer.parameters(), lr=learning_rate))
        else:
            # model.encode_layer.freeze()
            optimizers.append(torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.000))
    else:
        optimizers.append(torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0))

    if lr <= 0:
        return model

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    epoch_mae_list = []
    epoch_list = []
    # 记录每个epoch的mae
    epoch_err_df = pd.DataFrame(columns=['epoch', 'mae'])
    for epoch in range(num_epoch):
        model.train()
        # model.encode_layer.print_value()
        train_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            for optimizer in optimizers:
                optimizer.zero_grad()
            # optimizer_lstm.zero_grad()
            outputs = model(inputs)
            # outputs = model(inputs, add_noise=True)
            batch_loss = criterion(outputs, labels)

            batch_loss.backward()

            # 梯度裁剪
            # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1, norm_type=2)

            for optimizer in optimizers:
                optimizer.step()
            # optimizer_lstm.step()
            train_loss += (batch_loss)
            # print(train_loss)


        model.eval()  # set the model to evaluation mode
        val_loss = 0
        batch_mae = []
        batch_mae_each = None
        batch_real_each = None
        batch_pred_each = None

        with torch.no_grad():
            for i, data in enumerate(val_loader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                batch_loss = criterion(outputs, labels)
                val_loss += batch_loss.item()

                # 输出结果进行逆归一化
                pred = np.array(outputs[:, -1].cpu()).reshape(-1) * 180
                real = np.array(labels[:, -1].cpu()).reshape(-1) * 180
                batch_mae.append(mae(pred=pred, real=real))


                if batch_mae_each is None:
                    batch_mae_each = mae_each(pred=pred, real=real)
                    batch_real_each = real
                    batch_pred_each = pred
                else:
                    batch_mae_each = np.concatenate([batch_mae_each, mae_each(pred=pred, real=real)], axis=0)
                    batch_real_each = np.concatenate([batch_real_each, real], axis=0)
                    batch_pred_each = np.concatenate([batch_pred_each, pred], axis=0)


        err_df = pd.DataFrame(columns=['pred', 'real', 'mae'])
        err_df['mae'] = batch_mae_each
        err_df['real'] = batch_real_each
        err_df['pred'] = batch_pred_each

        # 记录每epoch的误差
        epoch_mae = np.array(batch_mae_each).mean()
        epoch_mae_list.append(epoch_mae)
        epoch_list.append(epoch)

        if verbose:
            print('[{:03d}/{:03d}] Train Loss: {:3.6f} | Val  loss: {:3.6f}'.format(
                epoch + 1, num_epoch, train_loss / len(train_loader), val_loss / len(val_loader)
            ))

            print('mae: {:3.4f} std: {:3.4f} corr: {:3.4f}'.format(epoch_mae, np.std(batch_mae_each), err_df['real'].corr(err_df['pred'])))
            print('percentiles: {:3.4f} {:3.4f} {:3.4f}'.format(np.percentile(batch_mae_each, 25), np.percentile(batch_mae_each, 50), np.percentile(batch_mae_each, 75)))

    epoch_err_df['epoch'] = epoch_list
    epoch_err_df['mae'] = epoch_mae_list
    # epoch_err_df.to_excel(f'./record/{filename}.xlsx', index=False)
    if save:
        torch.save(model.state_dict(), './LSTM.pth')

    # err_df.to_excel(f'./record/{filename}.xlsx')

    return model



