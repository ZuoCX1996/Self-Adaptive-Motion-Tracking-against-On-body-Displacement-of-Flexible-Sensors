import torch
from torch import nn


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
    将源域数据和目标域数据转化为核矩阵，即上文中的K
    Params:
	    source: 源域数据（n * len(x))
	    target: 目标域数据（m * len(y))
	    kernel_mul:
	    kernel_num: 取不同高斯核的数量
	    fix_sigma: 不同高斯核的sigma值
	Return:
		sum(kernel_val): 多个核矩阵之和
    '''
    n_samples = int(source.size()[0]) + int(target.size()[0])  # 求矩阵的行数，一般source和target的尺度是一样的，这样便于计算
    total = torch.cat([source, target], dim=0)  # 将source,target按列方向合并
    # 将total复制（n+m）份
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    # 将total的每一行都复制成（n+m）行，即每个数据都扩展成（n+m）份
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    # 求任意两个数据之间的和，得到的矩阵中坐标（i,j）代表total中第i行数据和第j行数据之间的l2 distance(i==j时为0）
    L2_distance = ((total0 - total1) ** 2).sum(2)
    # 调整高斯核函数的sigma值
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    # 以fix_sigma为中值，以kernel_mul为倍数取kernel_num个bandwidth值（比如fix_sigma为1时，得到[0.25,0.5,1,2,4]
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    # 高斯核函数的数学表达式
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    # 得到最终的核矩阵
    return sum(kernel_val)  # /len(kernel_val)


def MmdLoss(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
    计算源域数据和目标域数据的MMD距离
    Params:
	    source: 源域数据（n * len(x))
	    target: 目标域数据（m * len(y))
	    kernel_mul:
	    kernel_num: 取不同高斯核的数量
	    fix_sigma: 不同高斯核的sigma值
	Return:
		loss: MMD loss
    '''
    batch_size = int(source.size()[0])  # 一般默认为源域和目标域的batchsize相同
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    # 根据式（3）将核矩阵分成4部分
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY - YX)
    return loss  # 因为一般都是n==m，所以L矩阵一般不加入计算


def CORAL(source, target, **kwargs):
    d = source.data.shape[1]
    ns, nt = source.data.shape[0], target.data.shape[0]
    # source covariance
    xm = torch.mean(source, 0, keepdim=True) - source
    xc = xm.t() @ xm / (ns - 1)

    # target covariance
    xmt = torch.mean(target, 0, keepdim=True) - target
    xct = xmt.t() @ xmt / (nt - 1)

    # frobenius norm between source and target
    loss = torch.mul((xc - xct), (xc - xct))
    loss = torch.sum(loss) / (4*d*d)
    return loss


def Similarity_loss(target_out, alpha=0.5):
    outputs = target_out

    batch_size = target_out.shape[0]

    criterion_mean = nn.MSELoss()
    criterion_shape = nn.MSELoss()

    align_outputs_scaled_list = []
    align_outputs_mean_list = []

    # 输出裁剪&对齐
    channel_num = outputs.shape[1]
    for j in range(channel_num):
        # 归一化后对齐的序列
        align_outputs_scaled_list.append(
            min_max_scale(outputs[channel_num - j - 1: batch_size - j, j]).unsqueeze(dim=1))
        # 各对齐序列的均值
        align_outputs_mean_list.append(outputs[channel_num - j - 1: batch_size - j, j].mean().unsqueeze(dim=0))

    # align_outputs = torch.cat(align_outputs_list[:-1], dim=1)
    align_outputs_scaled = torch.cat(align_outputs_scaled_list, dim=1)
    # print(align_outputs.shape)
    align_outputs_mean = torch.cat(align_outputs_mean_list, dim=0)


    # 原版
    y_batch_scaled = align_outputs_scaled.mean(dim=1).unsqueeze(dim=1).detach()

    # y_batch_scaled = bidirectional_filter_for_tensor(tensor_X=y_batch_scaled,cutoff=5,fs=50, order=3).to(device)

    y_batch_mean = align_outputs_mean.mean().unsqueeze(dim=0).detach()

    # 复制10份
    y_batch_scaled = torch.cat([y_batch_scaled for k in range(channel_num)], dim=1)
    y_batch_mean = torch.cat([y_batch_mean for k in range(channel_num)], dim=0)

    # y_batch_smooth_scaled = torch.cat([min_max_scale(y_batch_smooth[:, k].unsqueeze(dim=1)) for k in range(channel_num)], dim=1)

    mean_loss = criterion_mean(align_outputs_mean, y_batch_mean)
    shape_loss = criterion_shape(align_outputs_scaled, y_batch_scaled)


    loss = alpha * mean_loss + (1 - alpha) * shape_loss

    return loss


def SD_loss(x, alpha=0.5, epsilon=1e-2):

    # x: seq_len x seq_num x feat_dim

    x_scaled_list = []
    x_mean_list = []

    seq_num = x.shape[1]
    for j in range(seq_num):
        x_scaled_list.append(min_max_scale(x[:, j], epsilon=epsilon).unsqueeze(dim=1))
        x_mean_list.append(x[:, j].mean().unsqueeze(dim=0))

    x_scaled = torch.cat(x_scaled_list, dim=1)
    x_mean = torch.cat(x_mean_list, dim=0)

    x_scaled_avg = x_scaled.mean(dim=1).unsqueeze(dim=1).detach()
    x_mean_avg = x_mean.mean().unsqueeze(dim=0).detach()


    x_scaled_avg = torch.cat([x_scaled_avg for _ in range(seq_num)], dim=1)
    x_mean_avg = torch.cat([x_mean_avg for _ in range(seq_num)], dim=0)

    mean_loss = torch.pow(x_scaled-x_scaled_avg, 2).mean()
    shape_loss = torch.pow(x_mean-x_mean_avg, 2).mean()

    loss = alpha * mean_loss + (1 - alpha) * shape_loss

    return loss

# 对tensor进行归一化
def min_max_scale(x, epsilon=1):
    min = torch.min(x).detach()
    max = torch.max(x).detach()
    x = (x-min) / (max-min+epsilon)
    return x

