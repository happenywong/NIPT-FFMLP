#!/usr/bin/python3
# -*- codeing = utf-8 -*-
# -------------------------------------------------------------------------------------#
import os
import sys
import time
import logging
import warnings
import argparse
# -------------------------------------------------------------------------------------#
import matplotlib
import matplotlib.pylab as plt
from matplotlib.pyplot import MultipleLocator
import seaborn as sns
# -------------------------------------------------------------------------------------#
import pandas as pd
import numpy as np
import math
from sklearn.model_selection import KFold  # K折交叉验证
from sklearn.decomposition import PCA
import joblib  # 保存sklearn模型
# -------------------------------------------------------------------------------------#
import torch
import torch.nn as nn  # 定义网络
import torch.nn.functional as functional  # 激励函数
import torch.utils.data as data  # 包装数据
# -------------------------------------------------------------------------------------#
# 作者信息 ##############################################################################
'''
author：hap
create data:2022/1/14
update data:2022/2/14
version：1.0.0.2
功能：
用每个窗口的read数作为特征、胎儿浓度作为标签，训练神经网络模型
'''
#  版本 ################################################################################
program = sys.argv[0]
version = '1.0.0.2'
# 默认路径 ##############################################################################
in_default = [r'../examples/database/datalist.txt']
out_default = [r'../examples/result/model']
# 运行信息 ##############################################################################
warnings.filterwarnings("ignore")  # 关闭警告信息提示
matplotlib.use('Agg')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 防止ipython崩溃
logger = logging.getLogger(program)
formatter = logging.Formatter('%(asctime)s - %(filename)s - %(levelname)s - %(message)s')
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)
# 可输入参数 #############################################################################
parser = argparse.ArgumentParser(description='训练胎儿浓度预测模型')
parser.add_argument('-V', '--version', action='version', version='%(prog)s ' + version)
# 路径信息
parser.add_argument('-D', '--datalink', nargs=1, default=in_default, help='用于训练的样本路径表')
parser.add_argument('-O', '--out', nargs=1, default=out_default, help='输出结果的目录')
# 模型参数
parser.add_argument('-RS', '--seed', nargs=1, default=[66], help='输入随机种子')
parser.add_argument('-TR', '--train', nargs=1, default=[0.9], help='输入作为数据集的百分比，默认为0.9')
parser.add_argument('-FN', '--feature', nargs=1, default=[4], help='输入神经网络的特征数量')
parser.add_argument('-EN', '--epoch', nargs=1, default=[5], help='输入神经网络的训练批次数')
parser.add_argument('-BS', '--batch', nargs=1, default=[2048], help='输入神经网络的每批次样本数')
parser.add_argument('-LR', '--learn', nargs=1, default=[0.001], help='输入学习率')
parser.add_argument('-L2', '--regularization', nargs=1, default=[0.0001], help='输入L2正则化系数')
parser.add_argument('-DR', '--dropout', nargs=1, default=[0.0001], help='输入神经元随机丢弃率，默认为0')
parser.add_argument('-KN', '--kfold', nargs=1, default=[10], help='输入K折交叉验证次数')
args = parser.parse_args()
#########################################################################################


class MLP(nn.Module):  # 继承 torch 的 Module
    def __init__(self, n_feature, DR=0):
        super(MLP, self).__init__()  # 继承 __init__ 功能
        # 定义每层用什么样的形式
        self.DR = DR
        self.hidden1 = nn.Linear(n_feature, n_feature * 3 + 1)  # 隐藏层1
        self.hidden2 = nn.Linear(n_feature * 3 + 1, n_feature)  # 隐藏层2
        self.predict = nn.Linear(n_feature, 1)  # 输出层

    def forward(self, n_input):
        # 正向传播输入值, 神经网络分析出输出值
        n_output = self.hidden1(n_input)
        n_output = functional.elu(n_output)
        n_output = functional.dropout(n_output, p=self.DR)

        n_output = self.hidden2(n_output)
        n_output = functional.elu(n_output)
        n_output = functional.dropout(n_output, p=self.DR)

        n_output = self.predict(n_output)
        n_output = n_output.squeeze(-1)
        return n_output


def random_parameter(n=66):
    random_seed = n
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    return 0


#  获取数据
def get_data(in_df):
    x_list = []
    x_list_a = x_list.append
    logger.info('读取数据')
    for index, row in in_df.iterrows():
        df_bed = pd.read_table(row['filepath'], header=0)
        bool_ruler = (df_bed['chr'] != 'Y') & (df_bed['chr'] != 'X') & \
                     (df_bed['chr'] != '13') & (df_bed['chr'] != '18') & (df_bed['chr'] != '21')
        median_ratio = np.median([i for i in df_bed[bool_ruler]['ratio'].values.tolist() if i != 0])
        df_ratio = [i / median_ratio for i in df_bed[bool_ruler]['ratio'].values.tolist()]
        x_list_a(df_ratio)
    del x_list_a, df_bed, bool_ruler, median_ratio, df_ratio
    logger.info('删除13,18,21,X,Y染色体的窗口')
    df = pd.DataFrame(x_list)
    df['FF'] = in_df['FF'].values.tolist()
    del in_df, x_list
    df = df.dropna(axis=0, how='any')  # 删除缺失值
    df = df.loc[:, (df != 0).any(axis=0)]  # 删除全为0窗口
    logger.info('删除存在缺失值的样本和全为0窗口')
    y_tensor = (torch.FloatTensor(df['FF']) - 20) / 15
    df = df.drop(columns=['FF'])
    win_index = pd.DataFrame(list(df))
    win_index.to_csv(os.path.join(out_dir, 'win_index.txt'), header=False, index=False, sep='\t')
    x_tensor = torch.FloatTensor(np.array(df).astype('float'))
    logger.info(f'过滤后的样本数:{len(y_tensor)}')
    del df, win_index
    return x_tensor, y_tensor


# 训练模型
def train_function(config, x, y):
    KFold_count = 1
    R_best = 0.85
    ME_best = 5
    df_loss = pd.DataFrame()
    df_R = pd.DataFrame()
    df_ME = pd.DataFrame()
    dataset_kf = KFold(n_splits=config['KN'], random_state=config['RS'], shuffle=True)  # K折交叉验证
    for train_index, test_index in dataset_kf.split(x, y):
        #  神经网络初始化
        logger.info('神经网络初始化')
        Model_nn = MLP(config['FN'])
        optimizer = torch.optim.Adam(Model_nn.parameters(), lr=config['LR'], weight_decay=config['L2'])  # 优化器
        MAD = torch.nn.L1Loss(reduction='mean')  # 平均绝对误差
        loss_func = torch.nn.SmoothL1Loss(reduction='mean')  # 损失函数
        ##########################################################################
        logger.info(fr'{config["KN"]}折交叉验证，进度:{KFold_count}/{config["KN"]}')
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        ##########################################################################
        #  PCA特征降低维度
        logger.info('主成分分析降低特征维度')
        pca = PCA(n_components=None, whiten=True)
        pca.fit(x_train)
        x_train = torch.FloatTensor(pca.fit_transform(x_train))[:, :config['FN']]
        x_test = torch.FloatTensor(pca.transform(x_test))[:, :config['FN']]
        ##########################################################################
        #  打包数据
        logger.info('打包数据，多线程读取数据')
        train_dataset = data.TensorDataset(x_train, y_train)
        loader_train = data.DataLoader(
            dataset=train_dataset,
            batch_size=config['BS'],  # 最小批量
            shuffle=True,
            num_workers=0,  # 多线程来读数据
        )
        test_dataset = data.TensorDataset(x_test, y_test)
        loader_test = data.DataLoader(
            dataset=test_dataset,
            batch_size=len(test_dataset),
            num_workers=0,
        )
        ##########################################################################
        #  循环训练
        logger.info(f'开始循环训练，循环次数：{config["EN"]}')

        loss_train_list, R_train_list, ME_train_list = [], [], []
        loss_train_list_a, R_train_list_a, ME_train_list_a = \
            loss_train_list.append, R_train_list.append, ME_train_list.append

        loss_test_list, R_test_list, ME_test_list = [], [], []
        loss_test_list_a, R_test_list_a, ME_test_list_a = \
            loss_test_list.append, R_test_list.append, ME_test_list.append

        for epoch in range(config["EN"]):  # 训练所有数据
            # 训练
            loss = 1
            R_corr = 0.8
            ME = 5
            Model_nn.train()
            for step, (batch_x, batch_y) in enumerate(loader_train):  # 训练集循环
                y_pre = Model_nn(batch_x)
                loss = loss_func(y_pre, batch_y)  # 正向传播求损失
                MAD_value = MAD(y_pre, batch_y)
                ME = round(MAD_value.item() * 15, 5)
                R_corr = np.corrcoef(y_pre.data.numpy(), batch_y.data.numpy())[0][1]
                optimizer.zero_grad()  # 清空上一步的残余更新参数值
                loss.backward()  # 误差反向传播, 计算参数更新值
                optimizer.step()  # 将参数更新值施加到 net 的 parameters 上
            logger.info(f'训练集结果 - ME：{ME}, R：{R_corr}')
            loss_train_list_a(round(loss.item(), 5))
            R_train_list_a(round(R_corr, 3))
            ME_train_list_a(ME)
            ##########################################################################
            # 验证
            Model_nn.eval()
            for step, (batch_x, batch_y) in enumerate(loader_test):  # 验证集循环
                y_pre = Model_nn(batch_x)
                loss = loss_func(y_pre, batch_y)
                MAD_value = MAD(y_pre, batch_y)
                ME = round(MAD_value.item() * 15, 5)
                R_corr = np.corrcoef(y_pre.data.numpy(), batch_y.data.numpy())[0][1]
                if ME < ME_best:
                    ME_best = ME
                    logger.info(f'保存效果最优模型，ME：{ME}，R：{R_corr}')
                    joblib.dump(pca, os.path.join(out_dir, f'PCA.pkl'))
                    torch.save(Model_nn.state_dict(), os.path.join(out_dir, 'MLP.pkl'))
            logger.info(f'测试集结果 - ME：{ME}, R：{R_corr}')
            loss_test_list_a(round(loss.item(), 5))
            R_test_list_a(round(R_corr, 3))
            ME_test_list_a(ME)
        ##########################################################################
        logger.info('对比并保存的模型\n'
                    '----------------------------------------------------------------------------------------')
        df_loss[f'train_{KFold_count}'] = loss_train_list
        df_R[f'train_{KFold_count}'] = R_train_list
        df_ME[f'train_{KFold_count}'] = ME_train_list

        df_loss[f'test_{KFold_count}'] = loss_test_list
        df_R[f'test_{KFold_count}'] = R_test_list
        df_ME[f'test_{KFold_count}'] = ME_test_list

        del loss_train_list_a, R_train_list_a, ME_train_list_a, \
            loss_test_list_a, R_test_list_a, ME_test_list_a
        del loss_train_list, R_train_list, ME_train_list, \
            loss_test_list, R_test_list, ME_test_list
        ##########################################################################
        KFold_count += 1
    logger.info('输出模型：PCA.pkl, MLP.pkl')
    logger.info('输出记录：loss.txt, R.txt, ME.txt')
    df_loss.to_csv(os.path.join(out_dir, 'loss.txt'), header=True, index=False, sep='\t')
    df_R.to_csv(os.path.join(out_dir, 'R.txt'), header=True, index=False, sep='\t')
    df_ME.to_csv(os.path.join(out_dir, 'ME.txt'), header=True, index=False, sep='\t')
    return 0


#  预测结果
def predict_function(config, x, y):
    logger.info('测试最终模型')
    pca = joblib.load(os.path.join(out_dir, 'PCA.pkl'))
    net = MLP(config['FN'])
    net.load_state_dict(torch.load(os.path.join(out_dir, 'MLP.pkl')))
    logger.info('可视化结果')
    result_visual(config, pca, net, x, y)
    return 0


#  可视化训练过程
def process_visual():
    # 画图颜色和节点类型
    color_list = ['#735245', '#CCA18D', '#6586B3', '#596D3D', '#8D89C2', '#84E4D0',
                  '#F97623', '#505BB6', '#DE5B7D', '#5B3F7B', '#ADE85B', '#FFA41A',
                  '#2C388E', '#4A9451', '#B32A32', '#FAE215', '#BF51A0', '#068EAC',
                  '#FCFCFC', '#E6E6E6', '#C8C8C8', '#8F8F8E', '#646464', '#323232']
    marker_list = ['.', ',', 'o', 'v', '^', '<', '>', '1',
                   '2', '3', '4', 's', 'p', '*', 'h', 'H',
                   '+', 'x', 'D', 'd', '|', '_', '.', ',']
    ##########################################################################
    logger.info('可视化训练过程')
    fig = plt.figure(figsize=(15, 9))
    plt.style.use('ggplot')
    path_loss = os.path.join(out_dir, 'loss.txt')
    path_corr = os.path.join(out_dir, 'R.txt')
    path_ME = os.path.join(out_dir, 'ME.txt')
    ##########################################################################
    df_loss = pd.read_table(path_loss, header=0)
    df_corr = pd.read_table(path_corr, header=0)
    df_ME = pd.read_table(path_ME, header=0)
    x = range(df_loss.shape[0] + 1)
    ##########################################################################
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.set_facecolor(color_list[-5])
    i = 0
    for col in df_loss.columns:
        if 'train' in col:
            y = [0] + df_loss[col].values.tolist()
            plt.plot(x, y, label=f'{col}',
                     color=color_list[i],
                     marker=marker_list[i], markersize=1.5, mec=color_list[i], mfc=color_list[i])
            i += 1
    plt.legend(loc='best', frameon=False)
    plt.xlim(0, len(x) + 5)
    plt.xlabel('times of repetition', fontsize=15)
    plt.ylabel('loss(MSE)', fontsize=15)
    plt.title('Loss function (train)', fontsize=15)
    plt.grid([])
    ##########################################################################
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.set_facecolor(color_list[-5])
    i = 0
    for col in df_corr.columns:
        if 'train' in col:
            y = [0] + df_corr[col].values.tolist()
            plt.plot(x, y, label=f'{col}',
                     color=color_list[i],
                     marker=marker_list[i], markersize=1.5, mec=color_list[i], mfc=color_list[i])
            i += 1
    plt.legend(loc='best', frameon=False)
    plt.xlim(0, len(x) + 5)
    plt.xlabel('times of repetition', fontsize=15)
    plt.ylabel('R', fontsize=15)
    plt.title('Pearson correlation coefficient (train)', fontsize=15)
    plt.grid([])
    ##########################################################################
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.set_facecolor(color_list[-5])
    i = 0
    for col in df_ME.columns:
        if 'train' in col:
            y = [0] + df_ME[col].values.tolist()
            plt.plot(x, y, label=f'{col}',
                     color=color_list[i],
                     marker=marker_list[i], markersize=1.5, mec=color_list[i], mfc=color_list[i])
            i += 1
    plt.legend(loc='best', frameon=False)
    plt.xlim(0, len(x) + 5)
    plt.xlabel('times of repetition', fontsize=15)
    plt.ylabel('mean error', fontsize=15)
    plt.title('Mean error (train)', fontsize=15)
    plt.grid([])
    ##########################################################################
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.set_facecolor(color_list[-5])
    i = 0
    for col in df_loss.columns:
        if 'test' in col:
            y = [0] + df_loss[col].values.tolist()
            plt.plot(x, y, label=f'{col}',
                     color=color_list[i],
                     marker=marker_list[i], markersize=1.5, mec=color_list[i], mfc=color_list[i])
            i += 1
    plt.legend(loc='best', frameon=False)
    plt.xlim(0, len(x) + 5)
    plt.xlabel('times of repetition', fontsize=15)
    plt.ylabel('loss(MSE)', fontsize=15)
    plt.title('Loss function (test)', fontsize=15)
    plt.grid([])
    ##########################################################################
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.set_facecolor(color_list[-5])
    i = 0
    for col in df_corr.columns:
        if 'test' in col:
            y = [0] + df_corr[col].values.tolist()
            plt.plot(x, y, label=f'{col}',
                     color=color_list[i],
                     marker=marker_list[i], markersize=1.5, mec=color_list[i], mfc=color_list[i])
            i += 1
    plt.legend(loc='best', frameon=False)
    plt.xlim(0, len(x) + 5)
    plt.xlabel('times of repetition', fontsize=15)
    plt.ylabel('R', fontsize=15)
    plt.title('Pearson correlation coefficient (test)', fontsize=15)
    plt.grid([])
    ##########################################################################
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.set_facecolor(color_list[-5])
    i = 0
    for col in df_ME.columns:
        if 'test' in col:
            y = [0] + df_ME[col].values.tolist()
            plt.plot(x, y, label=f'{col}',
                     color=color_list[i],
                     marker=marker_list[i], markersize=1.5, mec=color_list[i], mfc=color_list[i])
            i += 1
    plt.legend(loc='best', frameon=False)
    plt.xlim(0, len(x) + 5)
    plt.xlabel('times of repetition', fontsize=15)
    plt.ylabel('mean error', fontsize=15)
    plt.title('Mean error (test)', fontsize=15)
    plt.grid([])
    ##########################################################################
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                        wspace=0.25, hspace=0.45)
    plt.savefig(os.path.join(out_dir, '训练过程.png'), dpi=300)
    logger.info('输出：训练过程.png')
    return 0


#  可视化预测结果
def result_visual(config, pca, net, x, y):
    # 画图颜色和节点类型
    color_list = ['#735245', '#CCA18D', '#6586B3', '#596D3D', '#8D89C2', '#84E4D0',
                  '#F97623', '#505BB6', '#DE5B7D', '#5B3F7B', '#ADE85B', '#FFA41A',
                  '#2C388E', '#4A9451', '#B32A32', '#FAE215', '#BF51A0', '#068EAC',
                  '#FCFCFC', '#E6E6E6', '#C8C8C8', '#8F8F8E', '#646464', '#323232']
    ##########################################################################
    fig = plt.figure(figsize=(14, 4))
    plt.style.use('ggplot')
    ##########################################################################
    #  PCs解释方差比
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.set_facecolor(color_list[-5])
    x_pcs = [math.log2(i + 1) for i in range(pca.n_components_)]
    y_pcs = [math.log10(i) for i in pca.explained_variance_ratio_]
    plt.plot(x_pcs, y_pcs, label='PCs', color=color_list[2], linewidth=2.5)
    plt.axvline(x=math.log2(config['FN']), color=color_list[14], linewidth=2, linestyle='--',
                label=f'feature:{config["FN"]}')
    plt.legend(loc='best', frameon=False)
    x_tick = [str(i + 1) for i in range(pca.n_components_)]
    for i in range(len(x_tick)):
        if x_tick[i] not in ['1', '2', '4', '7', '13', '26', '52', '113', str(pca.n_components_)]:
            x_tick[i] = ''
    plt.xticks([i for i in x_pcs], x_tick)
    y_tick = [str(round(i, 3)) for i in pca.explained_variance_ratio_]
    for i in range(len(y_tick)):
        if i > 2:
            y_tick[i] = ''
    plt.yticks([i for i in y_pcs], y_tick)
    plt.ylim(-5.5, 0)
    plt.xlabel('components', fontsize=15)
    plt.ylabel('explained variance ratio', fontsize=15)
    plt.title('PCA', fontsize=18)
    plt.grid([])
    ##########################################################################
    #  相关系数
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.set_facecolor(color_list[-5])
    x_pre = torch.FloatTensor(pca.transform(x))[:, :config['FN']]
    y_pre = net(x_pre).tolist()
    R_corr = round(np.corrcoef(y_pre, y)[0][1], 3)

    y_r = [i * 15 + 20 for i in y]
    y_p = [i * 15 + 20 for i in y_pre]
    plt.xlim(0, max(y_r) + 5)
    plt.ylim(0, max(y_r) + 5)
    x_major_locator = MultipleLocator(5)  # 把x轴的刻度间隔设置为5
    y_major_locator = MultipleLocator(5)
    ax2.xaxis.set_major_locator(x_major_locator)  # 设置间隔
    ax2.yaxis.set_major_locator(y_major_locator)
    plt.scatter(y_p, y_r, color=color_list[-2], s=14, alpha=0.55)
    plt.plot([0, max(y_p)], [0, max(y_p)], color=color_list[-1], linewidth=2, linestyle='--',
             label=f'R: {R_corr}')
    plt.plot([0, max(y_r)], [0, max(y_r)], color=color_list[9], linewidth=2, linestyle='--')
    plt.legend(loc='best', frameon=False)
    plt.xlabel('model predict(%)', fontsize=15)
    plt.ylabel('chrY-base(%)', fontsize=15)
    plt.title('R', fontsize=18)
    ##########################################################################
    #  平均误差分布
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.set_facecolor(color_list[-5])
    error_list = []
    error_list_a = error_list.append
    for i in range(len(y_p)):
        error_value = y_p[i] - y_r[i].item()
        error_list_a(error_value)
    sns.histplot(error_list, bins=30, kde=True,
                 color=color_list[0], alpha=0.65,
                 ax=ax3)
    mean_error = np.mean(error_list)
    mean_error = round(float(mean_error), 3)
    plt.axvline(x=mean_error, color=color_list[14], linewidth=2, linestyle='--',
                label=f'mean error:{mean_error}')
    plt.legend(loc='best', frameon=False)
    plt.xlabel('predict - chrY base (%)', fontsize=15)
    plt.ylabel('Density', fontsize=15)
    plt.title('The error distribution', fontsize=18)
    plt.grid([])
    ##########################################################################
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                        wspace=0.25)
    plt.savefig(os.path.join(out_dir, '模型预测结果.png'), dpi=300)
    logger.info('输出：模型预测结果.png')
    logger.info(f'预测完毕，模型性能 R：{R_corr}，样本预测胎儿浓度与实际偏差平均为:{mean_error}\n'
                '----------------------------------------------------------------------------------------')
    return 0


#  主程序
def main_process(config):
    #  获取数据
    data_set = pd.read_table(in_file, header=0)
    feat_set, tag_set = get_data(data_set)
    TR = config['TR']
    pp = int(TR * len(tag_set))  # Partition point
    pp2 = int(1 * TR * len(tag_set))
    logger.info(f'划分数据集，参与训练样本数:{pp}')
    feat_train = feat_set[:pp, :]
    tag_train = tag_set[:pp]
    feat_test = feat_set[pp2:, :]
    tag_test = tag_set[pp2:]
    del data_set
    train_function(config, feat_train, tag_train)
    process_visual()
    predict_function(config, feat_test, tag_test)
    return 0


if __name__ == '__main__':
    # -------------------------------------------------------------------------------------------------------- #
    #  初始化
    logger.info('开始')
    start = time.time()
    device = torch.device("cpu")
    logger.info('获取路径参数')
    in_file = str(args.datalink[0])
    out_dir = str(args.out[0])
    if not os.path.exists(in_file):
        logger.error('缺少数据路径表')
        sys.exit(1)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    logger.info(f'训练数据路径表 {in_file}')
    logger.info(f'输出结果的目录 {out_dir}')
    # -------------------------------------------------------------------------------------------------------- #
    # 参数设置
    Random_seed = int(args.seed[0])  # 随机种子
    random_parameter(Random_seed)
    Train_ratio = float(args.train[0])  # 训练集比例
    Feature_num = int(args.feature[0])  # 特征数
    Epoch_num = int(args.epoch[0])  # 训练次数
    Batch_size = int(args.batch[0])  # 单批次数据量(批量)
    Learning_rate = float(args.learn[0])  # 学习率
    L2_weight_decay = float(args.regularization[0])  # L2正则化
    Dropout_ratio = float(args.dropout[0])  # 随机丢弃率
    KFold_num = int(args.kfold[0])  # K折交叉验证次数
    config_dict = {'RS': Random_seed,
                   'TR': Train_ratio,
                   'FN': Feature_num,
                   'EN': Epoch_num,
                   'BS': Batch_size,
                   'LR': Learning_rate,
                   'L2': L2_weight_decay,
                   'DR': Dropout_ratio,
                   'KN': KFold_num}
    # -------------------------------------------------------------------------------------------------------- #
    logger.info(f'模型参数：\n----------------------------------------------------------------------------------------\n'
                f'随机种子:{Random_seed}\n'
                f'训练集比例:{Train_ratio * 100}%\n'
                f'特征数:{Feature_num}\n'
                f'训练次数:{Epoch_num}\n'
                f'单批次数据量(批量):{Batch_size}\n'
                f'学习率:{Learning_rate}\n'
                f'L2正则化:{L2_weight_decay}\n'
                f'随机丢弃率:{Dropout_ratio}\n'
                f'K折交叉验证次数:{KFold_num}\n'
                f'----------------------------------------------------------------------------------------\n')
    del Random_seed, Train_ratio, Feature_num, Epoch_num, Batch_size, \
        Learning_rate, L2_weight_decay, Dropout_ratio, KFold_num
    # -------------------------------------------------------------------------------------------------------- #
    logger.info(f'运行主程序')
    main_process(config_dict)
    # -------------------------------------------------------------------------------------------------------- #
    end = time.time()
    elapsed_time = round((end-start)/60, 2)
    logger.info(f'耗时：{elapsed_time}分钟')
    logger.info(f'结束')
