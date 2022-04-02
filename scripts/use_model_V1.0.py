#!/usr/bin/python3
# -*- codeing = utf-8 -*-

import os
import sys
import time
import warnings
import argparse
import torch
import pandas as pd
import numpy as np
import joblib  # 保存sklearn模型
import torch.nn as nn  # 定义网络
import torch.nn.functional as functional  # 激励函数

################################################
# 作者信息
"""
author：hap
create data:2022/1/14
update data:2022/2/14
version：1.0.0.2
功能：
调用模型
"""
################################################
# 默认路径
in_default = [r'../examples/database/datalist.txt']
model_default = [r'../examples/result/model']
out_default = [r'../examples/result']

################################################
# 版本
program = sys.argv[0]
version = '1.0.0.2'

################################################
# 日志
warnings.filterwarnings("ignore")  # 关闭警告信息提示

################################################
# 可输入参数
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
parser = argparse.ArgumentParser(description='胎儿浓度预测')
parser.add_argument('-V', '--version', action='version', version='%(prog)s ' + version)
parser.add_argument('-D', '--datalink', nargs=1, default=in_default, help='用于训练的样本路径表')
parser.add_argument('-M', '--model', nargs=1, default=model_default, help='模型文件夹')
parser.add_argument('-O', '--out', nargs=1, default=out_default, help='输出结果的目录')
parser.add_argument('-FN', '--feature', nargs=1, default=[4], help='输入神经网络的特征数量')
args = parser.parse_args()


################################################

# 构建神经网络
class MLP(nn.Module):  # 继承 torch 的 Module
    def __init__(self, n_feature, DR=0):
        super(MLP, self).__init__()  # 继承 __init__ 功能
        # 定义每层用什么样的形式
        self.DR = DR
        self.hidden1 = nn.Linear(n_feature, n_feature * 3 + 1)  # 隐藏层1
        self.hidden2 = nn.Linear(n_feature * 3 + 1, n_feature)  # 隐藏层2
        self.predict = nn.Linear(n_feature, 1)  # 输出层

    def forward(self, n_input):  # 这同时也是 Module 中的 forward 功能
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


def fill_function(database_file):
    df = pd.read_table(database_file)
    df['predict'] = 0.0
    win_index = []
    win_index_a = win_index.append
    pca = joblib.load(pca_config)
    features_num = int(args.feature[0])
    net = MLP(features_num)
    net.load_state_dict(torch.load(mlp_config))
    with open(win_config, 'r', encoding='utf-8') as config:
        for win in config.readlines():
            win_index_a(int(win.strip()))
    for index, row in df.iterrows():
        try:
            df_bed = pd.read_table(row['filepath'], header=0)
            bool_ruler = (df_bed['chr'] != 'Y') & (df_bed['chr'] != 'X') & \
                         (df_bed['chr'] != '13') & (df_bed['chr'] != '18') & (df_bed['chr'] != '21')
            median_ratio = np.median([i for i in df_bed[bool_ruler]['ratio'].values.tolist() if i != 0])
            df_ratio = pd.DataFrame([[i / median_ratio for i in df_bed[bool_ruler]['ratio'].values.tolist()]])
            x_tensor = torch.FloatTensor(np.array(df_ratio[win_index]).astype('float'))
            x_pre = torch.FloatTensor(pca.transform(x_tensor))[:, :features_num]
            y_pre = net(x_pre).tolist()[0] * 15 + 20
            df.at[index, 'predict'] = y_pre
            ID = row["ID"]
            end = round(time.time() - start, 2)
            print(f'耗时{end}s 完成:{ID}')
        except:
            ID = row["ID"]
            print(f'错误:{ID}')
    df.to_csv(os.path.join(out_dir, f"{os.path.basename(database_file).split('.')[0]}_predict.txt")
              , sep='\t', index=False, header=True)
    return 0


if __name__ == '__main__':
    start = time.time()
    in_file = str(args.datalink[0])
    model_dir = str(args.model[0])
    out_dir = str(args.out[0])
    win_config = os.path.join(model_dir, r'win_index.txt')
    pca_config = os.path.join(model_dir, r'PCA.pkl')
    mlp_config = os.path.join(model_dir, r'MLP.pkl')
    fill_function(in_file)
