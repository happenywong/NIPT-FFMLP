#!/usr/bin/python3
# -*- codeing = utf-8 -*-


import os
import pandas as pd
import time
import argparse
from pandas.api.types import CategoricalDtype

################################################

'''
author：hap
version：1.0.0.2
功能：
生成bed文件
'''
# 默认路径
in_default = r'../examples/rawdata/readnum_afterGC_20kb' \
             r'/Auto_user_sn247560254-10-pooling-320flows-2022DSEQ0099_1452_2441.gccorrect.filtered'
out_default = r'../examples/rawdata/bednum_20kb'
win_fill = r'../config/20k_fill'
win_filter = r'../config/20k_filter'
################################################
# 可输入参数
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
parser = argparse.ArgumentParser(description='从run拆分每个barcode的窗口信息')
parser.add_argument('-P', '--path', type=str, default=in_default, help='run路径')
args = parser.parse_args()
################################################


def get_barcode(run_path):
    run_id = os.path.basename(run_path).split('.')[0]
    df = pd.read_table(run_path, header=0)
    chr_order = CategoricalDtype(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
                                  '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
                                  '21', '22', 'X', 'Y'],
                                 ordered=True)  # 自定义排序
    for index, row in df.iteritems():
        df_barcode = pd.read_table(win_filter, header=0).iloc[:, :3]
        df_barcode['ratio'] = df[index]
        df_fill = pd.read_table(win_fill, header=0, low_memory=False)
        df_barcode = pd.concat([df_barcode, df_fill])
        df_barcode.drop_duplicates(subset=['chr', 'start', 'end'], keep='first', inplace=True)
        df_barcode['chr'] = df_barcode['chr'].astype(chr_order)
        df_barcode.sort_values(by=['chr', 'start'], axis=0, ascending=True, inplace=True)
        df_barcode.to_csv(os.path.join(out_dir, fr'{run_id}_{df[index].name}'), sep='\t', header=True, index=False)
    return 0


if __name__ == '__main__':
    in_file = args.path
    out_dir = out_default
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    start_all = time.time()
    get_barcode(in_file)
    end_all = time.time()
    use_time_all = round(end_all - start_all, 2)
    print(f'耗时:{use_time_all}s')
