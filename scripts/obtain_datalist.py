#!/usr/bin/python3
# -*- codeing = utf-8 -*-


import os
import pandas as pd

################################################

'''
author：hap
version：1.0.0.2
功能：
获取训练的数据列表
'''
# 默认路径
in_default = r'../examples/database/database.txt'
out_default = r'../examples/database/datalist.txt'
bed_default = r'../examples/rawdata/bednum_60kb'
################################################


def read_database(in_file, out_file, bed_dir):
    df = pd.read_table(in_file, header=0)
    bed_list = os.listdir(bed_dir)
    flag_list, path_list, FF_list = [], [], []
    flag_list_a, path_list_a, FF_list_a = \
        flag_list.append, path_list.append, FF_list.append
    for index, row in df.iterrows():
        unique_id = f"{row['runid']}_{row['barcode']}"
        if unique_id in bed_list:
            flag_list_a(unique_id)
            path_list_a(os.path.join(bed_dir, unique_id))
            FF_list_a(float(row['FF']))
    df_list = pd.DataFrame(columns=['ID', 'filepath', 'FF'])
    df_list['ID'] = flag_list
    df_list['filepath'] = path_list
    df_list['FF'] = FF_list
    df_list.to_csv(out_file, sep='\t', index=False, header=True)
    return 0


if __name__ == '__main__':
    read_database(in_default, out_default, bed_default)
