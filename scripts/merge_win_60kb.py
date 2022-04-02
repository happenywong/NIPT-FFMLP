#!/usr/bin/python3
# -*- codeing = utf-8 -*-


import os
import pandas as pd
import time
import argparse
################################################

'''
author：hap
version：1.0.0.2
功能：
将20kb窗口合并为60kb
'''
# 默认路径
in_default = r'../examples/rawdata/bednum_20kb' \
             r'/Auto_user_sn247560254-10-pooling-320flows-2022DSEQ0099_1452_2441_IonXpress_006'
out_default = r'../examples/rawdata/bednum_60kb'

################################################
# 可输入参数
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
parser = argparse.ArgumentParser(description='合并20kb窗口为60kb')
parser.add_argument('-P', '--path', type=str, default=in_default, help='sample路径')
args = parser.parse_args()
################################################


def merge_function(sample_path):
    sample_id = os.path.basename(sample_path)
    df = pd.read_table(sample_path, header=0, low_memory=False)
    chr_id = df['chr'].unique().tolist()
    merge_list = []
    merge_list_a = merge_list.append
    for i in chr_id:
        df_chr = df.groupby('chr').get_group(i)
        chr_len = len(df_chr['chr'].values.tolist())
        index_list = range(0, chr_len, 3)
        for x in index_list:
            start_win = df_chr.iloc[x, :]['start']
            try:
                end_win = df_chr.iloc[x + 2, :]['end']
                ratio = int(df_chr.iloc[x:x + 2, :]['ratio'].sum())
                merge_list_a([i, start_win, end_win, ratio])
            except IndexError:
                end_win = df_chr.iloc[chr_len - 1, :]['end']
                ratio = int(df_chr.iloc[x:x + 2, :]['ratio'].sum())
                merge_list_a([i, start_win, end_win, ratio])
    del df
    df = pd.DataFrame(merge_list, columns=['chr', 'start', 'end', 'ratio'])
    df.to_csv(os.path.join(out_dir, sample_id), header=True, index=False, sep='\t')
    return 0


if __name__ == '__main__':
    in_file = args.path
    out_dir = out_default
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    start_all = time.time()
    merge_function(in_file)
    end_all = time.time()
    use_time_all = round(end_all - start_all, 2)
    print(f'耗时:{use_time_all}s')
