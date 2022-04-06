# NIPT-FFMLP使用说明

------



## 目录

[TOC]

--------------------------------------------------------------------------------------------

## 描述

### github地址

```
https://github.com/happenywong/NIPT-FFMLP.git
```

### 主要功能

```
用每个窗口的read数作为特征、胎儿浓度作为标签，训练神经网络模型
训练完成的神经网络模型可用于NIPT胎儿浓度的预测
```

### 环境需求

```
语言：python3,perl,shell
测试环境：windows10,windows11,Linux Ubuntu 16.04.3
#------系统------#
os
sys
time
logging
warnings
argparse
#------可视化------#
matplotlib	3.4.3
seaborn	0.11.2
#------文件处理------#
pandas	1.3.4
numpy	1.20.3
#------建模------#
math
sklearn	0.0
joblib	1.1.0
torch	1.10.0
```

------

## 工作说明

### 原始数据说明

#### 样本信息汇总

例：./examples/database/database.txt

需从数据库中获取

```
ID	runid	samid	barcode	plus	time	fetus_num	raw_reads	gender	readlen	GC	Q20	disease	dup_reads	dup_ratio	chr1_z_score	chr2_z_score	chr3_z_score	chr4_z_score	chr5_z_score	chr6_z_score	chr7_z_score	chr8_z_score	chr9_z_score	chr10_z_score	chr11_z_score	chr12_z_score	chr13_z_score	chr14_z_score	chr15_z_score	chr16_z_score	chr17_z_score	chr18_z_score	chr19_z_score	chr20_z_score	chr21_z_score	chr22_z_score	chrX_z_score	chrY_z_score	FF
...
```

#### 样本组20kb窗口信息（未进行GC矫正）

<img src=".\imgs\20k_GC未矫正.png" alt="20k_GC未矫正" style="zoom:24%;" />

例：./examples/rawdata/readnum_beforeGC_20kb

需从备份文件中获取

```
#chr	start	end	IonXpress_006	IonXpress_017	IonXpress_020
chr1	0	20000	0	0	1	
chr1	20000	40000	0	0	0
chr1	40000	60000	4	2	6
... ... ... ... ... ...
... ... ... ... ... ...
... ... ... ... ... ...
chrY	59320000	59340000	0	0	0
chrY	59340000	59360000	0	0	0
chrY	59360000	59380000	0	0	0
```

#### 样本组20kb窗口信息（GC矫正）

<img src=".\imgs\20k_GC矫正.png" alt="20k_GC矫正" style="zoom:24%;" />

例：./examples/rawdata/readnum_afterGC_20kb

通过调用脚本 gc_correct_run.pl 对样本组20kb窗口信息（./examples/rawdata/readnum_beforeGC_20kb/?*）进行GC矫正，例：./scripts/run_gc_correct.sh

```
perl ./gc_correct_run.pl ../examples/rawdata/readnum_afterGC_20kb/ ../examples/rawdata/readnum_beforeGC_20kb/?*
#Linux可多进程
perl ./multi-process.pl --cpu 40 ./run_gc_correct.sh
```

```
IonXpress_006	IonXpress_017	IonXpress_020
140.855	33.485	48.585
48.855	16.485	21.585
67.855	24.485	17.585
... ... ...
... ... ...
... ... ...
0.000	1.000	0.000
2.000	1.000	0.000
0.000	1.000	0.000
```

#### 单个样本20kb窗口信息

例：./examples/rawdata/bednum_20kb

通过调用脚本 obtain_bednum_20kb.py 对样本组20kb窗口信息（./examples/rawdata/readnum_afterGC_20kb/?*.gccorrect.filtered）进行拆分，例：

```
python3 obtain_bednum_20kb.py -P ../examples/rawdata/readnum_afterGC_20kb/?*.gccorrect.filtered
```

```
chr	start	end	ratio
1	0	20000	0.0
1	20000	40000	0.0
1	40000	60000	0.0
... ... ... ...
... ... ... ...
... ... ... ...
Y	59320000	59340000	0.0
Y	59340000	59360000	0.0
Y	59360000	59380000	0.0
```

#### 单个样本60kb窗口信息

如果20kb窗口已经进行过GC矫正，则不需要再次矫正

<img src=".\imgs\60k_GC未矫正.png" style="zoom:24%;" />

例：./examples/rawdata/bednum_60kb

通过调用脚本 merge_win_60kb.py 对单个样本20kb窗口信息（./examples/rawdata/bednum_20kb/?*）进行合并，例：

```
python3 merge_win_60kb.py -P ../examples/rawdata/bednum_20kb/?*
```

```
chr	start	end	ratio
1	0	60000	0
1	60000	120000	0
1	120000	180000	0
... ... ... ...
... ... ... ...
... ... ... ...
Y	59220000	59280000	0
Y	59280000	59340000	0
Y	59340000	59380000	0
```



### 输入文件说明

#### 样本路径列表datalist.txt

例：./examples/database/datalist.txt

通过调用脚本 obtain_datalist.py 对样本60kb窗口信息（./examples/rawdata/bednum_60kb/?*）获取，例：

```
python3 obtain_datalist.py
# 默认路径
in_default = r'../examples/database/database.txt'
out_default = r'../examples/database/datalist.txt'
bed_default = r'../examples/rawdata/bednum_60kb'
```

```
ID	filepath	FF
IonXpress_010	../examples/rawdata/bednum_60kb\IonXpress_010	21.78700066
IonXpress_020	../examples/rawdata/bednum_60kb\IonXpress_020	30.66057777
```

### 多进程说明

gc_correct_run.pl，obtain_bednum_20kb.py，merge_win_60kb.py，use_model_V1.0.py 

以上脚本都建议配合shell脚本进行多进程调用节省时间，例：

```
perl ./multi-process.pl --cpu 40 ./run_gc_correct.sh
```

### 训练模型

#### 帮助查看可输入参数

```
python3 ff_model_V1.0.py -h
```

例：

```
python3 ff_model_V1.0.py -D ../examples/database/datalist.txt -O ../examples/result/model -FN 4
```

| Optional argument     | Function                   | Default |
| :-------------------- | -------------------------- | ------- |
| -h，--help            | 查看帮助                   |         |
| -V, --version         | 查看脚本版本               | 1.0.0.2 |
| -D, --datalink        | 用于训练的样本路径表       |         |
| -O, --out             | 输出结果的目录             |         |
| -RS, --seed           | 输入随机种子               | 66      |
| -TR, --train          | 输入作为训练集的百分比     | 0.9     |
| -FN, --feature        | 输入神经网络的特征数量     | 4       |
| -EN, --epoch          | 输入神经网络的训练批次数   | 5       |
| -BS, --batch          | 输入神经网络的每批次样本数 | 2048    |
| -LR, --learn          | 输入学习率                 | 0.001   |
| -L2, --regularization | 输入L2正则化系数           | 0.0001  |
| -DR, --dropout        | 输入神经元随机丢弃率       | 0.0001  |
| -KN, --kfold          | 输入K折交叉验证次数        | 10      |



### 结果说明

输出文件目录， 例：./examples/result/model

#### 训练过程

##### 记录文件

loss.txt，R.txt，ME.txt 分别记录损失值，相关系数，平均绝对误差

训练过程.png为训练过程中损失值，相关系数，平均绝对误差的可视化

##### 描述模型性能文件

模型预测结果.png

##### 模型和配置文件

win_index.txt，PCA.pkl， MLP.pkl

需要调用模型时，以上三个文件不可以缺少

#### 性能评估

**训练过程.png** 展示模型是否训练完成，验证集损失函数不再明显下降说明模型训练达到瓶颈

<img src=".\imgs\训练过程.png" alt="训练过程" style="zoom:24%;" />

**模型预测结果.png** 展示PCA对特征造成的影响，相关系数R和平均误差可以直观表示模型性能

<img src=".\imgs\模型预测结果.png" alt="模型预测结果" style="zoom:24%;" />

#### 调用模型

```
python3 use_model_V1.0.py
# 默认路径
in_default = [r'../examples/database/datalist.txt']
model_default = [r'../examples/result/model']
out_default = [r'../examples/result']
```

#### 输出结果

```
ID	filepath	FF	predict
IonXpress_010	../examples/rawdata/bednum_60kb\IonXpress_010	21.78700066	21.17458324879408
IonXpress_020	../examples/rawdata/bednum_60kb\IonXpress_020	30.66057777	19.18106146156788
```

| Optional argument | Function               | Default |
| ----------------- | ---------------------- | ------- |
| -h，--help        | 查看帮助               |         |
| -V, --version     | 查看脚本版本           | 1.0.0.2 |
| -D, --datalink    | 用于训练的样本路径表   |         |
| -M, --model       | 模型文件夹             |         |
| -O, --out         | 输出结果的目录         |         |
| -FN, --feature    | 输入神经网络的特征数量 |         |

------

## 更新日志

### ff_model_V1.0.py

```
###############################################
'''
author：hap
create data:2022/1/14
update data:2022/2/14
version：1.0.0.2
功能：
用每个窗口的read数作为特征、胎儿浓度作为标签，训练神经网络模型
'''
更新：
优化PCA可视化
决策改为使用最优MAD
###############################################
'''
author：hap
create data:2022/1/14
update data:2022/1/18
version：1.0.0.1
功能：
用每个窗口的read数作为特征、胎儿浓度作为标签，训练神经网络模型
'''
更新：
添加训练过程可视化，展示训练过程中损失函数和相关系数的变化。
模块为def process_visual()
###############################################
'''
author：hap
create data:2022/1/14
update data:2022/1/14
version：1.0.0.0
功能：
用每个窗口的read数作为特征、胎儿浓度作为标签，训练神经网络模型
'''
更新：
测试完成，正式版的版本号为1.0.0.0
###############################################
```

