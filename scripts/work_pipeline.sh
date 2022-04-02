#------详细说明查看README.md------#
#---准备NIPT数据库信息,样本对应的20k原始文件
where database.txt
where readnum_beforeGC_20kb
#---多进程对20k原始数据进行GC矫正
perl ./multi-process.pl --cpu 40 ./run_gc_correct.sh
#---GC后的20k原始数据按照barcode拆分
python3 obtain_bednum_20kb.py -P ../examples/rawdata/readnum_afterGC_20kb/?*.gccorrect.filtered
#---拆分后单个样本的20k窗口合并为60k
python3 merge_win_60kb.py -P ../examples/rawdata/bednum_20kb/?*IonXpress_*
#---获取用于训练模型的datalist
python3 obtain_datalist.py
#---训练模型
python3 ff_model_V1.0.py
#---调用模型
python3 use_model_V1.0.py
#------详细说明查看README.md------#