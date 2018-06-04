Input: filter-raw-data.txt, word2vec_model
其中filter-raw-data.txt为初始的feature数据，但是里面并没有将title、body和comment转化为向量。
word2vec_model为训练好的词向量。
生成数据的python文件为：features.py、features_k.py、features_time.py、features_time_k.py。
features.py和features_k.py分别为生成Accept的数据集和除去k8s项目的Accept数据集。
features_time.py和features_time_k.py为生成Response数据集和除去k8s项目Response数据集。


训练文件为：model_predict.py 和model_predict_time.py，其余类似名字的文件为变种。但是功能基本一致，只是有少许改动。

model_predict.py为训练accept模型的，model_predict_time.py为训练Response模型的。其中包括SVM、LR、RF和XGBoost模型，每个模型跑十遍，算取AUC。
Step 1: 运行features.py和features_time.py生成相应的训练数据集。
Step 2: 运行model_predict.py和model_predict_time.py进行模型训练和AUC计算。



onedaymodel.py为使用保存的xgboost进行排序预测，该程序会在本目录下生成一个目录，其中包含k8s项目的2018年1月份的每天的排序结果。cal_*.py文件则是根据排序结果进行各项指标计算并画图的脚本文件。
