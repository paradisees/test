#-*- coding: utf-8 -*-
#使用ID3决策树算法预测销量高低
import pandas as pd
import codecs
#参数初始化
inputfile = 'C:/Users/hhy/Desktop/3.csv'
data = pd.read_csv(inputfile,encoding='gbk') #导入数据

#数据是类别标签，要将它转换为数据
#用1来表示“好”、“是”、“高”这三个属性，用-1来表示“坏”、“否”、“低”

x = data.iloc[:,:25].as_matrix().astype(int)
y = data.iloc[:,25].as_matrix().astype(int)

from sklearn.tree import DecisionTreeClassifier as DTC
dtc = DTC(criterion='entropy') #建立决策树模型，基于信息熵
dtc.fit(x, y) #训练模型

#导入相关函数，可视化决策树。
#导出的结果是一个dot文件，需要安装Graphviz才能将它转换为pdf或png等格式。
from sklearn.tree import export_graphviz
x = pd.DataFrame(x)
from sklearn.externals.six import StringIO
with open("tree.dot", 'w') as f:
  f = export_graphviz(dtc, feature_names = x.columns, out_file = f)
