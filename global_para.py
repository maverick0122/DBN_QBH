#-*- coding:utf-8 -*-   #允许文档中有中文
"""
全局变量
"""

import theano, copy, sys, json, cPickle
import numpy as np
reload(sys)
sys.setdefaultencoding('utf-8') #允许打印unicode字符

DATASET = './data'  #需要聚类数据所在文件夹
K = 500  #聚类数，设定为跟歌曲数一个量级即可
BORROW = True   # True makes it faster with the GPU
                #设置共享变量时的参数，为true能在GPU上运行更快
NUMPY_ARRAY_ONLY = False     #设置为False时共享变量，用于GPU

N_FRAMES = 20   #特征抽取的窗长(帧数)
DIMENSION = 1   #每帧的数据维数
N_OUTS = K      #输出长度(分类数)，因为使用聚类结果作为标签，所以要和聚类数相同

N_BATCHES_DATASET = 10 # number of batches in which we divide the dataset 
                      # (to fit in the GPU memory, only 2Gb at home)
                      # 每个小批量数据包含的查询数
DBN_PICKLED_FILE = DATASET+'/dbn_qbh.pickle'   #DBN pickle文件的路径
X_DTYPE = 'float64' #训练数据类型
Y_DTYPE = 'int32'   #标签数据类型

PRETRAIN_EPOCHS = 100     #DBN建立时的预训练迭代次数
FINETUNE_EPOCHS = 300     #DBN建立时的微调迭代次数