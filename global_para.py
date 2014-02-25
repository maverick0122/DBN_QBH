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

TRAIN_X_FILE = "/train_xdata.npy"   #训练集数据文件名
TRAIN_Y_SONG = "/train_ylabels_song.npy"    #歌曲标签文件
TRAIN_Y_KMEANS = "/train_ylabels_kmeans.npy"    #聚类标签文件
TRAIN_Y_FILE = TRAIN_Y_SONG  #训练集标签文件名

N_OUTS = 200      #输出长度(分类数)，若使用聚类结果作为标签，设置为聚类数
                #若使用所属歌曲作为标签，设置为歌曲数

BATCH_SIZE = 10 #训练时每个小批量数据的大小

DBN_PICKLED_FILE = DATASET+'/dbn_qbh_song.pickle'   #DBN pickle文件的路径
X_DTYPE = 'float64' #训练数据类型
Y_DTYPE = 'int32'   #标签数据类型

PRETRAIN_EPOCHS = 100     #DBN建立时的预训练迭代次数
FINETUNE_EPOCHS = 200     #DBN建立时的微调迭代次数