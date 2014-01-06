#-*- coding:utf-8 -*-   #允许文档中有中文
import theano, copy, sys, json, cPickle
import theano.tensor as T
import numpy as np
from scipy.cluster.vq import *

DATASET = './data'
K = 60

def prep_data(dataset):
    try:
        train_x = np.load(dataset + "/aligned_train_xdata.npy")
        #train_y = np.load(dataset + "/aligned_train_ylabels.npy")

    except:
        print >> sys.stderr, "you need the .npy python arrays"
        print >> sys.stderr, "you can produce them with txt_to_numpy.py"
        print >> sys.stderr, dataset + "/aligned_train_xdata.npy"
        sys.exit(-1)

    print "train_x shape:", train_x.shape

    return train_x


def kmeans(dataset,k):
    train_x = prep_data(dataset)

    centroid,label = kmeans2(train_x,k)
    #print centroid
    for i in label:
        print i
    from collections import Counter
    c = Counter(label)    #统计label的每个值的个数，即每个分类的类名和属于该分类的样本数
    print c
    

if __name__ == '__main__':
    kmeans(DATASET,K)