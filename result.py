#-*- coding:utf-8 -*-   #允许文档中有中文
import numpy as np
from numpy import linalg
import functools
import sys, math
import cPickle
import os
sys.path.append(os.getcwd())

import sys
reload(sys)
sys.setdefaultencoding('utf-8') #允许打印unicode字符

usage = """
python result.py OUTPUT[.mlf] INPUT_SCP INPUT_HMM  
        [--d DBN_PICKLED_FILE DBN_TO_INT_TO_STATE_DICTS_TUPLE]

Exclusive uses of these options:
    --d followed by a pickled DBN file
"""

N_BATCHES_DATASET = 32 # number of batches in which we divide the dataset 
                      # (to fit in the GPU memory, only 2Gb at home)
N_FRAMES = 20   #特征抽取的窗长(帧数)


def compute_likelihoods_dbn(dbn, mat, depth=np.iinfo(int).max, normalize=True, unit=False):
    """ compute the log-likelihoods of each states i according to the Deep 
    Belief Network (stacked RBMs) in dbn, for each line of mat (input data) 
    depth is the depth of the DBN at which the likelihoods will pop out,
    if None, the full DBN is used"""
    # first normalize or put in the unit ([0-1]) interval
    # TODO do that only if we did not do that at the full scale of the corpus
    if normalize:
        # if the first layer of the DBN is a Gaussian RBM, we need to normalize mat
        mat = (mat - np.mean(mat, 0)) / np.std(mat, 0)
    elif unit:
        # if the first layer of the DBN is a binary RBM, send mat in [0-1] range
        mat = (mat - np.min(mat, 0)) / np.max(mat, 0)

    import theano.tensor as T
    ret = np.ndarray((mat.shape[0], dbn.logLayer.b.shape[0].eval()), dtype="float32")
    from theano import shared#, scan
    # propagating through the deep belief net
    batch_size = mat.shape[0] / N_BATCHES_DATASET
    max_layer = dbn.n_layers
    out_ret = None
    if depth < dbn.n_layers:
        max_layer = min(dbn.n_layers, depth)
        print max_layer
        out_ret = np.ndarray((mat.shape[0], dbn.rbm_layers[max_layer].W.shape[1].eval()), dtype="float32")
    else:
        out_ret = np.ndarray((mat.shape[0], dbn.logLayer.b.shape[0].eval()), dtype="float32")

    for ind in xrange(0, mat.shape[0]+1, batch_size):
        output = shared(mat[ind:ind+batch_size])
        print "evaluating the DBN on all the test input"
        for layer_ind in xrange(max_layer):
            [pre, output] = dbn.rbm_layers[layer_ind].propup(output)
        if depth >= dbn.n_layers:
            print "dbn output shape", output.shape.eval()
            ret = T.nnet.softmax(T.dot(output, dbn.logLayer.W) + dbn.logLayer.b)
            out_ret[ind:ind+batch_size] = T.log(ret).eval()
        else:
            out_ret[ind:ind+batch_size] = T.log(output).eval()
    return out_ret


def process(ofname, iqueryfname, idbnfname):
    '''处理函数，建立DBN网络，进行查询，输出结果
    ofname: 输出文件
    iqueryfname: 查询文件，每行一个查询，维数必须为N_FRAMES
    idbnfname: DBN模型
    '''

    #建立DBN网络模型
    dbn = None
    if idbnfname != None:
        with open(idbnfname) as idbnf:
            dbn = cPickle.load(idbnf)
        #partial函数：先将不变的参数dbn导入函数，防止每次调用函数重复导入
        likelihoods_computer = functools.partial(compute_likelihoods_dbn, dbn)
        # like that = for GRBM first layer (normalize=True, unit=False)
        # TODO correct the normalize/unit to work on full test dataset

    if dbn != None:
        input_n_frames = dbn.rbm_layers[0].n_visible 
        print "this is a DBN with", input_n_frames, "frames on the input layer"
        try: 
            print "loading query from pickled file", iqueryfname
            #读取查询
            with open(iqueryfname) as iqueryf:
                query = np.load(iqueryf)
        except:
            #读取失败，初始化为0
            query = np.ndarray((0, N_FRAMES), dtype='float32')

    #计算似然性（查询属于每个类的概率）
    print "computing likelihoods"
    if dbn != None:
        likelihoods = likelihoods_computer(query)
        #mean_dbns = np.mean(tmp_likelihoods, 0)
        #tmp_likelihoods *= (mean_gmms / mean_dbns)
        print likelihoods
        print likelihoods.shape
        #print likelihoods[0]
        #print likelihoods[0].shape

    #保存查询结果
    with open(ofname, 'w') as of:
        of.write('#!MLF!#\n')
        for line in list_mlf_string:
            of.write(line)


if __name__ == "__main__":
    if len(sys.argv) > 3:
        if '--help' in sys.argv:
            print usage
            sys.exit(0)
        args = dict(enumerate(sys.argv))
        options = filter(lambda (ind, x): '--' in x[0:2], enumerate(sys.argv))
        dbn_fname = None # DBN cPickle
        if len(options): # we have options
            for ind, option in options:
                args.pop(ind)
                if option == '--d':
                    try:
                        from DBN_Gaussian_timit import DBN # not Gaussian if no GRBM
                    except:
                        print >> sys.stderr, "experimental: TO BE LAUNCHED FROM THE 'DBN/' DIR"
                        sys.exit(-1)
                    dbn_fname = args[ind+1]
                    args.pop(ind+1)
                    print "will use the following DBN to estimate states likelihoods", dbn_fname
        else:
            print "initialize the transitions between phones uniformly"
        output_fname = args.values()[1]
        input_scp_fname = args.values()[2]
        process(output_fname, input_scp_fname, dbn_fname)
    else:
        print usage
        sys.exit(-1)
