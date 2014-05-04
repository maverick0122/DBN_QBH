#-*- coding:utf-8 -*-   #允许文档中有中文
"""
读入训练数据和标签，建立DBN模型

训练数据需满足的要求：
1.所有数据维数相同，对应DBN模型的n_ins参数
2.每条数据对应一个标签
系统预处理时会调用prep_qbh.py将训练数据分成按训练集，验证集，测试集三部分

标签需满足的要求：
1.标签种类数必须和DBN输出维数n_outs相同
2.标签内容必须是0-(n_outs-1)范围内的数字

"""

import gzip
import os
import time
import numpy
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from logistic_sgd import LogisticRegression #, load_data
from mlp import HiddenLayer
from rbm import RBM
from prep_qbh import load_data
from global_para import *


class DBN(object):
    """Deep Belief Network

    A deep belief network is obtained by stacking several RBMs on top of each
    other. The hidden layer of the RBM at layer `i` becomes the input of the
    RBM at layer `i+1`. The first layer RBM gets as input the input of the
    network, and the hidden layer of the last RBM represents the output. When
    used for classification, the DBN is treated as a MLP, by adding a logistic
    regression layer on top.
    """

    '''深度置信网络(DBN)

    一个深度置信网络由若干叠加的受限波茨曼机(RBM)相互组成
    第i层RBM的输出是i+1层的输入，第1层的输入是网络输入，最后一层输出是网络输出
    用作分类时，通过在顶层加入一个logistic回归，DBN被当作一个多层感知器网络(MLP)
    '''

    def __init__(self, numpy_rng, theano_rng=None, n_ins=DIMENSION * N_FRAMES,
                 hidden_layers_sizes=[1024, 1024, 1024], n_outs=N_OUTS):
        """This class is made to support a variable number of layers.

        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: numpy random number generator used to draw initial
                    weights
                    numpy随机数生成器,用于初始化权重

        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given one is
                           generated based on a seed drawn from `rng`
                           theano随机数生成器

        :type n_ins: int
        :param n_ins: dimension of the input to the DBN
                        DBN的输入样本维数

        :type n_layers_sizes: list of ints
        :param n_layers_sizes: intermediate layers size, must contain
                               at least one value
                               每个隐层大小,至少一个数

        :type n_outs: int
        :param n_outs: dimension of the output of the network
                        输出维数
        """

        self.sigmoid_layers = []
        self.rbm_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)

        assert self.n_layers > 0

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        # allocate symbolic variables for the data
        # 为数据开辟符号变量
        self.x = T.matrix('x')  # the data is presented as rasterized images 数据表示为光栅图像
        self.y = T.ivector('y')  # the labels are presented as 1D vector 标签表示为一维整形数组
                                 # of [int] labels

        # The DBN is an MLP, for which all weights of intermediate
        # layers are shared with a different RBM.  We will first
        # construct the DBN as a deep multilayer perceptron, and when
        # constructing each sigmoidal layer we also construct an RBM
        # that shares weights with that layer. During pretraining we
        # will train these RBMs (which will lead to chainging the
        # weights of the MLP as well) During finetuning we will finish
        # training the DBN by doing stochastic gradient descent on the
        # MLP.

        #DBN是一个MLP,每个中间层的权重被一个RBM共享，且RBM互不相同.先建立一个MLP,
        #在建立MLP的若干sigmoid层时建立对应的RBM层，RBM层共享sigmoid层的权重.
        #预训练时训练RBM，并引发MLP的权重改变
        #微调时通过在MLP中做随机梯度下降完成训练

        for i in xrange(self.n_layers):
            # construct the sigmoidal layer

            # the size of the input is either the number of hidden
            # units of the layer below or the input size if we are on
            # the first layer
            # 每层的输入大小：在第一层，为输入数据大小
            # 不在第一层，为上一层输出大小
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layers_sizes[i - 1]

            # the input to this layer is either the activation of the
            # hidden layer below or the input of the DBN if you are on
            # the first layer
            # 每层的输入数据：在第一层，为输入数据
            # 不在第一层，为上一层输出
            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[-1].output

            sigmoid_layer = HiddenLayer(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=hidden_layers_sizes[i],
                                        activation=T.nnet.sigmoid)

            # add the layer to our list of layers
            self.sigmoid_layers.append(sigmoid_layer)

            # its arguably a philosophical question...  but we are
            # going to only declare that the parameters of the
            # sigmoid_layers are parameters of the DBN. The visible
            # biases in the RBM are parameters of those RBMs, but not
            # of the DBN.
            #sigmoid层的参数是DBN的参数
            #但是RBM的可见偏置是RBM的参数，不是DBM的参数
            self.params.extend(sigmoid_layer.params)

            # Construct an RBM that shared weights with this layer
            # 建立共享这一层权重的RBM
            rbm_layer = RBM(numpy_rng=numpy_rng,
                            theano_rng=theano_rng,
                            input=layer_input,
                            n_visible=input_size,
                            n_hidden=hidden_layers_sizes[i],
                            W=sigmoid_layer.W,
                            hbias=sigmoid_layer.b)
            self.rbm_layers.append(rbm_layer)

        # We now need to add a logistic layer on top of the MLP
        # 在MLP上方加一个logistic层
        self.logLayer = LogisticRegression(
            input=self.sigmoid_layers[-1].output,
            n_in=hidden_layers_sizes[-1],
            n_out=n_outs)
        self.params.extend(self.logLayer.params)

        # compute the cost for second phase of training, defined as the
        # negative log likelihood of the logistic regression (output) layer
        # 计算训练第二阶段的代价，定义为logistic回归的负对数似然性
        self.finetune_cost = self.logLayer.negative_log_likelihood(self.y)

        # compute the gradients with respect to the model parameters
        # symbolic variable that points to the number of errors made on the
        # minibatch given by self.x and self.y
        # 根据模型参数计算梯度
        # 指向有小批量数据产生的错误数的符号变量由self.x，self.y给出
        self.errors = self.logLayer.errors(self.y)

    def pretraining_functions(self, train_set_x, batch_size, k):
        '''Generates a list of functions, for performing one step of
        gradient descent at a given layer. The function will require
        as input the minibatch index, and to train an RBM you just
        need to iterate, calling the corresponding function on all
        minibatch indexes.
        生成一系列函数，在给定的层执行一步梯度下降。
        函数需要小批量索引作为输入，在所有小批量索引上执行相关函数

        :type train_set_x: theano.tensor.TensorType
        :param train_set_x: Shared var. that contains all datapoints used
                            for training the RBM
                            共享变量，包含训练RBM的所有数据点
        :type batch_size: int
        :param batch_size: size of a [mini]batch 小批量数据的大小
        :param k: number of Gibbs steps to do in CD-k / PCD-k 做Gibbs步骤的数目

        '''
        # index to a [mini]batch
        index = T.lscalar('index')  # index to a minibatch
        learning_rate = T.scalar('lr')  # learning rate to use

        # number of batches
        n_batches = train_set_x.get_value(borrow=BORROW).shape[0] / batch_size
        # begining of a batch, given `index`
        batch_begin = index * batch_size
        # ending of a batch given `index`
        batch_end = batch_begin + batch_size

        pretrain_fns = []
        for rbm in self.rbm_layers:

            # get the cost and the updates list
            # using CD-k here (persisent=None) for training each RBM.
            # TODO: change cost function to reconstruction error
            cost, updates = rbm.get_cost_updates(learning_rate,
                                                 persistent=None, k=k)

            # compile the theano function
            fn = theano.function(inputs=[index,
                            theano.Param(learning_rate, default=0.1)],
                                 outputs=cost,
                                 updates=updates,
                                 givens={self.x:
                                    train_set_x[batch_begin:batch_end]})
            # append `fn` to the list of functions
            pretrain_fns.append(fn)

        return pretrain_fns

    def build_finetune_functions(self, datasets, batch_size, learning_rate):
        '''Generates a function `train` that implements one step of
        finetuning, a function `validate` that computes the error on a
        batch from the validation set, and a function `test` that
        computes the error on a batch from the testing set
        建立一个train函数实现一步微调
        一个validate函数计算一批数据与校验集比较的错误
        一个test函数激素啊一批数据与测试集比较的错误

        :type datasets: list of pairs of theano.tensor.TensorType
        :param datasets: It is a list that contain all the datasets;
                        the has to contain three pairs, `train`,
                        `valid`, `test` in this order, where each pair
                        is formed of two Theano variables, one for the
                        datapoints, the other for the labels
                        包含所有数据集的列表，包含三对数据，train，valid，test
                        每对数据包含两个theano变量，数据和标签
        :type batch_size: int
        :param batch_size: size of a minibatch 小批量数据的大小
        :type learning_rate: float
        :param learning_rate: learning rate used during finetune stage 微调时用的学习率

        '''

        (train_set_x, train_set_y) = datasets[0]
        (valid_set_x, valid_set_y) = datasets[1]
        (test_set_x, test_set_y) = datasets[2]

        # compute number of minibatches for training, validation and testing
        n_valid_batches = valid_set_x.get_value(borrow=BORROW).shape[0]
        n_valid_batches /= batch_size
        n_test_batches = test_set_x.get_value(borrow=BORROW).shape[0]
        n_test_batches /= batch_size

        index = T.lscalar('index')  # index to a [mini]batch

        # compute the gradients with respect to the model parameters
        gparams = T.grad(self.finetune_cost, self.params)

        # compute list of fine-tuning updates
        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - gparam * learning_rate))

        train_fn = theano.function(inputs=[index],
              outputs=self.finetune_cost,
              updates=updates,
              givens={self.x: train_set_x[index * batch_size:
                                          (index + 1) * batch_size],
                      self.y: train_set_y[index * batch_size:
                                          (index + 1) * batch_size]})

        test_score_i = theano.function([index], self.errors,
                 givens={self.x: test_set_x[index * batch_size:
                                            (index + 1) * batch_size],
                         self.y: test_set_y[index * batch_size:
                                            (index + 1) * batch_size]})

        valid_score_i = theano.function([index], self.errors,
              givens={self.x: valid_set_x[index * batch_size:
                                          (index + 1) * batch_size],
                      self.y: valid_set_y[index * batch_size:
                                          (index + 1) * batch_size]})

        # Create a function that scans the entire validation set
        def valid_score():
            return [valid_score_i(i) for i in xrange(n_valid_batches)]

        # Create a function that scans the entire test set
        def test_score():
            return [test_score_i(i) for i in xrange(n_test_batches)]

        return train_fn, valid_score, test_score


def test_DBN(finetune_lr=0.1, pretraining_epochs=PRETRAIN_EPOCHS,
             pretrain_lr=0.01, k=1, training_epochs=FINETUNE_EPOCHS,
             dataset='/mnist.pkl.gz', batch_size=BATCH_SIZE,
             outputfile=DBN_PICKLED_FILE):
    """

    :type finetune_lr: float
    :param finetune_lr: learning rate used in the finetune stage
                        微调阶段用的学习率
    :type pretraining_epochs: int
    :param pretraining_epochs: number of epoch to do pretraining
                                预训练次数
    :type pretrain_lr: float
    :param pretrain_lr: learning rate to be used during pre-training
                        预训练时用的学习率
    :type k: int
    :param k: number of Gibbs steps in CD/PCD 
                做Gibbs步骤的数目
    :type training_epochs: int
    :param training_epochs: maximal number of iterations to run the optimizer
                            最大优化次数(微调次数)
    :type dataset: string
    :param dataset: path the the pickled dataset
                    数据文件路径
    :type batch_size: int
    :param batch_size: the size of a minibatch
                        小批量数据大小（运算时多少条数据合成一块进行运算）
    :type outputfile: string
    :param outputfile: 输出的DBN pickle的路径
    """

    datasets = load_data(DATASET+dataset)
    #读dataset文件夹下的训练集数据和标签train_xdata.npy,train_ylabels.npy
    #生成测试集数据和标签test_xdata.npy,test_ylabels.npy
    #生成校验集数据和标签valid_set_x,valid_set_y
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    # 计算小批量数据数量
    n_train_batches = train_set_x.get_value(borrow=BORROW).shape[0] / batch_size

    # numpy random generator
    # numpy随机数生成器
    numpy_rng = numpy.random.RandomState(123)
    print '... building the model'
    # construct the Deep Belief Network
    # 建立DBN
    dbn = DBN(numpy_rng=numpy_rng, n_ins=DIMENSION * N_FRAMES,
              hidden_layers_sizes=[1024,1024,1024,1024,1024,1024],
              n_outs=N_OUTS)

    #########################
    # PRETRAINING THE MODEL #
    #########################
    # 预训练模型

    #生成预训练函数
    print '... getting the pretraining functions'
    pretraining_fns = dbn.pretraining_functions(train_set_x=train_set_x,
                                                batch_size=batch_size,
                                                k=k)

    print '... pre-training the model'
    start_time = time.clock()
    ## Pre-train layer-wise
    #逐层预训练
    for i in xrange(dbn.n_layers):
        # go through pretraining epochs
        # 迭代预训练
        for epoch in xrange(pretraining_epochs):
            # go through the training set
            # 遍历训练集
            c = []
            #遍历每个小批量数据
            for batch_index in xrange(n_train_batches):
                #训练当前小批量数据并加入c
                c.append(pretraining_fns[i](index=batch_index,
                                            lr=pretrain_lr))
            print 'Pre-training layer %i, epoch %d, cost ' % (i, epoch),
            print numpy.mean(c)

        # with open(DATASET+'/dbn_pre-train_layer'+i+'.pickle', 'w') as f:
        #     cPickle.dump(dbn, f)

    end_time = time.clock()
    print >> sys.stderr, ('The pretraining code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

    ########################
    # FINETUNING THE MODEL #
    ########################
    # 微调模型

    # get the training, validation and testing function for the model
    # 生成训练、校验、测试函数
    print '... getting the finetuning functions'
    train_fn, validate_model, test_model = dbn.build_finetune_functions(
                datasets=datasets, batch_size=batch_size,
                learning_rate=finetune_lr)

    print '... finetunning the model'
    # early-stopping parameters 停止参数
    patience = 4 * n_train_batches  # look as this many examples regardless 最小迭代微调次数
    patience_increase = 2.    # wait this much longer when a new best is
                              # found
                              # 找到新的最佳时,微调迭代次数变为现在的patience_increase倍
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant 
                                   # 新的校验损失低于原来校验损失的improvement_threshold倍
                                   # 才认为有意义
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many 
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch 
                                  # 遍历这么多个小批量数据才在校验集上校验一次网络
                                  # 在这种情况下，每次迭代都校验

    best_params = None
    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = time.clock()

    done_looping = False
    epoch = 0

    # 迭代进行微调
    while (epoch < training_epochs) and (not done_looping):
        epoch = epoch + 1

        # if epoch % 50 == 0:
        #     with open(DATASET+'/dbn_fine-tuning_epoch'+epoch+'.pickle', 'w') as f:
        #         cPickle.dump(dbn, f)

        # 遍历每个小批量数据
        for minibatch_index in xrange(n_train_batches):

            minibatch_avg_cost = train_fn(minibatch_index)  #本次小批量数据的平均代价
            iter = epoch * n_train_batches + minibatch_index    #总体上是第几次计算小批量数据

            # 每validation_frequency次进行一次校验
            if (iter + 1) % validation_frequency == 0:

                validation_losses = validate_model()    #计算校验损失
                this_validation_loss = numpy.mean(validation_losses)    #平均损失
                print('epoch %i, minibatch %i/%i, validation error %f %%' % \
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                # 得到最佳校验得分
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    # 如果损失提升足够好，提升patience
                    # 即若损失提升足够好，证明微调还在寻找最佳，没有收敛
                    # 所以增大patience使微调迭代次数变为patience_increase倍
                    if (this_validation_loss < best_validation_loss *
                        improvement_threshold):
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    # 存储最佳校验结果和迭代次数
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    # 在测试集上测试
                    test_losses = test_model()  #计算测试损失
                    test_score = numpy.mean(test_losses)    #平均损失
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            #patient小于迭代次数，停止迭代
            # if patience <= iter:
            #     done_looping = True
            #     break

    end_time = time.clock()
    print(('Optimization complete with best validation score of %f %%,'
           'with test performance %f %%') %
                 (best_validation_loss * 100., test_score * 100.))
    print >> sys.stderr, ('The fine tuning code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time)
                                              / 60.))

    with open(outputfile, 'w') as f:
        cPickle.dump(dbn, f)

if __name__ == '__main__':
    test_DBN(dataset='')
    print 'Done'
