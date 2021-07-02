# -*- coding: utf-8 -*-
import pickle as pkl
import tensorflow as tf
import pandas as pd
import numpy as np
import math
import os
import numpy.linalg as la
from input_data_ex import *
# from tgcn import tgcnCell
#from gru import GRUCell
from models import GCN1
from utils import calculate_laplacian
# from models_pure import GCN

from visualization import plot_result,plot_error
from sklearn.metrics import mean_squared_error,mean_absolute_error
#import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.client import device_lib
"""
config = tf.compat.v1.ConfigProto(gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8))
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)
"""

time_start = time.time()
###### Settings ######
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.005, 'Initial learning rate.')
flags.DEFINE_integer('training_epoch', 200, 'Number of epochs to train.')
flags.DEFINE_integer('gru_units', 20, 'hidden units of gru.')
flags.DEFINE_integer('seq_len',20 , '  time length of inputs.')
flags.DEFINE_integer('pre_len', 5, 'time length of prediction.')
flags.DEFINE_float('train_rate', 0.5, 'rate of training set.')
flags.DEFINE_integer('batch_size', 32, 'batch size.')
flags.DEFINE_string('dataset', 'ILI_Region','ILI_Region or ILI_Stats')
flags.DEFINE_string('model_name', 'SAIFlu', 'SAIFlu')

flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

model_name = FLAGS.model_name
data_name = FLAGS.dataset
train_rate =  FLAGS.train_rate
seq_len = FLAGS.seq_len
output_dim = pre_len = FLAGS.pre_len
batch_size = FLAGS.batch_size
lr = FLAGS.learning_rate
training_epoch = FLAGS.training_epoch
gru_units = FLAGS.gru_units
validation_rate = 0.1

def GCN(inputs, num_nodes, output_size, _weights, _biases, adj):
    print(inputs, 'inputs')
    input_size = inputs.get_shape()[2].value  # seq_len
    _adj = []
    _adj.append(calculate_laplacian(adj))
    x0 = tf.transpose(inputs, perm=[1, 2, 0])  # (num_nodes, seq_len, bat)
    x0 = tf.reshape(x0, shape=[num_nodes, -1])  # (num_nodes, bat * seq_len)

    scope = tf.get_variable_scope()
    with tf.variable_scope(scope):
        for m in _adj:
            x1 = tf.sparse_tensor_dense_matmul(m, x0)  # AX
            #                print(x1)
        x = tf.reshape(x1, shape=[num_nodes, input_size, -1])  # (num_nodes, seq_len, bat)
        x = tf.transpose(x, perm=[2, 0, 1])  # (bat, num_nodes, seq_len)
        x = tf.reshape(x, shape=[-1, input_size])  # (bat * num_nodes, seq_len)
        weights = tf.get_variable(
            'weights', [input_size, output_size], initializer=tf.contrib.layers.xavier_initializer())  # output_size = 1
        # (batch_size * self._nodes, output_size)
        x = tf.matmul(x, weights)  # AXW
        biases = tf.get_variable(
            "biases", [output_size], initializer=tf.constant_initializer())
        x = tf.nn.bias_add(x, biases)  # AXW + B
        output = tf.reshape(x, shape=[-1, num_nodes])  # (batchsize, 10, gru_units)
    return output, m

def scaled_dot_product_attention_ori(Q, K, V,
                                     dropout_rate=0.,
                                     training=True,
                                     scope="scaled_dot_product_attention"):

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        d_k = Q.get_shape().as_list()[-1]

        # outputs = Q

        cell = tf.nn.rnn_cell.BasicLSTMCell(200)
        _X = tf.unstack(Q, axis=1)
        print(_X)
        region_vec_list = list()
        for i in range(num_nodes):
            temp_x = tf.expand_dims(_X[i], axis=2)
            temp_x = tf.unstack(temp_x, axis=1)
            outputs, states = tf.nn.static_rnn(cell, temp_x, dtype=tf.float32)
            region_vec_list.append(outputs[-1])

        Q = tf.stack(region_vec_list, axis=1)

        cell = tf.nn.rnn_cell.BasicLSTMCell(200)
        _X = tf.unstack(K, axis=1)
        print(_X)
        region_vec_list = list()
        for i in range(num_nodes):
            temp_x = tf.expand_dims(_X[i], axis=2)
            temp_x = tf.unstack(temp_x, axis=1)
            outputs, states = tf.nn.static_rnn(cell, temp_x, dtype=tf.float32)
            region_vec_list.append(outputs[-1])

        K = tf.stack(region_vec_list, axis=1)

        # dot product
        outputs = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))  # (N, T_q, T_k)

        # scale
        # outputs /= d_k ** 0.5

        outputs = tf.nn.softmax(outputs)
        att = outputs

        outputs = tf.matmul(outputs, V)  # (N, T_q, d_v)


    return outputs, att

def multihead_attention(queries, keys, values,
                        num_heads=8,
                        dropout_rate=0,
                        training=True,
                        scope="multihead_attention",
                        d_model=None):

    if d_model is None:
        d_model = queries.get_shape().as_list()[-1]

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # Linear projections

        Q = queries

        K = Q

        V = tf.layers.dense(values, 50, use_bias=True,
                            activation=tf.nn.relu)  # (N, T_k, d_model)

        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, d_model/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, d_model/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, d_model/h)

        # Attention
        outputs, att = scaled_dot_product_attention_ori(Q_, K_, V_, dropout_rate, training=training)


        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, d_model)


    return outputs, att

def ff1(inputs, num_units, residual=True, norm=True, scope="positionwise_feedforward",
        output_act=None, drop_out=0.0):

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # Inner layer
        if output_act:
            outputs = tf.layers.dropout(tf.layers.dense(inputs, num_units, activation=output_act),
                                        rate=drop_out)
        else:
            outputs = tf.layers.dropout(tf.layers.dense(inputs, num_units), rate=drop_out)

    return outputs

class Transformer:
    def __init__(self, embeddings, num_nodes, d_model, dropout_rate, list_num_hiddens,
                 num_heads, num_blocks, sequence_length, predict_period):
        # self.embeddings = get_token_embeddings(self.hp.vocab_size, self.hp.d_model, zero_pad=True)
        self.embeddings = embeddings
        self.d_model = d_model
        self.dropout_rate = dropout_rate
        self.num_nodes = num_nodes
        self.list_num_hiddens = list_num_hiddens
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.weight_list = list()
        self.sequence_length = sequence_length
        self.predict_period = predict_period

    def encode(self, training=True):
        '''
        Returns
        memory: encoder outputs. (N, T1, d_model)
        '''
        return_dict = dict()

        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):


            enc = self.embeddings


            ## Blocks
            for i in range(self.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                    # self-attention
                    enc, att = multihead_attention(queries=enc,
                                                   keys=enc,
                                                   values=enc,
                                                   num_heads=self.num_heads,
                                                   dropout_rate=self.dropout_rate,
                                                   training=training, scope="multi1",
                                                   d_model=d_model)
                    # enc  -> shape=(?, 10, 50)
                    print(enc, att,'Before ff1')
                    return_dict['block%d_output' % i] = enc
                    return_dict['block%d_att' % i] = att

                    # feed forward
            enc = ff1(enc, residual=False, norm=False, num_units=1, output_act=tf.nn.relu,
                      scope="pp1", drop_out=self.dropout_rate)
            enc = tf.reshape(enc, shape=[-1, self.num_nodes])
        memory = enc
        return memory, return_dict

    def train(self):
        '''
        Returns
        loss: scalar.
        train_op: training operation
        global_step: scalar.
        summaries: training summary node
        '''
        # forward
        memory, return_dict = self.encode()

        return memory, return_dict
###### load data ######

if data_name == 'ILI_Region':
    data, adj = load_ILI_Region_data('ILI_Region')

if data_name == 'ILI_states':
    data, adj = load_ILI_State_data('ILI_states')

print(adj)
scaler=MinMaxScaler()
time_len = data.shape[0]
num_nodes = data.shape[1]
print(time_len, num_nodes)
data1 =np.mat(data,dtype=np.float32)
max_value = np.max(data1)

scaler.fit(data1[0:int(len(data1))])
data1 = scaler.transform(data1)
#전처리할때 X는 transpose 해서 Y랑 형식이 다름
trainX, trainY, testX, testY = preprocess_data1(data1, time_len, train_rate, validation_rate, seq_len, pre_len)
# trainY = np.squeeze(trainY,axis=1)
# testY = np.squeeze(testY,axis=1)
training_data_count = len(trainX)


#with tf.Session(config=config) as session:
###### placeholders ######
inputs = tf.placeholder(tf.float32, shape=[None, num_nodes, seq_len])
labels = tf.placeholder(tf.float32, shape=[None, num_nodes])

# Graph weights
weights = {
    'out': tf.Variable(tf.random_normal([gru_units, 10], mean=1.0), name='weight_o'),
    'w_s': tf.Variable(tf.random_normal([10, 1], mean=0.0, stddev=0.01), name='weight_s'),
    'w_t': tf.Variable(tf.random_normal([10, 2], mean=0.0, stddev=0.01), name='weight_s'),
    'v_t': tf.Variable(tf.random_normal([2, 1], mean=0.0, stddev=0.01), name='v_t'),
    'w_m': tf.Variable(tf.random_normal([num_nodes, num_nodes], mean=0.0, stddev=0.01), name='w_m')}
biases = {
    'out': tf.Variable(tf.random_normal([10]), name='bias_o'),
    'b_s': tf.Variable(tf.random_normal([1]), name='bias_s')}
support = [calculate_laplacian(adj)]
print(support)
num_supports = 1
list_num_hiddens = [16, 16, 8, 8, 8, 20, 8, 8, 40]  # [12, 12]
num_heads = 1
num_blocks = 1
d_model = 4  # 12
dropout_rate = 0.2

tr = Transformer(inputs,  num_nodes, d_model, dropout_rate, list_num_hiddens, num_heads, num_blocks, seq_len, pre_len)
# model = GCN1(inputs, input_dim=seq_len, support=support, labels=labels, logging=True)


y_pred, return_dict = tr.train()
print('y_pred tensor shape:',y_pred)  # shape -> (bat, num_nodes)


###### optimizer ######
lambda_loss = 0.0015
Lreg = lambda_loss * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
label = tf.reshape(labels, [-1, num_nodes])
##loss
loss = tf.reduce_mean(tf.nn.l2_loss(y_pred - label) + Lreg)
##rmse
error = tf.sqrt(tf.reduce_mean(tf.square(y_pred - label)))
optimizer = tf.train.AdamOptimizer(lr).minimize(loss)

###### Initialize session ######
variables = tf.global_variables()
saver = tf.train.Saver(tf.global_variables())
#sess = tf.Session()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
sess.run(tf.global_variables_initializer())

out = 'out/%s' % (model_name)
# out = 'out/%s_%s'%(model_name,'perturbation')
path1 = '%s_%s_lr%r_batch%r_unit%r_seq%r_pre%r_epoch%r' % (
model_name, data_name, lr, batch_size, gru_units, seq_len, pre_len, training_epoch)
path = os.path.join(out, path1)
if not os.path.exists(path):
    os.makedirs(path)


###### evaluation ######
def evaluation(a,b):
    rmse = math.sqrt(mean_squared_error(a,b))
    mae = mean_absolute_error(a, b)
    F_norm = la.norm(a-b,'fro')/la.norm(a,'fro')

    a_mean = np.mean(a, axis=0)
    b_mean = np.mean(b, axis=0)
    t1 = np.sum(np.multiply(a - a_mean, b-b_mean), axis=0)
    t2 = np.multiply(np.sqrt(np.sum(np.multiply(a - a_mean, a - a_mean), axis=0)), np.sqrt(np.sum(np.multiply(b - b_mean, b - b_mean), axis=0)))
    pcc = np.mean(np.divide(t1, t2+1.0e-20))

    ab = np.absolute(np.subtract(a, b))

    div_a = np.where(a==0, np.ones_like(a), a)
    mape = np.mean(100 * np.divide(ab, div_a))
    Residual = pd.DataFrame(b_mean-a_mean)

    return rmse, mae, 1-F_norm, pcc, mape, Residual
def mape(Y_pred, Y_true):
    N=len(Y_pred)
    numerator=[abs(Y_true[i]-Y_pred[i])/Y_true[i] for i in range(N)]
    numerator=np.sum(numerator)
    val=(numerator/N)*100
    # val=numerator
    return val

x_axe, batch_loss, batch_rmse, batch_pred = [], [], [], []
test_loss, test_rmse, test_mae, test_acc, test_pcc1, test_var, test_pred, test_mape1 = [], [], [], [], [], [], [], []

test_labels = scaler.inverse_transform(testY)
train_labels = scaler.inverse_transform(trainY)
cur_time = time.time()
file_name = "./report_files/report_%s_%s_%f.csv" % (pre_len, model_name,cur_time)
#file_name = "./report_files/report_%s_%0.2d_%f.csv" % (pre_len, model_name,cur_time)
result_file = open(file_name, mode='w')
print("result file name : %s" % file_name)
result_file.write("EPOCH,RMSE,PCC,MAPE,MAE\n")
len_test_data = np.shape(testY)[0]


for epoch in range(training_epoch):

    len_data = np.shape(trainX)[0]
    loss_sum = 0.0
    error_sum = 0.0
    temp_ind = np.arange(np.shape(trainX)[0])
    np.random.shuffle(temp_ind)
    temp_trainX = trainX[temp_ind]
    temp_trainY = trainY[temp_ind]

    i = 0
    while i < len_data:
        if i + batch_size > len_data:
            break
        mini_batch = temp_trainX[i:i + batch_size]
        mini_label = temp_trainY[i:i + batch_size]


        _, loss1, train_output,error1 = sess.run([optimizer, loss, y_pred, error],
                                          feed_dict={inputs: mini_batch, labels: mini_label})

        loss_sum += loss1
        error_sum += error1
        i += batch_size
    loss_sum /= len_data
    print("%d epoch loss : %f, error: %f" % (epoch, loss_sum,error_sum))
    # print(temp_fc1)

    if epoch % 100 == 0:
        outputs = list()
        for i in range(0, len_test_data):
            mini_batch = testX[i:i + 1]
            mini_label = testY[i:i + 1]
            output = sess.run(y_pred, feed_dict={inputs: mini_batch, labels: mini_label})
            # print(output,i)
            outputs.append(output)

        outputs = np.array(outputs)
        tsize = np.shape(outputs)[0]
        # print(tsize,outputs)
        outputs = np.reshape(outputs, newshape=[tsize, num_nodes])
        predict = scaler.inverse_transform(outputs)

        rmse, mae, acc, test_pcc, test_mape,test_Res = evaluation(test_labels, predict)
        test_rmse.append(rmse )  # rmse
        test_mae.append(mae )
        test_acc.append(acc)
        test_pcc1.append(test_pcc)
        test_mape1.append(test_mape)
        test_pred.append(predict)

        print('Iter:{}'.format(epoch),
              'test_rmse:{:.4}'.format(rmse),
              'test_pcc:{:.4}'.format(test_pcc),  # rmse
              'test_mape:{:.4}'.format(test_mape),
              'test_mae:{:.4}'.format(mae),
              'test_acc:{:.4}'.format(acc))

        result_file.write("%d,%f,%f,%f,%f\n" % (epoch, rmse, test_pcc, test_mape, mae))

        outputs = list()
        for i in range(0, len_data):
            mini_batch = trainX[i:i + 1]
            mini_label = trainY[i:i + 1]
            output = sess.run(y_pred, feed_dict={inputs: mini_batch, labels: mini_label})
            outputs.append(output)

        outputs = np.array(outputs)
        tsize = np.shape(outputs)[0]
        outputs = np.reshape(outputs, newshape=[tsize, num_nodes])
        predict = scaler.inverse_transform(outputs)

        train_labels = train_labels[:tsize]

        rmse, mae, acc, train_pcc, train_mape,train_Res = evaluation(train_labels, predict)

        print('Iter:{}'.format(epoch),
              'train_rmse:{:.4}'.format(rmse),
              'train_pcc:{:.4}'.format(train_pcc),  # rmse
              'train_mape:{:.4}'.format(train_mape),
              'train_mae:{:.4}'.format(mae),
              'train_acc:{:.4}'.format(acc))

index = test_rmse.index(np.min(test_rmse))
test_result = test_pred[index]
var = pd.DataFrame(test_result)
var.to_csv(path + '/test_result.csv', index=False, header=False)
train_Res.to_csv(path + '/train_Residual.csv', index=False, header=False)
print('training_epoch:%r' % (training_epoch),
      'seq_len:%r' % (seq_len),
      'pre_len:%r' % (pre_len),
      'min_rmse:%r' % (np.min(test_rmse)),
      'min_mae:%r' % (test_mae[index]),
      'max_acc:%r' % (test_acc[index]),
      'pcc:%r' % (test_pcc1[index]),
      'best epoch:%r' % (index)
      )
