import csv
import numpy as np
import matplotlib.pyplot as plt
from numpy import loadtxt
from tensorflow.keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from tensorflow.keras import layers
from scipy.sparse import load_npz
import spektral
import scipy.sparse as sp
import pandas as pd
import sys
from spektral.utils.convolution import *
from scipy.sparse import csr_matrix
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Conv1D, LSTM, PReLU, GRU, Bidirectional
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy, MeanSquaredError
from tensorflow.keras.metrics import categorical_accuracy, sparse_categorical_accuracy, binary_accuracy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
from tcn import TCN
from spektral.data import Dataset, Graph, MixedLoader, DisjointLoader, BatchLoader, PackedBatchLoader
from spektral.layers import *
from spektral.transforms.normalize_adj import NormalizeAdj
import os
from scipy.sparse import save_npz, csr_matrix
import time
from tensorflow.keras.models import load_model

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# nohup python 500bus_split_train.py > output.log 2>&1 &
mode = 'test' # 'train' or 'data gen' or 'save_output' / 'save_excel' / 'test'

nBus = 500
nSamples = 1000
SysName = "500Bus"

nGen = 90
nBranch = 597
n_llel = 13


range = __builtins__.range  # range 함수 복구

np.random.seed(1)
np.set_printoptions(threshold=sys.maxsize)

nPrd_list = [24] # ,24
GNN_types = [ "GATConv"] # "APPNP", "ECCConv","ARMA" , "DiffusionConv", "GATConv", "GCNConv", "GCSConv", "GINconvBatch"
RNN_types = [ "TCN"] # "GRU", "LSTM", "BiGRU","BiLSTM", "TCN"
#
# GNN = "APPNP" #APPNP, ECCConv, ARMA, DiffusionConv, GATConv, GCNConv, GCSConv, GINconvBatch
# RNN = "TCN" #   BiGRU, BiLSTM,GRU, LSTM,TCN


def SMAPE(y_true, y_pred, weight_m=0.5, weight_A=0.5):
    # y_true와 y_pred는 각각 [m_true, A_true]와 [m_pred, A_pred] 형식으로 들어옴
    m_true = tf.cast(y_true[:, 0], tf.float32)
    A_true = tf.cast(y_true[:, 1], tf.float32)
    m_pred = tf.cast(y_pred[:, 0], tf.float32)
    A_pred = tf.cast(y_pred[:, 1], tf.float32)

    # SMAPE 계산
    smape_m = tf.reduce_mean(
        2 * tf.abs(m_true - m_pred) / (tf.abs(m_true) + tf.abs(m_pred))) * 100  # m에 대한 SMAPE
    smape_A = tf.reduce_mean(
        2 * tf.abs(A_true - A_pred) / (tf.abs(A_true) + tf.abs(A_pred))) * 100  # A에 대한 SMAPE

    # 가중 평균 SMAPE 계산
    total_smape = weight_m * smape_m + weight_A * smape_A
    return total_smape, smape_m, smape_A

def NMAE(y_true, y_pred, weight_m=0.5, weight_A=0.5):
    # y_true와 y_pred는 각각 [m_true, A_true]와 [m_pred, A_pred] 형식
    m_true = tf.cast(y_true[:, 0], tf.float32)
    A_true = tf.cast(y_true[:, 1], tf.float32)
    m_pred = tf.cast(y_pred[:, 0], tf.float32)
    A_pred = tf.cast(y_pred[:, 1], tf.float32)

    # 정규화에 필요한 최댓값과 최솟값
    m_range = tf.reduce_max(m_true) - tf.reduce_min(m_true)
    A_range = tf.reduce_max(A_true) - tf.reduce_min(A_true)

    # NMAE 계산
    mae_m = tf.reduce_mean(tf.abs(m_true - m_pred))
    mae_A = tf.reduce_mean(tf.abs(A_true - A_pred))
    nmae_m = mae_m / m_range * 100
    nmae_A = mae_A / A_range * 100


    # 가중 평균 NMAE
    total_nmae = weight_m * nmae_m + weight_A * nmae_A
    return total_nmae, nmae_m, nmae_A , mae_m, mae_A

def evaluateGNNLSTM(loader):
    total_loss = 0
    total_mae = 0
    total_samples = 0
    total_smape = 0
    total_smape_m = 0  # m의 sMAPE 총합
    total_smape_A = 0  # A의 sMAPE 총합
    total_nmae = 0
    total_nmae_m = 0
    total_nmae_A = 0
    total_mae_m = 0
    total_mae_A = 0

    for batch_va in loader:
        pred_va, targ_va = modelGNNLSTM(*batch_va, training=False)
        loss_va = loss_fn(targ_va, pred_va)

        smape_va, smape_m, smape_A = SMAPE(targ_va, pred_va)
        nmae_va, nmae_m, nmae_A, mae_m, mae_A = NMAE(targ_va, pred_va)
        mae_va = (mae_m + mae_A) / 2
        # if model ==modelM:
        #     smape = SMAPE_M(targ_va, pred_va)
        # elif model == modelA:
        #     smape = SMAPE_A(targ_va, pred_va)

        batch_size = tf.shape(targ_va)[0]
        batch_size_f = tf.cast(batch_size, tf.float32)

        total_loss += loss_va * batch_size_f
        total_mae += mae_va * batch_size_f
        total_smape += smape_va * batch_size_f
        total_smape_m += smape_m * batch_size_f
        total_smape_A += smape_A * batch_size_f
        total_nmae += nmae_va * batch_size_f
        total_nmae_m += nmae_m * batch_size_f
        total_nmae_A += nmae_A * batch_size_f
        total_mae_m += mae_m * batch_size_f
        total_mae_A += mae_A * batch_size_f
        total_samples += batch_size

    total_samples_f = tf.cast(total_samples, tf.float32)

    avg_mse = total_loss / total_samples_f
    avg_mae = total_mae / total_samples_f
    avg_smape = total_smape / total_samples_f
    avg_smape_m = total_smape_m / total_samples_f
    avg_smape_A = total_smape_A / total_samples_f
    avg_nmae = total_nmae / total_samples_f
    avg_nmae_m = total_nmae_m / total_samples_f
    avg_nmae_A = total_nmae_A / total_samples_f
    avg_mae_m = total_mae_m / total_samples_f
    avg_mae_A = total_mae_A / total_samples_f

    return avg_mse, avg_mae, avg_smape, avg_smape_m, avg_smape_A, avg_nmae, avg_nmae_m, avg_nmae_A, avg_mae_m, avg_mae_A



for Prd in nPrd_list:
    nPrd = Prd
    for gnn_type in GNN_types:
        GNN = gnn_type
        for rnn_type in RNN_types:
            RNN = rnn_type

            print(f"Testing GNN: {GNN}, RNN: {RNN}")

            if mode == 'data gen':

                Bra_dataFile = f'{PROJECT_ROOT}/Data/same proportions/500bus/{nPrd}/case500_edgedata.dat'
                Bra_data = pd.read_csv(Bra_dataFile)
                fBus = Bra_data["branch_fbus"]
                tBus = Bra_data["branch_tbus"]
                kRating = Bra_data["branch_rateA"]
                kBranch_b = Bra_data["branch_b"]

                Demand_fileName = f'{PROJECT_ROOT}/Data/same proportions/500bus/{nPrd}/demand500Bus_{nPrd}_all.txt'
                dfNDmd = loadtxt(Demand_fileName, delimiter=',')  # Nodal demand NBus*24-Hours sequences

                # 노드의 M A데이터셋
                madata = f'{PROJECT_ROOT}/Data/same proportions/500bus/{nPrd}/500_{nPrd}_ma_values_all.txt'
                dfLFlw = loadtxt(madata, delimiter=',')
                # each graph(sample) must have 24 Nodal values and edge connection(constant, topology does not chage)
                x_data = dfNDmd
                nSamples = len(x_data)
                y_dataFlow = dfLFlw
                y_data = dfLFlw

                #NF (Node features) include nodal demand
                N_data = np.zeros([nSamples,nBus,nPrd])
                MA_data = np.zeros((nSamples, nBus, nPrd, 2))  #1000, 24, 12 , 2

                for m in range(nSamples):
                    for t in range(nPrd):
                        for n in range(nBus):
                            N_data[m,n,t] = x_data[m,(t)*nBus+n]

                for m in range(nSamples):
                    for t in range(nPrd):
                        for n in range(nBus):
                            MA_data[m,n,t,0] = y_data[m, 2*t * nBus + 2*n]  #M
                            MA_data[m,n,t,1] = y_data[m, 2*t * nBus + 2*n+1]  #A

                # Create NF (Node features)
                NF = np.zeros([nSamples,nBus,nPrd])
                for m in range(nSamples):
                    for t in range(nPrd):
                        for n in range(nBus):
                            NF[m,n,t] = x_data[m, (t)*nBus+n] #Load Profile

                print ("NF :" , NF.shape)

                # normalize
                def min_max_normalize(data):
                    min_val = np.min(data)
                    max_val = np.max(data)
                    return (data - min_val) / (max_val - min_val), min_val, max_val

                # Normalize M and A
                MA_data_normalized = np.zeros_like(MA_data)
                MA_data_normalized[..., 0], M_min, M_max = min_max_normalize(MA_data[..., 0])  # Normalize M
                MA_data_normalized[..., 1], A_min, A_max = min_max_normalize(MA_data[..., 1])  # Normalize A

                print("M_normalized min:", np.min(MA_data_normalized[..., 0]))
                print("M_normalized max:", np.max(MA_data_normalized[..., 0]))
                print("A_normalized min:", np.min(MA_data_normalized[..., 1]))
                print("A_normalized max:", np.max(MA_data_normalized[..., 1]))

                e_flow = np.zeros([nSamples,nBranch,nPrd])
                for m in range(nSamples):
                    for t in range(nPrd):
                        for k in range(nBranch):
                            e_flow[m,k,t] = ((y_dataFlow[m,(t)*nBranch+k])/kRating[k])##########

                n_llel = 0
                for k in range(nBranch):
                    if k>0:
                        if ((fBus[k] == fBus[k-1]) and (tBus[k] == tBus[k-1])):
                            n_llel = n_llel + 1

                print ("n_llel : ", n_llel) #13
                # initialize vectors to fill
                kRating_wo_llel = np.zeros([nBranch-n_llel,1])
                kBranch_b_wo_llel = np.zeros([nBranch-n_llel,1])

                e_flow_wo_llel = np.zeros([nSamples,(nBranch-n_llel),nPrd])

                k_new = -1
                fBus_llel = np.zeros([nBranch-n_llel,1])
                tBus_llel = np.zeros([nBranch-n_llel,1])
                for k in range(nBranch):
                    if k==0:
                        k_new = k_new + 1
                        kRating_wo_llel[k_new] = kRating[k]
                        kBranch_b_wo_llel[k_new] = kBranch_b[k]
                        e_flow_wo_llel[:,k_new,:] = e_flow[:,k,:]
                        fBus_llel[k_new] = fBus[k]
                        tBus_llel[k_new] = tBus[k]
                    else:
                        if ((fBus[k] == fBus[k-1]) and (tBus[k] == tBus[k-1])):
                            kRating_wo_llel[k_new] = max(kRating[k],kRating[k-1])###################
                            kBranch_b_wo_llel[k_new] = max(kBranch_b[k],kBranch_b[k-1])#############
                            for m in range(nSamples):
                                for t in range(nPrd):
                                    e_flow_wo_llel[m,k_new,t] = max(e_flow[m,k,t],e_flow[m,k-1,t])##########
                        else:
                            k_new = k_new + 1
                            kRating_wo_llel[k_new] = kRating[k]
                            kBranch_b_wo_llel[k_new] = kBranch_b[k]
                            e_flow_wo_llel[:,k_new,:] = e_flow[:,k,:]
                            fBus_llel[k_new] = fBus[k]
                            tBus_llel[k_new] = tBus[k]

                # Create Edge features of nBus,nBus,nedgefeat
                EF2 = np.zeros([nSamples,nBus,nBus,2])
                for m in range(nSamples):
                    for k in range(nBranch - n_llel):
                        EF2[m,int(fBus_llel[k])-1,int(tBus_llel[k])-1,0] = kBranch_b_wo_llel[k] #Reactance
                        EF2[m,int(fBus_llel[k])-1,int(tBus_llel[k])-1,1] = kRating_wo_llel[k]  #Line Limit


                # 인접행렬 저장하는 코드

                # Adjacency matrix as sparse matrix
                AM_sparse = csr_matrix((np.ones(nBranch - n_llel), (fBus_llel[:, 0] - 1, tBus_llel[:, 0] - 1)), shape=[nBus, nBus])

                # Dense matrix 생성 (벡터화 적용)
                AM_dense = np.zeros([nBus, nBus])
                AM_dense[(fBus_llel[:, 0] - 1).astype(int), (tBus_llel[:, 0] - 1).astype(int)] = 1

                ############################################
                # 파일 경로
                # 경로가 존재하지 않으면 디렉토리 생성
                directory = f'{PROJECT_ROOT}/Data/same proportions/500bus/{nPrd}/AM'
                if not os.path.exists(directory):
                    os.makedirs(directory)

                file_path_sparse = os.path.join(directory, 'AM_sparse.npz')
                save_npz(file_path_sparse, AM_sparse)
                file_path_dense = os.path.join(directory, 'AM_dense.npy')
                np.save(file_path_dense, AM_dense)

                print("nSamples: ", nSamples)
                print("nBranch: ", nBranch)
                print("n_llel: ", n_llel)
                print("nBranch - n_llel: ", nBranch - n_llel)
                print("np.sum(AM_dense): ", np.sum(AM_dense))

                # # 데이터셋 저장

                # Create graph files for Node Prediction (M,A is output label)
                # and save them into the folder
                path = f'{PROJECT_ROOT}/Data/same proportions/500bus/{nPrd}/graph_data'
                if not os.path.exists(path):
                    os.makedirs(path)

                for i in range(nSamples):
                    filename = os.path.join(path, f'GNN_{i}')
                    np.savez_compressed(filename, x=NF[i, :, :], a=AM_sparse, e=EF2[i, :, :, :],
                                        y=MA_data_normalized[i, :, :, :])


            if mode == 'train':

                # 데이터셋 불러오기
                class Graphs_DataGNNLSTM(spektral.data.dataset.Dataset):
                    # class Graphs_DataGNNLSTM(spektral.data.Dataset):
                    def read(self):
                        output = []

                        path = f'{PROJECT_ROOT}/Data/same proportions/500bus/{nPrd}/graph_data'
                        # Load data from npz files into read()
                        for i in range(nSamples):
                            graph = np.load(os.path.join(path, f'GNN_{i}.npz'))
                            output.append(spektral.data.Graph(x=graph['x'], e=graph['e'], y=graph['y']))  # e필요없으면 제외 가능

                        return output

                # 인접행렬 로드
                def load_adjacency_matrix(file_path_sparse, file_path_dense):
                    # .npz 파일에서 sparse 행렬 로드
                    AM_sparse = load_npz(file_path_sparse)
                    # .npy 파일에서 dense 행렬 로드
                    AM_dense = np.load(file_path_dense)
                    return AM_sparse, AM_dense

                # 경로 설정
                adjacency_matrix_path_sparse = f'{PROJECT_ROOT}/Data/same proportions/500bus/{nPrd}/AM/AM_sparse.npz'
                adjacency_matrix_path_dense = f'{PROJECT_ROOT}/Data/same proportions/500bus/{nPrd}/AM/AM_dense.npy'

                # 인접행렬 로드
                AM_sparse_loaded, AM_dense = load_adjacency_matrix(
                    adjacency_matrix_path_sparse, adjacency_matrix_path_dense
                )


                # Load Data
                GNN_DataGNNLSTM = Graphs_DataGNNLSTM()
                bus = GNN_DataGNNLSTM[0].n_nodes
                # Adjacency matrix is intentionally avoided since static netowrk topology is used
                # mixed data => one AM for all data samples. i.e. topology does not change

                # 차수 행렬 계산
                D_sparse = degree_matrix(AM_sparse_loaded)

                l_sparse = laplacian(AM_sparse_loaded) # 차수행렬D - 인접행렬A = 라플라시안 행렬 L

                Modified_L = gcn_filter(AM_sparse_loaded, symmetric=True)

                norm_laplacian = normalized_laplacian(AM_sparse_loaded,  symmetric=True)
                norm_rescaled_L = rescale_laplacian(norm_laplacian)

                Chebyshev = chebyshev_filter(AM_sparse_loaded,1)

                normalized_sparse_A = normalized_adjacency(AM_sparse_loaded)
                ##########################################################################################
                # dense matrix
                normalized_A = normalized_adjacency(AM_dense, symmetric=True)

                ########################################################################################## AM
                # print(l_sparse)
                # 인접행렬 변경
                if GNN == "GINconvBatch":
                    # binary dense adjacency matrix  #GINConvbatch
                    GNN_DataGNNLSTM.a = AM_dense
                elif GNN == "GATConv" or GNN == "ECCConv":
                    # #  Binary adjacency matrix # GAT # ECC
                    GNN_DataGNNLSTM.a = AM_sparse_loaded
                elif GNN == "APPNP" or GNN == "GCNConv":
                    # Modified_AM_sparse # APPNP , GCN
                    GNN_DataGNNLSTM.a = Modified_L
                elif GNN == "ARMA":
                    # norm_rescaled # ARMA
                    GNN_DataGNNLSTM.a = norm_rescaled_L
                # elif GNN == "APPNP":
                #     # Chebyshev
                #     GNN_DataGNNLSTM.a = Chebyshev
                elif GNN == "DiffusionConv" or GNN == "GCSConv":
                    # DiffusionConv ,GCSConv
                    GNN_DataGNNLSTM.a = normalized_A

                dataGNNLSTM = GNN_DataGNNLSTM
                ################################################################################
                # Config
                ################################################################################
                # learning_rate = 0.003  # Learning rate
                epochs = 500  # Number of training epochs
                es_patience = 50  # Patience for early stopping
                batch_sizes = 32  # Batch size 더 커질시 작동 x


                ################################################################################
                # Build model
                ################################################################################

                class GNNLSTMCmt(Model):
                    def __init__(self, **kwargs):
                        super().__init__(**kwargs)

                        # 엣지 정보 필요한 layer

                        if GNN == "ECCConv":
                            # ECC #
                            self.conv1 = ECCConv(nPrd, activation=None, root=True, use_bias=False, kernel_initializer=None)
                            self.conv_middle = ECCConv(nPrd, activation=None, root=True, use_bias=False, kernel_initializer=None)
                            self.conv2 = ECCConv(nPrd, activation="PReLU", root=True, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros')

                        # 엣지 정보 필요없는 layer

                        elif GNN == "GINconvBatch":
                            # GINConvBatch
                            self.conv1 = GINConvBatch(nPrd, epsilon=None, mlp_hidden=None, mlp_activation='relu', mlp_batchnorm=True, aggregate='sum', activation=None, use_bias=False)
                            self.conv_middle = GINConvBatch(nPrd, epsilon=None, mlp_hidden=None, mlp_activation='relu', mlp_batchnorm=True, aggregate='sum', activation=None,use_bias=False)
                            self.conv2 = GINConvBatch(nPrd, epsilon=None, mlp_hidden=None, mlp_activation='relu', mlp_batchnorm=True, aggregate='sum', activation=PReLU(), use_bias=True)
                        elif GNN == "GATConv":
                            # GATConv
                            self.conv1 = GATConv(nPrd, attn_heads=3, concat_heads=True, dropout_rate=0.5, add_self_loops=False, activation=None, use_bias=False)
                            self.conv_middle = GATConv(nPrd, attn_heads=3, concat_heads=True, dropout_rate=0.5,add_self_loops=False, activation=None, use_bias=False)
                            self.conv2 = GATConv(nPrd, attn_heads=3, concat_heads=True, dropout_rate=0.5, add_self_loops=False, activation=PReLU(), use_bias=True)
                        elif GNN == "GCNConv":
                            # # GCNConv
                            self.conv1 = GCNConv(nPrd, activation=None, use_bias=False)
                            self.conv_middle = GCNConv(nPrd, activation=None, use_bias=False)
                            self.conv2 = GCNConv(nPrd, activation=PReLU(), use_bias=True)
                        elif GNN == "APPNP":
                            # APPNP
                            self.conv1 = APPNPConv(nPrd, alpha=0.2, propagations=1, mlp_hidden=None, mlp_activation='relu', dropout_rate=0.0, activation=None, use_bias=False, kernel_initializer='glorot_uniform', bias_initializer='zeros')
                            self.conv_middle = APPNPConv(nPrd, alpha=0.2, propagations=1, mlp_hidden=None,mlp_activation='relu', dropout_rate=0.0, activation=None,  use_bias=False, kernel_initializer='glorot_uniform',bias_initializer='zeros')
                            self.conv2 = APPNPConv(nPrd, alpha=0.2, propagations=1, mlp_hidden=None, mlp_activation='relu', dropout_rate=0.0, activation=PReLU(), use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros')

                        elif GNN == "ARMA":
                            # ARMA
                            self.conv1 = ARMAConv(channels=nPrd, order=1, iterations=1, share_weights=False, gcn_activation='relu', dropout_rate=0.0, activation=None, use_bias=False, kernel_initializer='glorot_uniform', bias_initializer='zeros')
                            self.conv_middle = ARMAConv(channels=nPrd, order=1, iterations=1, share_weights=False,  gcn_activation='relu', dropout_rate=0.0, activation=None, use_bias=False, kernel_initializer='glorot_uniform', bias_initializer='zeros')
                            self.conv2 = ARMAConv(channels=nPrd, order=1, iterations=1, share_weights=False, gcn_activation='relu', dropout_rate=0.0, activation=PReLU(), use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros')

                        elif GNN == "DiffusionConv":
                            # DiffusionConv
                            self.conv1 = DiffusionConv(nPrd, K=6, activation=None, kernel_initializer='glorot_uniform', kernel_regularizer=None, kernel_constraint=None)
                            self.conv_middle = DiffusionConv(nPrd, K=6, activation=None, kernel_initializer='glorot_uniform',  kernel_regularizer=None, kernel_constraint=None)
                            self.conv2 = DiffusionConv(nPrd, K=6, activation='tanh', kernel_initializer='glorot_uniform', kernel_regularizer=None, kernel_constraint=None)
                        elif GNN == "GCSConv":
                            # GCSConv
                            self.conv1 = GCSConv(nPrd, activation=None, use_bias=False, kernel_initializer='glorot_uniform', bias_initializer='zeros')
                            self.conv_middle = GCSConv(nPrd, activation=None, use_bias=False, kernel_initializer='glorot_uniform', bias_initializer='zeros')
                            self.conv2 = GCSConv(nPrd, activation=PReLU(), use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros')


                        ################################################################################################################################################################
                        if RNN == "GRU":
                            self.lstm1 = GRU(nBus, activation="tanh", recurrent_activation="sigmoid", return_sequences=True)

                        elif RNN == "LSTM":
                            self.lstm1 = LSTM(nBus, activation="tanh", recurrent_activation="sigmoid", use_bias=True,
                                              kernel_initializer="glorot_uniform", recurrent_initializer="orthogonal",
                                              bias_initializer="zeros", return_sequences=True)

                        elif RNN == "BiGRU":
                            self.lstm1 = Bidirectional(
                                GRU(nBus, activation="tanh", recurrent_activation="sigmoid", return_sequences=True))
                        elif RNN == "BiLSTM":
                            self.lstm1 = Bidirectional(LSTM(nBus, activation="tanh", recurrent_activation="sigmoid", use_bias=True,
                                                            kernel_initializer="glorot_uniform", recurrent_initializer="orthogonal",
                                                            bias_initializer="zeros", return_sequences=True))

                        elif RNN == "TCN": #, use_batch_norm=True,
                            if nPrd == 8:
                                self.lstm1 = TCN(nb_filters=nBus, kernel_size=2, dilations=(1, 2, 4), padding='causal',
                                                 activation=PReLU(), return_sequences=True, dropout_rate=0.3, kernel_initializer='glorot_uniform')
                            elif nPrd == 24:
                                self.lstm1 = TCN(nb_filters=nBus, kernel_size=3, dilations=(1, 3, 9), padding='causal',
                                                 activation=PReLU(), return_sequences=True, dropout_rate=0.3, kernel_initializer='glorot_uniform')
                        # self.drop = Dropout(0.05)
                        self.dense = Dense(nPrd*2, activation="sigmoid")

                        self.dense1 = Dense(nBus, activation="relu") # , activation="relu"
                        self.dense_middle = Dense(nBus*2, activation="relu")
                        self.dense2_middle = Dense(nBus * 2, activation="relu")
                        # self.dense2 = Dense(nPrd*2, activation="sigmoid")
                        self.dense2 = Dense(nPrd * 2)

                    def call(self, inputs, labels):
                        if GNN == "ECCConv": # edge 사용하는 layer

                            x, a, e = inputs
                            x = self.conv1([x, a, e])
                            x = self.conv_middle([x, a, e])
                            x = self.conv2([x, a, e])

                        else:# edge 사용 안하는 layer

                            x, a, e = inputs
                            x = self.conv1([x, a])
                            x = self.conv_middle([x, a])
                            x = self.conv2([x, a])

                        if RNN in  ["GRU", "LSTM"]: # , "TCN"
                            # # non bidirectional
                            x = tf.transpose(x,[0, 2, 1])
                            out = self.lstm1(x)
                            out = tf.transpose(out, [0, 2, 1])
                            out = self.dense_middle(out) # +
                            out = self.dense2_middle(out)  # +
                            output = self.dense(out)
                            output = tf.reshape(output, [tf.shape(output)[0], nBus, nPrd, 2])
                            return output, labels

                        else:
                            # bidirectional
                            x = tf.transpose(x, [0, 2, 1])
                            out = self.lstm1(x)
                            output = self.dense1(out)
                            out = tf.transpose(output, [0, 2, 1])
                            out = self.dense_middle(out)  # +
                            out = self.dense2_middle(out)  # +
                            output = self.dense2(out)
                            output = tf.reshape(output, [tf.shape(output)[0], nBus, nPrd, 2])
                            return output, labels


                modelGNNLSTM = GNNLSTMCmt()

                lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                    initial_learning_rate=0.001,
                    decay_steps=5000,
                    decay_rate=0.95)
                optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
                ################################################################################
                # loss

                loss_fn = tf.keras.losses.MeanSquaredError()
                mae_fn = tf.keras.losses.MeanAbsoluteError()

                # 데이터셋 분할

                #data=data_temp
                # Train/valid/test split
                idxsGNNLSTM = range(len(dataGNNLSTM))
                split_vaGNNLSTM, split_teGNNLSTM = int(0.70 * len(dataGNNLSTM)), int(0.85 * len(dataGNNLSTM))
                idx_trGNNLSTM, idx_vaGNNLSTM, idx_teGNNLSTM = np.split(idxsGNNLSTM, [split_vaGNNLSTM, split_teGNNLSTM])
                data_trGNNLSTM = dataGNNLSTM[idx_trGNNLSTM]
                data_vaGNNLSTM = dataGNNLSTM[idx_vaGNNLSTM]
                data_teGNNLSTM = dataGNNLSTM[idx_teGNNLSTM]

                # Data loaders
                loader_trGNNLSTM = MixedLoader(data_trGNNLSTM, batch_size=batch_sizes, epochs=epochs, shuffle=False)
                loader_vaGNNLSTM = MixedLoader(data_vaGNNLSTM, batch_size=batch_sizes, shuffle = False, epochs=1)
                loader_teGNNLSTM = MixedLoader(data_teGNNLSTM, batch_size=batch_sizes, shuffle = False, epochs=1)

                # 데이터 로더에서 가져오는 배치의 형태 확인
                for batch_va in loader_trGNNLSTM:  # 여기서 트레이닝 로더 사용
                    inputs, labels = batch_va

                    print("Inputs shape:", [x.shape for x in inputs])  # x, a, e의 형태
                    print("Labels shape:", labels.shape)  # (batch_size, 24, 12, 2) #batch size 64
                    break  # 한 번만 확인

                ###################################################################################################################
                #### TRAINING model
                ###################################################################################################################
                epoch = step = 0
                best_val_loss = np.inf
                best_val_acc = 0
                best_weightsM = None
                patience = es_patience
                results = []
                trackGNNLSTM = np.zeros([epochs, 8])

                for batch in loader_trGNNLSTM:
                    step += 1

                    with tf.GradientTape() as tape:
                        prediction, target = modelGNNLSTM(*batch, training=True)
                        loss = loss_fn(target, prediction) # + sum(modelGNNLSTM.losses)
                        # loss = mae_fn(target, prediction)
                        # loss = SMAPE_M(target, prediction)
                    gradients = tape.gradient(loss, modelGNNLSTM.trainable_variables,
                                              unconnected_gradients=tf.UnconnectedGradients.ZERO)
                    optimizer.apply_gradients(zip(gradients, modelGNNLSTM.trainable_variables))

                    mae = tf.reduce_mean(tf.abs(target - prediction))
                    smape_total, smape_m, smape_A = SMAPE(target, prediction)

                    results.append((loss.numpy(), mae.numpy(), smape_total.numpy(), smape_m.numpy(), smape_A.numpy()))

                    # Print out result after every epoch
                    if step == loader_trGNNLSTM.steps_per_epoch:

                        # Compute validation loss and accuracy
                        loader_vaGNNLSTM = MixedLoader(data_vaGNNLSTM, batch_size=batch_sizes, shuffle=False, epochs=1)
                        val_loss, val_mae, val_smape_total, val_smape_m, val_smape_A = evaluateGNNLSTM(loader_vaGNNLSTM)

                        # Save loss and accuracy for plotting
                        # val_loss와 val_mae를 numpy로 변환
                        val_loss_np = val_loss.numpy() if isinstance(val_loss, tf.Tensor) else val_loss
                        val_mae_np = val_mae.numpy() if isinstance(val_mae, tf.Tensor) else val_mae
                        val_smape_total_np = val_smape_total.numpy() if isinstance(val_smape_total, tf.Tensor) else val_smape_total
                        val_smape_m_np = val_smape_m.numpy() if isinstance(val_smape_m, tf.Tensor) else val_smape_m
                        val_smape_A_np = val_smape_A.numpy() if isinstance(val_smape_A, tf.Tensor) else val_smape_A

                        trackGNNLSTM[epoch, :] = [*np.mean(results, 0), val_loss_np, val_mae_np, val_smape_total_np]

                        step = 0
                        epoch += 1
                        # Print out result for each epoch
                        print(
                            "Ep. {} - loss: {:.5f} - MAE: {:.5f} - SMAPE: {:.5f} (m: {:.5f}, A: {:.5f}) - Val MSE: {:.5f} - Val MAE: {:.5f} - Val SMAPE: {:.5f} (m: {:.5f}, A: {:.5f})".format(
                                epoch, np.mean(results, 0)[0], np.mean(results, 0)[1], np.mean(results, 0)[2],
                                np.mean(results, 0)[3],
                                np.mean(results, 0)[4],
                                val_loss_np, val_mae_np, val_smape_total_np, val_smape_m_np, val_smape_A_np
                            )
                        )
                        results = []
                        # Check if loss improved for early stopping
                        # val_loss가 1차원 배열이라면, numpy로 변환한 후 첫 번째 원소를 가져옴
                        if isinstance(val_loss, tf.Tensor) and val_loss.shape.ndims > 0:
                            current_val_loss = val_loss_np  # 직접 numpy로 변환한 값을 사용
                        else:
                            current_val_loss = val_loss_np

                        # Calculate the change in validation loss
                        val_loss_diff = best_val_loss - current_val_loss

                        if val_loss_diff < 0.00000001:
                            patience -= 1  # 감소시키기
                            if patience == 0:
                                print("Early stopping triggered.")
                                break
                        else:
                            # If there is a meaningful improvement, reset patience
                            best_val_loss = current_val_loss
                            patience = es_patience
                            print("New best val_loss {:.8f}".format(current_val_loss))
                            best_weightsMA = modelGNNLSTM.get_weights()



                # NOTE: Saved model can be used in the future without training
                modelGNNLSTM.set_weights(best_weightsMA)
                # Compile model before saving to include training information
                modelGNNLSTM.compile(optimizer=optimizer, loss=loss_fn, metrics=[SMAPE])

                #save_path = DataDir + f"/{nBus}bus/{nPrd}/GNN_trained_model/Spatio-Temporal_NodePrediction_" + SysName + "_SavedModel"
                save_path_MA = f"{PROJECT_ROOT}/Data/same proportions/500bus/{nPrd}/GNN_trained_model/" + GNN +"_"+ RNN + "_modelMA_SavedModel"
                os.makedirs(save_path_MA, exist_ok=True)  # exist_ok=True이면 폴더가 이미 있어도 오류가 나지 않음
                modelGNNLSTM.save(save_path_MA)

                ################################################################################
                # Evaluate model
                ################################################################################
                # modelGNNLSTM.set_weights(best_weightsGNNLSTM)  # Load best model


                # 모델 불러오기
                # model_path = f"Data/IEEE{nBus}Bus/{nBus}bus/Spatio-Temporal_NodePrediction_{nBus}Bus_SavedModel"
                model_path_M = save_path_MA
                modelMA = load_model(model_path_M,compile=False )

                loader_teGNNLSTM = MixedLoader(data_teGNNLSTM, batch_size=batch_sizes, shuffle = False, epochs=1)

                start_time = time.time()
                test_loss, test_mae, test_smape, test_smape_m, test_smape_A = evaluateGNNLSTM(loader_teGNNLSTM)
                end_time = time.time()

                # 테스트가 완료된 시간 계산
                elapsed_time = end_time - start_time
                # print("Total Test Time: {:.5f} seconds".format(elapsed_time))
                dataset_size = len(data_teGNNLSTM)
                time_per_dataset = elapsed_time / dataset_size

                # 출력할 내용 저장
                output_text = f"""
                Test MSE: {test_loss:.5f} 
                Test MAE: {test_mae:.5f} 
                Test SMAPE: {test_smape:.5f}
                Test SMAPE_M: {test_smape_m:.5f}
                Test SMAPE_A: {test_smape_A:.5f}
                Total Test Time: {elapsed_time:.5f} 
                Time per dataset: {time_per_dataset:.5f} 
                """

                # 파일로 저장
                output_file = f"{PROJECT_ROOT}/Data/same proportions/500bus/{nPrd}/result/{GNN}_{RNN}_results.txt"
                with open(output_file, "w") as f:
                    f.write(output_text)

                print(f"Results saved to {output_file}")
                print(f"Test SMAPE: {test_smape:.5f}")
                print(f" Test SMAPE_M: {test_smape_m:.5f}")
                print(f" Test SMAPE_A: {test_smape_A:.5f}")


            if mode == 'save_output':

                model_path = f"{PROJECT_ROOT}/Data/same proportions/500bus/{nPrd}/GNN_trained_model/GATConv_TCN_modelMA_SavedModel"
                output_dir = f"{PROJECT_ROOT}/Data/same proportions/500bus/{nPrd}/failed_predictions"
                batch_sizes = 500  # Batch size 더 커질시 작동 x

                Bra_dataFile = f'{PROJECT_ROOT}/Data/same proportions/500bus/{nPrd}/case500_edgedata.dat'
                Bra_data = pd.read_csv(Bra_dataFile)
                fBus = Bra_data["branch_fbus"]
                tBus = Bra_data["branch_tbus"]
                kRating = Bra_data["branch_rateA"]
                kBranch_b = Bra_data["branch_b"]

                Demand_fileName = f'{PROJECT_ROOT}/Data/same proportions/500bus/{nPrd}/demand500Bus_{nPrd}_failed_all.txt'

                dfNDmd = loadtxt(Demand_fileName, delimiter=',')  # Nodal demand NBus*24-Hours sequences

                # 노드의 M A데이터셋
                madata = f'{PROJECT_ROOT}/Data/same proportions/500bus/{nPrd}/500_{nPrd}_ma_values_all.txt'
                dfLFlw = loadtxt(madata, delimiter=',')
                # each graph(sample) must have 24 Nodal values and edge connection(constant, topology does not chage)
                x_data = dfNDmd
                nSamples = len(x_data)
                y_dataFlow = dfLFlw
                y_data = dfLFlw

                # NF (Node features) include nodal demand
                N_data = np.zeros([nSamples, nBus, nPrd])
                MA_data = np.zeros((nSamples, nBus, nPrd, 2))  # 1000, 24, 12 , 2

                for m in range(nSamples):
                    for t in range(nPrd):
                        for n in range(nBus):
                            N_data[m, n, t] = x_data[m, (t) * nBus + n]

                for m in range(nSamples):
                    for t in range(nPrd):
                        for n in range(nBus):
                            MA_data[m, n, t, 0] = y_data[m, 2 * t * nBus + 2 * n]  # M
                            MA_data[m, n, t, 1] = y_data[m, 2 * t * nBus + 2 * n + 1]  # A

                # Create NF (Node features)
                NF = np.zeros([nSamples, nBus, nPrd])
                for m in range(nSamples):
                    for t in range(nPrd):
                        for n in range(nBus):
                            NF[m, n, t] = x_data[m, (t) * nBus + n]  # Load Profile

                print("NF :", NF.shape)


                # normalize
                def min_max_normalize(data):
                    min_val = np.min(data)
                    max_val = np.max(data)
                    return (data - min_val) / (max_val - min_val), min_val, max_val


                # Normalize M and A
                MA_data_normalized = np.zeros_like(MA_data)
                MA_data_normalized[..., 0], M_min, M_max = min_max_normalize(MA_data[..., 0])  # Normalize M
                MA_data_normalized[..., 1], A_min, A_max = min_max_normalize(MA_data[..., 1])  # Normalize A

                print("M_normalized min:", np.min(MA_data_normalized[..., 0]))
                print("M_normalized max:", np.max(MA_data_normalized[..., 0]))
                print("A_normalized min:", np.min(MA_data_normalized[..., 1]))
                print("A_normalized max:", np.max(MA_data_normalized[..., 1]))

                e_flow = np.zeros([nSamples, nBranch, nPrd])
                for m in range(nSamples):
                    for t in range(nPrd):
                        for k in range(nBranch):
                            e_flow[m, k, t] = ((y_dataFlow[m, (t) * nBranch + k]) / kRating[k])  ##########

                n_llel = 0
                for k in range(nBranch):
                    if k > 0:
                        if ((fBus[k] == fBus[k - 1]) and (tBus[k] == tBus[k - 1])):
                            n_llel = n_llel + 1

                print("n_llel : ", n_llel)  # 13
                # initialize vectors to fill
                kRating_wo_llel = np.zeros([nBranch - n_llel, 1])
                kBranch_b_wo_llel = np.zeros([nBranch - n_llel, 1])

                e_flow_wo_llel = np.zeros([nSamples, (nBranch - n_llel), nPrd])

                k_new = -1
                fBus_llel = np.zeros([nBranch - n_llel, 1])
                tBus_llel = np.zeros([nBranch - n_llel, 1])
                for k in range(nBranch):
                    if k == 0:
                        k_new = k_new + 1
                        kRating_wo_llel[k_new] = kRating[k]
                        kBranch_b_wo_llel[k_new] = kBranch_b[k]
                        e_flow_wo_llel[:, k_new, :] = e_flow[:, k, :]
                        fBus_llel[k_new] = fBus[k]
                        tBus_llel[k_new] = tBus[k]
                    else:
                        if ((fBus[k] == fBus[k - 1]) and (tBus[k] == tBus[k - 1])):
                            kRating_wo_llel[k_new] = max(kRating[k], kRating[k - 1])  ###################
                            kBranch_b_wo_llel[k_new] = max(kBranch_b[k], kBranch_b[k - 1])  #############
                            for m in range(nSamples):
                                for t in range(nPrd):
                                    e_flow_wo_llel[m, k_new, t] = max(e_flow[m, k, t], e_flow[m, k - 1, t])  ##########
                        else:
                            k_new = k_new + 1
                            kRating_wo_llel[k_new] = kRating[k]
                            kBranch_b_wo_llel[k_new] = kBranch_b[k]
                            e_flow_wo_llel[:, k_new, :] = e_flow[:, k, :]
                            fBus_llel[k_new] = fBus[k]
                            tBus_llel[k_new] = tBus[k]

                # Create Edge features of nBus,nBus,nedgefeat
                EF2 = np.zeros([nSamples, nBus, nBus, 2])
                for m in range(nSamples):
                    for k in range(nBranch - n_llel):
                        EF2[m, int(fBus_llel[k]) - 1, int(tBus_llel[k]) - 1, 0] = kBranch_b_wo_llel[k]  # Reactance
                        EF2[m, int(fBus_llel[k]) - 1, int(tBus_llel[k]) - 1, 1] = kRating_wo_llel[k]  # Line Limit

                # 인접행렬 저장하는 코드

                # Adjacency matrix as sparse matrix
                AM_sparse = csr_matrix((np.ones(nBranch - n_llel), (fBus_llel[:, 0] - 1, tBus_llel[:, 0] - 1)),
                                       shape=[nBus, nBus])

                # Dense matrix 생성 (벡터화 적용)
                AM_dense = np.zeros([nBus, nBus])
                AM_dense[(fBus_llel[:, 0] - 1).astype(int), (tBus_llel[:, 0] - 1).astype(int)] = 1

                path = f'{PROJECT_ROOT}/Data/same proportions/500bus/{nPrd}/failed_graph_data'
                if not os.path.exists(path):
                    os.makedirs(path)

                for i in range(nSamples):
                    filename = os.path.join(path, f'GNN_{i}')
                    np.savez_compressed(filename, x=NF[i, :, :], a=AM_sparse, e=EF2[i, :, :, :],
                                        y=MA_data_normalized[i, :, :, :])


                # 데이터셋 불러오기
                class Graphs_DataGNNLSTM(spektral.data.dataset.Dataset):
                    # class Graphs_DataGNNLSTM(spektral.data.Dataset):
                    def read(self):
                        output = []

                        path = f'{PROJECT_ROOT}/Data/same proportions/500bus/{nPrd}/failed_graph_data'
                        # Load data from npz files into read()
                        for i in range(nSamples):
                            graph = np.load(os.path.join(path, f'GNN_{i}.npz'))
                            output.append(
                                spektral.data.Graph(x=graph['x'], e=graph['e'], y=graph['y']))  # e필요없으면 제외 가능

                        return output


                # 인접행렬 로드
                def load_adjacency_matrix(file_path_sparse, file_path_dense):
                    # .npz 파일에서 sparse 행렬 로드
                    AM_sparse = load_npz(file_path_sparse)
                    # .npy 파일에서 dense 행렬 로드
                    AM_dense = np.load(file_path_dense)
                    return AM_sparse, AM_dense


                # 경로 설정
                adjacency_matrix_path_sparse = f'{PROJECT_ROOT}/Data/same proportions/500bus/{nPrd}/AM/AM_sparse.npz'
                adjacency_matrix_path_dense = f'{PROJECT_ROOT}/Data/same proportions/500bus/{nPrd}/AM/AM_dense.npy'

                # 인접행렬 로드
                AM_sparse_loaded, AM_dense = load_adjacency_matrix(
                    adjacency_matrix_path_sparse, adjacency_matrix_path_dense
                )

                # Load Data
                GNN_DataGNNLSTM = Graphs_DataGNNLSTM()
                bus = GNN_DataGNNLSTM[0].n_nodes
                # Adjacency matrix is intentionally avoided since static netowrk topology is used
                # mixed data => one AM for all data samples. i.e. topology does not change

                # 차수 행렬 계산
                D_sparse = degree_matrix(AM_sparse_loaded)

                l_sparse = laplacian(AM_sparse_loaded)  # 차수행렬D - 인접행렬A = 라플라시안 행렬 L

                Modified_L = gcn_filter(AM_sparse_loaded, symmetric=True)

                norm_laplacian = normalized_laplacian(AM_sparse_loaded, symmetric=True)
                norm_rescaled_L = rescale_laplacian(norm_laplacian)

                Chebyshev = chebyshev_filter(AM_sparse_loaded, 1)

                normalized_sparse_A = normalized_adjacency(AM_sparse_loaded)
                ##########################################################################################
                # dense matrix
                normalized_A = normalized_adjacency(AM_dense, symmetric=True)

                ########################################################################################## AM
                # print(l_sparse)
                # 인접행렬 변경
                if GNN == "GINconvBatch":
                    # binary dense adjacency matrix  #GINConvbatch
                    GNN_DataGNNLSTM.a = AM_dense
                elif GNN == "GATConv" or GNN == "ECCConv":
                    # #  Binary adjacency matrix # GAT # ECC
                    GNN_DataGNNLSTM.a = AM_sparse_loaded
                elif GNN == "APPNP" or GNN == "GCNConv":
                    # Modified_AM_sparse # APPNP , GCN
                    GNN_DataGNNLSTM.a = Modified_L
                elif GNN == "ARMA":
                    # norm_rescaled # ARMA
                    GNN_DataGNNLSTM.a = norm_rescaled_L
                # elif GNN == "APPNP":
                #     # Chebyshev
                #     GNN_DataGNNLSTM.a = Chebyshev
                elif GNN == "DiffusionConv" or GNN == "GCSConv":
                    # DiffusionConv ,GCSConv
                    GNN_DataGNNLSTM.a = normalized_A

                dataGNNLSTM = GNN_DataGNNLSTM
                ################################################################################
                # Config
                ################################################################################
                # learning_rate = 0.003  # Learning rate
                epochs = 500  # Number of training epochs
                es_patience = 50  # Patience for early stopping


                loss_fn = tf.keras.losses.MeanSquaredError()
                mae_fn = tf.keras.losses.MeanAbsoluteError()


                idxsGNNLSTM = list(range(len(dataGNNLSTM)))

                data_teGNNLSTM = dataGNNLSTM[idxsGNNLSTM]

                print(len(data_teGNNLSTM))

                loader_teGNNLSTM = MixedLoader(data_teGNNLSTM, batch_size=batch_sizes, shuffle=False, epochs=1)


                # model_path = save_path
                modelGNNLSTM = load_model(model_path, compile=False)


                def denormalize(value, min_val, max_val):
                    """
                    Normalize된 값을 실제 값으로 복원하는 함수.
                    """
                    return value * (max_val - min_val) + min_val


                def save_denormalized_predictions_to_files(loader, model, output_dir, M_min, M_max, A_min, A_max, nBus):
                    """
                    모델의 예측값(M, A)을 Denormalize하여 각 데이터셋별로 텍스트 파일로 저장하는 함수.
                    """
                    os.makedirs(output_dir, exist_ok=True)  # 결과 저장 경로 생성

                    for batch_va in loader:
                        # 모델 예측 수행
                        pred_va, targ_va = model(*batch_va, training=False)

                        # 예측값의 Shape 확인
                        num_samples, num_buses, num_times, _ = pred_va.shape
                        for i in range(num_samples):
                            print(f"Dataset {i + 1} - Shape of pred_va: {pred_va.shape}")

                            # 파일 경로 생성
                            output_file = os.path.join(output_dir, f"failed_predictions_{i + 1}.txt")
                            with open(output_file, "w") as file:
                                file.write("voltages and angles:\n")  # 헤더 작성

                                for bus_idx in range(num_buses):
                                    for time_idx in range(num_times):
                                        # 예측값 M, A 추출
                                        M = pred_va[i, bus_idx, time_idx, 0].numpy()
                                        A = pred_va[i, bus_idx, time_idx, 1].numpy()

                                        # Denormalize 수행
                                        M_denormalized = M * (M_max - M_min) + M_min
                                        A_denormalized = A * (A_max - A_min) + A_min

                                        # 파일에 저장
                                        file.write(
                                            f"bus {bus_idx + 1} M {M_denormalized:.16f} A {A_denormalized:.16f} k {time_idx}\n"
                                        )

                            print(f"Saved predictions to {output_file}")


                # Denormalized된 예측값을 각 데이터셋 파일에 저장

                save_denormalized_predictions_to_files(
                    loader_teGNNLSTM,
                    modelGNNLSTM,
                    output_dir,
                    M_min,
                    M_max,
                    A_min,
                    A_max,
                    nBus=nBus  # 총 버스 수
                )

            if mode == 'save_excel':

                # model_path = f"{PROJECT_ROOT}/Data/same proportions/500bus/{nPrd}/GNN_trained_model/!_GATConv_TCN_modelMA_SavedModel"
                model_path = f"{PROJECT_ROOT}/Data/same proportions/500bus/{nPrd}/GNN_trained_model/sigx_GATConv_TCN_modelMA_SavedModel"
                output_dir = f"{PROJECT_ROOT}/Data/same proportions/500bus/sigmoid_x/500bus/{nPrd}/{nPrd}_test_excel/"
                batch_sizes = 150

                Bra_dataFile = f'{PROJECT_ROOT}/Data/same proportions/500bus/{nPrd}/case500_edgedata.dat'
                Bra_data = pd.read_csv(Bra_dataFile)
                fBus = Bra_data["branch_fbus"]
                tBus = Bra_data["branch_tbus"]
                kRating = Bra_data["branch_rateA"]
                kBranch_b = Bra_data["branch_b"]

                Demand_fileName = f'{PROJECT_ROOT}/Data/same proportions/500bus/{nPrd}/demand500Bus_{nPrd}_all.txt'

                dfNDmd = loadtxt(Demand_fileName, delimiter=',')  # Nodal demand NBus*24-Hours sequences

                # 노드의 M A데이터셋
                madata = f'{PROJECT_ROOT}/Data/same proportions/500bus/{nPrd}/500_{nPrd}_ma_values_all.txt'
                dfLFlw = loadtxt(madata, delimiter=',')
                # each graph(sample) must have 24 Nodal values and edge connection(constant, topology does not chage)
                x_data = dfNDmd
                nSamples = len(x_data)
                y_dataFlow = dfLFlw
                y_data = dfLFlw

                # NF (Node features) include nodal demand
                N_data = np.zeros([nSamples, nBus, nPrd])
                MA_data = np.zeros((nSamples, nBus, nPrd, 2))  # 1000, 24, 12 , 2

                for m in range(nSamples):
                    for t in range(nPrd):
                        for n in range(nBus):
                            N_data[m, n, t] = x_data[m, (t) * nBus + n]

                for m in range(nSamples):
                    for t in range(nPrd):
                        for n in range(nBus):
                            MA_data[m, n, t, 0] = y_data[m, 2 * t * nBus + 2 * n]  # M
                            MA_data[m, n, t, 1] = y_data[m, 2 * t * nBus + 2 * n + 1]  # A

                # Create NF (Node features)
                NF = np.zeros([nSamples, nBus, nPrd])
                for m in range(nSamples):
                    for t in range(nPrd):
                        for n in range(nBus):
                            NF[m, n, t] = x_data[m, (t) * nBus + n]  # Load Profile

                print("NF :", NF.shape)


                # normalize
                def min_max_normalize(data):
                    min_val = np.min(data)
                    max_val = np.max(data)
                    return (data - min_val) / (max_val - min_val), min_val, max_val


                # Normalize M and A
                MA_data_normalized = np.zeros_like(MA_data)
                MA_data_normalized[..., 0], M_min, M_max = min_max_normalize(MA_data[..., 0])  # Normalize M
                MA_data_normalized[..., 1], A_min, A_max = min_max_normalize(MA_data[..., 1])  # Normalize A

                print("M_normalized min:", np.min(MA_data_normalized[..., 0]))
                print("M_normalized max:", np.max(MA_data_normalized[..., 0]))
                print("A_normalized min:", np.min(MA_data_normalized[..., 1]))
                print("A_normalized max:", np.max(MA_data_normalized[..., 1]))

                e_flow = np.zeros([nSamples, nBranch, nPrd])
                for m in range(nSamples):
                    for t in range(nPrd):
                        for k in range(nBranch):
                            e_flow[m, k, t] = ((y_dataFlow[m, (t) * nBranch + k]) / kRating[k])  ##########

                n_llel = 0
                for k in range(nBranch):
                    if k > 0:
                        if ((fBus[k] == fBus[k - 1]) and (tBus[k] == tBus[k - 1])):
                            n_llel = n_llel + 1

                print("n_llel : ", n_llel)  # 13
                # initialize vectors to fill
                kRating_wo_llel = np.zeros([nBranch - n_llel, 1])
                kBranch_b_wo_llel = np.zeros([nBranch - n_llel, 1])

                e_flow_wo_llel = np.zeros([nSamples, (nBranch - n_llel), nPrd])

                k_new = -1
                fBus_llel = np.zeros([nBranch - n_llel, 1])
                tBus_llel = np.zeros([nBranch - n_llel, 1])
                for k in range(nBranch):
                    if k == 0:
                        k_new = k_new + 1
                        kRating_wo_llel[k_new] = kRating[k]
                        kBranch_b_wo_llel[k_new] = kBranch_b[k]
                        e_flow_wo_llel[:, k_new, :] = e_flow[:, k, :]
                        fBus_llel[k_new] = fBus[k]
                        tBus_llel[k_new] = tBus[k]
                    else:
                        if ((fBus[k] == fBus[k - 1]) and (tBus[k] == tBus[k - 1])):
                            kRating_wo_llel[k_new] = max(kRating[k], kRating[k - 1])  ###################
                            kBranch_b_wo_llel[k_new] = max(kBranch_b[k], kBranch_b[k - 1])  #############
                            for m in range(nSamples):
                                for t in range(nPrd):
                                    e_flow_wo_llel[m, k_new, t] = max(e_flow[m, k, t], e_flow[m, k - 1, t])  ##########
                        else:
                            k_new = k_new + 1
                            kRating_wo_llel[k_new] = kRating[k]
                            kBranch_b_wo_llel[k_new] = kBranch_b[k]
                            e_flow_wo_llel[:, k_new, :] = e_flow[:, k, :]
                            fBus_llel[k_new] = fBus[k]
                            tBus_llel[k_new] = tBus[k]

                # Create Edge features of nBus,nBus,nedgefeat
                EF2 = np.zeros([nSamples, nBus, nBus, 2])
                for m in range(nSamples):
                    for k in range(nBranch - n_llel):
                        EF2[m, int(fBus_llel[k]) - 1, int(tBus_llel[k]) - 1, 0] = kBranch_b_wo_llel[k]  # Reactance
                        EF2[m, int(fBus_llel[k]) - 1, int(tBus_llel[k]) - 1, 1] = kRating_wo_llel[k]  # Line Limit

                # 인접행렬 저장하는 코드

                # Adjacency matrix as sparse matrix
                AM_sparse = csr_matrix((np.ones(nBranch - n_llel), (fBus_llel[:, 0] - 1, tBus_llel[:, 0] - 1)),
                                       shape=[nBus, nBus])

                # Dense matrix 생성 (벡터화 적용)
                AM_dense = np.zeros([nBus, nBus])
                AM_dense[(fBus_llel[:, 0] - 1).astype(int), (tBus_llel[:, 0] - 1).astype(int)] = 1

                path = f'{PROJECT_ROOT}/Data/same proportions/500bus/{nPrd}/graph_data'
                if not os.path.exists(path):
                    os.makedirs(path)

                for i in range(nSamples):
                    filename = os.path.join(path, f'GNN_{i}')
                    np.savez_compressed(filename, x=NF[i, :, :], a=AM_sparse, e=EF2[i, :, :, :],
                                        y=MA_data_normalized[i, :, :, :])


                # 데이터셋 불러오기
                class Graphs_DataGNNLSTM(spektral.data.dataset.Dataset):
                    # class Graphs_DataGNNLSTM(spektral.data.Dataset):
                    def read(self):
                        output = []

                        path = f'{PROJECT_ROOT}/Data/same proportions/500bus/{nPrd}/graph_data'
                        # Load data from npz files into read()
                        for i in range(nSamples):
                            graph = np.load(os.path.join(path, f'GNN_{i}.npz'))
                            output.append(
                                spektral.data.Graph(x=graph['x'], e=graph['e'], y=graph['y']))  # e필요없으면 제외 가능

                        return output


                # 인접행렬 로드
                def load_adjacency_matrix(file_path_sparse, file_path_dense):
                    # .npz 파일에서 sparse 행렬 로드
                    AM_sparse = load_npz(file_path_sparse)
                    # .npy 파일에서 dense 행렬 로드
                    AM_dense = np.load(file_path_dense)
                    return AM_sparse, AM_dense


                # 경로 설정
                adjacency_matrix_path_sparse = f'{PROJECT_ROOT}/Data/same proportions/500bus/{nPrd}/AM/AM_sparse.npz'
                adjacency_matrix_path_dense = f'{PROJECT_ROOT}/Data/same proportions/500bus/{nPrd}/AM/AM_dense.npy'

                # 인접행렬 로드
                AM_sparse_loaded, AM_dense = load_adjacency_matrix(
                    adjacency_matrix_path_sparse, adjacency_matrix_path_dense
                )

                # Load Data
                GNN_DataGNNLSTM = Graphs_DataGNNLSTM()
                bus = GNN_DataGNNLSTM[0].n_nodes
                # Adjacency matrix is intentionally avoided since static netowrk topology is used
                # mixed data => one AM for all data samples. i.e. topology does not change

                # 차수 행렬 계산
                D_sparse = degree_matrix(AM_sparse_loaded)

                l_sparse = laplacian(AM_sparse_loaded)  # 차수행렬D - 인접행렬A = 라플라시안 행렬 L

                Modified_L = gcn_filter(AM_sparse_loaded, symmetric=True)

                norm_laplacian = normalized_laplacian(AM_sparse_loaded, symmetric=True)
                norm_rescaled_L = rescale_laplacian(norm_laplacian)

                Chebyshev = chebyshev_filter(AM_sparse_loaded, 1)

                normalized_sparse_A = normalized_adjacency(AM_sparse_loaded)
                ##########################################################################################
                # dense matrix
                normalized_A = normalized_adjacency(AM_dense, symmetric=True)

                ########################################################################################## AM
                # print(l_sparse)
                # 인접행렬 변경
                if GNN == "GINconvBatch":
                    # binary dense adjacency matrix  #GINConvbatch
                    GNN_DataGNNLSTM.a = AM_dense
                elif GNN == "GATConv" or GNN == "ECCConv":
                    # #  Binary adjacency matrix # GAT # ECC
                    GNN_DataGNNLSTM.a = AM_sparse_loaded
                elif GNN == "APPNP" or GNN == "GCNConv":
                    # Modified_AM_sparse # APPNP , GCN
                    GNN_DataGNNLSTM.a = Modified_L
                elif GNN == "ARMA":
                    # norm_rescaled # ARMA
                    GNN_DataGNNLSTM.a = norm_rescaled_L
                # elif GNN == "APPNP":
                #     # Chebyshev
                #     GNN_DataGNNLSTM.a = Chebyshev
                elif GNN == "DiffusionConv" or GNN == "GCSConv":
                    # DiffusionConv ,GCSConv
                    GNN_DataGNNLSTM.a = normalized_A

                dataGNNLSTM = GNN_DataGNNLSTM
                ################################################################################
                # Config
                ################################################################################
                # learning_rate = 0.003  # Learning rate
                epochs = 500  # Number of training epochs
                es_patience = 50  # Patience for early stopping

                loss_fn = tf.keras.losses.MeanSquaredError()
                mae_fn = tf.keras.losses.MeanAbsoluteError()

                modelGNNLSTM = load_model(model_path, compile=False)

                # 데이터셋 분할
                # Train/valid/test split
                idxsGNNLSTM = range(len(dataGNNLSTM))
                split_vaGNNLSTM, split_teGNNLSTM = int(0.70 * len(dataGNNLSTM)), int(0.85 * len(dataGNNLSTM))
                idx_trGNNLSTM, idx_vaGNNLSTM, idx_teGNNLSTM = np.split(idxsGNNLSTM, [split_vaGNNLSTM, split_teGNNLSTM])
                # data_trGNNLSTM = dataGNNLSTM[idx_trGNNLSTM]
                # data_vaGNNLSTM = dataGNNLSTM[idx_vaGNNLSTM]
                data_teGNNLSTM = dataGNNLSTM[idx_teGNNLSTM]

                # # Data loaders
                # loader_trGNNLSTM = MixedLoader(data_trGNNLSTM, batch_size=batch_sizes, epochs=epochs, shuffle=False)
                # loader_vaGNNLSTM = MixedLoader(data_vaGNNLSTM, batch_size=batch_sizes, shuffle = False, epochs=1)
                loader_teGNNLSTM = MixedLoader(data_teGNNLSTM, batch_size=batch_sizes, shuffle=False, epochs=1)



                def save_denormalized_predictions_to_excels(loader, model, output_dir, M_min, M_max, A_min, A_max,
                                                            nBus):
                    """
                    각 샘플별로 denormalized된 예측값(M, A)을 엑셀 파일로 저장하는 함수.
                    """
                    os.makedirs(output_dir, exist_ok=True)

                    for i, batch_va in enumerate(loader):
                        pred_va, targ_va = model(*batch_va, training=False)
                        num_samples, num_buses, num_times, _ = pred_va.shape

                        for sample_idx in range(num_samples):
                            data = []
                            for bus_idx in range(num_buses):
                                for time_idx in range(num_times):
                                    M = pred_va[sample_idx, bus_idx, time_idx, 0].numpy()
                                    A = pred_va[sample_idx, bus_idx, time_idx, 1].numpy()
                                    M_denorm = M * (M_max - M_min) + M_min
                                    A_denorm = A * (A_max - A_min) + A_min

                                    data.append({
                                        'Bus': bus_idx + 1,
                                        'Time': time_idx,
                                        'M': M_denorm,
                                        'A': A_denorm
                                    })

                            df = pd.DataFrame(data)
                            output_file = os.path.join(output_dir, f"predictions_sample_{i + sample_idx + 1}.xlsx")
                            df.to_excel(output_file, index=False)
                            print(f"Saved predictions to {output_file}")


                save_denormalized_predictions_to_excels(
                    loader_teGNNLSTM,
                    modelGNNLSTM,
                    output_dir,
                    M_min,
                    M_max,
                    A_min,
                    A_max,
                    nBus=nBus  # 총 버스 수
                )

            if mode == 'test':

                model_path = f"{PROJECT_ROOT}/Data/same proportions/500bus/{nPrd}/GNN_trained_model/!_GATConv_TCN_modelMA_SavedModel"
                output_dir = f"{PROJECT_ROOT}/Data/25_1017 access review/500bus/{nPrd}"

                # 데이터셋 불러오기
                class Graphs_DataGNNLSTM(spektral.data.dataset.Dataset):
                    # class Graphs_DataGNNLSTM(spektral.data.Dataset):
                    def read(self):
                        output = []

                        path = f'{PROJECT_ROOT}/Data/same proportions/500bus/{nPrd}/graph_data'
                        # Load data from npz files into read()
                        for i in range(nSamples):
                            graph = np.load(os.path.join(path, f'GNN_{i}.npz'))
                            output.append(spektral.data.Graph(x=graph['x'], e=graph['e'], y=graph['y']))  # e필요없으면 제외 가능

                        return output

                # 인접행렬 로드
                def load_adjacency_matrix(file_path_sparse, file_path_dense):
                    # .npz 파일에서 sparse 행렬 로드
                    AM_sparse = load_npz(file_path_sparse)
                    # .npy 파일에서 dense 행렬 로드
                    AM_dense = np.load(file_path_dense)
                    return AM_sparse, AM_dense

                # 경로 설정
                adjacency_matrix_path_sparse = f'{PROJECT_ROOT}/Data/same proportions/500bus/{nPrd}/AM/AM_sparse.npz'
                adjacency_matrix_path_dense = f'{PROJECT_ROOT}/Data/same proportions/500bus/{nPrd}/AM/AM_dense.npy'

                # 인접행렬 로드
                AM_sparse_loaded, AM_dense = load_adjacency_matrix(
                    adjacency_matrix_path_sparse, adjacency_matrix_path_dense
                )


                # Load Data
                GNN_DataGNNLSTM = Graphs_DataGNNLSTM()
                bus = GNN_DataGNNLSTM[0].n_nodes
                # Adjacency matrix is intentionally avoided since static netowrk topology is used
                # mixed data => one AM for all data samples. i.e. topology does not change

                # 차수 행렬 계산
                D_sparse = degree_matrix(AM_sparse_loaded)

                l_sparse = laplacian(AM_sparse_loaded) # 차수행렬D - 인접행렬A = 라플라시안 행렬 L

                Modified_L = gcn_filter(AM_sparse_loaded, symmetric=True)

                norm_laplacian = normalized_laplacian(AM_sparse_loaded,  symmetric=True)
                norm_rescaled_L = rescale_laplacian(norm_laplacian)

                Chebyshev = chebyshev_filter(AM_sparse_loaded,1)

                normalized_sparse_A = normalized_adjacency(AM_sparse_loaded)
                ##########################################################################################
                # dense matrix
                normalized_A = normalized_adjacency(AM_dense, symmetric=True)

                ########################################################################################## AM
                # print(l_sparse)
                # 인접행렬 변경
                if GNN == "GINconvBatch":
                    # binary dense adjacency matrix  #GINConvbatch
                    GNN_DataGNNLSTM.a = AM_dense
                elif GNN == "GATConv" or GNN == "ECCConv":
                    # #  Binary adjacency matrix # GAT # ECC
                    GNN_DataGNNLSTM.a = AM_sparse_loaded
                elif GNN == "APPNP" or GNN == "GCNConv":
                    # Modified_AM_sparse # APPNP , GCN
                    GNN_DataGNNLSTM.a = Modified_L
                elif GNN == "ARMA":
                    # norm_rescaled # ARMA
                    GNN_DataGNNLSTM.a = norm_rescaled_L
                # elif GNN == "APPNP":
                #     # Chebyshev
                #     GNN_DataGNNLSTM.a = Chebyshev
                elif GNN == "DiffusionConv" or GNN == "GCSConv":
                    # DiffusionConv ,GCSConv
                    GNN_DataGNNLSTM.a = normalized_A

                dataGNNLSTM = GNN_DataGNNLSTM

                ################################################################################
                # Config
                ################################################################################
                # learning_rate = 0.003  # Learning rate
                epochs = 500  # Number of training epochs
                es_patience = 50  # Patience for early stopping
                batch_sizes = 32  # Batch size 더 커질시 작동 x

                loss_fn = tf.keras.losses.MeanSquaredError()
                mae_fn = tf.keras.losses.MeanAbsoluteError()


                idxsGNNLSTM = range(len(dataGNNLSTM))
                split_vaGNNLSTM, split_teGNNLSTM = int(0.70 * len(dataGNNLSTM)), int(0.85 * len(dataGNNLSTM))
                idx_trGNNLSTM, idx_vaGNNLSTM, idx_teGNNLSTM = np.split(idxsGNNLSTM, [split_vaGNNLSTM, split_teGNNLSTM])
                data_trGNNLSTM = dataGNNLSTM[idx_trGNNLSTM]
                data_vaGNNLSTM = dataGNNLSTM[idx_vaGNNLSTM]
                data_teGNNLSTM = dataGNNLSTM[idx_teGNNLSTM]
                data_allGNNLSTM = dataGNNLSTM

                loader_trGNNLSTM = MixedLoader(data_trGNNLSTM, batch_size=batch_sizes, shuffle=False, epochs=1)
                loader_vaGNNLSTM = MixedLoader(data_vaGNNLSTM, batch_size=batch_sizes, shuffle=False, epochs=1)
                loader_teGNNLSTM = MixedLoader(data_teGNNLSTM, batch_size=batch_sizes, shuffle=False, epochs=1)
                loader_allGNNLSTM = MixedLoader(data_allGNNLSTM, batch_size=batch_sizes, shuffle=False, epochs=1)


                modelGNNLSTM = load_model(model_path, compile=False)

                # 평가할 데이터셋 딕셔너리로 관리
                datasets = {
                    "Train": loader_trGNNLSTM,
                    "Validation": loader_vaGNNLSTM,
                    "Test": loader_teGNNLSTM,
                    "Whole Dataset": loader_allGNNLSTM
                }

                results = {}

                # 각 세트에 대해 evaluate 수행
                for name, loader in datasets.items():
                    print(f"\nEvaluating on {name} set...")
                    start_time = time.time()
                    loss, mae, smape, smape_m, smape_a, nmae, nmae_m, nmae_a, mae_m, mae_a = evaluateGNNLSTM(loader)
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    dataset_size = len(loader.dataset) if hasattr(loader, 'dataset') else len(loader.data)
                    time_per_dataset = elapsed_time / dataset_size

                    results[name] = {
                        "Loss": loss,
                        "MAE": mae,
                        "MAE_M": mae_m,
                        "MAE_A": mae_a,
                        "SMAPE": smape,
                        "SMAPE_M": smape_m,
                        "SMAPE_A": smape_a,
                        "NMAE": nmae,
                        "NMAE_M": nmae_m,
                        "NMAE_A": nmae_a,
                        "Total Time": elapsed_time,
                        "Time per Dataset": time_per_dataset
                    }

                # 파일로 저장
                output_file = f"{PROJECT_ROOT}/Data/25_1017 access review/500bus/{nPrd}/{GNN}_{RNN}_results_all.txt"
                with open(output_file, "w") as f:
                    for name, metrics in results.items():
                        f.write(f"\n===== {name} Set Results =====\n")
                        for k, v in metrics.items():
                            f.write(f"{k}: {v:.5f}\n")
                print(f"All results saved to {output_file}")

                # 콘솔 출력 요약
                for name, metrics in results.items():
                    print(f"--- {name} Set ---")
                    print(f"Loss: {metrics['Loss']:.5f}, SMAPE: {metrics['SMAPE']:.5f}")



