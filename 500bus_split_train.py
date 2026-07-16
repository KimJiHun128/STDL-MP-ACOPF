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

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


# nohup python 500bus_split_train.py > output.log 2>&1 &
mode = 'train' # 'train' or 'data gen'

# nPrd = 8 #8, 24
nBus = 500
nSamples = 1000
SysName = "500Bus"

nGen = 90
nBranch = 597
n_llel = 13

np.random.seed(1)
np.set_printoptions(threshold=sys.maxsize)

nPrd_list = [8,24] # ,24
GNN_types = ["APPNP", "ECCConv","ARMA" , "DiffusionConv", "GATConv", "GCNConv", "GCSConv", "GINconvBatch"] # "APPNP", "ECCConv","ARMA" , "DiffusionConv", "GATConv", "GCNConv", "GCSConv", "GINconvBatch"
RNN_types = [ "LSTM"] # "GRU", "LSTM", "BiGRU","BiLSTM", "TCN"

# GNN = "APPNP" #APPNP, ECCConv, ARMA, DiffusionConv, GATConv, GCNConv, GCSConv, GINconvBatch
# RNN = "TCN" # GRU, LSTM, BiGRU, BiLSTM, TCN

for Prd in nPrd_list:
    nPrd = Prd
    for gnn_type in GNN_types:
        GNN = gnn_type
        for rnn_type in RNN_types:
            RNN = rnn_type

            print(f"Testing GNN: {GNN}, RNN: {RNN}")

            ###################################################################
            ######data gen#######
            ###################################################################
            if mode == 'data gen':


                Bra_dataFile = f'{PROJECT_ROOT}/Data/same proportions/500bus/{nPrd}/case500_edgedata.dat'
                Bra_data = pd.read_csv(Bra_dataFile)
                # print (Bra_data.info)
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
                    filename = os.path.join(path,f'GNN_{i}')
                    np.savez_compressed(filename, x = NF[i,:,:], a = AM_sparse, e = EF2[i,:,:,:], y = MA_data_normalized[i,:,:,:])


            ######train#######
            ###################################################################




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
                epochs = 1000  # Number of training epochs
                es_patience = 100  # Patience for early stopping
                batch_sizes = 64  # Batch size


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
                            self.conv2 = ECCConv(nPrd, activation="PReLU", root=True, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros')

                        # 엣지 정보 필요없는 layer

                        elif GNN == "GINconvBatch":
                            # GINConvBatch
                            self.conv1 = GINConvBatch(nPrd, epsilon=None, mlp_hidden=None, mlp_activation='relu', mlp_batchnorm=True, aggregate='sum', activation=None, use_bias=False)
                            self.conv2 = GINConvBatch(nPrd, epsilon=None, mlp_hidden=None, mlp_activation='relu', mlp_batchnorm=True, aggregate='sum', activation=PReLU(), use_bias=True)
                        elif GNN == "GATConv":
                            # GATConv
                            self.conv1 = GATConv(nPrd, attn_heads=3, concat_heads=True, dropout_rate=0.5, add_self_loops=True, activation=None, use_bias=False)
                            self.conv2 = GATConv(nPrd, attn_heads=3, concat_heads=True, dropout_rate=0.5, add_self_loops=True, activation=PReLU(), use_bias=True)
                        elif GNN == "GCNConv":
                            # # GCNConv
                            self.conv1 = GCNConv(nPrd, activation=None, use_bias=False)
                            self.conv2 = GCNConv(nPrd, activation=PReLU(), use_bias=True)
                        elif GNN == "APPNP":
                            # APPNP
                            self.conv1 = APPNPConv(nPrd, alpha=0.2, propagations=1, mlp_hidden=None, mlp_activation='relu', dropout_rate=0.0, activation=None, use_bias=False, kernel_initializer='glorot_uniform', bias_initializer='zeros')
                            self.conv2 = APPNPConv(nPrd, alpha=0.2, propagations=1, mlp_hidden=None, mlp_activation='relu', dropout_rate=0.0, activation=PReLU(), use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros')

                        elif GNN == "ARMA":
                            # ARMA
                            self.conv1 = ARMAConv(channels=nPrd, order=1, iterations=1, share_weights=False, gcn_activation='relu', dropout_rate=0.0, activation=None, use_bias=False, kernel_initializer='glorot_uniform', bias_initializer='zeros')
                            self.conv2 = ARMAConv(channels=nPrd, order=1, iterations=1, share_weights=False, gcn_activation='relu', dropout_rate=0.0, activation=PReLU(), use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros')

                        elif GNN == "DiffusionConv":
                            # DiffusionConv
                            self.conv1 = DiffusionConv(nPrd, K=6, activation=None, kernel_initializer='glorot_uniform', kernel_regularizer=None, kernel_constraint=None)
                            self.conv2 = DiffusionConv(nPrd, K=6, activation='tanh', kernel_initializer='glorot_uniform', kernel_regularizer=None, kernel_constraint=None)
                        elif GNN == "GCSConv":
                            # GCSConv
                            self.conv1 = GCSConv(nPrd, activation=None, use_bias=False, kernel_initializer='glorot_uniform', bias_initializer='zeros')
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
                        elif RNN == "TCN":
                            self.lstm1 = TCN(nb_filters=nBus, kernel_size=3, dilations=(1, 3, 9), padding='causal',
                                             activation="tanh", return_sequences=True)


                        self.drop = Dropout(0.05)
                        self.dense = Dense(nPrd, activation="sigmoid")  # V, A

                        # bidirectional
                        self.dense1 = Dense(nBus)  # V, A
                        self.dense2 = Dense(nPrd, activation="sigmoid")  # V, A

                        self.bn1 = BatchNormalization()
                        self.bn2 = BatchNormalization()
                        self.bn3 = BatchNormalization()

                    def call(self, inputs, labels):
                        if GNN == "ECCConv": # edge 사용하는 layer

                            x, a, e = inputs
                            x = self.conv1([x, a, e])
                            x = self.bn1(x)
                            x = self.conv2([x, a, e])
                            x = self.bn2(x)

                        else:# edge 사용 안하는 layer

                            x, a, e = inputs
                            x = self.conv1([x, a])
                            x = self.bn1(x)
                            x = self.conv2([x, a])
                            x = self.bn2(x)

                        if RNN in  ["GRU", "LSTM"]: # , "TCN"
                            # # non bidirectional
                            x = tf.transpose(x,[0, 2, 1])  # (batch_size, 24, 12) -> (batch_size, 12, 24) (batch_size, time_steps, features)
                            # print(f"Before LSTM (after transpose): {x.shape}")  # LSTM 입력 전 출력
                            out = self.lstm1(x)
                            out = self.bn3(out)
                            # print(f"After LSTM: {out.shape}")  # LSTM 출력
                            out = tf.transpose(out, [0, 2, 1])  # (batch_size, 12, 24) -> (batch_size, 24, 12)
                            # print(f"After transpose (before Dense): {out.shape}")  # Dense 입력 전 출력
                            output = self.dense(out)
                            # print(f"After Dense: {output.shape}")  # Dense 출력
                            output = tf.reshape(output, [tf.shape(output)[0], nBus, nPrd, 1])  # 출력 형태를 (batch_size, 24, 12, 1)로 조정
                            return output, labels  # labels의 형태는 (batch_size, 24, 12, 1)여야 함

                        else:
                            # bidirectional
                            x = tf.transpose(x, [0, 2, 1])  # (batch_size, 24, 12) -> (batch_size, 12, 24) (batch_size, time_steps, features) 500,8 -> 8,500
                            # print(f"Before LSTM (after transpose): {x.shape}")  # LSTM 입력 전 출력
                            out = self.lstm1(x) # (batch_size, nPrd, bus*2) 8,500*2
                            out = self.bn3(out)
                            output = self.dense1(out) # (batch_size, nPrd, 24) 8,500
                            out = tf.transpose(output, [0, 2, 1])   # (batch_size, 12, 24) -> (batch_size, 24, 12) 500 , 8
                            output = self.dense2(out) #(batch_size, 24, 12*2) 500, 8*2
                            output = tf.reshape(output, [tf.shape(output)[0], nBus, nPrd, 1]) # 출력 형태를 (batch_size, 24, 12, 2)로 조정 500, 8, 2
                            return output, labels    # labels의 형태는 (batch_size, 24, 12, 1)여야 함


                # modelGNNLSTM = GNNLSTMCmt()

                modelM = GNNLSTMCmt()  # M 예측용
                modelA = GNNLSTMCmt()  # A 예측용

                lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                    initial_learning_rate=0.001,
                    decay_steps=5000,
                    decay_rate=0.95)
                optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
                ################################################################################
                # loss

                loss_fn = tf.keras.losses.MeanSquaredError()
                mae_fn = tf.keras.losses.MeanAbsoluteError()


                def SMAPE_A(y_true, y_pred):
                    # y_true와 y_pred는 각각 [m_true, A_true]와 [m_pred, A_pred] 형식으로 들어옴
                    A_true = tf.cast(y_true[:, 1], tf.float32)
                    A_pred = tf.cast(y_pred[:, 0], tf.float32)

                    smape_A = tf.reduce_mean(2 * tf.abs(A_true - A_pred) / (tf.abs(A_true) + tf.abs(A_pred))) * 100  # A에 대한 SMAPE

                    return  smape_A

                def SMAPE_M(y_true, y_pred):
                    # y_true와 y_pred는 각각 [m_true, A_true]와 [m_pred, A_pred] 형식으로 들어옴
                    m_true = tf.cast(y_true[:, 0], tf.float32)
                    m_pred = tf.cast(y_pred[:, 0], tf.float32)

                    # SMAPE 계산
                    smape_m = tf.reduce_mean(2 * tf.abs(m_true - m_pred) / (tf.abs(m_true) + tf.abs(m_pred))) * 100  # m에 대한 SMAPE

                    return  smape_m


                def evaluateGNNLSTM(loader,model):
                    total_loss = 0
                    total_mae = 0
                    total_samples = 0
                    total_smape = 0


                    for batch_va in loader:
                        pred_va, targ_va = model(*batch_va, training=False)
                        loss_va = loss_fn(targ_va, pred_va)
                        mae_va = mae_fn(targ_va, pred_va)
                        if model ==modelM:
                            smape = SMAPE_M(targ_va, pred_va)
                        elif model == modelA:
                            smape = SMAPE_A(targ_va, pred_va)

                        batch_size = tf.shape(targ_va)[0]
                        # 배치 크기에 비례한 손실의 총합
                        total_loss += loss_va * tf.cast(batch_size, tf.float32)
                        total_mae += mae_va * tf.cast(batch_size, tf.float32)
                        total_smape += smape * tf.cast(batch_size, tf.float32)
                        total_samples += batch_size

                    # 평균 손실 및 MAE, MAPE 계산
                    avg_loss = total_loss / tf.cast(total_samples, tf.float32)
                    avg_mae = total_mae / tf.cast(total_samples, tf.float32)
                    avg_smape = total_smape / tf.cast(total_samples, tf.float32)


                    return avg_loss, avg_mae, avg_smape
                # 데이터셋 분할

                #data=data_temp
                # Train/valid/test split
                idxsGNNLSTM = range(len(dataGNNLSTM))
                split_vaGNNLSTM, split_teGNNLSTM = int(0.70 * len(dataGNNLSTM)), int(0.85 * len(dataGNNLSTM))
                idx_trGNNLSTM, idx_vaGNNLSTM, idx_teGNNLSTM = np.split(idxsGNNLSTM, [split_vaGNNLSTM, split_teGNNLSTM])
                data_trGNNLSTM = dataGNNLSTM[idx_trGNNLSTM]
                data_vaGNNLSTM = dataGNNLSTM[idx_vaGNNLSTM]
                data_teGNNLSTM = dataGNNLSTM[idx_teGNNLSTM]
                #print (idx_te)
                print("train:",len(data_trGNNLSTM))
                print("val:",len(data_vaGNNLSTM))
                print("test:",len(data_teGNNLSTM))

                #print(data_tr.n_node_features)

                # Data loaders
                loader_trGNNLSTM1 = MixedLoader(data_trGNNLSTM, batch_size=batch_sizes, epochs=epochs, shuffle = False)
                loader_trGNNLSTM2 = MixedLoader(data_trGNNLSTM, batch_size=batch_sizes, epochs=epochs, shuffle=False)
                loader_vaGNNLSTM = MixedLoader(data_vaGNNLSTM, batch_size=batch_sizes, shuffle = False, epochs=1)
                loader_teGNNLSTM = MixedLoader(data_teGNNLSTM, batch_size=batch_sizes, shuffle = False, epochs=1)

                # # Data loaders
                # loader_trGNNLSTM = BatchLoader(data_trGNNLSTM, batch_size=batch_sizes, epochs=epochs, shuffle = False ,node_level= False)
                # loader_vaGNNLSTM = BatchLoader(data_vaGNNLSTM, batch_size=batch_sizes, shuffle = False, epochs=1,node_level= False)
                # loader_teGNNLSTM = BatchLoader(data_teGNNLSTM, batch_size=batch_sizes, shuffle = False, epochs=1,node_level= False)

                # # Data loaders
                # loader_trGNNLSTM = SingleLoader(data_trGNNLSTM,  epochs=epochs, shuffle = False)
                # loader_vaGNNLSTM = SingleLoader(data_vaGNNLSTM,  shuffle = False, epochs=1)
                # loader_teGNNLSTM = SingleLoader(data_teGNNLSTM,  shuffle = False, epochs=1)

                # 데이터 로더에서 가져오는 배치의 형태 확인
                for batch_va in loader_trGNNLSTM1:  # 여기서 트레이닝 로더 사용
                    inputs, labels = batch_va

                    print("Inputs shape:", [x.shape for x in inputs])  # x, a, e의 형태
                    print("Labels shape:", labels.shape)  # (batch_size, 24, 12, 2) #batch size 64
                    break  # 한 번만 확인

                ###################################################################################################################
                #### TRAINING modelM
                ###################################################################################################################
                epoch = step = 0
                best_val_loss = np.inf
                best_val_acc = 0
                best_weightsM = None
                patience = es_patience
                results = []
                trackGNNLSTM = np.zeros([epochs, 6])

                for batch in loader_trGNNLSTM1:
                    step += 1

                    with tf.GradientTape() as tape:
                        prediction, target = modelM(*batch, training=True)
                        loss = loss_fn(target, prediction) # + sum(modelM.losses)  # mse 만 적용 정규화 손실 적용 x

                    gradients = tape.gradient(loss, modelM.trainable_variables,
                                              unconnected_gradients=tf.UnconnectedGradients.ZERO)
                    optimizer.apply_gradients(zip(gradients, modelM.trainable_variables))

                    mae = tf.reduce_mean(tf.abs(target - prediction))
                    smape_total = SMAPE_M(target, prediction)

                    results.append((loss.numpy(), mae.numpy(), smape_total.numpy(),))

                    # Print out result after every epoch
                    if step == loader_trGNNLSTM1.steps_per_epoch:

                        # Compute validation loss and accuracy
                        loader_vaGNNLSTM = MixedLoader(data_vaGNNLSTM, batch_size=batch_sizes, shuffle=False, epochs=1)
                        val_loss, val_mae, val_smape_total = evaluateGNNLSTM(loader_vaGNNLSTM,modelM)

                        # Save loss and accuracy for plotting
                        # val_loss와 val_mae를 numpy로 변환
                        val_loss_np = val_loss.numpy() if isinstance(val_loss, tf.Tensor) else val_loss
                        val_mae_np = val_mae.numpy() if isinstance(val_mae, tf.Tensor) else val_mae
                        val_smape_total_np = val_smape_total.numpy() if isinstance(val_smape_total, tf.Tensor) else val_smape_total
                        # val_smape_m_np = val_smape_m.numpy() if isinstance(val_smape_m, tf.Tensor) else val_smape_m
                        # val_smape_A_np = val_smape_A.numpy() if isinstance(val_smape_A, tf.Tensor) else val_smape_A

                        trackGNNLSTM[epoch, :] = [*np.mean(results, 0), val_loss_np, val_mae_np, val_smape_total_np]

                        step = 0
                        epoch += 1
                        # Print out result for each epoch
                        print(
                            "Ep. {} - Loss: {:.5f} - MAE: {:.5f} - SMAPE_M: {:.5f} - Val loss: {:.5f} - Val MAE: {:.5f} - Val SMAPE: {:.5f}".format(
                                epoch, np.mean(results, 0)[0], np.mean(results, 0)[1], np.mean(results, 0)[2],
                                val_loss_np, val_mae_np, val_smape_total_np
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

                        if val_loss_diff < 0.000001:
                            patience -= 1  # 감소시키기
                            if patience == 0:
                                print("Early stopping triggered.")
                                break
                        else:
                            # If there is a meaningful improvement, reset patience
                            best_val_loss = current_val_loss
                            patience = es_patience
                            print("New best val_loss {:.6f}".format(current_val_loss))
                            best_weightsM = modelM.get_weights()

                # NOTE: Saved model can be used in the future without training
                modelM.set_weights(best_weightsM)
                # Compile model before saving to include training information
                modelM.compile(optimizer=optimizer, loss=loss_fn, metrics=[SMAPE_M])

                #save_path = DataDir + f"/{nBus}bus/{nPrd}/GNN_trained_model/Spatio-Temporal_NodePrediction_" + SysName + "_SavedModel"
                save_path_M = f"{PROJECT_ROOT}/Data/same proportions/500bus/{nPrd}/GNN_trained_model/" + GNN +"_"+ RNN + "_modelM_SavedModel"
                os.makedirs(save_path_M, exist_ok=True)  # exist_ok=True이면 폴더가 이미 있어도 오류가 나지 않음
                modelM.save(save_path_M)

                ###################################################################################################################
                #### TRAINING modelA
                ###################################################################################################################
                epoch = step = 0
                best_val_loss2 = np.inf
                best_val_acc = 0
                best_weightsA = None
                patience2 = es_patience
                results = []
                trackGNNLSTM = np.zeros([epochs, 6])

                for batch in loader_trGNNLSTM2:
                    step += 1

                    with tf.GradientTape() as tape:
                        prediction, target = modelA(*batch, training=True)
                        loss = loss_fn(target, prediction)  # + sum(modelA.losses)  # mse 만 적용 정규화 손실 적용 x

                    gradients = tape.gradient(loss, modelA.trainable_variables,
                                              unconnected_gradients=tf.UnconnectedGradients.ZERO)
                    optimizer.apply_gradients(zip(gradients, modelA.trainable_variables))

                    mae = tf.reduce_mean(tf.abs(target - prediction))
                    smape_total = SMAPE_A(target, prediction)

                    results.append((loss.numpy(), mae.numpy(), smape_total.numpy(),))

                    # Print out result after every epoch
                    if step == loader_trGNNLSTM2.steps_per_epoch:

                        # Compute validation loss and accuracy
                        loader_vaGNNLSTM = MixedLoader(data_vaGNNLSTM, batch_size=batch_sizes, shuffle=False, epochs=1)
                        val_loss, val_mae, val_smape_total = evaluateGNNLSTM(loader_vaGNNLSTM, modelA)

                        # Save loss and accuracy for plotting
                        # val_loss와 val_mae를 numpy로 변환
                        val_loss_np = val_loss.numpy() if isinstance(val_loss, tf.Tensor) else val_loss
                        val_mae_np = val_mae.numpy() if isinstance(val_mae, tf.Tensor) else val_mae
                        val_smape_total_np = val_smape_total.numpy() if isinstance(val_smape_total, tf.Tensor) else val_smape_total
                        # val_smape_m_np = val_smape_m.numpy() if isinstance(val_smape_m, tf.Tensor) else val_smape_m
                        # val_smape_A_np = val_smape_A.numpy() if isinstance(val_smape_A, tf.Tensor) else val_smape_A

                        trackGNNLSTM[epoch, :] = [*np.mean(results, 0), val_loss_np, val_mae_np, val_smape_total_np]

                        step = 0
                        epoch += 1
                        # Print out result for each epoch
                        print(
                            "Ep. {} - Loss: {:.5f} - MAE: {:.5f} - SMAPE_A: {:.5f} - Val loss: {:.5f} - Val MAE: {:.5f} - Val SMAPE: {:.5f}".format(
                                epoch, np.mean(results, 0)[0], np.mean(results, 0)[1], np.mean(results, 0)[2],
                                val_loss_np, val_mae_np, val_smape_total_np
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
                        val_loss_diff2 = best_val_loss2 - current_val_loss

                        if val_loss_diff2 < 0.000001:
                            patience -= 1  # 감소시키기
                            if patience == 0:
                                print("Early stopping triggered.")
                                break
                        else:
                            # If there is a meaningful improvement, reset patience
                            best_val_loss2 = current_val_loss
                            patience = es_patience
                            print("New best val_loss {:.6f}".format(current_val_loss))
                            best_weightsA = modelA.get_weights()



                # NOTE: Saved model can be used in the future without training
                modelA.set_weights(best_weightsA)
                # Compile model before saving to include training information
                modelA.compile(optimizer=optimizer, loss=loss_fn, metrics=[SMAPE_A])

                # save_path = DataDir + f"/{nBus}bus/{nPrd}/GNN_trained_model/Spatio-Temporal_NodePrediction_" + SysName + "_SavedModel"
                save_path_A = f"{PROJECT_ROOT}/Data/same proportions/500bus/{nPrd}/GNN_trained_model/"  + GNN +"_"+ RNN + "_modelA_SavedModel"
                os.makedirs(save_path_A, exist_ok=True)
                modelA.save(save_path_A)

                ################################################################################
                # Evaluate model
                ################################################################################
                # modelGNNLSTM.set_weights(best_weightsGNNLSTM)  # Load best model
                import time
                from tensorflow.keras.models import load_model

                # 모델 불러오기
                # model_path = f"Data/IEEE{nBus}Bus/{nBus}bus/Spatio-Temporal_NodePrediction_{nBus}Bus_SavedModel"
                model_path_M = save_path_M
                modelM = load_model(model_path_M,compile=False )
                model_path_A = save_path_A
                modelA = load_model(model_path_A, compile=False)
                loader_teGNNLSTM = MixedLoader(data_teGNNLSTM, batch_size=batch_sizes, shuffle = False, epochs=1)
                loader_teGNNLSTM2 = MixedLoader(data_teGNNLSTM, batch_size=batch_sizes, shuffle=False, epochs=1)
                start_time = time.time()
                # 모델 M 평가
                test_loss_M, test_mae_M, test_smape_M = evaluateGNNLSTM(loader_teGNNLSTM, modelM)
                # 모델 A 평가
                test_loss_A, test_mae_A, test_smape_A = evaluateGNNLSTM(loader_teGNNLSTM2, modelA)
                end_time = time.time()

                # 테스트가 완료된 시간 계산
                elapsed_time = end_time - start_time
                print("Total Test Time: {:.5f} seconds".format(elapsed_time))
                dataset_size = len(data_teGNNLSTM)
                time_per_dataset = elapsed_time / dataset_size

                # 출력할 내용 저장
                output_text = f"""          
                Model M 
                Test Loss: {test_loss_M:.5f} 
                Test MAE: {test_mae_M:.5f} 
                Test SMAPE: {test_smape_M:.5f}
                Model A 
                Test Loss: {test_loss_A:.5f} 
                Test MAE: {test_mae_A:.5f} 
                Test SMAPE: {test_smape_A:.5f}
                
                Total Test Time: {elapsed_time:.5f} 
                Time per dataset: {time_per_dataset:.5f} 
                """

                # 파일로 저장
                output_file = f"{PROJECT_ROOT}/Data/same proportions/500bus/{nPrd}/result/{GNN}_{RNN}_results.txt"
                with open(output_file, "w") as f:
                    f.write(output_text)

                print(f"Results saved to {output_file}")

                print("Time per dataset: {:.5f} seconds".format(time_per_dataset))
                # 모델 M의 결과 출력
                print(f"Model M - Test Loss: {test_loss_M:.5f}, Test MAE: {test_mae_M:.5f}, Test SMAPE: {test_smape_M:.5f}")
                # 모델 A의 결과 출력
                print(f"Model A - Test Loss: {test_loss_A:.5f}, Test MAE: {test_mae_A:.5f}, Test SMAPE: {test_smape_A:.5f}")



                # # #### TEST
                # # # denormalization
                # # # model_path = f"Data/IEEE500Bus/500bus/Spatio-Temporal_NodePrediction_{nBus}Bus_SavedModel"
                # # model_path = save_path
                # # modelGNNLSTM = load_model(model_path, compile=False)
                # #
                # loader_teGNNLSTM = MixedLoader(data_teGNNLSTM, batch_size=1, shuffle=False, epochs=1)
                # predCmt = np.zeros([len(data_teGNNLSTM), nBus, nPrd, 2])
                # targCmt = np.zeros([len(data_teGNNLSTM), nBus, nPrd, 2])
                # # targCmt = np.zeros([len(data_teGNNLSTM),nBus,nPrd])
                #
                # j = 0
                #
                # for i in range(len(data_teGNNLSTM)):
                #     targCmt[i, :, :] = data_teGNNLSTM[i].y[:, :, :]
                #
                # for batch_va in loader_teGNNLSTM:
                #     pred_va, targ_va = modelGNNLSTM(*batch_va, training=False)
                #
                #     for n in range(nBus):
                #         for t in range(nPrd):
                #             predCmt[j, n, t, :] = pred_va[0, n, t, :]
                #
                #     j = j + 1
                #
                # # # # Denormalization
                # # # denormalized_targ_M = targCmt[..., 0] * (M_max - M_min) + M_min  # M 값 denormalization
                # # # denormalized_targ_A = targCmt[..., 1] * (A_max - A_min) + A_min  # A 값 denormalization
                # # #
                # # # denormalized_M = predCmt[..., 0] * (M_max - M_min) + M_min  # M 값 denormalization
                # # # denormalized_A = predCmt[..., 1] * (A_max - A_min) + A_min  # A 값 denormalization
                # #
                # # # 회귀 성능 평가 (예: MSE 계산)
                # # mse = np.mean(np.square(targCmt - predCmt))  # MSE 계산
                # # print("Test MSE: {:.5f}".format(mse))
                # #
                # # mae = np.mean(np.abs(targCmt - predCmt))  # MAE 계산
                # # print("Test MAE: {:.5f}".format(mae))
                # #
                # # # SMAPE 계산 (각각 m과 A에 대해서)
                # # smape_total, smape_m, smape_A = SMAPE(targCmt, predCmt)
                # # print("Test SMAPE: {:.5f}".format(smape_total))
                # # print("Test SMAPE for M: {:.5f}".format(smape_m))
                # # print("Test SMAPE for A: {:.5f}".format(smape_A))
                #
                #
                #
                # # 노드별 SMAPE를 저장할 리스트 초기화
                # node_smape_results = []
                #
                # # 각 노드에 대해 SMAPE 계산
                # for n in range(nBus):
                #     # 특정 노드의 실제값과 예측값 추출 (모든 샘플과 시간대에 대해)
                #
                #     y_true_node = targCmt[:, n, :, :]  # shape: [150, 12, 2]
                #     y_pred_node = predCmt[:, n, :, :]  # shape: [150, 12, 2]
                #
                #     # 2차원 배열을 1차원으로 변환하여 SMAPE 함수에 전달할 수 있게 reshape
                #
                #     y_true_node_flat = y_true_node.reshape(-1, 2)  # shape: [150*12, 2]
                #     y_pred_node_flat = y_pred_node.reshape(-1, 2)  # shape: [150*12, 2]
                #
                #     # 노드별 SMAPE 계산
                #     total_smape, smape_m, smape_A = SMAPE(y_true_node_flat, y_pred_node_flat)
                #
                #     # 결과를 딕셔너리 형태로 저장
                #     node_smape_results.append({
                #         "node": n,
                #         "total_smape": total_smape.numpy(),
                #         "smape_m": smape_m.numpy(),
                #         "smape_A": smape_A.numpy()
                #     })
                #
                # # 노드별 SMAPE 결과에서 값을 추출
                # total_smape_values = [result["total_smape"] for result in node_smape_results]
                # smape_m_values = [result["smape_m"] for result in node_smape_results]
                # smape_A_values = [result["smape_A"] for result in node_smape_results]
                #
                # # 평균, 최소, 최대, 중간값 계산
                # def calculate_stats(values):
                #     return {
                #         "mean": np.mean(values),
                #         "min": np.min(values),
                #         "max": np.max(values),
                #         "median": np.median(values)
                #     }
                # # 계산된 통계 출력
                # total_smape_stats = calculate_stats(total_smape_values)
                # smape_m_stats = calculate_stats(smape_m_values)
                # smape_A_stats = calculate_stats(smape_A_values)
                #
                # # 결과 출력
                # print("Total SMAPE: Mean = {:.2f}%, Min = {:.2f}%, Max = {:.2f}%, Median = {:.2f}%".format(
                #     total_smape_stats["mean"], total_smape_stats["min"], total_smape_stats["max"], total_smape_stats["median"]
                # ))
                # print("SMAPE_M: Mean = {:.2f}%, Min = {:.2f}%, Max = {:.2f}%, Median = {:.2f}%".format(
                #     smape_m_stats["mean"], smape_m_stats["min"], smape_m_stats["max"], smape_m_stats["median"]
                # ))
                # print("SMAPE_A: Mean = {:.2f}%, Min = {:.2f}%, Max = {:.2f}%, Median = {:.2f}%".format(
                #     smape_A_stats["mean"], smape_A_stats["min"], smape_A_stats["max"], smape_A_stats["median"]
                # ))
                # #
                # # # 노드별 MAPE 결과 출력
                # # for result in node_smape_results:
                # #     print("Node {}: Total SMAPE = {:.2f}%, SMAPE_M = {:.2f}%, SMAPE_A = {:.2f}%".format(
                # #         result["node"], result["total_smape"], result["smape_m"], result["smape_A"]
                # #     ))
