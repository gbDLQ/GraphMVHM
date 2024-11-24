import argparse
import os
import random
from copy import deepcopy

import networkx as nx
import pandas as pd
import torch.multiprocessing as mp
import numpy as np
import scipy as sp
import torch
import torch.optim as optim
from torch_sparse import SparseTensor
import utils
from model import DAV
from dataloader import load_data
from utils import setup_seed
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch.nn.functional as F

def main(args):
    acc=[]
    bacc=[]
    f1=[]
    gm=[]
    epoch_list=[]
    cl_1_list=[]
    cl_2_list=[]
    theta_list=[]
    Loss_all=[]
    epoch_all=[]
    for seeds in range(10):
        seed = 100 + seeds
        setup_seed(seed)
        device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')

        # load dataset
        data = load_data(args=args,name=args.dataset, is_random_split=True)


        num_classes = len(set(data.y.tolist()))


        if args.dataset in ['cora', 'CiteSeer', 'PubMed']:
            cls_num_list = np.zeros((num_classes)).astype(int)
            for i in range(num_classes):
                c_idx = (data.y[data.train_mask] == i).nonzero()[:, -1].tolist()
                cls_num_list[i] = len(c_idx)
            (class_num_list, data_train_mask, idx_info,train_node_mask,
                     train_edge_mask) = utils.make_longtailed_data_remove(
                data.edge_index, data.y, cls_num_list, num_classes,
                args.imbalance_ratio, data.train_mask.clone())

            data.train_edge_index = data.edge_index[:, train_edge_mask]
            data.train_data_index=data_train_mask
            data_nodes=list(range(data.x.shape[0]))
            train_list=[data_nodes[i] for i in range(len(data_nodes)) if data_train_mask[i]]
            train_num_classes_list=[]
            train_num_classes_number = []
            train_data_labels = data.y[data.train_data_index]
            train_num_classes = len(set(train_data_labels.tolist()))

            for i in range(train_num_classes):
                c_idx = (train_data_labels == i).nonzero()[:, -1].tolist()
                train_num_classes_list.append([train_list[j] for j in c_idx])
                train_num_classes_number.append(len(c_idx))
            train_label_list = train_data_labels.flatten().tolist()

        elif args.dataset in ['CS', 'Computers', 'Photo']:
            ##node传回不同labels的节点,train_idx是全部的训练节点
            (train_idx, val_idx,test_idx,train_node) = utils.make_longtailed_data_remove_2(
                args.imbalance_ratio,
                valid_each=int(data.x.shape[0] * (1-args.train_ratio-args.test_ratio)),
                labeling_ratio=args.train_ratio,
                all_idx=[i for i in range(data.x.shape[0])],
                all_label=data.y.cpu().detach().numpy(),
                nclass=num_classes,
                k=data.y)
            data.train_edge_index = data.edge_index
            train_data_labels = data.y[train_idx]
            train_num_classes= len(set(train_data_labels.tolist()))
            train_list=train_idx
            train_label_list=train_data_labels.tolist()
            train_num_classes_list=train_node
            train_num_classes_number = []
            for i in range(train_num_classes):
                train_num_classes_number.append(len(train_node[i]))
            data.val_mask=val_idx
            data.test_mask=test_idx


        data.edge_sparse = SparseTensor(row=data.edge_index[0], col=data.edge_index[1],
                                        value=torch.ones(data.edge_index.size(1)),
                                        sparse_sizes=(data.x.shape[0], data.x.shape[0]))
        features_3=utils.feature_propagation(data.edge_sparse,data.x,args.k, args.alpha)
        tp=utils.toplogy_enhance(data.edge_index,data.x,args.num)


        (train_new_edge_index,data_new_x,
         data_new_y,train_new_list,train_new_label_list)=utils.train_generated_data(data.train_edge_index,data.y,
                                                                train_num_classes,train_list,
                                                        train_label_list,data.x,train_num_classes_number,train_num_classes_list)

        model = DAV(args, data,tp.shape[1])
        model.to(device)

        optimizer_model_cl1= optim.Adam([model.cl_1], lr=args.lr/args.small, weight_decay=args.wd/args.small)
        optimizer_model_cl2 = optim.Adam([model.cl_2], lr=args.lr / args.small,
                                         weight_decay=args.wd/args.small)
        optimizer_model_theta = optim.Adam([model.theta], lr=args.lr/args.small ,
                                         weight_decay=args.wd/args.small)
        optimizer_mlp = optim.Adam(model.MLP_encoder.parameters(), lr=args.lr, weight_decay=args.wd)
        optimizer_cls = optim.Adam(model.MLP_classifier.parameters(), lr=args.lr, weight_decay=args.wd)
        optimizer_gnn = optim.Adam(model.GNN_encoder.parameters(),lr=args.lr,weight_decay=args.wd)

        val_bacc=[]
        val_acc = []
        test_results = []
        best_metric = 0
        tsne_embed=0
        tsne_embed_2=0

        Now_Seed_result={}
        Now_Seed_result['acc'] = []
        Now_Seed_result['macro_F'] = []
        Now_Seed_result['gmeans'] = []
        Now_Seed_result['bacc'] = []

        data.data_new_x=data_new_x
        data.features_3=features_3
        data.data_new_y=data_new_y
        data.train_list=train_list
        data.train_new_list=train_new_list
        data.train_new_label_list=train_new_label_list
        data.train_new_edge_index=train_new_edge_index
        data.tp=tp

        data.to(device)

        with open('Finally.txt', 'a') as f:
            # 向文件中写入内容
            f.write("{:d} {:s} {:s} ".format(seed,args.dataset, args.layer))
        best_epoch = 0
        best_loss=0
        best_cl1=0
        best_cl2=0
        best_theta=0

        for epoch in range(args.epochs):
            model.train()
            # optimizer_model.zero_grad()
            optimizer_model_cl1.zero_grad()
            optimizer_model_cl2.zero_grad()
            # optimizer_model_theta.zero_grad()
            optimizer_mlp.zero_grad()
            optimizer_cls.zero_grad()
            optimizer_gnn.zero_grad()
            loss = model(data.x, data.data_new_x,data.features_3,
                         data.y, data.data_new_y,
                         data.train_list, data.train_new_list,
                         data.edge_index, data.train_new_edge_index,
                         data.tp,
                         finetune=False)

            Loss_all.append(loss.item())
            epoch_all.append(epoch)

            loss.backward()

            # optimizer_model.step()
            optimizer_model_cl1.step()
            optimizer_model_cl2.step()
            optimizer_model_theta.step()
            optimizer_mlp.step()
            optimizer_cls.step()
            optimizer_gnn.step()
            model.eval()

            embed_1=model.GNN_encoder(data.x,data.edge_index)
            embed_2=model.GNN_encoder(data.data_new_x,data.train_new_edge_index)
            output=model.MLP_classifier(embed_1)

            acc_val, macro_F_val, gmeans_val, bacc_val= utils.performance_measure(output[data.val_mask],
                                                                                   data.y[data.val_mask]
                                                                                       )

            val_acc.append((acc_val+macro_F_val+bacc_val)/3)
            max_idx = val_acc.index(max(val_acc))

            if best_metric < ((acc_val+macro_F_val+bacc_val)/3):
                best_metric = ((acc_val+macro_F_val+bacc_val)/3)
                best_epoch=epoch
                best_model = deepcopy(model)
                best_cl1=model.cl_1.item()
                best_cl2 = model.cl_2.item()
                best_theta = model.theta.item()
                # best_theta = model.theta

            acc_test, macro_F_test, gmeans_test, bacc_test = utils.performance_measure(output[data.test_mask],
                                                                                       data.y[data.test_mask]
                                                                                       )

            test_results.append([acc_test, macro_F_test, gmeans_test, bacc_test])
            best_test_result = test_results[max_idx]

            Now_Seed_result['acc'] = float(best_test_result[0])
            Now_Seed_result['macro_F'] = float(float(best_test_result[1]))
            Now_Seed_result['gmeans'] = float(float(best_test_result[2]))
            Now_Seed_result['bacc'] = float(float(best_test_result[3]))

        acc.append(Now_Seed_result['acc'])
        f1.append(Now_Seed_result['macro_F'])
        gm.append(Now_Seed_result['gmeans'])
        bacc.append(Now_Seed_result['bacc'])
        epoch_list.append(best_epoch)
        cl_1_list.append(best_cl1)
        cl_2_list.append(best_cl2)
        theta_list.append(best_theta)

    print('ACC: {:.2f}, Macro-F: {:.2f}, bACC: {:.2f}'.format(np.mean(acc), np.mean(f1),np.mean(bacc)))

if __name__ == '__main__':


    parser = argparse.ArgumentParser()

    parser.add_argument('-dataset', type=str, default='Computers')
    parser.add_argument('-gpu_id', type=int, default=0)

    # Model Hyper-param
    parser.add_argument("-layer", nargs="?", default="gcn", help="(default: gcn)")

    parser.add_argument("-encoder_activation", nargs="?", default="relu",
                        help="Activation function for Dual-view encoder, (default: relu)")

    parser.add_argument('-encoder_channels', type=int, default=128,
                        help='Channels of Dual-view encoder layers. (default: 128)')

    parser.add_argument('-hidden_channels', type=int, default=256, help='Channels of embedding size. (default: 256)')

    parser.add_argument('-encoder_layers', type=int, default=2,
                        help='Number of layers for Dual-view encoder. (default: 2)')

    parser.add_argument('-encoder_dropout', type=float, default=0.8,
                        help='Dropout probability of encoder. (default: 0.8)')

    parser.add_argument('-Classifier_dropout', type=float, default=0.8,
                        help='Dropout probability of encoder. (default: 0.8)')

    parser.add_argument('-Classifier_hidden_channels', type=int, default=128, help='Channels of classifier_hidden_size. (default: 128)')

    parser.add_argument('-lr', type=float, default=0.01, help='Learning rate for autoencoding. (default: 0.01)')

    parser.add_argument('-wd', type=float, default=5e-5,help='weight_decay for autoencoding. (default: 5e-5)')

    parser.add_argument('-k', type=int, default=2)
    parser.add_argument('-alpha', type=float, default=0.8)


    parser.add_argument('-epochs', type=int, default=501, help='Number of training epochs. (default: 500)')
    parser.add_argument('-runs', type=int, default=10, help='Number of runs. (default: 10)')

    parser.add_argument('-norm',default=True, action='store_true', help='(default: False)')
    parser.add_argument('-theta', type=float, default=0.4)

    parser.add_argument('-cl_1', type=float, default=0.7)
    parser.add_argument('-cl_2', type=float, default=0.25)

    parser.add_argument('-num', type=int, default=5)
    parser.add_argument('-train_ratio', type=float,default=0.1)
    parser.add_argument('-test_ratio', type=float,default=0.8)
    parser.add_argument('-imbalance_ratio', type=int,default=20)
    parser.add_argument('-small', type=float,default=50)
    args = parser.parse_args()
    print(args)
    main(args)