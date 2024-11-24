import random
import numpy as np
import torch
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import f1_score, balanced_accuracy_score
import torch.nn.functional as F
from torch_geometric.utils import dense_to_sparse
from torch_sparse import SparseTensor
import math
import copy
import warnings

# 忽略UserWarning
warnings.filterwarnings("ignore", category=UserWarning)
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    np.random.seed(seed)
    random.seed(seed)
def get_split(num_samples, train_ratio=0.2, test_ratio=0.4, num_splits=10):
    # random split
    assert train_ratio + test_ratio < 1
    train_size = int(num_samples * train_ratio)
    test_size = int(num_samples * test_ratio)

    for _ in range(num_splits):
        indices = torch.randperm(num_samples)

        train_mask = torch.zeros(num_samples, dtype=torch.bool)
        train_mask.fill_(False)
        train_mask[indices[:train_size]] = True

        test_mask = torch.zeros(num_samples, dtype=torch.bool)
        test_mask.fill_(False)
        test_mask[indices[train_size: test_size + train_size]] = True

        val_mask = torch.zeros(num_samples, dtype=torch.bool)
        val_mask.fill_(False)
        val_mask[indices[test_size + train_size:]] = True

    train_mask_all = train_mask
    val_mask_all = val_mask
    test_mask_all = test_mask


    return train_mask_all, val_mask_all, test_mask_all

def make_longtailed_data_remove(edge_index, label, n_data, n_cls, ratio, train_mask):
    n_data = torch.tensor(n_data)
    sorted_n_data, indices = torch.sort(n_data, descending=True)
    inv_indices = np.zeros(n_cls, dtype=np.int64)
    for i in range(n_cls):
        inv_indices[indices[i].item()] = i
    assert (torch.arange(len(n_data))[indices][torch.tensor(inv_indices)] - torch.arange(len(n_data))).sum().abs() < 1e-12

    mu = np.power(1/ratio, 1/(n_cls - 1))
    n_round = []
    class_num_list = []
    for i in range(n_cls):

        assert int(sorted_n_data[0].item() * np.power(mu, i)) >= 1
        class_num_list.append(int(min(sorted_n_data[0].item() * np.power(mu, i), sorted_n_data[i])))
        if i < 1:
            n_round.append(1)
        else:
            n_round.append(10)

    class_num_list = np.array(class_num_list)
    class_num_list = class_num_list[inv_indices]
    n_round = np.array(n_round)[inv_indices]

    remove_class_num_list = [n_data[i].item()-class_num_list[i] for i in range(n_cls)]
    remove_idx_list = [[] for _ in range(n_cls)]
    cls_idx_list = []
    index_list = torch.arange(len(train_mask))
    original_mask = train_mask.clone()
    for i in range(n_cls):
        cls_idx_list.append(index_list[(label == i) * original_mask])

    for i in indices.numpy():
        for r in range(1,n_round[i]+1):
            node_mask = label.new_ones(label.size(), dtype=torch.bool)
            node_mask[sum(remove_idx_list,[])] = False
            row, col = edge_index[0], edge_index[1]
            row_mask = node_mask[row]
            col_mask = node_mask[col]
            edge_mask = (row_mask * col_mask).type(torch.bool)
            degree = torch.bincount(row[edge_mask]).to(row.device)
            if len(degree) < len(label):
                degree = torch.cat([degree, degree.new_zeros(len(label)-len(degree))], dim=0)
            degree = degree[cls_idx_list[i]]

            _, remove_idx = torch.topk(degree, (r*remove_class_num_list[i])//n_round[i], largest=False)
            remove_idx = cls_idx_list[i][remove_idx]
            remove_idx_list[i] = list(remove_idx.numpy())

    node_mask = label.new_ones(label.size(), dtype=torch.bool)
    node_mask[sum(remove_idx_list,[])] = False

    row, col = edge_index[0], edge_index[1]
    row_mask = node_mask[row]
    col_mask = node_mask[col]
    edge_mask = (row_mask * col_mask).type(torch.bool)

    train_mask = (node_mask * train_mask).type(torch.bool)
    idx_info = []
    for i in range(n_cls):
        cls_indices = index_list[(label == i) * train_mask]
        idx_info.append(cls_indices)
    # print(train_mask.sum().item())
    # print(edge_mask.sum().item())
    return list(class_num_list), train_mask, idx_info, node_mask, edge_mask


def make_longtailed_data_remove_2(imb_ratio, valid_each, labeling_ratio, all_idx, all_label, nclass,k):
    cls_num_list_0 = np.zeros((nclass)).astype(int)
    for i in range(nclass):
        c_idx = (k == i).nonzero()[:, -1].tolist()
        cls_num_list_0[i] = len(c_idx)

    cls_num_list = cls_num_list_0.copy()
    ShunXu_Pailie=[i for i in range(nclass)]
    for i in range(0,nclass):
        for j in range (0,nclass-i-1):
            if cls_num_list[j]<cls_num_list[j+1]:
                x=cls_num_list[j]
                cls_num_list[j]=cls_num_list[j+1]
                cls_num_list[j+1]=x
                # w=ShunXu_Pailie[j]
                # ShunXu_Pailie[j]=j+1
                # ShunXu_Pailie[j+1]=w
    for i in range(0,nclass):
        for j in range(0,nclass):
            if cls_num_list[i]==cls_num_list_0[j]:
                ShunXu_Pailie[i]=j
    head_list = [ShunXu_Pailie[i] for i in range(nclass//2)]

    all_class_list = [i for i in range(nclass)]
    tail_list = list(set(all_class_list) - set(head_list))

    h_num = len(head_list)
    t_num = len(tail_list)

    base_train_each = int( len(all_idx) * labeling_ratio / (t_num + h_num * imb_ratio) )
    base_valid_each = int(valid_each / (t_num + h_num * imb_ratio) )

    idx2train,idx2valid = {},{}

    total_train_size = 0
    total_valid_size = 0

    for i_h in head_list:
        idx2train[i_h] = int(base_train_each * imb_ratio)
        idx2valid[i_h] = int(base_valid_each * imb_ratio)

        total_train_size += idx2train[i_h]
        total_valid_size += idx2valid[i_h]

    for i_t in tail_list:
        idx2train[i_t] = int(base_train_each * 1)
        idx2valid[i_t] = int(base_valid_each * 1)

        total_train_size += idx2train[i_t]
        total_valid_size += idx2valid[i_t]

    train_list = [0 for _ in range(nclass)]
    train_node = [[] for _ in range(nclass)]
    train_idx  = []

    for iter1 in all_idx:
        iter_label = all_label[iter1]
        if train_list[iter_label] < idx2train[iter_label]:
            train_list[iter_label]+=1
            train_node[iter_label].append(iter1)
            train_idx.append(iter1)

        if sum(train_list)==total_train_size:
            break

    # a=sum(train_list)
    # b=total_train_size
    assert sum(train_list)==total_train_size

    after_train_idx = list(set(all_idx)-set(train_idx))

    valid_list = [0 for _ in range(nclass)]
    valid_idx  = []
    for iter2 in after_train_idx:
        iter_label = all_label[iter2]
        if valid_list[iter_label] < idx2valid[iter_label]:
            valid_list[iter_label]+=1
            valid_idx.append(iter2)
        if sum(valid_list)==total_valid_size:break

    test_idx = list(set(after_train_idx)-set(valid_idx))

    return train_idx, valid_idx, test_idx, train_node


def feature_propagation(adj_, features, k, alpha):
    # device = torch.device(f'cuda:1' if torch.cuda.is_available() else 'cpu')
    # features = features.to(device)
    # adj = adj.to(device)
    features_prop = features.clone()
    adj=adj_.to_dense()
    # print(adj.dtype)
    # adj.is_sparse=True
    for i in range(1, k + 1):
        features_prop = torch.sparse.mm(adj, features_prop)
        features_prop = (1 - alpha) * features_prop + alpha * features
        # (1 − λ)AXˆ(k−1) + λXˆ (0)
    features_p = features_prop
    del features_prop
    del adj
    del features
    torch.cuda.empty_cache()
    return features_p



def get_sim(embeds1, embeds2):
    embeds1 = F.normalize(embeds1)
    embeds2 = F.normalize(embeds2)
    sim = torch.mm(embeds1, embeds2.t())
    return sim

def get_other_node(node, edge_index):
    device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
    # node.to(device)
    edge_index.to(device)
    other_node_idx = []
    # 使用逻辑运算检查节点是否在 edge_index 中出现
    mask = torch.any(edge_index == node, dim=0)
    if torch.sum(mask) > 0:
        if (node in edge_index[0][mask]):
            other_node_idx.append(edge_index[1][mask].to('cpu'))
        if (node in edge_index[1][mask]):
            other_node_idx.append(edge_index[0][mask].to('cpu'))

    # # 返回另一个节点的索引
    return other_node_idx

def combine_nodes(feature1, feature2, similarity):
    # 计算加权平均特征
    combined_feature = (feature1  + feature2* similarity) / (similarity + 1)
    return combined_feature
def remove_edge(edge_index,node_1,node_2):
    mask_1 = (edge_index[0] != node_1) | (edge_index[1] != node_2)
    mask_2 = (edge_index[0] != node_2) | (edge_index[1] != node_1)
    mask=mask_1 & mask_2
    edge_index = edge_index[:, mask]
    return edge_index
def train_generated_data(train_edge_index_,labels,
                      train_num_classes_,
                      train_list_,train_label_list_,
                      x,train_num_classes_number_,train_num_classes_list_,minority=0.4,min_ratio=0.8):

    device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
    data_new_x=x.clone()
    data_new_y=labels.clone()
    train_num_classes = train_num_classes_
    train_num_classes_number=list(train_num_classes_number_)
    train_num_classes_list=list(train_num_classes_list_)
    train_num_classes_number_max=0
    train_num_classes_number_max_idx=0
    train_num_classes_minority=[]
    train_num_classes_minority_list=[]

    train_num_classes_majority=[]
    train_num_classes_majority_list=[]
    train_label_list=list(train_label_list_)
    train_list=list(train_list_)
    train_edge_index=train_edge_index_.clone()

    for i in range(train_num_classes):
        if train_num_classes_number[i]>train_num_classes_number_max:
            train_num_classes_number_max=train_num_classes_number[i]
            train_num_classes_number_max_idx=i
    for i in range(train_num_classes):
        if train_num_classes_number_max*minority<float(train_num_classes_number[i]) <= train_num_classes_number_max:
            train_num_classes_majority.append(i)
            train_num_classes_majority_list.append(train_num_classes_list[i])
        #if float(train_num_classes_number[i])<train_num_classes_number_max*minority:
        else:
            train_num_classes_minority.append(i)
            a=copy.deepcopy(train_num_classes_list[i])
            train_num_classes_minority_list.append(a)

    Majority_con_labels=[]
    Majority_labels=[]
    '''
    多数类加工
    '''
    for i in range(len(train_num_classes_majority)):
        for node in train_num_classes_majority_list[i]:
            list_score=[]
            node_y=labels[node].item()
            con_node_list_label=[]
            con_node_list_index=get_other_node(node,train_edge_index)
            if len(con_node_list_index) > 0:
                con_node_list_index = torch.cat(con_node_list_index).tolist()
                for con_node_index in con_node_list_index:
                    node_x = x[node]
                    con_node_x = x[con_node_index]
                    # sim=get_sim(node_x,con_node_x)
                    sim = F.cosine_similarity(node_x, con_node_x, dim=0)
                    score = (sim.mean().item() + 1) / 2 * 100
                    list_score.append(score)
                    con_node_label = labels[con_node_index].item()
                    con_node_list_label.append(con_node_label)
                    if node_y != con_node_label:
                        train_edge_index = remove_edge(train_edge_index, node, con_node_index)
            Majority_con_labels.append(con_node_list_label)
            Majority_labels.append(node_y)

    train_num_classes_minority_list_1 = copy.deepcopy(train_num_classes_minority_list)

    for i in range(len(train_num_classes_minority)):
        for node in train_num_classes_minority_list[i]:
            node_y=data_new_y[node].item()
            con_node_list_label=[]
            con_node_list_index=get_other_node(node,train_edge_index)
            if len(con_node_list_index) >0:
                con_node_list_index=torch.cat(con_node_list_index).tolist()
                # print(len(con_node_list_index))
                list_score=[]
                for con_node_index in con_node_list_index:

                    now_minority_class_number = train_num_classes_number[train_num_classes_minority[i]]
                    node_x=data_new_x[node]
                    con_node_x=data_new_x[con_node_index]
                    # sim=get_sim(node_x,con_node_x)
                    sim=F.cosine_similarity(node_x, con_node_x, dim=0)
                    score = (sim.mean().item() + 1) / 2 * 100
                    list_score.append(round(score,2))
                    con_node_label=data_new_y[con_node_index].item()
                    con_node_list_label.append(con_node_label)
                    if node_y!=con_node_label:
                        train_edge_index=remove_edge(train_edge_index,node,con_node_index)
                    else:
                        if now_minority_class_number<=train_num_classes_number_max*minority:
                            new_node_x=combine_nodes(node_x,con_node_x,score/100)
                            new_sim = F.cosine_similarity(node_x, new_node_x, dim=0)
                            new_sim_score = (new_sim.mean().item() + 1) / 2 * 100
                            if new_sim_score>80:
                                new_node_x=(torch.tensor(new_node_x)).clone().detach()
                                data_new_x=torch.cat((data_new_x,new_node_x.unsqueeze(0)),dim=0)
                                new_edge=torch.tensor([[node],[data_new_x.shape[0]-1]])
                                train_edge_index=torch.cat((train_edge_index,new_edge),dim=1)
                                data_new_y=torch.cat((data_new_y,torch.tensor(node_y).unsqueeze(0)),dim=0)
                                train_list.append(data_new_x.shape[0]-1)
                                train_label_list.append(node_y)

                                train_num_classes_minority_list_1[i].append(data_new_x.shape[0]-1)

                                train_num_classes_list[train_num_classes_minority[i]].append(data_new_x.shape[0]-1)
                                train_num_classes_number[train_num_classes_minority[i]]=(
                                        train_num_classes_number[train_num_classes_minority[i]]+1)

    Second_minority=[]
    Second_minority_list=[]
    for i in range(len(train_num_classes_minority)):
        if train_num_classes_number[train_num_classes_minority[i]]<minority*train_num_classes_number_max:
            Second_minority.append(train_num_classes_minority[i])
            a=copy.deepcopy(train_num_classes_minority_list_1[i])
            Second_minority_list.append(a)


    for i in range(len(Second_minority)):
        for node in Second_minority_list[i]:
            for j in range(len(train_num_classes_minority)):
                if(train_num_classes_minority[j]==Second_minority[i]):
                    cidx=j
            node_y = data_new_y[node].item()
            con_node_list_label = []
            con_node_list_index = get_other_node(node, train_edge_index)
            if len(con_node_list_index) > 0:
                con_node_list_index = torch.cat(con_node_list_index).tolist()
                # print(len(con_node_list_index))
                list_score = []
                for con_node_index in con_node_list_index:

                    now_minority_class_number = train_num_classes_number[Second_minority[i]]
                    node_x = data_new_x[node]
                    con_node_x = data_new_x[con_node_index]
                    # sim=get_sim(node_x,con_node_x)
                    sim = F.cosine_similarity(node_x, con_node_x, dim=0)
                    score = (sim.mean().item() + 1) / 2 * 100
                    list_score.append(round(score, 2))
                    con_node_label = data_new_y[con_node_index].item()
                    con_node_list_label.append(con_node_label)
                    if node_y != con_node_label:
                        train_edge_index = remove_edge(train_edge_index, node, con_node_index)
                    else:
                        if now_minority_class_number <= train_num_classes_number_max * minority:
                            new_node_x = combine_nodes(node_x, con_node_x, score / 100)
                            new_sim = F.cosine_similarity(node_x, new_node_x, dim=0)
                            new_sim_score = (new_sim.mean().item() + 1) / 2 * 100
                            if new_sim_score > 80:
                                new_node_x = (torch.tensor(new_node_x)).clone().detach()
                                data_new_x = torch.cat((data_new_x, new_node_x.unsqueeze(0)), dim=0)
                                new_edge = torch.tensor([[node], [data_new_x.shape[0]-1]])
                                train_edge_index = torch.cat((train_edge_index, new_edge), dim=1)
                                data_new_y = torch.cat((data_new_y, torch.tensor(node_y).unsqueeze(0)), dim=0)
                                train_list.append(data_new_x.shape[0]-1)
                                train_label_list.append(node_y)
                                train_num_classes_minority_list_1[cidx].append(data_new_x.shape[0] - 1)
                                # train_num_classes_minority_list_1[Second_minority[i]].append(data_new_x.shape[0]-1)
                                train_num_classes_list[Second_minority[i]].append(data_new_x.shape[0]-1)
                                train_num_classes_number[Second_minority[i]] = (
                                        train_num_classes_number[Second_minority[i]] + 1)

    return train_edge_index,data_new_x,data_new_y,train_list,train_label_list

def toplogy_enhance(edge_index,x,num):
    A=SparseTensor(row=edge_index[0],col=edge_index[1],
                   value=torch.ones(edge_index.size(1)),
                   sparse_sizes=(x.shape[0],x.shape[0]))
    D_inv=(A.sum(1).squeeze()+1e-10)**-1.0
    I=torch.eye(x.shape[0],device=x.device)
    row,col=dense_to_sparse(I)[0]
    D_inv=SparseTensor(row=row,col=col,value=D_inv,sparse_sizes=(x.shape[0],x.shape[0]))

    P=A @ D_inv
    M=P
    TE=[M.get_diag().float()]
    M_power=M
    for _ in range(num-1):
        M_power=M_power@M
        TE.append(M_power.get_diag().float())
    TE=torch.stack(TE,dim=-1)
    return TE

def accuracy(output, labels, sep_point=None, sep=None, pre=None):
    if sep in ['T', 'TH', 'TT']:
        labels = labels - sep_point

    if output.shape != labels.shape:
        if len(labels) == 0:
            return np.nan
        preds = output.max(1)[1].type_as(labels)
    else:
        preds= output

    correct = preds.eq(labels).double()
    correct = correct.sum()

    return correct / len(labels)

def performance_measure(output, labels):
    acc= accuracy(output, labels)*100

    if len(labels) == 0:
        return np.nan

    if output.shape != labels.shape:
        output = torch.argmax(output, dim=-1)

    macro_F = f1_score(labels.cpu().detach(), output.cpu().detach(), average='macro') * 100
    gmean = geometric_mean_score(labels.cpu().detach(), output.cpu().detach(), average='macro') * 100
    bacc = balanced_accuracy_score(labels.cpu().detach(), output.cpu().detach()) * 100

    return acc, macro_F, gmean, bacc