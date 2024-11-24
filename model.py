import torch
from torch.nn import Linear
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import Linear, GCNConv, SAGEConv, GATConv, GINConv, GATv2Conv
from torch_sparse import SparseTensor
import math

def get_activation_layer(activation):
    if activation is None:
        return nn.Identity()
    elif activation == 'relu':
        return nn.ReLU()
    elif activation == 'elu':
        return nn.ELU()
    elif activation == 'tanh':
        return nn.Tanh()
    else:
        raise ValueError('Unknown activation')

def get_gnn_layer(name,in_channels,out_channels,heads):
    if name=='gcn':
        layer=GCNConv(in_channels,out_channels)
    elif name=='gat':
        layer=GATConv(-1,out_channels,heads)
    elif name=='sage':
        layer=SAGEConv(in_channels,out_channels)
    elif name=='gin':
        layer=GINConv(Linear(in_channels,out_channels),train_eps=True)
    elif name=='gat2':
        layer=GATv2Conv(-1,out_channels,heads)
    else:
        raise  ValueError(name)
    return layer

def to_sparse_tensor(edge_index,num_nodes):
    return SparseTensor.from_edge_index(
        edge_index,sparse_sizes=(num_nodes,num_nodes)
    ).to(edge_index.device)

def get_sim(embed_1,embed_2):
    embed_1s=F.normalize(embed_1)
    embed_2s=F.normalize(embed_2)
    sim=torch.mm(embed_1s,embed_2s.t())
    return sim

def get_mutual(embed_1,embed_2):
    # 计算联合分布
    joint_distribution = torch.mm(embed_1, embed_2.t())
    # 计算边缘分布
    marginal_distribution_1 = torch.mm(embed_1, embed_1.t())
    marginal_distribution_2 = torch.mm(embed_2, embed_2.t())

    joint_entropy = -torch.sum(torch.log(joint_distribution),dim=0,keepdim=True)
    marginal_entropy_1 = -torch.sum(torch.log(marginal_distribution_1),dim=0,keepdim=True)
    marginal_entropy_2 = -torch.sum(torch.log(marginal_distribution_2),dim=0,keepdim=True)
    # 计算互信息
    mutual_information = marginal_entropy_1+marginal_entropy_2-joint_entropy
    return mutual_information
def compute_contra_loss(embed_1,embed_3,str):
    if str == 'sim':
        f = lambda x: torch.exp(x / 1.0)
        refl_sim=f(get_sim(embed_1,embed_1))
        between_sim=f(get_sim(embed_1,embed_3))
        x1=refl_sim.sum(1)+between_sim.sum(1)-refl_sim.diag()
        sim_loss=-torch.log(between_sim.diag()/x1)
        sim_loss=sim_loss.mean()
        return sim_loss
    elif str == 'mutual':
        f = lambda x: torch.exp(x / 1.0)
        refl_sim=get_mutual(embed_1,embed_1).squeeze()
        between_sim=get_mutual(embed_1,embed_3).squeeze()
        x1=refl_sim+between_sim
        mutual_loss=-torch.log(between_sim/x1)
        mutual_loss=mutual_loss.mean()
        return mutual_loss
class MLP_encoder(nn.Module):
    def __init__(self,in_channels,pe_dim,hidden_channels,out_channels,
                 dropout=0.5,norm=False,activation='tanh'):
        super().__init__()
        self.dropout=nn.Dropout(dropout)
        self.activation=get_activation_layer(activation)

        bn = nn.BatchNorm1d if norm else nn.Identity
        self.mlpX= nn.ModuleList()
        self.mlpP= nn.ModuleList()

        self.mlpX.append(nn.Linear(in_channels,hidden_channels))
        self.mlpX.append(nn.Linear(hidden_channels*2,hidden_channels))
        self.bnX1=bn(hidden_channels)
        self.mlpX.append(nn.Linear(hidden_channels,out_channels))
        self.bnX2=bn(out_channels)

        self.mlpP.append(nn.Linear(pe_dim,hidden_channels))

    def forward(self,x,p):
        x=self.activation(self.mlpX[0](x))
        p=self.activation(self.mlpP[0](p))

        x=torch.cat([x,p],dim=-1)
        x=self.mlpX[1](x)
        x=self.activation(self.bnX1(x))

        x=self.dropout(x)
        x=self.mlpX[2](x)
        x=self.activation(self.bnX2(x))

        return x

class GNN_encoder_1(nn.Module):
    def __init__(self,in_channels,hidden_channels,out_channels,
                num_layers=2,dropout=0.5,norm=False,
                lay='gcn',activation='elu'):
        super().__init__()

        self.convs=nn.ModuleList()
        self.bns=nn.ModuleList()
        bn=nn.BatchNorm1d if norm else nn.Identity

        for i in range(num_layers):
            first_channels=in_channels if i==0 else hidden_channels
            second_channels=out_channels if i==num_layers-1 else hidden_channels
            heads=1 if i==num_layers-1 or 'gat' not in lay else 8
            self.convs.append(get_gnn_layer(lay,first_channels,second_channels,heads))
            self.bns.append(bn(second_channels*heads))
        self.dropout=nn.Dropout(dropout)
        self.activation=get_activation_layer(activation)

    def forward(self,x,edge_index):
        edge_sparse= to_sparse_tensor(edge_index,x.size(0))

        for i,conv in enumerate(self.convs[:-1]):
            x=self.dropout(x)
            x=conv(x,edge_sparse)
            x=self.bns[i](x)
            x=self.activation(x)
        x=self.dropout(x)
        x=self.convs[-1](x,edge_sparse)
        x=self.bns[-1](x)
        x=self.activation(x)
        return x

class MLP_classifier(nn.Module):
    def __init__(self,in_channels,hidden_channels,out_channels=1,
                 num_layrs=2,dropout=0.5,activation='elu'):
        super().__init__()
        self.mlpc=nn.ModuleList()
        for i in range(num_layrs):
            first_channels=in_channels if i==0 else hidden_channels
            second_channels=out_channels if i ==num_layrs-1 else hidden_channels

            self.mlpc.append(nn.Linear(first_channels,second_channels))
        self.dropout=nn.Dropout(dropout)
        self.activation=get_activation_layer(activation)

    def forward(self,x,sigmoid=False):
        for i,mlp in enumerate(self.mlpc[:-1]):
            x=self.dropout(x)
            x=mlp(x)
            x=self.activation(x)
        x=self.mlpc[-1](x)

        if sigmoid:
            return x.sigmoid()
        else:
            return x

class DAV(nn.Module):
    def __init__(self,args,data,tp_dim):
        device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
        super(DAV, self).__init__()
        self.data=data.clone()
        self.args=args
        self.MLP_encoder=MLP_encoder(data.num_features,tp_dim,
                                    args.encoder_channels,
                                    args.hidden_channels,
                                    dropout=args.encoder_dropout,
                                    norm=args.norm,
                                    activation=args.encoder_activation)

        self.GNN_encoder = GNN_encoder_1(self.data.x.shape[1],
                                         args.encoder_channels,
                                         args.hidden_channels,
                                        args.encoder_layers,
                                        args.encoder_dropout,
                                        args.norm,
                                        args.layer,
                                        args.encoder_activation)

        self.MLP_classifier = MLP_classifier(in_channels=args.hidden_channels,
                                             hidden_channels=args.Classifier_hidden_channels,
                                             out_channels=data.y.max().item() + 1,
                                             dropout=args.Classifier_dropout)

        self.cl_1=nn.Parameter(torch.tensor(self.args.cl_1,device=device),requires_grad=True)
        self.cl_2=nn.Parameter(torch.tensor(self.args.cl_2,device=device),requires_grad=True)
        self.theta=nn.Parameter(torch.tensor(self.args.theta,device=device),requires_grad=True)

    def forward(self,features_1,features_2,features_3,labels_1,labels_2,
                idx_train_1,idx_train_2,train_edge_index_1,train_edge_index_2,tp,finetune=False):

        device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')

        weight_1=features_1.new((labels_1.max().item()+1)).fill_(1)
        weight_2=features_2.new((labels_2.max().item()+1)).fill_(1)

        num_class_1=len(set(labels_1.tolist()))

        a_list=[]
        b_list=[]
        for i in range(num_class_1):
            cidx=(labels_1[idx_train_1] == i).nonzero()[:,-1].tolist()
            a_list.append(cidx)
            weight_1[i]=1/(len(cidx)+1)

        num_class_2=len(set(labels_2.tolist()))
        for i in range(num_class_2):
            cidx=(labels_2[idx_train_2] == i).nonzero()[:,-1].tolist()
            b_list.append(cidx)
            weight_2[i]=1/(len(cidx)+1)

        # for i in range(len(a_list)):
        #     print("Label: {:d}, number: {:d}".format(i,len(a_list[i])))
        # print('\n')
        # for i in range(len(b_list)):
        #     print("Label: {:d}, number: {:d}".format(i,len(b_list[i])))
        embed_1=self.GNN_encoder(features_1,train_edge_index_1)
        embed_1=F.normalize(embed_1,dim=1)
        embed_2=self.GNN_encoder(features_2,train_edge_index_2)
        embed_2=F.normalize(embed_2,dim=1)

        # embed_1=self.GNN_encoder_2(features_1,train_edge_index_1)
        # embed_1=F.normalize(embed_1,dim=1)
        # embed_2=self.GNN_encoder_2(features_2,train_edge_index_2)
        # embed_2=F.normalize(embed_2,dim=1)

        output_1=self.MLP_classifier(embed_1)
        output_1=F.normalize(output_1,dim=1)
        # print(output_1.device)

        output_2=self.MLP_classifier(embed_2)
        output_2=F.normalize(output_2,dim=1)

        embed_3=self.MLP_encoder(features_3,tp)

        output_3=self.MLP_classifier(embed_3)
        output_3=F.normalize(output_3,dim=1)

        loss_node_cls_1=F.cross_entropy(output_1[idx_train_1],labels_1[idx_train_1],
                                        weight=(weight_1.clone().detach()).to(device))
        loss_node_cls_2=F.cross_entropy(output_2[idx_train_2],labels_2[idx_train_2],
                                        weight=(weight_2.clone().detach()).to(device))
        loss_node_cls_3=F.cross_entropy(output_3[idx_train_1],labels_1[idx_train_1],
                                        weight=(weight_1.clone().detach()).to(device))
        a=loss_node_cls_1+loss_node_cls_2+loss_node_cls_3

        loss_node_cls = (self.cl_1*loss_node_cls_1 + self.cl_2*loss_node_cls_2
                         + (1-self.cl_1-self.cl_2)*loss_node_cls_3)
        loss_contra = compute_contra_loss(embed_1, embed_3, 'sim')
        loss= (1-self.theta)*loss_contra + self.theta*loss_node_cls
        return loss
