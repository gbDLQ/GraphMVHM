import torch
from torch_geometric.datasets import Planetoid, WebKB, Actor, WikipediaNetwork, Amazon, WikiCS, Coauthor
import torch_geometric.transforms as T
from torch_geometric.utils import remove_self_loops,to_undirected

from utils import get_split


def load_data(args,root='./data', name='cora',is_random_split=False):

    print("dataloading..."+name)
    if name in ['cora', 'CiteSeer','PubMed']:
        dataset = Planetoid(root=root + '/Planetoid', name=name,split='full')
    elif name in ['cornell', 'texas', 'wisconsin']:
        dataset = WebKB(root=root + '/WebKB', name = name)
    elif name in ['actor']:
        dataset = Actor(root=root + '/Actor')
    elif name in ['chameleon', 'squirrel']:
        dataset = WikipediaNetwork(root=root + '/WikipediaNetwork', name=name, transform=T.NormalizeFeatures())
    elif name in ['Computers', 'Photo']:
        dataset = Amazon(root=root + '/Amazon', name=name, transform=T.NormalizeFeatures())
    elif name in ['CS', 'physics']:
        dataset = Coauthor(root=root + '/Coauthor', name=name, transform=T.NormalizeFeatures())
    elif name in ['wiki-cs']:
        dataset = WikiCS(root=root + '/Wiki-CS')
    data = dataset[0]

    data.edge_index = remove_self_loops(data.edge_index)[0]
    data.edge_index = to_undirected(data.edge_index)
    edge_index = torch.sort(data.edge_index, dim=0).values
    # 使用 torch.unique() 函数对排序后的边进行去重
    edge_index, _ = torch.unique(edge_index, dim=1, return_inverse=True)
    data.edge_index=edge_index
    # 数据集划分
    # if name in ['Computers', 'Photo', 'CS'] or is_random_split == True:
    # , 'Computers', 'Photo', 'CS', 'physics', 'wiki-cs'
    if name in ['cora', 'CiteSeer','PubMed'] and is_random_split == True:
    # if is_random_split == True:
        train_mask, val_mask, test_mask = get_split(data.x.shape[0],args.train_ratio,args.test_ratio)
        data.train_mask, data.val_mask, data.test_mask = train_mask, val_mask, test_mask
    # else:
    #     train_mask, val_mask, test_mask = data.train_mask, data.val_mask, data.test_mask

    # data.train_mask, data.val_mask, data.test_mask = train_mask, val_mask, test_mask

    return data