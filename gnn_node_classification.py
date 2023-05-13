from constants_and_utils import *
from graph_methods import *
from vaccine_intent import *

import argparse
from collections import Counter
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import CountVectorizer
import string
from torch_geometric.data import Data, HeteroData
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GraphConv, HeteroConv
import torch as t
import torch.nn.functional as F
import time
from tqdm import tqdm


class CharCNNLayers(t.nn.Module):
    """
    Wrapper for character-level convolutional layers. Meant to be used in other models; not as stand-alone model.
    """
    def __init__(self, num_chars, max_length, num_layers=2, num_channels=256):
        super().__init__()
        self.num_chars = num_chars
        self.max_length = max_length
        self.num_layers = num_layers
        self.convs = t.nn.ModuleList()
        for l in range(self.num_layers):
            if l == 0:
                conv = t.nn.Sequential(
                    t.nn.Conv1d(num_chars, num_channels, kernel_size=7, padding=0),
                    t.nn.ReLU(),
                    t.nn.MaxPool1d(3))
            else:
                conv = t.nn.Sequential(
                    t.nn.Conv1d(num_channels, num_channels, kernel_size=3, padding=0),
                    t.nn.ReLU(),
                    t.nn.MaxPool1d(3))
            self.convs.append(conv)        
    
    def forward(self, x):
        """
        x: tensor, initial representation of data
        """
        x = x.transpose(1, 2)
        for l in range(self.num_layers):
            x = self.convs[l](x)
        x = x.view(x.shape[0], -1)  # flatten last dimension
        return x
    
    def get_output_dim(self):
        """
        Uses dummy tensor to return output dimension.
        """
        input_shape = (1, self.max_length, self.num_chars)
        x = t.rand(input_shape)
        x = self.forward(x)
        output_dim = x.shape[1]
        return output_dim
    
    
class CharCNN(t.nn.Module):
    """
    Character-level CNN for query-url vaccine intent classification. 
    Ignores graph structure.
    Based on https://github.com/ahmedbesbes/character-based-cnn/blob/593197610498bf0b4898b3bdf2e1f6730f954613/src/model.py
    """
    def __init__(self, num_chars, max_length, num_layers=2, num_channels=256):
        super().__init__()
        self.num_chars = num_chars
        self.max_length = max_length
        self.num_layers = num_layers
        self.convs = CharCNNLayers(num_chars, max_length, num_layers)
        output_dim = self.convs.get_output_dim()
        self.fc = t.nn.Sequential(
                    t.nn.Linear(output_dim, 1024), 
                    t.nn.ReLU(),
                    t.nn.Dropout(0.5))
        self.out = t.nn.Linear(1024, 1)
    
    def forward(self, batch):
        """
        batch: output from graph sampler. For CNN, we only use .x attribute and ignore graph
        attributes like .edge_index.
        """
        x = F.one_hot(batch.x.long(), num_classes=self.num_chars).float()
        x = self.convs(batch.x.float())
        x = self.fc(x)
        return self.out(x).reshape(-1)
    

class HeteroCharCNN(t.nn.Module):
    """
    Heterogeneous character-level CNN for query-url vaccine intent classification. 
    Ignores graph structure.
    """
    def __init__(self, num_chars, max_length, metadata, num_layers=2, num_channels=256):
        super().__init__()
        self.num_chars = num_chars
        self.max_length = max_length
        self.num_layers = num_layers
        self.conv_dict = t.nn.ModuleDict()
        for node_type in metadata[0]:
            self.conv_dict[node_type] = CharCNNLayers(num_chars, max_length, num_layers)
        output_dim = next(iter(self.conv_dict.values())).get_output_dim()
        self.fc = t.nn.Sequential(
                    t.nn.Linear(output_dim, 1024), 
                    t.nn.ReLU(),
                    t.nn.Dropout(0.5))
        self.out = t.nn.Linear(1024, 1)
    
    def forward(self, batch):
        """
        batch: output from heterogeneous graph sampler. For hetero CNN, we only use .x_dict attribute and
        ignore graph attributes like .edge_index.
        """
        x_dict = batch.x_dict
        for node_type, x in x_dict.items():
            x = F.one_hot(x.long(), num_classes=self.num_chars).float()
            x = self.conv_dict[node_type](x)
            x = self.fc(x)
            x_dict[node_type] = self.out(x).reshape(-1)
        return x_dict 
    
    
class QueryUrlGCN(t.nn.Module):
    """
    Character-level GCN for query-url vaccine intent classification. 
    If use_char_cnn: character-level sequence is first passed through CNN. Else, we use concatenation of 
    character-level one-hot encodings.
    """
    def __init__(self, num_chars, max_length, hidden_dim=32, num_gcn_layers=3, num_cnn_layers=2, 
                 dropout=0.5, use_char_cnn=True):
        super().__init__()
        self.num_chars = num_chars
        self.max_length = max_length
        self.num_gcn_layers = num_gcn_layers
        self.dropout = dropout
        
        self.use_char_cnn = use_char_cnn
        if self.use_char_cnn:
            self.num_cnn_layers = num_cnn_layers
            self.text_convs = CharCNNLayers(num_chars, max_length, num_layers=num_cnn_layers)
            input_dim = self.text_convs.get_output_dim()
        else:
            input_dim = self.num_chars * self.max_length  # concatenation of one-hot encodings
        
        self.convs = t.nn.ModuleList()
        self.bns = t.nn.ModuleList()
        for l in range(num_gcn_layers):
            self.convs.append(GraphConv(input_dim, hidden_dim))
            input_dim = hidden_dim  # input dim is hidden dim after l=0
            if l < (num_gcn_layers-1):
                self.bns.append(t.nn.BatchNorm1d(hidden_dim))
                # no batch normalization for final layer 
        self.out = t.nn.Linear(hidden_dim, 1)
        print('Initalized QueryUrlGCN with use_char_cnn =', self.use_char_cnn)
        
    def forward(self, batch):
        """
        batch: output from graph sampler.
        """
        x, edge_index, edge_weight = batch.x, batch.edge_index, batch.edge_attr.reshape(-1).float()
        x = F.one_hot(x.long(), num_classes=self.num_chars).float()
        if self.use_char_cnn:
            x = self.text_convs(x)
        else:
            x = t.flatten(x, start_dim=1)  # concatenate one-hot encodings
        for l in range(self.num_gcn_layers-1):
            x = self.convs[l](x, edge_index, edge_weight=edge_weight)
            x = self.bns[l](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return self.out(x).reshape(-1)

    
class HeteroQueryUrlGCN(t.nn.Module):
    """
    Character-level heterogeneous GCN for query-url vaccine intent classification. 
    If use_char_cnn: character-level sequence is first passed through CNN. Else, we use concatenation of 
    character-level one-hot encodings.
    See example here: https://github.com/pyg-team/pytorch_geometric/blob/master/examples/hetero/hetero_conv_dblp.py
    """
    def __init__(self, num_chars, max_length, metadata, hidden_dim=32, num_gcn_layers=3, num_cnn_layers=2, 
                 dropout=0.5, use_char_cnn=True):
        super().__init__()
        self.num_chars = num_chars
        self.max_length = max_length
        self.metadata = metadata
        self.num_gcn_layers = num_gcn_layers
        self.dropout = dropout
        
        self.use_char_cnn = use_char_cnn
        if self.use_char_cnn:
            self.num_cnn_layers = num_cnn_layers
            self.text_conv_dict = t.nn.ModuleDict()
            for node_type in metadata[0]:
                self.text_conv_dict[node_type] = CharCNNLayers(num_chars, max_length, num_cnn_layers)
            input_dim = next(iter(self.text_conv_dict.values())).get_output_dim()
        else:
            input_dim = self.num_chars * self.max_length  # concatenation of one-hot encodings
        
        self.convs = t.nn.ModuleList()
        self.bns = t.nn.ModuleList()
        for l in range(num_gcn_layers):
            conv = HeteroConv({et: GraphConv(input_dim, hidden_dim) for et in metadata[1]}, aggr='sum')
            self.convs.append(conv)
            input_dim = hidden_dim  # input dim is hidden dim after l=0
            if l < (num_gcn_layers-1):
                self.bns.append(t.nn.BatchNorm1d(hidden_dim))
            # no batch normalization for final layer 
        self.out = t.nn.Linear(hidden_dim, 1)
        
    def forward(self, batch):
        x_dict, edge_index_dict = batch.x_dict, batch.edge_index_dict
        edge_weights = {et: a.reshape(-1).float() for et, a in batch.edge_attr_dict.items()}
        for node_type, x in x_dict.items():
            x = F.one_hot(x.long(), num_classes=self.num_chars).float()
            if self.use_char_cnn:
                x_dict[node_type] = self.text_conv_dict[node_type](x)
            else:
                x_dict[node_type] = t.flatten(x, start_dim=1)  # concatenate one-hot encodings
        for l in range(self.num_gcn_layers-1):
            x_dict = self.convs[l](x_dict, edge_index_dict, edge_weight_dict=edge_weights)
            for node_type, x in x_dict.items():
                x = self.bns[l](x)
                x = F.relu(x)
                x_dict[node_type] = F.dropout(x, p=self.dropout, training=self.training)
        x_dict = self.convs[-1](x_dict, edge_index_dict, edge_weight_dict=edge_weights)
        x_dict = {node_type:self.out(x).reshape(-1) for node_type, x in x_dict.items()}
        return x_dict

    
def convert_pair_counts_to_pyg_data(pair_counts, feature_type='char', hetero=True, max_length=200, max_tokens=10000):
    """
    Convert list of pair counts, which are in the form ((node1, node2), count), to PyG data.
    Note: first part of function is identical to convert_pair_counts_to_sparse_matrix.
    """
    assert feature_type in ['char', 'bow']
    node1 = [t[0][0] for t in pair_counts]
    node2 = [t[0][1] for t in pair_counts]
    node_list = sorted(set(node1 + node2))
    N = len(node_list)
    print('num unique nodes:', N)
    print('num edges:', len(pair_counts))
    
    # create initial node features
    if feature_type == 'char':
        ts = time.time()
        strings = [n.rsplit('_', 1)[0] for n in node_list]  # drop QUERY or URL tag
        char2idx = {c:i+2 for i, c in enumerate(string.printable)}  # leave 0 for padding and 1 for unknown
        char2idx['__PAD__'] = 0
        char2idx['__UNK__'] = 1
        x = convert_chars_to_int_seq(strings, char2idx, max_length, one_hot=False)
        feature_dict = char2idx
    else:
        strings = [n.rsplit('_', 1)[0].lower() for n in node_list]  # drop QUERY or URL tag, lowercase
        cv = CountVectorizer(max_features=max_tokens, token_pattern=r'[a-z]{2,}')  # token is 2+ consecutive letters
        x = t.tensor(cv.fit_transform(strings).toarray(), dtype=t.float)
        feature_dict = cv.vocabulary_
    
    if hetero:  # construct heterogeneous graph
        data = HeteroData()
        is_query = []
        queries = []
        urls = []
        for n in node_list:
            if n.endswith('###QUERY###'):
                is_query.append(True)
                queries.append(n)
            else:
                is_query.append(False)
                urls.append(n)
        is_query = t.tensor(is_query).bool()
        data['query'].x = x[is_query]
        assert len(data['query'].x) == len(queries)
        data['url'].x = x[~is_query]
        assert len(data['url'].x) == len(urls)
        
        # need separate mapping for queries and urls
        query2idx = {q:i for i,q in enumerate(queries)}
        url2idx = {u:i for i,u in enumerate(urls)} 
        # create data for each edge type (there are no URL-URL edges)
        edge_types = [('query', 'query'),
                      ('query', 'url'),
                      ('url', 'query')]
        et_to_data = {et:{'edge_index':[], 'edge_weights':[]} for et in edge_types}
        for (n1, n2), w in pair_counts:
            if n1.endswith('###QUERY###') and n2.endswith('###QUERY###'):
                et_to_data[('query', 'query')]['edge_index'].append([query2idx[n1], query2idx[n2]])
                et_to_data[('query', 'query')]['edge_weights'].append(w)
            elif n1.endswith('###QUERY###') and n2.endswith('###URL###'):
                et_to_data[('query', 'url')]['edge_index'].append([query2idx[n1], url2idx[n2]])
                et_to_data[('query', 'url')]['edge_weights'].append(w)
            else:
                assert n1.endswith('###URL###') and n2.endswith('###QUERY###')
                et_to_data[('url', 'query')]['edge_index'].append([url2idx[n1], query2idx[n2]])
                et_to_data[('url', 'query')]['edge_weights'].append(w)
        for (type1, type2), et_data in et_to_data.items():
            edge_index = t.transpose(t.tensor(et_data['edge_index'], dtype=t.long), 0, 1)
            edge_attr = t.tensor(np.array(et_data['edge_weights']).reshape(-1, 1))
            assert edge_index.shape[1] == edge_attr.shape[0]
            print('%s -> %s: found %d edges' % (type1, type2, edge_index.shape[1]))
            data[type1, '->', type2].edge_index = edge_index
            data[type1, '->', type2].edge_attr = edge_attr
    else:
        node2idx = {n:i for i,n in enumerate(node_list)}
        node1 = [node2idx[n] for n in node1]
        node2 = [node2idx[n] for n in node2]
        edge_index = t.tensor([node1, node2], dtype=t.long)
        edge_weights = [t[1] for t in pair_counts]
        edge_attr = t.tensor(np.array(edge_weights).reshape(-1, 1))
        data = Data(edge_index=edge_index, edge_attr=edge_attr, x=x)
        data.n_id = t.arange(data.num_nodes)  # include this so we can map nodes in batches back to original graph
    return data, node_list, feature_dict


def convert_chars_to_int_seq(str_list, char2idx, max_length, one_hot=True):
    """
    Helper function to convert list of strings into sequences of character indices, possibly as one-hot encoding.
    """
    assert char2idx['__PAD__'] == 0
    assert char2idx['__UNK__'] == 1
    N = len(str_list)
    x = np.zeros((N, max_length))
    truncated = 0
    ts = time.time()
    for i, s in enumerate(str_list):
        if len(s) > max_length:
            truncated += 1
            s = s[:max_length]
        seq = [char2idx.get(c, 1) for c in s]  # convert to int seq, use 1 for unknown
        x[i, :len(seq)] = seq
    x = t.tensor(x, dtype=t.long)
    print('Converted strings to character int sequences [time=%.2fs]' % (time.time()-ts))
    print('%d (%.2f%%) strings were truncated with max length %d' % (truncated, 100. * truncated / N, max_length))
    if one_hot:
        ts = time.time()
        x = F.one_hot(x, num_classes=len(char2idx))
        assert x.shape == (N, max_length, len(char2idx))
        print('Converted int sequences to one-hot [time=%.2fs]' % (time.time()-ts))
    return x
    
    
def convert_int_seq_to_chars(x, char2idx, one_hot=True):
    """
    Helper function to convert sequences of character indices, possibly as one-hot encoding, back to list of strings.
    """
    assert char2idx['__PAD__'] == 0
    assert char2idx['__UNK__'] == 1
    if one_hot:
        N, max_length, num_chars = x.shape
        assert num_chars == len(char2idx)
        ts = time.time()
        x = x.argmax(-1)
        assert x.shape == (N, max_length)
        print('Converted one-hot to int sequences [time=%.2fs]' % (time.time()-ts))
    if type(x) == t.Tensor:
        x = x.detach().numpy()
    ordered_chars = np.array(sorted(char2idx.keys(), key=lambda x: char2idx[x]))  # sort chars by index
    str_list = []
    ts = time.time()
    for i in range(len(x)):
        seq = x[i]
        seq = seq[seq > 0]  # drop padding
        chars = ordered_chars[seq]
        str_list.append(''.join(chars))
    print('Converted character int sequences to strings [time=%.2fs]' % (time.time()-ts))
    return str_list


def add_sppr_labels_to_pyg_data(data, node_list, state, dates, min_n=1000):
    """
    Add SPPR scores and SPPR "reverse rank" to PyG data. A node's reverse rank = max(0, N-rank), where rank is its rank from SPPR.
    The max reverse rank is N, for the node at rank=0, and all nodes with rank >= N have reverse rank 0.
    If we have multiple states, we take the max SPPR score and reverse rank over states.
    """
    if type(state) == list:
        states = state
    elif state == 'All States':
        states = get_state_names(drop_dc=True)
    else:
        states = [state]
    
    all_sppr_scores = np.zeros(len(node_list))
    all_sppr_reverse = np.zeros(len(node_list))
    for s in states:
        sppr_df = load_nodes_to_label_for_state_and_date_range(s, dates, filter_queries=False, filter_urls=False)
        node2sppr = dict(zip(sppr_df['node'], sppr_df['score']))
        sppr_scores = np.array([node2sppr.get(n, 0) for n in node_list])
        all_sppr_scores = np.maximum(sppr_scores, all_sppr_scores)  # pointwise max

        last_seed_set_rank = sppr_df[sppr_df.label == 1]['rank'].values[-1]
        N = max(min_n, last_seed_set_rank)
        top_n_sppr_df = sppr_df.iloc[:N]
        node2sppr_reverse = dict(zip(top_n_sppr_df['node'], N-top_n_sppr_df['rank']))  # max is N, everything below rank N-1 is 0
        sppr_reverse = np.array([node2sppr_reverse.get(n, 0) for n in node_list])
        all_sppr_reverse = np.maximum(sppr_reverse, all_sppr_reverse)
        
    print('Found SPPR scores for %d nodes' % np.sum(all_sppr_scores > 0)) 
    data.sppr = t.tensor(all_sppr_scores).float()
    print('Found SPPR reverse ranks for %d nodes' % np.sum(all_sppr_reverse > 0))
    data.sppr_reverse = t.tensor(all_sppr_reverse).float()
    return data
    

def pretrain_model_on_sppr(data, mdl, device, hetero=True, save_prefix=None, epochs=100, 
                           batch_size=32, num_neighbors=20, target_correlation=0.81):
    """
    Pre-train the GNN model on SPPR reverse rank.
    """
    for p in mdl.parameters():
        assert p.device.type == device.type
    if save_prefix is not None:
        print('Saving results with prefix', save_prefix)
    top_n = data.sppr_reverse > 0
    other_idx = t.arange(data.num_nodes)[~top_n]
    num_to_sample = top_n.sum()
    print('Will sample %d for train in each epoch' % num_to_sample)
    
    opt = t.optim.Adam(mdl.parameters(), lr=0.001, weight_decay=5e-4)
    top_n_loader = NeighborLoader(data, num_neighbors=[num_neighbors]*mdl.num_gcn_layers, input_nodes=top_n, 
                                  batch_size=batch_size)  
    train_losses = []
    correlations = []
    for epoch in range(epochs):
        print('============== EPOCH %d ==============' % epoch)
        ts = time.time()
        train_mask = data.sppr_reverse > 0  # always train on top n
        sample = t.randperm(len(other_idx))[:num_to_sample]
        train_mask[other_idx[sample]] = True  # add random equal-sized sample of other nodes to train
        assert train_mask.sum() == (2*num_to_sample)
        train_loader = NeighborLoader(data, num_neighbors=[num_neighbors]*mdl.num_gcn_layers, shuffle=True, input_nodes=train_mask, 
                                      batch_size=batch_size)        
        train_loss = train_single_epoch(mdl, opt, device, train_loader, pretrain=True)
        print('Train loss = %.4f [time = %.2fs]' % (train_loss, time.time()-ts))
        train_losses.append(train_loss)
        
        # test correlation on top n nodes
        _, _, pred, sppr_reverse = test_single_epoch(mdl, device, top_n_loader, pretrain=True, return_pred_and_labels=True)
        pred = pred.cpu().detach().numpy()
        sppr_reverse = sppr_reverse.cpu().detach().numpy()
        r, p = spearmanr(pred, sppr_reverse)
        print('Correlation with SPPR = %.4f' % r)
        correlations.append(r)
            
        if save_prefix is not None:
            t.save(mdl.state_dict(), './model_results/%s.weights' % save_prefix)  # save model state
            t.save(opt.state_dict(), './model_results/%s.opt' % save_prefix)  # save optimizer state
            with open('./model_results/%s_losses.pkl' % save_prefix, 'wb') as f:
                pickle.dump((train_losses, correlations), f)
        if r >= target_correlation:
            print('Early stopping; reached target correlation of %s' % target_correlation)
            break
    return mdl, opt, train_losses
    
    
def get_labels_for_nodes(node_list, only_url=True):
    """
    Get AMT and regex labels for URLs; also for queries, if only_url=False.
    """
    url2label = load_AMT_labeled_urls(only_nonnan=True)
    node_idx = []
    labels = []
    label_types = []
    conflicting_queries = []
    for i, n in enumerate(node_list):
        node_str, node_type = n.rsplit('_', 1)
        if node_type == '###QUERY###' and not only_url:
            node_wants_vaccine = wants_vaccine(node_str)
            node_not_about_vaccine = not_about_vaccine(node_str)
            if node_wants_vaccine and not node_not_about_vaccine:
                node_idx.append(i)
                labels.append(True)
                label_types.append('regex positive query')
            elif not node_wants_vaccine and node_not_about_vaccine:
                node_idx.append(i)
                labels.append(False)
                label_types.append('regex negative query')
            elif node_wants_vaccine and node_not_about_vaccine:
                conflicting_queries.append(node_str)
        elif node_type == '###URL###':
            if is_vaccine_intent_url(node_str):
                node_idx.append(i)
                labels.append(True)
                label_types.append('regex positive url')
            # no need to check for conflicts since negative function already checks positive function is false 
            elif is_not_vaccine_intent_url(node_str):  
                node_idx.append(i)
                labels.append(False)
                label_types.append('regex negative url')
            elif node_str in url2label:
                node_idx.append(i)
                if url2label[node_str] == 1:
                    labels.append(True)
                    label_types.append('AMT positive url')
                elif url2label[node_str] == 0:
                    labels.append(False)
                    label_types.append('AMT negative url')
    node_idx = np.array(node_idx)
    labels = np.array(labels)
    print('Found %d positive nodes and %d negative nodes' % (np.sum(labels == True), np.sum(labels == False)))
    print(Counter(label_types))
    if len(conflicting_queries) > 0:
        print('Conflicting queries:', conflicting_queries)
    return node_idx, labels


def add_labels_to_pyg_data(data, node_list, node_idx, labels, train_prop=0.6, val_prop=0.15, seed=0, hetero=True):
    """
    Do train/test split on labeled nodes. Add y, is_url, has_real_label, val_mask, test_mask, and heldout_mask to data.
    """
    assert data.num_nodes == len(node_list)
    assert len(node_idx) == len(labels)  # node_idx contains the indices of the labeled nodes
    order = np.arange(len(node_idx))
    np.random.seed(seed)
    np.random.shuffle(order)
    node_idx = node_idx[order]
    labels = labels[order]
    num_train = int(len(node_idx) * train_prop)
    train_idx = node_idx[:num_train]
    train_labels = labels[:num_train]
    print('Train:', Counter(train_labels))
    num_val = int(len(node_idx) * val_prop)
    val_idx = node_idx[num_train:num_train+num_val]
    val_labels = labels[num_train:num_train+num_val]
    print('Val:', Counter(val_labels))
    test_idx = node_idx[num_train+num_val:]
    test_labels = labels[num_train+num_val:]
    print('Test:', Counter(test_labels))
    
    y = np.zeros(data.num_nodes)  # treat unlabeled as negative 
    y[node_idx] = labels
    data.y = t.tensor(y).float()
    is_url = np.array([n.endswith('###URL###') for n in node_list]).astype(bool)
    data.is_url = t.tensor(is_url)
    has_real_label = np.zeros(data.num_nodes).astype(bool)
    has_real_label[node_idx] = True
    data.has_real_label = t.tensor(has_real_label)
    val_mask = np.zeros(data.num_nodes).astype(bool)
    val_mask[val_idx] = True
    data.val_mask = t.tensor(val_mask)
    test_mask = np.zeros(data.num_nodes).astype(bool)
    test_mask[test_idx] = True
    data.test_mask = t.tensor(test_mask)
    data.heldout_mask = data.val_mask | data.test_mask  # all heldout indices
    return data


def train_model(data, mdl, device, hetero=True, only_url=True, save_prefix=None, epochs=100, patience=5, unlabeled_weight=0.2, 
                batch_size=32, num_neighbors=20):
    """
    Train the GNN model.
    """
    for p in mdl.parameters():
        assert p.device.type == device.type
    if save_prefix is not None:
        print('Saving results with prefix', save_prefix)
    # we train on all nodes but hold out the labels of the val+test nodes (and treat them as negative-unlabeled)
    val_loader = NeighborLoader(data, num_neighbors=[num_neighbors]*mdl.num_gcn_layers, input_nodes=data.val_mask, 
                                 batch_size=batch_size)
    seen_real_label = data.has_real_label & (~data.heldout_mask)
    if only_url:
        assert all(data.is_url[data.has_real_label])  # only URLs should be labeled
        unlabeled_idx = t.arange(data.num_nodes)[(~seen_real_label) & data.is_url]
    else: 
        unlabeled_idx = t.arange(data.num_nodes)[~seen_real_label]
    num_real_label = seen_real_label.sum()
    print('Will sample %d unlabeled for train in each epoch' % num_real_label)
    
    opt = t.optim.Adam(mdl.parameters(), lr=0.001, weight_decay=5e-4)
    train_losses = []
    val_losses = []
    no_improvement = 0    
    for epoch in range(epochs):
        print('============== EPOCH %d ==============' % epoch)
        ts = time.time()
        train_mask = data.has_real_label & (~data.heldout_mask)
        sample = t.randperm(len(unlabeled_idx))[:num_real_label]
        train_mask[unlabeled_idx[sample]] = True  # add random equal-sized sample of unlabeled nodes to train
        assert train_mask.sum() == (2*num_real_label)
        train_loader = NeighborLoader(data, num_neighbors=[num_neighbors]*mdl.num_gcn_layers, shuffle=True, input_nodes=train_mask, 
                                      batch_size=batch_size)
        train_loss = train_single_epoch(mdl, opt, device, train_loader, unlabeled_weight=unlabeled_weight)
        print('Train loss = %.4f [time = %.2fs]' % (train_loss, time.time()-ts))
        train_losses.append(train_loss)
        ts = time.time()
        val_loss = test_single_epoch(mdl, device, val_loader)
        print('Val loss = %.4f [time = %.2fs]' % (val_loss, time.time()-ts))
        val_losses.append(val_loss)
        if val_loss > min(val_losses):
            no_improvement += 1
        else:
            print('New minimum val loss!')
            no_improvement = 0
            
        if save_prefix is not None:
            t.save(mdl.state_dict(), './model_results/%s.weights' % save_prefix)  # save model state
            t.save(opt.state_dict(), './model_results/%s.opt' % save_prefix)  # save optimizer state
            with open('./model_results/%s_losses.pkl' % save_prefix, 'wb') as f:
                pickle.dump((train_losses, val_losses), f)
            if val_loss == min(val_losses):  # save if this is the best val loss we've seen so far
                t.save(mdl.state_dict(), './model_results/%s.best_weights' % save_prefix)
        if (epoch >= 10) and (no_improvement >= patience):
            print('Early stopping; min val loss has not improved at least %s times' % patience)
            break
    return mdl, opt, train_losses, val_losses


def train_single_epoch(mdl, opt, device, loader, pretrain=False, unlabeled_weight=0.2):
    """
    Train the model for a single epoch and update parameters.
    """
    assert unlabeled_weight <= 1
    mdl.train()
    total_loss = 0
    for batch in tqdm(loader):
        opt.zero_grad()
        batch = batch.to(device)
        s = batch.batch_size
        # From PyG doc: "Sampled nodes are sorted based on the order in which they were sampled. 
        # In particular, the first batch_size nodes represent the set of original mini-batch nodes."
        pred = mdl(batch)[:s]
        if pretrain:
            pred = pred.relu()  # reverse rank is non-negative
            y = batch.sppr_reverse[:s]
            criterion = t.nn.MSELoss(reduction='sum')
        else:
            y = batch.y[:s]
            y[batch.heldout_mask[:s]] = 0  # held-out points are assumed to be 0s during training
            seen_real_label = batch.has_real_label[:s] & (~batch.heldout_mask[:s])
            weights = t.ones(s)
            weights[~seen_real_label] = unlabeled_weight  # downweight unlabeled data points
            criterion = t.nn.BCEWithLogitsLoss(weight=weights.to(device), reduction='sum')  # combines sigmoid and BCELoss
        loss = criterion(pred, y)
        loss.backward()
        opt.step()
        total_loss += float(loss)
    return total_loss


@t.no_grad()
def test_single_epoch(mdl, device, loader, pretrain=False, return_pred_and_labels=False):
    """
    Test the model for a single epoch.
    """
    mdl.eval()
    total_loss = 0
    node_ids = []
    all_pred = []
    all_labels = []
    for batch in tqdm(loader):
        s = batch.batch_size
        batch = batch.to(device)
        pred = mdl(batch)[:s]
        if pretrain:
            pred = pred.relu()  # reverse rank is non-negative
            y = batch.sppr_reverse[:s]
            criterion = t.nn.MSELoss(reduction='sum')
        else:
            y = batch.y[:s]
            criterion = t.nn.BCEWithLogitsLoss(reduction='sum')  # combines sigmoid and BCELoss
        loss = criterion(pred, y)
        total_loss += float(loss)
        if return_pred_and_labels:
            node_ids.append(batch.n_id[:s])
            all_pred.append(pred)
            all_labels.append(y)
    if return_pred_and_labels:
        node_ids = t.concat(node_ids)
        all_pred = t.concat(all_pred)
        all_labels = t.concat(all_labels)
        assert len(node_ids) == len(all_pred)
        assert len(node_ids) == len(all_labels)
        return total_loss, node_ids, all_pred, all_labels
    return total_loss


def identify_urls_past_cutoff(state, dates, num_seeds, device, use_pretrain=True, hetero=False, data_bundle=None, min_precision=0.9):
    """
    For each seed, compute two cutoffs: 1) the min cutoff necessary to maintain near-perfect precision on the
    test URLs, 2) the median prediction for the positive test URLs. Then, for all "candidate" URLs (within SPPR top-N),
    store whether they pass the cutoffs for this seed. Saves and returns a summary result dataframe that records,
    for each candidate URL, the number of seeds for which it passes each cutoff.
    """
    print('Identifying URLs past cutoffs per seed for %s, %s (with pretraining = %s)...' % (state, dates, use_pretrain))
    if data_bundle is None:
        data, node_list, mdl = run_pretraining_experiment(state, dates, hetero=hetero, epochs=0, save_results=False)
        node_idx, labels = get_labels_for_nodes(node_list, only_url=True)
        data = add_labels_to_pyg_data(data, node_list, node_idx, labels, seed=0, hetero=hetero)  # any seed is fine; not using labels here
    else:
        assert len(data_bundle) == 5
        data, node_list, mdl, node_idx, labels = data_bundle
    nid2label = dict(zip(node_idx, labels))
    candidates = (data.sppr_reverse > 0) & (data.is_url)  # candidates are all URLs in SPPR top N
    cand_loader = NeighborLoader(data, num_neighbors=[20]*mdl.num_gcn_layers, input_nodes=candidates, batch_size=32)
    
    min_cutoffs = np.zeros(num_seeds)
    median_cutoffs = np.zeros(num_seeds)
    num_candidates = candidates.sum().detach().numpy()
    print('Testing %d candidate URLs from SPPR top N' % num_candidates)
    all_preds = np.zeros((num_seeds, num_candidates))
    node_ids_0 = None
    for seed in range(num_seeds):
        prefix = '%s_%s_QueryUrlGCN_s%d' % (state.replace(' ', ''), dates, seed)
        if not use_pretrain:
            prefix += '_wo_pretrain'
        weights_fn = './model_results/%s.best_weights' % prefix
        mdl.load_state_dict(t.load(weights_fn))
        # get model predictions on candidate URLs
        _, node_ids, pred, _ = test_single_epoch(mdl, device, cand_loader, return_pred_and_labels=True)
        if seed == 0:
            node_ids_0 = node_ids
        else:
            assert all(node_ids == node_ids_0)
        all_preds[seed] = pred.cpu().detach().numpy()
        
        # get saved model results on held-out test set
        with open('./model_results/%s_results_on_test.pkl' % prefix, 'rb') as f:
            labeled_nodes, test_labels, test_pred = pickle.load(f)
        thresholds = np.arange(100)  # percentiles
        cutoffs, prec_vec, recall_vec, pred_auc = make_curves_and_measure_auc(
            test_pred, test_labels, thresholds=thresholds, mode='prc')
        passes_min_precision = prec_vec >= min_precision
        if np.sum(passes_min_precision) == 0:  # cannot reach minimum precision with any threshold
            print('Warning: no threshold reaches minimum precision of %s in seed %d' % (min_precision, seed))
            min_cutoff = np.max(all_preds[seed]) + 1  # nothing will pass this cutoff
        else:
            min_cutoff = np.min(cutoffs[passes_min_precision])
        min_cutoffs[seed] = min_cutoff
        median_cutoff = np.median(test_pred[test_labels == True])
        median_cutoffs[seed] = median_cutoff
        print('Seed %d: num URLs above cutoffs: min=%d, median=%d' % (
            seed, np.sum(all_preds[seed] >= min_cutoff), np.sum(all_preds[seed] >= median_cutoff)))
    
    sppr_reverse_rank = data.sppr_reverse[node_ids_0]
    node_ids_0 = node_ids_0.cpu().detach().numpy()
    urls = [node_list[nid].rsplit('_', 1)[0] for nid in node_ids_0]
    found_labels = [nid2label.get(nid, np.nan) for nid in node_ids_0]
    results_df = pd.DataFrame({'url': urls, 'label': found_labels, 'sppr_reverse': sppr_reverse_rank.detach().numpy()})

    above_cutoff = (all_preds.T >= min_cutoffs).T.astype(int)
    print('URLs above min cutoff per seed:', np.sum(above_cutoff, axis=1))
    results_df['above_min_cutoff'] = np.sum(above_cutoff, axis=0)
    above_cutoff = (all_preds.T >= median_cutoffs).T.astype(int)
    print('URLs above median cutoff per seed:', np.sum(above_cutoff, axis=1))
    results_df['above_median_cutoff'] = np.sum(above_cutoff, axis=0)
    results_df = results_df.sort_values(['above_min_cutoff', 'above_median_cutoff'], ascending=False)
    return results_df


def run_pretraining_experiment(state, dates, hetero=False, use_char_cnn=True, max_length=200, epochs=10,
                               save_results=True, gpu=0):
    """
    Main function to pretrain model for a state and dates.
    """
    print('Pretraining model for %s, %s' % (state, dates))
    device = t.device('cuda:%d' % gpu if t.cuda.is_available() else 'cpu')
    print('Using device', device)
    
    # load graph, features, and labels
    if dates == '3months':
        date_ranges = ['2021-04-01_2021-04-30', '2021-06-15_2021-07-14', '2021-09-01_2021-09-30']
    elif dates == '2months':
        date_ranges = ['2021-04-01_2021-04-30', '2021-08-01_2021-08-31']
    else:
        date_ranges = [date_range]
    pair_counts = load_pair_counts_for_state_and_date_ranges(state, date_ranges)
    data, node_list, char2idx = convert_pair_counts_to_pyg_data(pair_counts, hetero=hetero, feature_type='char', max_length=max_length)
    num_chars = len(char2idx)
    data = add_sppr_labels_to_pyg_data(data, node_list, state, dates)
    
    # pretrain model
    if hetero:
        mdl = HeteroQueryUrlGCN(num_chars, max_length, data.metadata(), use_char_cnn=use_char_cnn).to(device)
    else:
        mdl = QueryUrlGCN(num_chars, max_length, use_char_cnn=use_char_cnn).to(device)
    if epochs == 0:  # just return data and model
        return data, node_list, mdl
    mdl_name = mdl.__class__.__name__
    if save_results:
        prefix = '%s_%s_%s_pretrain' % (state.replace(' ', ''), dates, mdl_name)
    else:
        prefix = None
    mdl, opt, train_losses = pretrain_model_on_sppr(data, mdl, device, hetero=hetero, save_prefix=prefix, epochs=epochs)
    
    # get predictions on labeled URLs - saves time on analysis later
    node_idx, all_labels = get_labels_for_nodes(node_list, only_url=True)
    data = add_labels_to_pyg_data(data, node_list, node_idx, all_labels, hetero=hetero)
    loader = NeighborLoader(data, num_neighbors=[20]*mdl.num_gcn_layers, input_nodes=data.has_real_label, batch_size=32)  
    _, node_ids, pred, _ = test_single_epoch(mdl, device, loader, pretrain=True, return_pred_and_labels=True)
    pred = pred.cpu().detach().numpy()
    sppr = data.sppr[node_ids].cpu().detach().numpy()
    labels = data.y[node_ids].cpu().detach().numpy().astype(bool)
    node_ids = node_ids.cpu().detach().numpy()
    labeled_nodes = [node_list[ni] for ni in node_ids]
    with open('./model_results/%s_results_on_labeled.pkl' % prefix, 'wb') as f:
        pickle.dump((labeled_nodes, labels, pred, sppr), f)
    _, _, _, pred_auc = make_curves_and_measure_auc(pred, labels)
    print('AUROC, pretrained GNN: %.4f' % pred_auc)
    _, _, _, sppr_auc = make_curves_and_measure_auc(sppr, labels)
    print('AUROC, SPPR: %.4f' % sppr_auc)
    return data, node_list, mdl, opt, train_losses


def run_experiment(state, dates, hetero=False, use_char_cnn=True, max_length=200, epochs=10, num_seeds=10, 
                   patience=5, use_pretrain=True, save_results=True, only_url=True, gpu=0):
    """
    Main function to train model for a state and dates.
    """
    print('Training model for %s, %s' % (state, dates))
    device = t.device('cuda:%d' % gpu if t.cuda.is_available() else 'cpu')
    print('Using device', device)
    
    # load graph, features, and labels
    if dates == '3months':
        date_ranges = ['2021-04-01_2021-04-30', '2021-06-15_2021-07-14', '2021-09-01_2021-09-30']
    elif dates == '2months':
        date_ranges = ['2021-04-01_2021-04-30', '2021-08-01_2021-08-31']
    else:
        date_ranges = [date_range]
    pair_counts = load_pair_counts_for_state_and_date_ranges(state, date_ranges)
    data, node_list, char2idx = convert_pair_counts_to_pyg_data(pair_counts, hetero=hetero, feature_type='char', max_length=max_length)
    num_chars = len(char2idx)
    node_idx, labels = get_labels_for_nodes(node_list, only_url=only_url)
    mdl = None
    
    seed2results = {}
    for seed in range(num_seeds):
        print('STARTING SEED %d...' % seed)
        # do train/test split for this seed
        data = add_labels_to_pyg_data(data, node_list, node_idx, labels, seed=seed, hetero=hetero)
        if hetero:
            mdl = HeteroQueryUrlGCN(num_chars, max_length, data.metadata(), use_char_cnn=use_char_cnn).to(device)
        else:
            mdl = QueryUrlGCN(num_chars, max_length, use_char_cnn=use_char_cnn).to(device)
        mdl_name = mdl.__class__.__name__
        if use_pretrain:
            pretrain_fn = './model_results/%s_%s_%s_pretrain.weights' % (state.replace(' ', ''), dates, mdl_name)
            print('Loading pretrained weights from', pretrain_fn)
            print(mdl.load_state_dict(t.load(pretrain_fn)))
        if epochs == 0:  # just return data and model
            return data, node_list, mdl

        # train model
        t.nn.init.xavier_normal_(mdl.out.weight)  # initialize final linear layer to normal
        if save_results:
            prefix = '%s_%s_%s_s%d' % (state.replace(' ', ''), dates, mdl_name, seed)
            if not use_pretrain:
                prefix += '_wo_pretrain'
        else:
            prefix = None
        mdl, opt, train_losses, val_losses = train_model(data, mdl, device, hetero=hetero, epochs=epochs, 
                                                          save_prefix=prefix, patience=patience, only_url=only_url)
        mdl.load_state_dict(t.load('./model_results/%s.best_weights' % prefix))  # reload best weights 
        seed2results[seed] = (mdl, opt, train_losses, val_losses)
        
        # get predictions on test URLs - saves time on analysis later
        if save_results:
            loader = NeighborLoader(data, num_neighbors=[20]*mdl.num_gcn_layers, input_nodes=data.test_mask, batch_size=32)
            _, node_ids, pred, test_labels = test_single_epoch(mdl, device, loader, return_pred_and_labels=True)
            pred = pred.cpu().detach().numpy()
            test_labels = test_labels.cpu().detach().numpy().astype(bool)
            node_ids = node_ids.cpu().detach().numpy()
            labeled_nodes = [node_list[ni] for ni in node_ids]
            with open('./model_results/%s_results_on_test.pkl' % prefix, 'wb') as f:
                pickle.dump((labeled_nodes, test_labels, pred), f)
            _, _, _, pred_auc = make_curves_and_measure_auc(pred, test_labels)
            print('AUROC on test: %.4f' % pred_auc)
        print()
    
    if save_results:
        # record URLs that pass cutoff - saves time on analysis later 
        data = add_sppr_labels_to_pyg_data(data, node_list, state, dates)
        data_bundle = data, node_list, mdl, node_idx, labels
        results_df = identify_urls_past_cutoff(state, dates, num_seeds, device, use_pretrain=use_pretrain, 
                                               hetero=hetero, data_bundle=data_bundle)
        postfix = '%s_%s' % (state.replace(' ', ''), dates)
        if not use_pretrain:
            postfix += '_wo_pretrain'
        results_df.to_csv(os.path.join(PATH_TO_PROCESSED_DATA, 'gnn_cutoffs_%s.csv' % postfix), index=False)
    return data, node_list, seed2results


def pretrain_all_small_states(dates, gpu=0, node_max=5e6):
    """
    Helper function to pretrain for all small states, skipping those we already have results for.
    """
    states = get_state_names(drop_dc=True)
    for state in states:
        state = state.replace(' ', '')
        prefix = '%s_%s_QueryUrlGCN_pretrain' % (state, dates)
        num_nodes = get_state_graph_size(state)
        if os.path.isfile('./model_results/%s_results_on_labeled.pkl' % prefix):
            print('Skipping %s since we already have results' % state)
        elif num_nodes > node_max:
            print('Skipping %s since too many nodes (N=%d)' % (state, num_nodes))
        else:
            cmd = 'nohup python -u gnn_node_classification.py %s %s --mode pretrain --gpu %d --epochs 100 ' \
                     '> ./logs/%s_pretrain.out 2>&1' % (state, dates, gpu, state)
            print(cmd)
            os.system(cmd)
            
                                    
def train_all_states(dates, state_type, gpu=0, node_max=5e6):
    """
    Helper function to train all states. For large states, we don't use pretraining; for small states, we do.
    """
    assert state_type in {'small', 'large'}
    states = get_state_names(drop_dc=True)
    for state in states:
        state = state.replace(' ', '')
        prefix = '%s_%s_QueryUrlGCN_s0' % (state, dates)  # check first seed
        num_nodes = get_state_graph_size(state)
        if num_nodes <= node_max and state_type == 'small':
            cmd = 'nohup python -u gnn_node_classification.py %s %s --gpu %d --epochs 100 --num_seeds 10 ' \
                      '> ./logs/%s_train.out 2>&1' % (state, dates, gpu, state)
        elif num_nodes > node_max and state_type == 'large':
            prefix += '_wo_pretrain'  # pretraining is unnecesary for large states
            cmd = 'nohup python -u gnn_node_classification.py %s %s --gpu %d --epochs 100 --num_seeds 10 ' \
                      '--use_pretrain 0 > ./logs/%s_train_wo_pretrain.out 2>&1' % (state, dates, gpu, state)
        else:
            continue 
        if os.path.isfile('./model_results/%s_results_on_test.pkl' % prefix):
            print('Skipping %s since we already have results' % state)
        else:
            print(cmd)
            os.system(cmd)

            
def get_gnn_expanded_urls(dates, above_median_cutoff=6, above_min_cutoff=6, verbose=True):
    """
    Gets all unlabeled URLs that pass cutoffs.
    """
    states = get_state_names(drop_dc=True)
    state2urls = {}
    url2states = {}
    url2median_cutoffs = {}
    url2min_cutoffs = {}
    for state in states:
        new_urls = get_gnn_expanded_urls_for_state(state, dates, 
                       above_median_cutoff=above_median_cutoff, above_min_cutoff=above_min_cutoff,
                       verbose=verbose)
        state2urls[state] = new_urls
        for _, row in new_urls.iterrows():
            url = row['url']
            if url in url2states:
                url2states[url].append(state)
                url2median_cutoffs[url].append(row['above_median_cutoff'])
                url2min_cutoffs[url].append(row['above_min_cutoff'])
            else:
                url2states[url] = [state]
                url2median_cutoffs[url] = [row['above_median_cutoff']]
                url2min_cutoffs[url] = [row['above_min_cutoff']]
    print('Found %d new URLs overall' % len(url2states))
    
    urls = list(url2states.keys())
    num_states = [len(url2states[u]) for u in urls]
    associated_states = [', '.join(sorted(url2states[u])) for u in urls]
    max_median_cutoffs = [np.max(url2median_cutoffs[u]) for u in urls]
    max_min_cutoffs = [np.max(url2min_cutoffs[u]) for u in urls]
    url_df = pd.DataFrame({'url': urls, 'num_states': num_states, 'associated_states': associated_states,
                           'max_median_cutoff': max_median_cutoffs, 'max_min_cutoff': max_min_cutoffs}).sort_values('url')
    return url_df, state2urls


def get_gnn_expanded_urls_for_state(state, dates, above_median_cutoff=6, above_min_cutoff=6, 
                                    return_candidates=False, verbose=True, with_pretrain=None):
    """
    Gets all unlabeled URLs that pass cutoffs for state.
    """
    postfix = '%s_%s' % (state.replace(' ', ''), dates)  # default to using pretrained weights
    if with_pretrain is None and get_state_graph_size(state) > LARGE_STATE_CUTOFF:  
        # if unspecified, don't use pretraining for large states
        postfix += '_wo_pretrain'
    elif with_pretrain == False:
        postfix += '_wo_pretrain'
    results_df = pd.read_csv(os.path.join(PATH_TO_PROCESSED_DATA, 'gnn_cutoffs_%s.csv' % postfix))
    candidates = results_df[results_df.label.isnull()]
    is_internal_search = candidates.url.apply(lambda x: x.startswith('/') and '?q=' in x)  # these show up in queries
    candidates = candidates[~is_internal_search]
    new_urls = candidates[(candidates.above_median_cutoff >= above_median_cutoff) &
                          (candidates.above_min_cutoff >= above_min_cutoff)]
    if verbose:
        print('%s: keeping %d out of %d (%.2f%%) candidate URLs --> min above_min_cutoff = %d, min above_median_cutoff = %d' % 
          (state, len(new_urls), len(candidates), 100.*len(new_urls)/len(candidates), 
           new_urls.above_min_cutoff.min(), new_urls.above_median_cutoff.min()))
    if return_candidates:
        return candidates, new_urls
    return new_urls

    
def compare_pretrained_gnn_vs_sppr(states, state2size, dates, save_results=True):
    """
    Evaluate AUROCs of pretrained GNN vs S-PPR on AMT labels.
    """
    auc_table = []
    for state in states:
        prefix = '%s_%s_QueryUrlGCN_pretrain' % (state.replace(' ', ''), dates)
        with open('./model_results/%s_results_on_labeled.pkl' % prefix, 'rb') as f:
            out = pickle.load(f)
            if len(out) == 4:
                labeled_nodes, labels, pred, sppr = out
            else:  # original version
                pred, sppr, labels = out
        state_aucs = {'state': state, 'size': state2size[state]}
        plt.figure(figsize=(5,5))
        plt.title('%s' % state, fontsize=12)
        _, tpr, fpr, auc = make_curves_and_measure_auc(pred, labels, mode='roc')
        state_aucs['gnn_pretrained_auroc'] = auc
        plt.plot(fpr, tpr, label='pretrained GNN')
        _, tpr, fpr, auc = make_curves_and_measure_auc(sppr, labels, mode='roc')
        state_aucs['sppr_auroc'] = auc
        auc_table.append(state_aucs)
        
        plt.plot(fpr, tpr, label='SPPR')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.grid(alpha=0.2)
        plt.legend(fontsize=12)
        plt.xlabel('False positive rate', fontsize=12)
        plt.ylabel('True positive rate', fontsize=12)
        if save_results:
            plt.savefig(os.path.join(PATH_TO_RESULTS, 
               '%s_pretrain_vs_sppr_aucs.pdf' % state.replace(' ', '')), bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    auc_table = pd.DataFrame(auc_table, columns=list(auc_table[-1].keys()))
    auc_table['diff'] = auc_table.gnn_pretrained_auroc - auc_table.sppr_auroc
    auc_table['ratio'] = auc_table.gnn_pretrained_auroc / auc_table.sppr_auroc
    print(auc_table[['diff', 'ratio']].mean())
    if save_results:
        auc_table.to_csv(os.path.join(PATH_TO_RESULTS, 'pretrain_vs_sppr_aucs.csv'), index=False)
    return auc_table

    
def compare_results_with_and_without_pretraining(states, state2size, dates,
                                                 num_seeds=10, save_results=True):
    """
    Evaluate AUROCs on held-out AMT labels with pretraining on S-PPR and without.
    """
    auc_table = []
    for state in states:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))  # left is with pretraining, right is without
        num_test = None
        pretrain_aucs = []
        wo_pretrain_aucs = []
        for seed in range(num_seeds):
            prefix = '%s_%s_QueryUrlGCN_s%d' % (state.replace(' ', ''), dates, seed)
            with open('./model_results/%s_results_on_test.pkl' % prefix, 'rb') as f:
                labeled_nodes, test_labels, pred = pickle.load(f)
            if seed == 0:
                num_test = len(test_labels)
                print('%s: %d test URLs' % (state, num_test))
            else: 
                assert len(test_labels) == num_test  # there should be an equal number of test URLs across seeds
            _, tpr, fpr, auc = make_curves_and_measure_auc(pred, test_labels, mode='roc')
            axes[1].plot(fpr, tpr)
            pretrain_aucs.append(auc)

            prefix += '_wo_pretrain'
            with open('./model_results/%s_results_on_test.pkl' % prefix, 'rb') as f:
                labeled_nodes, test_labels, pred = pickle.load(f)
            assert num_test == len(test_labels)
            _, tpr, fpr, auc = make_curves_and_measure_auc(pred, test_labels, mode='roc')
            axes[0].plot(fpr, tpr)
            wo_pretrain_aucs.append(auc)
            
        auc_table.append({'state': state, 'size': state2size[state],
                          'mean_auc_with_pretrain': np.mean(pretrain_aucs),
                          'std_auc_with_pretrain': np.std(pretrain_aucs),
                          'mean_auc_wo_pretrain': np.mean(wo_pretrain_aucs),
                          'std_auc_wo_pretrain': np.std(wo_pretrain_aucs)})
        plt.suptitle(state, fontsize=14)
        axes[0].set_title('Without pretraining', fontsize=12)
        axes[1].set_title('With pretraining', fontsize=12)
        for ax in axes:
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.grid(alpha=0.2)
            ax.set_xlabel('False positive rate', fontsize=12)
            ax.set_ylabel('True positive rate', fontsize=12)
        if save_results:
            plt.savefig(os.path.join(PATH_TO_RESULTS, 
               '%s_with_vs_without_pretrain_aucs.pdf' % state.replace(' ', '')), bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    auc_table = pd.DataFrame(auc_table, columns=list(auc_table[-1].keys()))
    auc_table['diff'] = auc_table.mean_auc_with_pretrain - auc_table.mean_auc_wo_pretrain
    auc_table['ratio'] = auc_table.mean_auc_with_pretrain / auc_table.mean_auc_wo_pretrain
    print(auc_table[['diff', 'ratio']].mean())
    if save_results:
        auc_table.to_csv(os.path.join(PATH_TO_RESULTS, 'with_vs_without_pretrain_aucs.csv'), index=False)
    return auc_table


def compute_final_results_for_all_states(states, dates, state2size, num_seeds=10, min_precision=0.9,
                                         state_url2freq=None, wo_pretrain=False, verbose=True):
    """
    Evaluate final AUCs for all states. For small states, use results with pretraining; 
    unnecessary for large states.
    """
    results_table = []
    for state in states:
        results = {'state': state, 'size': state2size[state]}
        num_with_cutoff = 0
        for seed in range(num_seeds):
            prefix = '%s_%s_QueryUrlGCN_s%d' % (state.replace(' ', ''), dates, seed)
            if state2size[state] > LARGE_STATE_CUTOFF or wo_pretrain:  # for large states we don't need pretraining
                prefix += '_wo_pretrain'
            with open('./model_results/%s_results_on_test.pkl' % prefix, 'rb') as f:
                labeled_nodes, test_labels, pred = pickle.load(f)
            results['s%d_test_pos' % seed] = np.sum(test_labels == True)
            matches_regex = np.array([is_vaccine_intent_url(n.rsplit('_', 1)[0]) for n in labeled_nodes])
            results['s%d_test_regex' % seed] = np.sum(matches_regex)
            results['s%d_test_neg' % seed] = np.sum(test_labels == False)
            
            _, tpr, fpr, auc = make_curves_and_measure_auc(pred, test_labels, mode='roc')
            results['s%d_auc' % seed] = auc
            
            thresholds = np.arange(100)  # percentiles
            cutoffs, prec_vec, _, _ = make_curves_and_measure_auc(pred, test_labels, thresholds=thresholds, mode='prc')
            passes_min_precision = prec_vec >= min_precision
            median_cutoff = np.median(pred[test_labels == True])
            equals_median = np.sum(np.isclose(pred, median_cutoff))
            if np.sum(passes_min_precision) == 0:  # can't reach minimum precision with any threshold
                if verbose:
                    print('Warning: %s, seed %d -> no threshold reaches minimum precision of %s' % (state, seed, min_precision))
            elif equals_median > 3:
                if verbose:
                    print('Warning: %s, seed %d -> %d predictions equal to median' % (state, seed, equals_median))
            else:
                num_with_cutoff += 1
                min_cutoff = np.min(cutoffs[passes_min_precision])   
                cutoff = np.max([min_cutoff, median_cutoff])
                test_pos = test_labels == True
                true_pos = test_pos & (pred >= cutoff)
                test_neg = test_labels == False
                false_pos = test_neg & (pred >= cutoff)
                if state_url2freq is not None:  # weight recall by URL frequency
                    freqs = np.array([state_url2freq[state][n.rsplit('_', 1)[0]] for n in labeled_nodes])
                else:
                    freqs = np.ones(len(labeled_nodes))
                results['s%d_tpr' % seed] = np.sum(true_pos.astype(int) * freqs) / np.sum(test_pos.astype(int) * freqs)
                results['s%d_fpr' % seed] = np.sum(false_pos.astype(int) * freqs) / np.sum(test_neg.astype(int) * freqs)
        results['num_with_cutoff'] = num_with_cutoff  # num seeds for which we evaluated TPR and FPR
        results_table.append(results)
    
    results_table = pd.DataFrame(results_table)
    for metric in ['auc', 'tpr', 'fpr']:
        seed_cols = ['s%d_%s' % (s, metric) for s in range(num_seeds)]
        results_table['mean_%s' % metric] = np.nanmean(results_table[seed_cols].values, axis=1)
        results_table['se_%s' % metric] = np.nanstd(results_table[seed_cols].values, axis=1, ddof=1)
    return results_table
    
                                    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('state', type=str)
    parser.add_argument('dates', type=str)
    parser.add_argument('--mode', type=str, choices=['train', 'pretrain'], default='train')
    parser.add_argument('--gpu', type=int, choices=[0, 1, 2, 3], default=0)
    parser.add_argument('--num_seeds', type=int, default=10)
    parser.add_argument('--use_pretrain', type=int, choices=[0, 1], default=1)
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()
    
    if args.mode == 'train':
        if args.state == 'all_small_states':
            train_all_states(args.dates, 'small', gpu=args.gpu)
        elif args.state == 'all_large_states':
            train_all_states(args.dates, 'large', gpu=args.gpu)
        else:
            use_pretrain = args.use_pretrain == 1
            run_experiment(args.state, args.dates, gpu=args.gpu, epochs=args.epochs, num_seeds=args.num_seeds,
                           use_pretrain=use_pretrain)
    else:
        if args.state == 'all_small_states':
            pretrain_all_small_states(args.dates, gpu=args.gpu)
        else:
            run_pretraining_experiment(args.state, args.dates, gpu=args.gpu, epochs=args.epochs)
