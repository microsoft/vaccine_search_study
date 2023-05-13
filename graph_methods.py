from constants_and_utils import *
from vaccine_intent import wants_vaccine

import argparse
import csv
from fast_pagerank import pagerank_power
from itertools import combinations
import matplotlib.pyplot as plt
import os
import pickle        
import random
from scipy.sparse import csr_matrix
from sklearn.metrics import cohen_kappa_score

                
def get_query_pairs_from_session(session_df):
    """
    Given a dataframe representing all queries from session, return all query pairs among "valid" queries.
    """
    session_queries = list(session_df.NormQuery.unique())
    if len(session_queries) > 1:
        return list(combinations(session_queries, 2))  # all unordered pairs, order within pair is alphabetical
    return [('', '')]  # return dummy pair so that we always return list of tuples


def get_consecutive_query_pairs_from_session(session_df):
    """
    Given a dataframe representing all queries from session, return consecutive (query, query) pairs.
    """
    session_queries = session_df.NormQuery.values
    query_pairs = list(zip(session_queries[:-1], session_queries[1:]))  # consecutive queries 
    kept_pairs = [(q1, q2) for q1, q2 in query_pairs if q1 != q2]
    if len(kept_pairs) > 0:
        return kept_pairs
    return [('', '')]  # return dummy pair so that we always return list of tuples
    

def get_edges_in_query_query_graph(df, qq_func=get_consecutive_query_pairs_from_session, sequential_by_day=False):
    """
    Get query-query edges in graph.
    """
    ts = time.time()
    df_dropped = df.dropna(subset=['NormQuery'])
    print('Kept %d/%d (%.2f%%) rows after dropping nan NormQuery' % (len(df_dropped), len(df), 100. * len(df_dropped) / len(df)))
    too_long = df_dropped.NormQuery.apply(lambda x:len(x) > 100)  # drop queries that are unrealistically long
    df_dropped = df_dropped[~too_long]
    print('Kept %d/%d (%.2f%%) rows after dropping queries with len > 100' % (len(df_dropped), len(df), 100. * len(df_dropped) / len(df)))
    if sequential_by_day:  # split by day so we don't run into memory errors
        df_dropped['day'] = df_dropped.Request_RequestTime.apply(lambda x: x.split()[0])
        total_counter = Counter()
        for day, day_df in df_dropped.groupby('day'):
            print(day)
            qq_pairs = day_df.groupby('SessionId').apply(qq_func).values.tolist()
            qq_pairs = np.concatenate(qq_pairs)
            to_keep = np.sum(qq_pairs != ('', ''), axis=1) > 0  # sum will be 2 for non-dummy and 0 for dummy pairs
            qq_pairs = qq_pairs[to_keep]
            qq_pairs = [tuple(t) for t in qq_pairs]  # need tuple for hashing
            total_counter += Counter(qq_pairs)
        pair_counts = total_counter.most_common()
    else:
        qq_pairs = df_dropped.groupby('SessionId').apply(qq_func).values.tolist()
        qq_pairs = np.concatenate(qq_pairs)
        to_keep = np.sum(qq_pairs != ('', ''), axis=1) > 0  # sum will be 2 for non-dummy and 0 for dummy pairs
        qq_pairs = qq_pairs[to_keep]
        qq_pairs = [tuple(t) for t in qq_pairs]  # need tuple for hashing
        pair_counts = Counter(qq_pairs).most_common()
    print('Finished getting query pairs [time=%.2fs]' % (time.time()-ts))
    return pair_counts

    
def get_query_url_pairs_for_query(query_df):
    """
    Given a dataframe representing all query instances of a query, return (query, URL) pairs 
    representing all the URLs clicked on from this query.
    """
    query = query_df.iloc[0].NormQuery
    extracted = query_df.ClickedURLs.apply(extract_url_list_from_clicked_urls).values
    urls = np.concatenate([t[1] for t in extracted])
    query_url_pairs = [(query, u) for u in urls]
    if len(query_url_pairs) > 0:
        return query_url_pairs
    return [('', '')]  # return dummy pair so that we always return list of tuples

    
def get_edges_in_query_url_graph(df):
    """
    Get query-URL edges in graph.
    """
    def drop_url(url):
        """
        Some URLs from Bing logs are meaningless, like javascript:void(0).
        """
        keywords = ['javascript:void', '/?FORM=']
        if any([k in url for k in keywords]):
            return True
        return False

    ts = time.time()
    if 'ClickUrl' in df.columns:  # one row per clicked URL and its corresponding query
        df_dropped = df.dropna(subset=['ClickUrl', 'NormQuery'])
        print('Kept %d/%d (%.2f%%) rows after dropping nan ClickUrl and NormQuery' % (len(df_dropped), len(df), 100. * len(df_dropped) / len(df)))
        bad_urls = df_dropped.ClickUrl.apply(drop_url)
        too_long = df_dropped.NormQuery.apply(lambda x:len(x) > 100)
        df_dropped = df_dropped[(~bad_urls) & (~too_long)]
        print('Kept %d/%d (%.2f%%) rows after dropping bad ClickUrl and queries with len > 100' % (len(df_dropped), len(df), 100. * len(df_dropped) / len(df)))        
        qu_pairs = list(zip(df_dropped.NormQuery, df_dropped.ClickUrl))
    else:  # original format that had list of URLs concatenated in single column and one row per query
        qu_pairs = df.groupby('NormQuery').apply(get_query_url_pairs_for_query).values.tolist()
        qu_pairs = np.concatenate(qu_pairs)  # query-url
        to_keep = np.sum(qu_pairs != ('', ''), axis=1) > 0  # sum will be 2 for non-dummy and 0 for dummy pairs
        qu_pairs = qu_pairs[to_keep]
        qu_pairs = [tuple(t) for t in qu_pairs]  # need tuple for hashing
    pair_counts = Counter(qu_pairs).most_common()
    print('Found %d query-URL pairs [time=%.2fs]' % (len(qu_pairs), time.time()-ts))
    return pair_counts


def get_top_and_bottom_quartiles_for_state(state, date_range, query_df=None, click_df=None, zcta_df=None):
    """
    Get separate files for queries/clicks coming from top and bottom quartile zip codes (by income).
    """    
    if zcta_df is None:
        zcta_df = get_zcta_df(include_lat_lon=False)
    zip2zcta = get_zip_to_zcta()
    abbr = get_state_to_abbr()[state]
    state_zcta_df = zcta_df[zcta_df.state == abbr].dropna(subset=['median_income'])
    print('Found %d ZCTAs with income data in %s (%s)' % (len(state_zcta_df), state, abbr))
    incomes = state_zcta_df.median_income.values
    quartile1 = np.percentile(incomes, 25)
    bottom_zctas = state_zcta_df[state_zcta_df.median_income < quartile1].index.values
    quartile3 = np.percentile(incomes, 75)
    top_zctas = state_zcta_df[state_zcta_df.median_income > quartile3].index.values
    
    postfix = '%s_%s' % (date_range, state.replace(' ', ''))
    if query_df is None:
        query_df = pd.read_csv(os.path.join(PATH_TO_BING_DATA, 'covid_sessions_%s.csv' % postfix))
    if click_df is None:
        click_df = pd.read_csv(os.path.join(PATH_TO_BING_DATA, 'covid_clicks_%s.csv' % postfix))
    print(state, date_range, '%d queries and %d clicks' % (len(query_df), len(click_df)))
    query_df['ZCTA_from_zip'] = query_df.DataSource_PostalCode.apply(lambda x: zip2zcta.get(x, np.nan))
    query_df = query_df.dropna(subset=['ZCTA_from_zip'])
    print('Kept %d queries with ZCTA' % len(query_df))

    bottom_queries = query_df[query_df.ZCTA_from_zip.isin(bottom_zctas)]
    bottom_session_ids = bottom_queries.SessionId.unique()
    bottom_clicks = click_df[click_df.SessionId.isin(bottom_session_ids)]
    top_queries = query_df[query_df.ZCTA_from_zip.isin(top_zctas)]
    top_session_ids = top_queries.SessionId.unique()
    top_clicks = click_df[click_df.SessionId.isin(top_session_ids)]
    print('Top quartile: %d ZCTAs, %d ZCTAs found, %d queries, %d clicks' % (
            len(top_zctas), len(top_queries.ZCTA_from_zip.unique()), len(top_queries), len(top_clicks)))    
    print('Bottom quartile: %d ZCTAs, %d ZCTAs found, %d queries, %d clicks' % (
        len(bottom_zctas), len(bottom_queries.ZCTA_from_zip.unique()), len(bottom_queries), len(bottom_clicks)))
    return top_zctas, top_queries, top_clicks, bottom_zctas, bottom_queries, bottom_clicks


def load_pair_counts_from_fn(path_to_pair_counts, pair_type='qu'):
    """
    Load saved pair counts and tag as query/URL.
    """
    assert pair_type in {'qq', 'qu'}
    with open(path_to_pair_counts, 'rb') as f:
        pair_counts = pickle.load(f)
    counter = Counter()
    # tag so that query and url nodes are kept separate, even if string is the same
    for (n1, n2), ct in pair_counts:
        n1 += '_###QUERY###'
        if pair_type == 'qu':
            n2 += '_###URL###'
        else:
            n2 += '_###QUERY###'
        counter[(n1, n2)] = ct
        # we need directed edges coming out of URLs too; otherwise random walk gets stuck
        if pair_type == 'qu':
            counter[(n2, n1)] = ct
    return counter 

    
def load_pair_counts_for_state_and_date_ranges(state, date_ranges, graph_type='qq_qu', min_deg=1):
    """
    Load pair counts for a given state and date ranges.
    Can return query-query ('qq'), query-url ('qu'), or combined ('qq_qu') graph.
    """
    assert graph_type in {'qq', 'qu', 'qq_qu'}
    assert min_deg >= 1
    # try loading directly from file
    if type(state) == str:
        possible_fn = '%s_pairs_%s_%dmonths.pkl' % (graph_type, state.replace(' ', ''), len(date_ranges))
        full_fn = os.path.join(PATH_TO_PROCESSED_DATA, possible_fn)
        if os.path.isfile(full_fn):
            print('Loading pair counts from', possible_fn)
            with open(full_fn, 'rb') as f:
                pair_counts = pickle.load(f)
                return pair_counts
    
    if type(state) == list:
        states = state
    elif state == 'All States':
        states = get_state_names(drop_dc=True)
    else:
        states = [state]
    print('Loading pair counts for %d states' % len(states))
    pair_counter = Counter()
    for s in states:
        if len(states) > 1:
            print(s)
        for date_range in date_ranges:
            postfix = '%s_%s' % (s.replace(' ', ''), date_range)
            if 'qq' in graph_type:
                pair_counts_fn = os.path.join(PATH_TO_PROCESSED_DATA, 'qq_pairs_%s.pkl' % postfix)
                pair_counter += load_pair_counts_from_fn(pair_counts_fn, pair_type='qq')
            if 'qu' in graph_type:
                pair_counts_fn = os.path.join(PATH_TO_PROCESSED_DATA, 'qu_pairs_%s.pkl' % postfix)
                pair_counter += load_pair_counts_from_fn(pair_counts_fn, pair_type='qu')
    pair_counts = pair_counter.most_common()
    if min_deg > 1:
        ts = time.time()
        weighted_deg = {}  # add in and out edges
        for (n1, n2), count in pair_counts:
            weighted_deg[n1] = count + weighted_deg.get(n1, 0)
            weighted_deg[n2] = count + weighted_deg.get(n2, 0)
        kept_pair_counts = []
        for (n1, n2), count in pair_counts:
            if weighted_deg[n1] >= min_deg and weighted_deg[n2] >= min_deg:
                kept_pair_counts.append(((n1, n2), count))
        print('Kept %d out of %d pairs (%.2f%%) where both nodes have degree >= %d [time=%.2fs]' % (
            len(kept_pair_counts), len(pair_counts), 100. * len(kept_pair_counts) / len(pair_counts), 
            min_deg, time.time()-ts))
        pair_counts = kept_pair_counts
    return pair_counts 

    
def convert_pair_counts_to_sparse_matrix(pair_counts):
    """
    Convert list of pair counts, which are in the form ((node1, node2), count), to sparse matrix representing graph.
    """
    node1 = []
    node2 = []
    edge_weights = []
    for (n1, n2), ct in pair_counts:
        node1.append(n1)
        node2.append(n2)
        edge_weights.append(ct)
    node_list = sorted(set(node1 + node2))
    N = len(node_list)
    print('num unique nodes:', N)
    print('num edges:', len(edge_weights))
    node2idx = {n:i for i,n in enumerate(node_list)}
    node1 = [node2idx[n] for n in node1]
    node2 = [node2idx[n] for n in node2]
    G = csr_matrix((1.0 * np.array(edge_weights), (node1, node2)), shape=(N, N))
    assert G.count_nonzero() == len(edge_weights)  # edge indices should've been unique
    return G, node_list


def load_qu_pairs_for_matched_users_as_bipartite_matrix(date_ranges, urls_to_keep=None, keep_url_func=None):
    """
    Load query-url graph for matched users as a bipartite matrix, so rows are queries and columns are URLs.
    If url_features is provided, only normalized URLs that match feature are kept.
    """
    print('Loading pair counts for %d months' % len(date_ranges))
    counter = Counter()
    for date_range in date_ranges:
        postfix = 'matched_users_%s' % date_range
        pair_counts_fn = os.path.join(PATH_TO_PROCESSED_DATA, 'qu_pairs_%s.pkl' % postfix)
        with open(pair_counts_fn, 'rb') as f:
            pair_counts = pickle.load(f)
        print('%s: loaded %d pair counts' % (date_range, len(pair_counts)))
        pairs, counts = zip(*pair_counts)
        queries, urls = zip(*pairs)
        temp_df = pd.DataFrame({'query': queries, 'url': urls, 'count': counts})
        temp_df['norm_url'] = temp_df.url.apply(normalize_url)
        if urls_to_keep is not None:
            to_keep = temp_df.norm_url.isin(urls_to_keep)  # only keep URLs that match url_feattures
            temp_df = temp_df[to_keep]
        elif keep_url_func is not None:
            to_keep = temp_df.norm_url.apply(keep_url_func)
            temp_df = temp_df[to_keep]
        print('Keeping %d pair counts' % len(temp_df))
        new_counts = temp_df.groupby(['query', 'norm_url'])['count'].sum()  # group common pairs after normalizing URL
        print('%d pairs counts after grouping' % len(new_counts))
        queries = new_counts.index.get_level_values(0).values
        urls = new_counts.index.get_level_values(1).values
        pairs = list(zip(queries, urls))
        counter += Counter(dict(zip(pairs, new_counts.values)))
    pair_counts = counter.most_common()
    print('Finished loading pair counts')

    pairs, edge_weights = zip(*pair_counts)
    queries, urls = zip(*pairs)
    unique_queries = sorted(set(queries))
    N_q = len(unique_queries)
    query2idx = {q:i for i,q in enumerate(unique_queries)}    
    unique_urls = sorted(set(urls))
    N_u = len(unique_urls)
    url2idx = {u:i for i,u in enumerate(unique_urls)}
    query_idx = []
    url_idx = []
    for q, u in zip(queries, urls):
        query_idx.append(query2idx[q])
        url_idx.append(url2idx[u])
    
    G = csr_matrix((1.0 * np.array(edge_weights), (query_idx, url_idx)), shape=(N_q, N_u))
    print('Num queries: %d. Num urls: %d. Num edges: %d' % (N_q, N_u, len(edge_weights)))
    assert G.count_nonzero() == len(edge_weights)
    return G, unique_queries, unique_urls

    
def run_sppr_for_state(state, date_ranges, seed_set_func=wants_vaccine, graph_type='qq_qu', 
                       pair_counts=None, weight_seed_set=False, query2count=None):
    """
    Run personalized Pagerank from seed set for graph in state.
    """
    if pair_counts is None:
        pair_counts = load_pair_counts_for_state_and_date_ranges(state, date_ranges, graph_type=graph_type)
    G, node_list = convert_pair_counts_to_sparse_matrix(pair_counts)
    seed_set_bools = []  # whether node is in seed set
    seed_set_queries = []  # just the set of seed set queries 
    for n in node_list:
        node_str = n.rsplit('_', 1)[0]
        if n.endswith('_###QUERY###') and seed_set_func(node_str):
            seed_set_bools.append(True)
            seed_set_queries.append(node_str)
        else:
            seed_set_bools.append(False)
    print('%d nodes in seed set' % len(seed_set_queries))
    
    p_vec = np.zeros(len(node_list))
    if weight_seed_set:  # weight teleportation probabilities by frequencies
        assert query2count is not None
        weights = np.array([query2count[q] if q in query2count else 1 for q in seed_set_queries])
    else:
        weights = np.ones(len(seed_set_queries))
    p_vec[seed_set_bools] = weights / np.sum(weights)
    ppr = pagerank_power(G, p=0.85, personalize=p_vec, tol=1e-6)
    ranking = get_ranking_from_scores(ppr)
    labels = np.zeros(len(node_list))
    labels[seed_set_bools] = 1.0
    results = {'node': node_list,
               'score': ppr,
               'rank': ranking,
               'label': labels,
               'p': p_vec}
    columns = list(results.keys())
    df = pd.DataFrame(results, columns=columns)
    df = df.sort_values('rank')
    return df

def skip_query_label(query, query2count, min_users):
    """
    Whether to skip a query.
    """
    if query2count.get(query, 0) < min_users:
        return True
    if len(query) < 2:
        return True 
    return False    
    
def load_nodes_to_label_for_state_and_date_range(state, date_range, graph_type='qq_qu', sppr_df=None, 
                        filter_queries=True, query2count=None, min_users=50,
                        filter_urls=True, verbose=True):
    """
    Load nodes from SPPR and filter out nodes that we don't want to label.
    """
    if sppr_df is None:
        postfix = '%s_%s' % (state.replace(' ', ''), date_range)
        sppr_fn = os.path.join(PATH_TO_PROCESSED_DATA, 'sppr_%s_%s.csv' % (graph_type, postfix))
        df = pd.read_csv(sppr_fn).sort_values('rank')
    else:
        df = sppr_df.copy()
    if filter_queries:  # drop queries without enough users
        assert query2count is not None
        queries_to_drop = df.node.apply(lambda x: x.endswith('_###QUERY###') and 
                                          skip_query_label(x.rsplit('_', 1)[0], query2count, min_users))
        if verbose:
            print('Dropping %d queries' % np.sum(queries_to_drop))
        df = df[~queries_to_drop]
    if filter_urls:  # drop URLs that are ads or internal clicks
        urls_to_drop = df.node.apply(lambda x: x.endswith('_###URL###') and 
                                     skip_url_label(x.rsplit('_', 1)[0]))
        if verbose:
            print('Dropping %d urls' % np.sum(urls_to_drop))
        df = df[~urls_to_drop]
    return df 


def get_union_of_top_unlabeled_nodes_per_state(state_names, date_range, N, state2nodes=None, 
                                               query2count=None, drop_repetitive=True, verbose=True):
    """
    Gets union over top N unlabeled nodes per state. Drops repetitive nodes if they appear.
    state2nodes, if provided, should map only to _unlabeled_ nodes in the state.
    """
    if state2nodes is None:
        print('Loading and filtering top 100K nodes for all states...')
        state2nodes = {}
        for state in state_names:
            node_df = load_nodes_to_label_for_state_and_date_range(state, date_range, query2count=query2count, verbose=False)
            state2nodes[state] = node_df[node_df.label == 0].node.values
        
    node2states = {}
    node2ranks = {}
    common_prefixes = ['www.cvs.com/store-locator/cvs-pharmacy-locations/covid-vaccine', 
                       'www.cvs.com/minuteclinic/clinic-locator/covid-19-testing',
                       'usafacts.org/visualizations/coronavirus-covid-19-spread-map/state',
                       'usafacts.org/visualizations/covid-vaccine-tracker-states/state',
                       'www.kingsoopers.com/stores/details/',
                       'www.fredmeyer.com/stores/details/',
                       'www.heb.com/heb-store/US/',
                       'corporate.walmart.com/media-library/document/covid-19-vaccine-locations-walmart-and-sams-clubs'] 
    prefix2nodes = {p:[] for p in common_prefixes}
    for state, nodes in state2nodes.items():
        for r, node in enumerate(nodes[:N]):
            if node in node2states:  # already seen
                node2states[node].append(state)
                node2ranks[node].append(r)
            else:  # new node
                node2states[node] = [state]
                node2ranks[node] = [r]
                if node.endswith('_###URL###'):
                    url = node.rsplit('_', 1)[0]
                    elems = url.split('/')
                    prefix = '/'.join(elems[2:6])  # drop https:// or http://
                    if prefix in common_prefixes:
                        prefix2nodes[prefix].append(node)
    union_size = len(node2states)
    num_url = np.sum([n.endswith('_###URL###') for n in node2states])
    if verbose:
        print('Union of top N=%d nodes has %d nodes (%.2f%% of max size) -> %.2f%% are URL' % 
          (N, union_size, 100. * union_size / (len(state2nodes) * N), 100. * num_url / union_size))
    
    if drop_repetitive:
        for prefix, nodes in prefix2nodes.items():
            if verbose and len(nodes) > 0: 
                print(prefix, len(nodes))
            if len(nodes) > 5:
                # sort by min rank, so that set for N is always subset of set for N', if N < N'
                ordered_nodes = sorted(nodes, key=lambda n:np.min(node2ranks[n]))  
                if verbose:
                    print('Keeping:', ordered_nodes[:5])
                for n in ordered_nodes[5:]:  # drop everything after first 5
                    del node2states[n]
        union_size = len(node2states)
        num_url = np.sum([n.endswith('_###URL###') for n in node2states])
        if verbose:
            print('After removing repetitive prefixes...')
            print('Union of top N=%d nodes has %d nodes (%.2f%% of max size) -> %.2f%% are URL' % 
                  (N, union_size, 100. * union_size / (len(state2nodes) * N), 100. * num_url / union_size))
    
    node_union = list(node2states.keys())
    node_val = [n.rsplit('_', 1)[0] for n in node_union]
    node_type = [n.rsplit('_', 1)[1].strip('#') for n in node_union]
    assert all([t in {'QUERY', 'URL'} for t in node_type])
    num_states = [len(node2states[n]) for n in node_union]
    associated_states = [', '.join(sorted(node2states[n])) for n in node_union]
    min_ranks = [np.min(node2ranks[n]) for n in node_union]
    mean_ranks = [np.mean(node2ranks[n]) for n in node_union]
    node_df = pd.DataFrame({'node':node_val, 'node_type':node_type, 'num_states':num_states, 
                            'associated_states':associated_states, 'min_rank':min_ranks,
                            'mean_rank':mean_ranks}, 
                           columns=['node', 'node_type', 'num_states', 'associated_states',
                                    'min_rank', 'mean_rank'])
    assert np.sum(node_df.node_type == 'URL') == num_url
    node_df['specificity'] = 1 / node_df.num_states  # temporary col for sorting
    node_df = node_df.sort_values(by=['specificity', 'associated_states', 'min_rank'])
    node_df = node_df.drop('specificity', axis='columns')
    return node_df


def get_union_of_seed_set_nodes_per_state(state_names, date_range, N, state2nodes=None, 
                                          query2count=None):
    """
    Gets union over top _and_ bottom N seed set nodes per state.
    state2nodes, if provided, should map only to _labeled_ nodes in the state.
    """
    if state2nodes is None:
        state2nodes = {}
        for state in state_names:
            node_df = load_nodes_to_label_for_state_and_date_range(state, date_range, query2count=query2count)
            state2nodes[state] = node_df[node_df.label == 1].node.values
        
    node2states = {}
    for state, nodes in state2nodes.items():
        for node in nodes[:N]:  # in top N nodes for state
            if node in node2states:  # already seen
                node2states[node].append('%s (top)' % state)
            else:  # new node
                node2states[node] = ['%s (top)' % state]
        for node in nodes[-N:]:  # in bottom N nodes for state
            if node in node2states:  # already seen
                node2states[node].append('%s (bottom)' % state)
            else:  # new node
                node2states[node] = ['%s (bottom)' % state]
    union_size = len(node2states)
    print('Union of top and bottom N=%d nodes has %d nodes (%.2f%% of max size)' % 
          (N, union_size, 100. * union_size / (len(state2nodes) * 2 * N)))
    
    node_union = list(node2states.keys())
    node_val = [n.rsplit('_', 1)[0] for n in node_union]
    node_type = [n.rsplit('_', 1)[1].strip('#') for n in node_union]
    assert all([t == 'QUERY' for t in node_type])  # seed set nodes should all be queries
    num_states = [len(node2states[n]) for n in node_union]
    associated_states = [', '.join(sorted(node2states[n])) for n in node_union]
    node_df = pd.DataFrame({'node':node_val, 'node_type':node_type, 'num_states':num_states, 
                            'associated_states':associated_states}, 
                           columns=['node', 'node_type', 'num_states', 'associated_states'])
    node_df['specificity'] = 1 / node_df.num_states  # temporary col for sorting
    node_df = node_df.sort_values(by=['specificity', 'associated_states'])
    node_df = node_df.drop('specificity', axis='columns')
    return node_df

    
def create_new_assignment_sheet(person, volume, update_assignments=True):
    """
    Creates new spreadsheet for person.
    """
    fn = os.path.join(PATH_TO_ANNOTATIONS, 'pilot_assignments.csv')
    ann_df = pd.read_csv(fn)
    unlabeled = ann_df.annotator == 'NOT ASSIGNED'
    indices_to_label = np.arange(len(ann_df))[unlabeled][:volume]
    print('Assigning %s %d samples between index %d and %d' % (person, volume, 
                                             indices_to_label[0], indices_to_label[-1]))
    if update_assignments:
        annotators = ann_df.annotator.values
        annotators[indices_to_label] = person
        ann_df['annotator'] = annotators  # save with new assignments added
        ann_df.to_csv(fn, index=False)
 
    new_df = ann_df.iloc[indices_to_label][['string', 'type']]
    new_columns = ['wants_vaccine', 'vaccine_safety', 'vaccine_eligibility', 'vaccine_mandates',
                   'covid_testing', 'covid_stats', 'covid_symptoms', 'other']
    for c in new_columns:
        new_df[c] = ''
    options_df = pd.DataFrame({'options':['[1] Highly likely', '[2] Likely', '[3] Ambiguous', '[4] Unlikely', '[5] Not sure']})
    
    fn = os.path.join(PATH_TO_ANNOTATIONS, '%s_annotations.xlsx' % person)
    with pd.ExcelWriter(fn, engine='xlsxwriter') as writer:
        new_df.to_excel(writer, sheet_name='data', index=False)
        options_df.to_excel(writer, sheet_name='options', index=False) 


def get_vaccine_intent_labels_from_AMT_results(amt_df, positive_threshold=2, negative_threshold=2,
                                               positive_set=['likely', 'highly-likely'],
                                               negative_set=['ambiguous', 'unlikely']):
    """
    Gets list of nodes and their vaccine intent labels \in {0, 1, np.nan}.
    """
    def get_vaccine_intent_label_for_node(subdf):
        positive_total = subdf[positive_set].sum().sum()
        negative_total = subdf[negative_set].sum().sum()
        if positive_total >= positive_threshold:
            return 1
        if negative_total >= negative_threshold:
            return 0
        return np.nan
    
    assert len(set(positive_set).intersection(set(negative_set))) == 0
    positive_set = ['Answer.wants-vaccine.%s' % a for a in positive_set]
    negative_set = ['Answer.wants-vaccine.%s' % a for a in negative_set]
    labels = amt_df.groupby('Input.node').apply(get_vaccine_intent_label_for_node)
    return labels.index, labels.values


def get_other_intent_labels_from_AMT_results(amt_df, threshold=3):
    """
    Gets list of nodes and their list of labels \in {0, 1} per column.
    """
    def get_other_intent_labels_for_node(subdf):
        sums = subdf[amt_column_names].sum()
        labels = (sums >= threshold).astype(int)
        return labels
    
    other_intents = ['covid-stats', 'covid-symptoms', 'covid-tests', 
                     'vaccine-eligibility', 'vaccine-incentive', 'vaccine-reqs', 
                     'vaccine-safety']
    amt_column_names = ['Answer.%s.on' % s for s in other_intents]
    labels = amt_df.groupby('Input.node').apply(get_other_intent_labels_for_node)
    label_df = pd.DataFrame(labels.values, columns=other_intents)
    label_df['node'] = labels.index
    label_df = label_df.set_index('node')
    return label_df


def get_interannotator_agreement(amt_df, binarize, num_annotators=3):
    """
    Get percent agreement and Cohen's kappa scores.
    """
    def get_annotations(subdf):
        assert len(subdf) == num_annotators
        responses = subdf[amt_column_names].values
        sums = np.sum(responses, axis=1)
        assert all(sums == 1)
        responses_idx = [np.argmax(x) for x in responses]
        return tuple(responses_idx)
    
    if binarize:
        amt_df['positive'] = amt_df['Answer.wants-vaccine.likely'] | amt_df['Answer.wants-vaccine.highly-likely']
        amt_df['negative'] = ~amt_df.positive
        amt_column_names = ['positive', 'negative']
    else:
        amt_column_names = [col for col in amt_df.columns if col.startswith('Answer.wants-vaccine')]
        assert len(amt_column_names) == 5
    annotations = amt_df.groupby('Input.node').apply(get_annotations)
    annotations = np.asarray(list(annotations.values))  # convert to N x num_annotators matrix
    num_annotators = annotations.shape[1]
    for i in np.arange(num_annotators):
        if i < num_annotators:
            for j in np.arange(i+1, num_annotators):
                x = annotations[:, i]
                y = annotations[:, j]
                prop_agree = np.mean(x == y)
                kappa = cohen_kappa_score(annotations[:, i], annotations[:, j])
                print('%d vs %d: prop agree = %.4f, Cohen\'s kappa = %.4f' % (i, j, prop_agree, kappa))

    
def save_graphs_for_all_states(dates, graph_type):
    """
    Outer call that issues internal save graph command for all states.
    """
    state_names = get_state_names()
    for state in state_names:
        cmd = 'nohup python -u graph_methods.py save_graph_for_state %s %s --state %s > ./logs/%s_%s_%s_graph.out 2>&1 &' % (
            dates, graph_type, state.replace(' ', '\ '), state.replace(' ', ''), dates, graph_type)
        print('Command: %s' % cmd)
        os.system(cmd)
        time.sleep(1)

    
def save_qu_graph_for_state(state, date_range):
    """
    Computes and saves query-url graph for state + date range.
    """
    fn = os.path.join(PATH_TO_BING_DATA, 'covid_clicks_%s_%s.csv' % (date_range, state.replace(' ', '')))
    state_df = pd.read_csv(fn)       
    print(state, 'num rows:', len(state_df))  
    qu_pairs = get_edges_in_query_url_graph(state_df)
    print(qu_pairs[:20])
    out_fn = os.path.join(PATH_TO_PROCESSED_DATA, 
                 'qu_pairs_%s_%s.pkl' % (state.replace(' ', ''), date_range)) 
    with open(out_fn, 'wb') as f:
        pickle.dump(qu_pairs, f)
        
        
def save_qu_graph_for_matched_users(date_range):
    """
    Computes and saves query-url graph for holdouts/nonholdouts + date range.
    """
    fn = os.path.join(PATH_TO_BING_DATA, 'holdout_clicks_%s.tsv' % date_range)
    ho_clicks_df = pd.read_csv(fn, delimiter='\t', header=0, quoting=csv.QUOTE_NONE, on_bad_lines='skip')
    ho_clicks_df = ho_clicks_df.dropna(subset=['ClientId', 'ClickUrl'])
    print('Loaded holdout clicks:', len(ho_clicks_df))
    fn = os.path.join(PATH_TO_BING_DATA, 'nonholdout_clicks_%s.tsv' % date_range)
    nh_clicks_df = pd.read_csv(fn, delimiter='\t', header=0, quoting=csv.QUOTE_NONE, on_bad_lines='skip')
    nh_clicks_df = nh_clicks_df.dropna(subset=['ClientId', 'ClickUrl'])
    print('Loaded nonholdout clicks:', len(nh_clicks_df))
    clicks_df = pd.concat([ho_clicks_df, nh_clicks_df])

    fn = os.path.join(PATH_TO_BING_DATA, 'holdout_queries_%s.tsv' % date_range)
    ho_queries_df = pd.read_csv(fn, delimiter='\t', header=0, quoting=csv.QUOTE_NONE, on_bad_lines='skip')
    print('Loaded holdout queries:', len(ho_queries_df))
    fn = os.path.join(PATH_TO_BING_DATA, 'nonholdout_queries_%s.tsv' % date_range)
    nh_queries_df = pd.read_csv(fn, delimiter='\t', header=0, quoting=csv.QUOTE_NONE, on_bad_lines='skip')
    print('Loaded nonholdout queries:', len(nh_queries_df))
    queries_df = pd.concat([ho_queries_df, nh_queries_df])
    
    clicks_df = clicks_df.merge(queries_df[['ClientId', 'ImpressionGuid', 'NormQuery']], how='left', 
                                left_on='ImpressionGuid', right_on='ImpressionGuid')
    assert clicks_df.ClientId_y.isnull().sum() == 0  # all clicks should have matching row from queries
    qu_pairs = get_edges_in_query_url_graph(clicks_df)
    print(qu_pairs[:20])
    out_fn = os.path.join(PATH_TO_PROCESSED_DATA, 'qu_pairs_matched_users_%s.pkl' % date_range)
    with open(out_fn, 'wb') as f:
        pickle.dump(qu_pairs, f)
    
    
def save_qq_graph_for_state(state, date_range):
    """
    Computes and saves query-query graph for state + date range.
    """
    fn = os.path.join(PATH_TO_BING_DATA, 'covid_sessions_%s_%s.csv' % (date_range, state.replace(' ', '')))
    state_df = pd.read_csv(fn)
    print(state, 'num rows:', len(state_df))  
    qq_pairs = get_edges_in_query_query_graph(state_df)
    print(qq_pairs[:20])
    out_fn = os.path.join(PATH_TO_PROCESSED_DATA, 
                 'qq_pairs_%s_%s.pkl' % (state.replace(' ', ''), date_range))
    with open(out_fn, 'wb') as f:
        pickle.dump(qq_pairs, f)


def save_qu_graph_for_states_combined(date_range):
    """
    Save combined query-url graph from all states.
    """
    print('Computing and saving combined query-url graph over all states for %s' % date_range)
    state_names = get_state_names()
    pair_counter = Counter()
    for state in state_names:
        print(state)
        postfix = '%s_%s' % (state.replace(' ', ''), date_range)
        pair_counts_fn = os.path.join(PATH_TO_PROCESSED_DATA, 'qu_pairs_%s.pkl' % postfix)
        node1, node2, edge_weights = load_pair_counts_from_fn(pair_counts_fn, tag_queries_and_urls=False)
        pair_counter += Counter({(n1, n2):e for n1, n2, e in zip(node1, node2, edge_weights)})
    pair_counts = pair_counter.most_common()
    out_fn = os.path.join(PATH_TO_PROCESSED_DATA, 'qu_pairs_all_states_%s.pkl' % date_range)
    with open(out_fn, 'wb') as f:
        pickle.dump(pair_counts, f)
    
    
def save_qq_graph_for_states_combined(date_range):
    """
    Save combined query-query graph from all states.
    """
    print('Computing and saving combined query-query graph over all states for %s' % date_range)
    state_names = get_state_names()
    pair_counter = Counter()
    for state in state_names:
        print(state)
        postfix = '%s_%s' % (state.replace(' ', ''), date_range)
        pair_counts_fn = os.path.join(PATH_TO_PROCESSED_DATA, 'qq_pairs_%s.pkl' % postfix)
        node1, node2, edge_weights = load_pair_counts_from_fn(pair_counts_fn, tag_queries_and_urls=False)
        pair_counter += Counter({(n1, n2):e for n1, n2, e in zip(node1, node2, edge_weights)})
    pair_counts = pair_counter.most_common()
    out_fn = os.path.join(PATH_TO_PROCESSED_DATA, 'qq_pairs_all_states_%s.pkl' % date_range)
    with open(out_fn, 'wb') as f:
        pickle.dump(pair_counts, f)
        
        
def save_graphs_for_state_quartiles(state, date_range):
    """
    Computes and saves query-query and query-url graphs for state's top and bottom quartiles (by zip code income), for date range.
    """
    _, top_queries, top_clicks, _, bottom_queries, bottom_clicks = get_top_and_bottom_quartiles_for_state(
                                                                        state, date_range)
    qq_pairs = get_edges_in_query_query_graph(top_queries)
    out_fn = os.path.join(PATH_TO_PROCESSED_DATA, 
                 'qq_pairs_%s_top_income_%s.pkl' % (state.replace(' ', ''), date_range)) 
    with open(out_fn, 'wb') as f:
        pickle.dump(qq_pairs, f)    
    qq_pairs = get_edges_in_query_query_graph(bottom_queries)
    out_fn = os.path.join(PATH_TO_PROCESSED_DATA, 
                 'qq_pairs_%s_bottom_income_%s.pkl' % (state.replace(' ', ''), date_range)) 
    with open(out_fn, 'wb') as f:
        pickle.dump(qq_pairs, f)  
        
    qu_pairs = get_edges_in_query_url_graph(top_clicks)
    out_fn = os.path.join(PATH_TO_PROCESSED_DATA, 
                 'qu_pairs_%s_top_income_%s.pkl' % (state.replace(' ', ''), date_range)) 
    with open(out_fn, 'wb') as f:
        pickle.dump(qu_pairs, f)
    qu_pairs = get_edges_in_query_url_graph(bottom_clicks)
    out_fn = os.path.join(PATH_TO_PROCESSED_DATA, 
                 'qu_pairs_%s_bottom_income_%s.pkl' % (state.replace(' ', ''), date_range)) 
    with open(out_fn, 'wb') as f:
        pickle.dump(qu_pairs, f)  
            
            
def save_sppr_for_all_states(dates, graph_type):
    """
    Outer call that issues internal SPPR command for all states.
    """
    state_names = get_state_names()
    for state in state_names:
        cmd = 'nohup python -u graph_methods.py save_sppr_for_state %s %s --state %s > ./logs/%s_sppr.out 2>&1 &' % (
            dates, graph_type, state.replace(' ', '\ '), state.replace(' ', ''))
        print('Command: %s' % cmd)
        os.system(cmd)
        time.sleep(1)

        
def save_sppr_for_state(state, dates, graph_type):
    """
    Executable that saves S-PPR results for a state.
    """
    if dates == '3months':
        date_ranges = ['2021-04-01_2021-04-30', '2021-06-15_2021-07-14', '2021-09-01_2021-09-30']
    elif dates == '2months':
        date_ranges = ['2021-04-01_2021-04-30', '2021-08-01_2021-08-31']
    else:
        date_ranges = [date_range]
    print('Running S-PPR for %s on %s graph with date ranges:' % (state, graph_type), date_ranges)
    sppr_df = run_sppr_for_state(state, date_ranges, graph_type=graph_type, weight_seed_set=False)
    print(sppr_df.head(30))
    sppr_df = sppr_df.iloc[:100000]  # save up to top 100K nodes
    out_fn = os.path.join(PATH_TO_PROCESSED_DATA, 
               'sppr_%s_%s_%s.csv' % (graph_type, state.replace(' ', ''), date_range))
    sppr_df.to_csv(out_fn, index=False)
    
    
def get_state_graph_stats(state):
    """
    Return number of nodes, number of edges, and number of seed queries in state graph.
    """
    sppr_log = './logs/%s_sppr.out' % state.replace(' ', '')
    with open(sppr_log, 'r') as f:
        lines = f.readlines()
    stats = {}
    for line in lines:
        if line.startswith('num unique nodes'):
            stats['num_nodes'] = int(line.strip().split()[-1])
        elif line.startswith('num edges'):
            stats['num_edges'] = int(line.strip().split()[-1])
        elif line.strip().endswith('nodes in seed set'):
            stats['num_seed_queries'] = int(line.strip().split()[0])
        if len(stats) == 3:
            break
    return stats

    
def get_state_graph_size(state):
    """
    Return number of nodes in graph.
    """
    stats = get_state_graph_stats(state)
    return stats['num_nodes']


def get_url_indeg(state, date_ranges):
    """
    Get indegree per URL in state graph (summing over query->URL edges).
    """
    url_deg = {}
    for date_range in date_ranges:
        postfix = '%s_%s' % (state.replace(' ', ''), date_range)
        pair_counts_fn = os.path.join(PATH_TO_PROCESSED_DATA, 'qu_pairs_%s.pkl' % postfix)
        with open(pair_counts_fn, 'rb') as f:
            pair_counts = pickle.load(f)
        for (q, u), ct in pair_counts:
            url_deg[u] = url_deg.get(u, 0) + ct
    return url_deg
                    
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('program', type=str, choices=['save_graph_for_state', 
                                                      'save_graph_for_matched_users', 
                                                      'save_graphs_for_all_states', 
                                                      'save_graphs_for_state_quartiles', 
                                                      'save_graph_for_states_combined',
                                                      'save_sppr_for_state', 
                                                      'save_sppr_for_all_states'])
    parser.add_argument('dates', type=str, choices=['04', '05', '06', '07', '08', '09', '2months', '3months'])
    parser.add_argument('graph_type', type=str, choices=['qq', 'qu', 'qq_qu'])
    parser.add_argument('--state', type=str, default='')
    args = parser.parse_args()
    
    if args.dates.endswith('months'):
        date_range = args.dates
    else:
        month2range = get_month_to_date_range()
        date_range = month2range[args.dates]  # convert month to full date range
        
    if args.program.startswith('save_graph'):
        assert 'months' not in args.dates  # we save separate graph per month
        if args.program == 'save_graph_for_state':
            assert args.graph_type != 'qq_qu'
            if args.graph_type == 'qu':
                save_qu_graph_for_state(args.state, date_range)
            else:
                save_qq_graph_for_state(args.state, date_range)
        elif args.program == 'save_graph_for_matched_users':
            assert args.graph_type == 'qu'  # we only use qu graph for matched user analysis
            save_qu_graph_for_matched_users(date_range)
        elif args.program == 'save_graphs_for_all_states':
            save_graphs_for_all_states(args.dates, args.graph_type)
        elif args.program == 'save_graphs_for_state_quartiles':
            assert args.graph_type == 'qq_qu'  # we don't save separate qq and qu graphs for quartiles
            save_graphs_for_state_quartiles(args.state, date_range)
        elif args.program == 'save_graph_for_states_combined':
            assert args.graph_type != 'qq_qu'
            if args.graph_type == 'qu':
                save_qu_graph_for_states_combined(date_range)
            else:
                save_qq_graph_for_states_combined(date_range)
    elif args.program.startswith('save_sppr'):        
        if args.program == 'save_sppr_for_state':
            save_sppr_for_state(args.state, date_range, args.graph_type)
        elif args.program == 'save_sppr_for_all_states':
            save_sppr_for_all_states(args.dates, args.graph_type)