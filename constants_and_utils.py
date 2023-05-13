import argparse
from collections import Counter
import csv
import datetime
import geopandas as gpd
import numpy as np
import os 
import pandas as pd
import time
from scipy.stats import pearsonr
from sklearn.metrics import auc, roc_auc_score, roc_curve, precision_recall_curve
import time


PATH_TO_BING_DATA = '../data/bing_data'
PATH_TO_PROCESSED_DATA = '../data/processed_bing_data'
PATH_TO_ANNOTATIONS = '../data/node_annotation'
PATH_TO_RESULTS = '../data/results'

DATE_RANGES = ['2020-12-01_2020-12-31', '2021-01-01_2021-01-31', '2021-02-01_2021-02-28', 
               '2021-03-01_2021-03-31', '2021-04-01_2021-04-30', '2021-05-01_2021-05-31', 
               '2021-06-01_2021-06-30', '2021-07-01_2021-07-31', '2021-08-01_2021-08-31']
LARGE_STATE_CUTOFF = 5e6
STATES_TO_COMPARE = ['Wyoming', 'Alaska', 'Delaware', 'Montana', 'Connecticut', 'Tennessee']


# from https://www.statista.com/statistics/381569/leading-news-and-media-sites-usa-by-share-of-visits/
# removed finance.yahoo.com and people.com
NEWS_DOMAINS = ['cnn.com', 'msn.com', 'foxnews.com', 'nytimes.com', 'news.google.com',
                'washingtonpost.com', 'nypost.com', 'cnbc.com',
                'news.yahoo.com', 'dailymail.co.uk', 'bbc.com', 'bbc.co.uk',
                'usatoday.com', 'nbcnews.com', 'theguardian.com',
                'businessinsider.com', 'wsj.com', 'forbes.com', 'huffpost.com']
# https://www.pewresearch.org/journalism/2021/01/12/news-use-across-social-media-platforms-in-2020
SOCIAL_MEDIA_DOMAINS = ['facebook.com', 'twitter.com', 'reddit.com', 'linkedin.com',
                        'whatsapp.com', 'twitch.tv', 'youtube.com', 'instagram.com',
                        'snapchat.com', 'tiktok.com', 'tumblr.com']

DP05_COLS = {'DP05_0001E': 'total_pop',  # Estimate!!SEX AND AGE!!Total population
             'DP05_0086E': 'housing_units',
             'DP05_0002PE': 'percent_male',  
             'DP05_0003PE': 'percent_female',
             'DP05_0005PE': 'percent_under_5',
             'DP05_0006PE': 'percent_5_to_9',
             'DP05_0007PE': 'percent_10_to_14',
             'DP05_0008PE': 'percent_15_to_19',
             'DP05_0009PE': 'percent_20_to_24',
             'DP05_0010PE': 'percent_25_to_34',
             'DP05_0011PE': 'percent_35_to_44',
             'DP05_0012PE': 'percent_45_to_54',
             'DP05_0013PE': 'percent_55_to_59',
             'DP05_0014PE': 'percent_60_to_64',
             'DP05_0015PE': 'percent_65_to_74',
             'DP05_0016PE': 'percent_75_to_84',
             'DP05_0017PE': 'percent_85_and_over',
             'DP05_0019PE': 'percent_under_18',
             'DP05_0024PE': 'percent_65_and_over',
             'DP05_0037PE': 'percent_white',
             'DP05_0038PE': 'percent_black',
             'DP05_0039PE': 'percent_american_indian',
             'DP05_0044PE': 'percent_asian',
             'DP05_0071PE': 'percent_hispanic'}

SUBCAT_KEY_TO_NAME = {
    'S-Normal': 'Normal side effects', 
    'S-Severe': 'Severe side effects', 
    'S-Reproductive': 'Reproductive health', 
    'S-Deaths': 'Vaccine-caused deaths', 
    'S-Eerie': 'Eerie fears',
    'S-Development': 'Vaccine development', 
    'S-Approval': 'FDA approval',
    'E-Efficacy': 'Efficacy from studies',
    'E-Variants': 'Efficacy against variants',
    'E-Breakthrough': 'Breakthrough cases', 
    'E-Natural': 'Natural immunity', 
    'C-Stats': 'Vaccine rates', 
    'C-Hesitancy': 'News about hesitancy', 
    'C-ExpertAnti': 'Expert anti-vax', 
    'C-FigureAnti': 'High-profile anti-vax', 
    'C-Religious': 'Religious concerns', 
    'A-Locations': 'Locations',
    'A-Children': 'Children',
    'A-Boosters': 'Boosters', 
    'I-Decision': 'Decision-making', 
    'I-Comparison': 'Comparing vaccines', 
    'I-Moderna': 'Info about Moderna',
    'I-Pfizer': 'Info about Pfizer', 
    'I-J&J': 'Info about J&J', 
    'I-Special': 'Special populations', 
    'I-PostVax': 'Post-vax guidelines', 
    'R-Travel': 'Travel restrictions',
    'R-Employment': 'Employee mandates', 
    'R-Proof': 'Vaccine proof', 
    'R-Exemption': 'Exemption', 
    'R-FakeProof': 'Fake vaccine proof',
    'R-AntiMandate': 'Anti-mandate', 
    'N-Incentives': 'Vaccine incentives' 
}

ABBR2CAT = {'S': 'Safety', 'E': 'Effectiveness', 'C': 'Community',
            'A': 'Availability', 'I': 'Information', 'R': 'Requirements',
            'N': 'Incentives', 'O': 'Other'}

##########################################################################
# Utility functions that are not specific to data source
##########################################################################
def get_new_counts_from_cumulative(ts):
    """
    Convert cumulative counts to new counts. First "new" count is the second nonzero entry minus the first. 
    For the first entry, we don't know if it is new or cumulative, so we drop it. Everything before
    second nonzero entry is nan.
    """
    indices = np.arange(len(ts))
    nonzero = ts > 0
    if np.sum(nonzero) == 0:
        return ts  # all zeros
    first_nnz = indices[nonzero][0]
    new_ts = np.ones(len(ts)) * np.nan
    new_ts[first_nnz+1:] = ts[first_nnz+1:] - ts[first_nnz:-1]
    return new_ts

def smooth_new_counts(ts, interval=7):
    """
    Smooth new counts by taking average in interval leading up to t.
    """
    trend = np.ones(len(ts)) * np.nan 
    for i in range(interval-1, len(ts)):
        trend[i] = np.mean(ts[i-interval+1:i+1])
    return trend 

def get_log_odds_ratio(list1, list2, denom1=None, denom2=None):
    """
    Compute log odds ratios between two lists.
    """
    counts1 = Counter(list1)
    if denom1 is None:
        denom1 = len(list1)
    counts2 = Counter(list2)
    if denom2 is None:
        denom2 = len(list2)
        
    all_vals = set(list1).union(set(list2))
    results = []
    for v in all_vals:
        ct1 = counts1[v] if v in counts1 else 0
        freq1 = min(0.999, max(ct1 / denom1, 1 / denom1))
        odds1 = freq1 / (1 - freq1)
        ct2 = counts2[v] if v in counts2 else 0
        freq2 = min(0.999, max(ct2 / denom2, 1 / denom2))
        odds2 = freq2 / (1 - freq2)
        log_or = np.log(odds1 / odds2)
        results.append({'value': v,
                        'left_count': ct1,
                        'left_freq': freq1,
                        'right_count': ct2,
                        'right_freq': freq2,
                        'log_or': log_or})
    cols = list(results[-1].keys())
    df = pd.DataFrame(results, columns=cols).sort_values(by='log_or', ascending=False)
    return df

def logistic_func(x):
    """
    Standard logistic function.
    """
    return 1 / (1 + np.exp(-x))
    
def get_ranking_from_scores(scores, to_keep=None):
    """
    Given scores, returns each entry i's ranking.
    """
    if to_keep is not None:
        assert len(to_keep) == len(scores)
        scores[~to_keep] = min(scores) - 1
    order = np.argsort(-scores)
    idx2ranking = {i:r for r,i in enumerate(order)}
    ranking = np.array([idx2ranking[i] for i in range(len(scores))])
    return ranking

def get_pearsonr_with_95_cis(x, y, w=None):
    """
    Return Pearson r with confidence intervals.
    See Wiki page for weighted correlation: https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
    CIs from https://faculty.washington.edu/gloftus/P317-318/Useful_Information/r_to_z/PearsonrCIs.pdf
    """
    def weighted_cov(x, y, w):
        x_mean = np.average(x, weights=w)
        y_mean = np.average(y, weights=w)
        num = np.sum(w * (x-x_mean) * (y-y_mean))
        denom = np.sum(w)
        return num / denom
        
    def weighted_pearsonr(x, y, w):
        num = weighted_cov(x, y, w)
        denom = np.sqrt(weighted_cov(x, x, w) * weighted_cov(y, y, w))
        return num / denom
        
    def get_r_to_z(r):
        return 0.5 * np.log((1+r) / (1-r))
    
    def get_z_to_r(z):
        num = np.exp(2*z) - 1
        denom = np.exp(2*z) + 1
        return num / denom
    
    assert len(x) == len(y)
    n = len(x)
    if n <= 3:
        return np.nan, np.nan, np.nan
    if w is None:
        r, p = pearsonr(x, y)
    else:
        assert len(w) == n, 'Weights length %d does not match expected n=%d' % (len(w), n)
        r = weighted_pearsonr(x, y, w)
    z = get_r_to_z(r)
    std = np.sqrt(1 / (n-3))
    z_lower = z - (std * 1.96)
    z_upper = z + (std * 1.96)
    r_lower = get_z_to_r(z_lower)
    r_upper = get_z_to_r(z_upper)
    return r, r_lower, r_upper

def get_cis_for_risk_ratio(p1, N1, p0, N0):
    """
    Return risk ratio with confidence intervals.
    See https://www.sjsu.edu/faculty/gerstman/StatPrimer/rr.pdf
    """
    rr = p1 / p0
    log_rr = np.log(rr)
    se_left = (1 - p1) / (N1 * p1)
    se_right = (1 - p0) / (N0 * p0)
    se = np.sqrt(se_left + se_right)
    log_lower = log_rr - (1.96 * se)
    log_upper = log_rr + (1.96 * se)
    lower = np.exp(log_lower)
    upper = np.exp(log_upper)
    return rr, lower, upper

def get_datetimes_in_range(start, end):
    """
    Get list of datetimes from start to end dates, inclusive.
    """
    assert start <= end
    dts = []
    curr = start
    while curr <= end:
        dts.append(curr)
        curr = curr + datetime.timedelta(days=1)
    return np.array(dts)

def convert_idf_to_df(idf_scores):
    """
    Convert IDF from TF-IDF to document frequency.
    See here for how IDF is computed:
    https://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting
    """
    df_scores = idf_scores - 1
    df_scores = np.exp(df_scores)
    df_scores = 1 / df_scores
    return df_scores

def make_curves_and_measure_auc(preds, labels, thresholds=None, mode='roc'):
    """
    Make the ROC or PRC curve and measure AUC.
    """
    assert type(labels[0]) == np.bool_
    assert mode in {'roc', 'prc'}
    if thresholds is None:
        labels = labels.astype(int)
        if mode == 'roc':
            x, y, cutoffs = roc_curve(labels, preds)
            area_under_curve = roc_auc_score(labels, preds)
        else:
            x, y, cutoffs = precision_recall_curve(labels, preds)
            area_under_curve = auc(x, y)
    else:  # use custom thresholds as percentiles
        assert all(thresholds >= 0) and all(thresholds <= 100)
        cutoffs = []
        x = []
        y = [] 
        for t in thresholds:
            cutoff = np.percentile(preds, t)
            cutoffs.append(cutoff)
            preds_t = preds >= cutoff  # predicted positive at this threshold
            if mode == 'roc':
                x_t = np.sum((~labels) & preds_t) / np.sum(~labels)  # false positive rate: FP/(TN + FP)
                y_t = np.sum(labels & preds_t) / np.sum(labels)  # true positive rate/recall: TP / (TP+FN) 
            else:
                x_t = np.sum(labels & preds_t) / np.sum(labels)  # true positive rate/recall: TP / (TP+FN) 
                y_t = np.sum(labels & preds_t) / np.sum(preds_t)  # precision: TP / (TP+FP)
            x.append(x_t)
            y.append(y_t)
        order = np.argsort(x)
        cutoffs = np.array(cutoffs)[order]
        x = np.array(x)[order]
        y = np.array(y)[order]
        area_under_curve = auc(x, y)  # auc requires x to be monotonic
    return cutoffs, y, x, area_under_curve 

def get_month_to_date_range():
    """
    Create mapping from month to date_range.
    """
    month2range = {}
    for date_range in DATE_RANGES:
        month = date_range.split('-')[1]
        month2range[month] = date_range
    return month2range


##########################################################################
# Utility functions that are mostly used with Bing logs
##########################################################################
def extract_datetime(string, with_hour=False):
    """
    Extract datetime from RequestTime string.
    """
    date_str, time, am_pm = string.split()
    dt = datetime.datetime.strptime(date_str, '%m/%d/%Y')
    if with_hour:
        hour = int(time.split(':')[0])
        if am_pm == 'PM' and hour <= 11:
            hour += 12
        elif am_pm == 'AM' and hour == 12:
            hour = 0
        dt = datetime.datetime(year=dt.year, month=dt.month, day=dt.day, hour=hour)
    return dt

def extract_url_list_from_clicked_urls(clicked_urls):
    """
    Parse clicked URLs string and return list of URLs and their corresponding ranks.
    This is from the old way of storing clicked URLs.
    """
    if type(clicked_urls) == str:
        urls = clicked_urls.split('###SEP###')
        ranks = []
        cleaned_urls = []
        for u in urls:
            if len(u) > 2:  # rank, '.', URL
                r, u = u.split('.', 1) 
                ranks.append(int(r))
                cleaned_urls.append(u)
        return ranks, cleaned_urls
    return [], []

def normalize_url(url):
    """
    Normalize URL.
    """
    url = url.lower().strip()
    if url.startswith('https'):  # so we don't treat http and https differently
        url = 'http' + url[len('https'):]
    url = url.rstrip('/')
    return url

def get_domain_from_http_url(url):
    """
    Get domain from URL that starts with http:.
    """
    assert url.startswith('http:')
    url = url[len('http:'):].strip('/')
    domain = url.split('/')[0]
    if domain.startswith('www.'):
        domain = domain[len('www.'):]
    return domain

def skip_url_label(url):
    """
    Whether to skip URL. Returns True if URL is internal click (does not start with http)
    and/or an ad.
    """
    if not url.startswith('http'):
        return True
    if 'bing.com/aclk' in url:  # ad
        return True
    return False

def get_single_val(df, groupby_col, val_col):
    """
    Returns groupby_col mapped to single value in val_col, for only the groupby_col 
    values that have a unique, non-nan value in val_col.
    """
    kept_df = df.dropna(subset=[groupby_col, val_col])
    n_vals = kept_df.groupby(groupby_col)[val_col].nunique()
    groups_kept = n_vals[n_vals.values == 1].index.values
    vals = kept_df[kept_df[groupby_col].isin(groups_kept)].groupby(groupby_col)[val_col].min()
    return vals

def get_query_to_count(fn='query_counts_3months', col='UserCount'):
    """
    Get mapping of query to count, default is user count.
    """
    tsv_fn = os.path.join(PATH_TO_BING_DATA, '%s.tsv' % fn)
    df = pd.read_csv(tsv_fn, delimiter='\t', header=0, quoting=csv.QUOTE_NONE, on_bad_lines='skip')
    query2count = dict(zip(df.NormQuery, df[col]))
    return query2count

def get_zip_to_count(fn='zip_code_counts_over50', col='UserCount'):
    """
    Get mapping of zip to count, default is user count.
    """
    tsv_fn = os.path.join(PATH_TO_BING_DATA, '%s.tsv' % fn)
    df = pd.read_csv(tsv_fn, delimiter='\t', header=0, quoting=csv.QUOTE_NONE, on_bad_lines='skip')
    df['zip_code'] = pd.to_numeric(df.ReverseIpPostalCode, errors='coerce')
    has_multiple_rows = np.sum(df.groupby('zip_code').size() > 1)
    if has_multiple_rows > 0:
        print('Warning: %d zip codes have more than one row' % has_multiple_rows)
    df = df.sort_values(by=['zip_code', col])
    df = df.drop_duplicates(subset=['zip_code'], keep='last')  # keep row with highest count
    print('Loaded zip code to count; found %d zip codes' % len(df))
    return dict(zip(df.zip_code, df[col]))
    

def split_tsv_file_into_states(fn, state_col='DataSource_State', save_other=False):
    """
    Used for preprocessing. Splits a large tsv file into one per state, so that in future 
    state-specific processing, only the state needs to be loaded (enabling parallelization
    across states too). 
    """
    tsv_fn = os.path.join(PATH_TO_BING_DATA, '%s.tsv' % fn)
    print('Loading %s...' % tsv_fn)
    ts = time.time()
    df = pd.read_csv(tsv_fn, delimiter='\t', header=0, quoting=csv.QUOTE_NONE, on_bad_lines='skip')
    time_to_load = time.time()-ts
    print('Took %.2f seconds to load %d rows (%.4f row/ms) to load' % (time_to_load, len(df), 1000. * time_to_load / len(df)))
    
    total_rows = 0
    other_dfs = []
    state_names = get_state_names()
    for state, state_df in df.groupby(state_col, dropna=False):
        if state in state_names:
            print(state, len(state_df))
            total_rows += len(state_df)
            state_fn = os.path.join(PATH_TO_BING_DATA, '%s_%s.csv' % (fn, state.replace(' ', '')))
            state_df.to_csv(state_fn, index=False)
        else:
            other_dfs.append(state_df)
    other_df = pd.concat(other_dfs)
    print('Other', len(other_df))
    total_rows += len(other_df)
    assert total_rows == len(df)
    if save_other:
        other_fn = os.path.join(PATH_TO_BING_DATA, '%s_Other.csv' % fn)
        other_df.to_csv(other_fn, index=False)
        
        
def map_user_to_single_val(df, val_col, user_col='ClientId', thresh=0.5):
    """
    Maps user to their most common, non-nan value in val_col *if* that value accounts for at least the threshold 
    proportion of non-nan values in val_col for that user.
    """
    kept_df = df.dropna(subset=[user_col, val_col])
    print('Kept %.2f%% rows after dropping NA in %s or %s' % (100. * len(kept_df)/len(df), user_col, val_col))
    pairs = list(zip(kept_df[user_col].values, kept_df[val_col].values))
    counter = Counter(pairs)
    tuples = counter.most_common()
    users = [t[0][0] for t in tuples]
    vals = [t[0][1] for t in tuples]
    counts = [t[1] for t in tuples]
    user_df = pd.DataFrame({user_col:users, val_col:vals, 'pair_count':counts})
    user_df = user_df.sort_values(by=[user_col, 'pair_count'])
    unique_user_df = user_df.drop_duplicates(subset=[user_col], keep='last')  # keep val with max count for user
    user_counts = user_df.groupby(user_col)['pair_count'].sum()  # get total count per user
    users = unique_user_df[user_col].values
    coverage = unique_user_df['pair_count'].values / user_counts.loc[users].values  # what proportion the top val accounts for
    print('Top val coverage: median = %.2f (%.2f-%.2f)' % (np.median(coverage), 
                                      np.percentile(coverage, 25), np.percentile(coverage, 75)))
    coverage_too_low = coverage < thresh
    print('Dropping %.2f%% users with coverage below %s' % (100. * np.sum(coverage_too_low) / len(users), thresh))
    unique_user_df = unique_user_df[~coverage_too_low]
    return unique_user_df
    

##########################################################################
# Functions to load external data sources related to geographics
# and demographics
##########################################################################
def get_state_code_from_fips(f):
    """
    Get state code from county FIPS code.
    """
    if np.isnan(f):
        return np.nan
    return int(f / 1000)  # first two digits

def get_state_names(drop_dc=False):
    """
    Load names of states.
    """
    states = pd.read_csv('../data/state_abbrs.csv')
    if drop_dc:
        state_names = states[states.State != 'District of Columbia'].State.values
        assert len(state_names) == 50
    else:
        state_names = states.State.values
        assert len(state_names) == 51
    return state_names

def get_state_to_abbr():
    """
    Load mapping of state names to two-letter abbreviations.
    """
    states = pd.read_csv('../data/state_abbrs.csv')
    return dict(zip(states.State, states.Code))
    
def get_state_code_to_state_abbr():
    """
    Load mapping of state codes to state abbreviations.
    """
    states = pd.read_csv('../data/state_abbrs.txt', sep='|')
    return dict(zip(states.STATE, states.STUSAB))

    
def get_zip_to_zcta(return_dict=True):
    """
    Load zip code to ZCTA dataframe.
    """
    fn = '../data/ZiptoZcta_Crosswalk_2021.xlsx'
    zip_df = pd.read_excel(fn)
    zip_df = zip_df[zip_df.ZCTA != 'No ZCTA']  # drop zip codes without ZCTA
    assert len(zip_df.ZIP_CODE.unique()) == len(zip_df)  # there should be one row per zip code
    zip_df['ZIP_CODE'] = pd.to_numeric(zip_df.ZIP_CODE)
    zip_df['ZCTA'] = pd.to_numeric(zip_df.ZCTA)
    print('Loaded zip code to ZCTA dataframe; found %d zip codes' % len(zip_df))
    if return_dict:
        return dict(zip(zip_df.ZIP_CODE, zip_df.ZCTA))
    return zip_df

def get_zcta_to_state(return_dict=True):
    """
    Get mapping of ZCTA to state.
    """
    zip_df = get_zip_to_zcta(return_dict=False)
    num_states_per_zcta = zip_df.groupby('ZCTA')['STATE'].nunique()
    print('Warning: %d ZCTAs map to more than one state' % (num_states_per_zcta > 1).sum())
    zcta_df = zip_df.drop_duplicates(subset=['ZCTA'])
    print('Loaded ZCTA to state mapping; found %d ZCTAs' % len(zcta_df))
    if return_dict:
        return dict(zip(zcta_df.ZCTA, zcta_df.STATE))
    return zcta_df
    
def get_zcta_to_county(return_dict=True):
    """
    Get mapping of ZCTA to county. Some ZCTAs overlap with multiple counties, so keep the county
    that has max overlap with the ZCTA.
    Documentation of area columns:
    - https://www.census.gov/programs-surveys/geography/technical-documentation/records-layout/2020-zcta-record-layout.html 
    - https://www2.census.gov/geo/pdfs/maps-data/data/rel2020/zcta520/explanation_tab20_zcta520_county20_natl.pdf
    """
    fn = '../data/tab20_zcta520_county20_natl.txt'
    map_df = pd.read_csv(fn, sep='|')
    col_mapping = {'GEOID_ZCTA5_20': 'ZCTA',
                   'AREALAND_ZCTA5_20': 'land_area',
                   'AREAWATER_ZCTA5_20': 'water_area',
                   'AREALAND_PART': 'land_overlap',
                   'AREAWATER_PART': 'water_overlap',
                   'GEOID_COUNTY_20': 'county',
                   'NAMELSAD_COUNTY_20': 'county_name'}
    map_df = map_df[list(col_mapping.keys())].rename(columns=col_mapping)
    numeric_cols = [col for col in map_df.columns if col != 'county_name']
    map_df[numeric_cols] = map_df[numeric_cols].apply(pd.to_numeric)
    
    map_df['total_overlap'] = map_df.land_overlap + map_df.water_overlap
    num_counties_per_zcta = map_df.groupby('ZCTA')['county'].nunique()
    print('Warning: %d ZCTAs map to more than one county' % (num_counties_per_zcta > 1).sum())
    # keep county that has max overlap with ZCTA
    map_df = map_df.dropna(subset=['ZCTA', 'total_overlap']).sort_values(['ZCTA', 'total_overlap'])
    map_df = map_df.drop_duplicates(subset=['ZCTA'], keep='last')
    prop_overlap = map_df['total_overlap'] / (map_df['land_area'] + map_df['water_area'])
    print('Warning: %d ZCTAs map to county with overlap proportion < 0.6' % (prop_overlap < 0.6).sum())
    print('Loaded ZCTA to county mapping; found %d ZCTAs' % len(map_df))
    if return_dict:
        return dict(zip(map_df.ZCTA, map_df.county))
    return map_df
    
    
def get_acs_data(acs_fn, col_mapping, geo='zcta'):
    """
    Helper function to load ACS data for ZCTA or county.
    """
    def string_to_float(s):
        if type(s) in {int, float}:
            return s
        if type(s) == str:
            if s == '-':  # missing
                return np.nan
            s = s.replace(',', '')
            if s.endswith('+'):  # this appears in income cols
                s = s.strip('+')
            elif s.endswith('-'):  # this appears in income cols
                s = s.strip('-')
            return float(s)
        raise Exception('Could not parse string in ACS data')
    
    acs_df = pd.read_csv(acs_fn, low_memory=False).iloc[1:]  # skip first row, which are descriptions of headers
    assert geo in ['zcta', 'county']
    if geo == 'zcta':
        geo_name = 'ZCTA'
        acs_df[geo_name] = acs_df['NAME'].apply(lambda x: int(x.split()[1]))  # convert ZCTA to int, eg, "ZCTA5 XXXXX"
    else:
        geo_name = 'FIPS'
        acs_df[geo_name] = acs_df['GEO_ID'].apply(lambda x: int(x.split('S')[1]))  # convert GEOID to int, eg, "0500000USXXXXX"
    
    for col in col_mapping:
        acs_df[col] = acs_df[col].apply(string_to_float)
        col_vals = acs_df[col].values
        col_vals = col_vals[~np.isnan(col_vals)]
        assert all(col_vals >= 0)  # parsed floats should always be non-negative
        if col_mapping[col].startswith('percent'):  # make sure percents are in the expected range
            assert all(col_vals <= 100)
    acs_df = acs_df.rename(columns=col_mapping)
    cols_to_keep = [geo_name] + list(col_mapping.values())
    return acs_df[cols_to_keep].apply(pd.to_numeric)  # every column should be numeric


def get_zcta_df(include_lat_lon=False):
    """
    Load data for ZCTAs: mappings to state and county, ACS data, land and water area, and, optionally, 
    lat/lon from SimpleMaps. 
    """
    # 1. map to state and county
    zcta_df = get_zcta_to_county(return_dict=False)[['ZCTA', 'county', 'county_name', 'land_area']]
    code2abbr = get_state_code_to_state_abbr()
    zcta_df['state'] = zcta_df.county.apply(lambda x: code2abbr[get_state_code_from_fips(x)])
    # check that state from county and state from ZCTA are very similar
    # use state from county for 100% coverage and consistency with county
    zcta_state_df = get_zcta_to_state(return_dict=False).rename(columns={'STATE': 'state_from_zcta'})
    comparison = zcta_df[['ZCTA', 'state']].merge(zcta_state_df[['ZCTA', 'state_from_zcta']],
                                                  on='ZCTA', how='inner')    
    agree = (comparison.state == comparison.state_from_zcta).mean()
    assert agree > 0.99
        
    # 2. load ACS data
    # income data
    fn = '../data/ACSST5Y2020.S1903/ACSST5Y2020.S1903_data_with_overlays_2022-04-26T163933.csv'
    col_mapping = {'S1903_C03_001E':'median_income'}  # Estimate!!Median income (dollars)!!HOUSEHOLD INCOME BY RACE AND HISPANIC OR LATINO ORIGIN OF HOUSEHOLDER!!Households
    acs_df = get_acs_data(fn, col_mapping, geo='zcta')
    zcta_df = zcta_df.merge(acs_df, on='ZCTA', how='left')
    # demographic data
    fn = '../data/ACSDP5Y2020.DP05/ACSDP5Y2020.DP05_data_with_overlays_2022-04-26T174048.csv'
    acs_df = get_acs_data(fn, DP05_COLS, geo='zcta')
    acs_df['percent_20_to_34'] = acs_df['percent_20_to_24'] + acs_df['percent_25_to_34']
    acs_df['percent_35_to_64'] = acs_df['percent_35_to_44'] + acs_df['percent_45_to_54'] + acs_df['percent_55_to_59'] + acs_df['percent_60_to_64']
    zcta_df = zcta_df.merge(acs_df, on='ZCTA', how='left')
    zcta_df['pop_per_sq_meter'] = zcta_df.total_pop / zcta_df.land_area
    # education data
    fn = '../data/ACSST5Y2020.S1501/ACSST5Y2020.S1501_data_with_overlays_2022-04-26T130946.csv'
    col_mapping = {'S1501_C01_006E': 'num_25+',
                   'S1501_C01_015E': 'num_25+_bachelor_or_higher'}
    acs_df = get_acs_data(fn, col_mapping, geo='zcta')
    zcta_df = zcta_df.merge(acs_df, on='ZCTA', how='left')
    zcta_df['percent_bachelor_or_higher'] = np.round(100. * zcta_df['num_25+_bachelor_or_higher'].values / zcta_df['num_25+'].values, 1)  # to match other percents
    zcta_df['percent_without_bachelor'] = 100 - zcta_df.percent_bachelor_or_higher    
    
    assert len(zcta_df.ZCTA.unique()) == len(zcta_df)  # should be unique
    zcta_df = zcta_df.sort_values('ZCTA').set_index('ZCTA')
    print('Finished loading all ZCTA data; found %d ZCTAs in total' % len(zcta_df))
    
    if include_lat_lon:
        zcta_df = merge_simple_maps_lat_lon_with_zcta_df(zcta_df)
    return zcta_df

def merge_simple_maps_lat_lon_with_zcta_df(zcta_df):
    """
    Merge lat/lon info from SimpleMaps into ZCTA dataframe. Assumed that ZCTA index is ZCTA.
    """
    sm_df = pd.read_csv('../data/simplemaps_us_zips.csv')
    assert len(sm_df.zip.unique()) == len(sm_df)
    sm_df = sm_df.set_index('zip')
    sm_df = sm_df[sm_df.zcta == True]  # only keep rows where zip code and ZCTA match
    zctas = sm_df.index  
    pairs = [('population', 'total_pop'), ('density', 'pop_per_sq_meter')]
    for sm_col, acs_col in pairs:  # check correlation between ACS and SimpleMaps
        x = sm_df.loc[zctas][sm_col].values
        y = zcta_df.loc[zctas][acs_col].values
        isnan = np.isnan(x) | np.isnan(y)
        r, p = pearsonr(x[~isnan], y[~isnan])
        print('SimpleMaps %s vs ACS %s: N=%d, correlation = %.3f' % (sm_col, acs_col, np.sum(~isnan), r))
    different_counties = np.sum(sm_df.loc[zctas].county_fips.values != zcta_df.loc[zctas].county.values)
    if different_counties > 0:
        print('Warning: %d ZCTAs have different matched county from ACS vs SimpleMaps' % different_counties)
    return zcta_df.merge(sm_df[['lat', 'lng']], how='left', left_index=True, right_index=True)  # don't add new ZCTAs from SimpleMaps

def get_print_name_for_col(col):
    """
    Convert ACS col name to pretty name for labeling figures, eg, 'percent_65_and_over' -> '% 65 and over'
    """
    toks = []
    for t in col.split('_'):
        if t == 'percent' or t == 'prop':
            toks.append('%')
        elif t == 'rep':
            toks.append('Republican')
        elif t in {'bachelor', 'white', 'black', 'asian', 'hispanic'}:
            toks.append(t.capitalize())
        else:
            toks.append(t)
    s = ' '.join(toks)
    s = s.replace(' or higher', '')
    return s


def get_county_df():
    """
    Load data for counties: ACS data, election results.
    """
    # 1. load ACS data
    fn = '../data/ACSDP5Y2020.DP05-Data.csv'
    county_df = get_acs_data(fn, DP05_COLS, geo='county')
    county_df['percent_65_and_over'] = county_df['percent_65_to_74'] + county_df['percent_75_to_84'] + county_df['percent_85_and_over']
    
    # 2. load election data
    election_df = get_county_election_df()[['prop_dem', 'prop_rep']]
    county_df = county_df.merge(election_df, left_on='FIPS', right_index=True, how='outer')
    return county_df.set_index('FIPS')

def get_county_election_df(year=2020):
    """
    Load county-level votes from presidential election.
    """
    assert year in {2020, 2016}
    if year == 2020:
        fn = '../data/2020_election_data.csv'
        df = pd.read_csv(fn).iloc[1:]  # skip first row
        dem_candidate = 'Joseph R. Biden Jr.'
        rep_candidate = 'Donald J. Trump'
    else:
        fn = '../data/2016_election_county.xlsx'
        df = pd.read_excel(fn).iloc[1:]  # skip first row, which are descriptions of headers
        dem_candidate = 'Hillary Clinton'
        rep_candidate = 'Donald J. Trump'
    dem_votes = pd.to_numeric(df[dem_candidate], errors='coerce')
    rep_votes = pd.to_numeric(df[rep_candidate], errors='coerce')
    total_votes = pd.to_numeric(df['Total Vote'], errors='coerce')
    df['prop_dem'] = dem_votes / total_votes
    df['prop_rep'] = rep_votes / total_votes
    df['FIPS'] = pd.to_numeric(df.FIPS)
    assert len(df.FIPS.unique()) == len(df)
    df = df.sort_values('FIPS').set_index('FIPS')
    print('Loaded county-level election data; found %d FIPS' % len(df))
    return df


def load_zcta_shapefile():
    """
    Load the shapefile for ZCTAs 2020.
    """ 
    df = gpd.read_file('../data/tl_2020_us_zcta520/tl_2020_us_zcta520.shp')
    df['ZCTA'] = pd.to_numeric(df.ZCTA5CE20)
    df = df.set_index('ZCTA')
    df['INTPTLAT20'] = df.INTPTLAT20.apply(lambda x: float(x.strip('+')))
    df['INTPTLON20'] = df.INTPTLON20.apply(lambda x: float(x.strip('+')))
    print('Loaded ZCTA shapefile, found %s ZCTAs' % len(df))
    return df 

def load_county_shapefile():
    """
    Load 2020 shapefile for counties.
    """
    df = gpd.read_file('../data/tl_2020_us_county/tl_2020_us_county.shp')
    df = df.set_index('GEOID')
    df['INTPTLAT'] = df.INTPTLAT.apply(lambda x: float(x.strip('+')))
    df['INTPTLON'] = df.INTPTLON.apply(lambda x: float(x.strip('+')))
    print('Loaded county shapefile, found %d counties' % len(df))
    return df 

def load_state_shapefile():
    """
    Load 2020 shapefile for states.
    """
    df = gpd.read_file('../data/tl_2020_us_state/tl_2020_us_state.shp')
    df = df.set_index('NAME')
    df['INTPTLAT'] = df.INTPTLAT.apply(lambda x: float(x.strip('+')))
    df['INTPTLON'] = df.INTPTLON.apply(lambda x: float(x.strip('+')))
    print('Loaded state shapefile, found %d states and state-equivalents' % len(df))
    return df 


def get_newsguard_domain_to_scores():
    """
    Get mapping of domain to Newsguard score (numerical) and rating (categorical).
    """
    fn = '../data/newsguard_complete_data.xlsx'
    newsguard_df = pd.read_excel(fn)
    # tie break based on which country
    country_priority = []
    for c in newsguard_df.Country.values:
        if c == 'US':
            country_priority.append(0)
        elif c == 'ALL':
            country_priority.append(1)
        else:
            country_priority.append(2)
    newsguard_df['country_priority'] = country_priority
    newsguard_df = newsguard_df.sort_values(by=['Domain', 'country_priority'])
    domain_sizes = newsguard_df.groupby('Domain').size()
    # check domains with multiple rows, make sure top priority is unique
    multiple_row_domains = domain_sizes.index[domain_sizes > 1]
    for d in multiple_row_domains:
        rows = newsguard_df[newsguard_df.Domain == d]
        top_priority = rows.iloc[0].country_priority
        if top_priority == 2:
            print(d, 'missing US or ALL')
        elif (rows.country_priority == top_priority).sum() > 1:
            # found multiple rows with top priority
            matches_top = rows[rows.country_priority == top_priority]
            scores = set(matches_top.Score.unique())
            if len(scores) > 1:
                print('Warning: %s has multiple scores (%s) with priority %d' % (d, scores, top_priority))
            ratings = set(matches_top.Rating.unique())
            if len(ratings) > 1:
                print('Warning: %s has multiple ratings (%s) with priority %d' % (d, ratings, top_priority))
                
    newsguard_df = newsguard_df.drop_duplicates(subset=['Domain'], keep='first')  # keep highest priority row
    domain2score = dict(zip(newsguard_df.Domain, newsguard_df.Score))
    domain2rating = dict(zip(newsguard_df.Domain, newsguard_df.Rating))
    print('Found %d domains with scores, %d domains with ratings' % (
        len(newsguard_df)-newsguard_df.Score.isna().sum(),
        len(newsguard_df)-newsguard_df.Rating.isna().sum()))
    return domain2score, domain2rating

def get_trusted_and_untrusted_news_domains():
    """
    Get lists of trusted and untrusted news from MSR domain tracker.
    """
    fn = '../data/domain_tracker_17082022.csv'
    domain_df = pd.read_csv(fn)
    trusted_domains = domain_df[domain_df.trusted == True].domain.values
    untrusted_domains = domain_df[domain_df.trusted == False].domain.values
    return trusted_domains, untrusted_domains


def map_users_to_zcta_and_county(user_df=None, user_fn=None, save_fn=None, lat_col='MedianLat', 
                                 long_col='MedianLong', zcta_geo=None, county_geo=None):
    """
    Map users (from merged US users) to ZCTA and county. 
    """
    if user_df is None:
        assert user_fn is not None
        user_df = pd.read_csv(os.path.join(PATH_TO_PROCESSED_DATA, user_fn))
        print('Loaded user dataframe:', len(user_df))
    if zcta_geo is None:
        zcta_geo = load_zcta_shapefile()
    if county_geo is None:
        county_geo = load_county_shapefile()
    for col in ['ClientId', lat_col, long_col]:  # must have cols
        assert col in user_df.columns
    
    isnan = user_df[lat_col].isnull().values | user_df[long_col].isnull().values
    kept_users = user_df.iloc[~isnan]
    print('Found %d users with nonnan lat/lon' % len(kept_users))
    points = gpd.points_from_xy(x=kept_users[long_col].values, y=kept_users[lat_col].values, crs='EPSG:4269')
    user_geo = gpd.GeoDataFrame({'ClientId': kept_users.ClientId.values,
                                 'geometry': points})    
    level2shapefile = {'zcta': zcta_geo, 'county': county_geo}
    for level, level_df in level2shapefile.items():
        ts = time.time()
        user_geo = user_geo.sjoin(level_df[['geometry']], how='left')
        user_geo = user_geo.rename(columns={'index_right': level})
        more_than_1 = (user_geo.ClientId.value_counts() > 1).sum()
        if more_than_1 > 0:
            print('Warning: %d lat/lons map to multiple %s; keeping first' % (more_than_1, level))
        user_geo = user_geo.drop_duplicates(subset=['ClientId'])
        assert all(kept_users.ClientId.values == user_geo.ClientId.values)  # ordering should remain the same
        found_mapping = np.sum(~user_geo[level].isnull())
        print('Mapped %d (%.2f%%) users to %s [time=%.3f]' % (found_mapping, 
                                  100. * found_mapping / len(user_geo), level, time.time()-ts))
    if save_fn is not None:
        user_geo = pd.DataFrame(user_geo[['ClientId', 'zcta', 'county']])
        user_geo.to_csv(os.path.join(PATH_TO_PROCESSED_DATA, save_fn), index=False)
    return user_geo
    

def load_bing_coverage_per_zcta(zcta_df, zip2zcta=None, min_pop_size=50):
    """
    Load Bing coverage per ZCTA. Drops ZCTAs with bing_count or pop_size < min_pop_size.
    Returns each ZCTA's average number of active Bing users (bing_count), Census pop size (pop_size),
    coverage, county, and state.
    """
    fn = os.path.join(PATH_TO_PROCESSED_DATA, 'active_user_counts_Zip.csv')
    zip_counts = pd.read_csv(fn).set_index('MaxZip')
    if zip2zcta is None:
        zip2zcta = get_zip_to_zcta()
    zip_counts['mean'] = zip_counts.mean(axis=1).values  # mean over months
    zip_counts['ZCTA'] = np.array([zip2zcta.get(z, np.nan) for z in zip_counts.index]) # map ZIP to ZCTA
    zcta_counts = zip_counts.groupby('ZCTA')['mean'].sum()  # we can sum bc no overlap between users assigned to zips
    
    kept_bing_zctas = zcta_counts[zcta_counts >= min_pop_size].index
    kept_acs_zctas = zcta_df[zcta_df.total_pop >= min_pop_size].index
    zctas = sorted(set(kept_bing_zctas).intersection(set(kept_acs_zctas)))
    print('%s ZCTAs have >= %d Bing users, %s ZCTAs have >= %d population count -> keeping %d' % (
            len(kept_bing_zctas), min_pop_size, len(kept_acs_zctas), min_pop_size, len(zctas)))
    kept_zcta_df = pd.DataFrame({'bing_count': zcta_counts.loc[zctas], 
                                 'pop_size': zcta_df.loc[zctas].total_pop,
                                 'county': zcta_df.loc[zctas].county,
                                 'state': zcta_df.loc[zctas].state})
    coverage = kept_zcta_df.bing_count / kept_zcta_df.pop_size
    kept_zcta_df['coverage'] = coverage
    print('%.2f%% ZCTAs have coverage greater than 1' % (100 * (kept_zcta_df.coverage > 1).mean()))
    return kept_zcta_df
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fn', type=str)
    parser.add_argument('--month', type=str)
    args = parser.parse_args()
    split_tsv_file_into_states(args.fn)