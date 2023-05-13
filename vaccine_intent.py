from constants_and_utils import *
from adjustText import adjust_text
import datetime
import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms import bipartite
import networkx.algorithms.community as nx_comm
import numpy as np
import pandas as pd
import pickle
import re
import time
from scipy.stats import pearsonr, linregress, zscore, pointbiserialr, spearmanr
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, LinearRegression
import statsmodels.api as sm
import tomotopy as tp
import torch
from torch.utils.data import Dataset
    

##########################################################################
# Functions to identify vaccine intent in queries and URLs 
# (keywords and regex).
##########################################################################
def get_all_pharmacy_variants():
    # from https://www.cdc.gov/vaccines/covid-19/retail-pharmacy-program/participating-pharmacies.html
    pharmacies = [
        # albertsons companies - skipping united and long s (too ambiguous)
        'albertsons', 'osco', 'safeway', 'tom thumb', 'star market', 'shaw\'s', 'haggen',
        'acme', 'randalls', 'carrs', 'market street', 'vons', 'pavilions', 'amigos',
        'lucky\'s', 'pak n save', 'sav-on',
        'costco', 'cpesn', 'cvs', 'gerimed', 'good neighbor', 'health mart', 'heb', 'hy-vee', 'innovatix', 
        # kroger co.
        'kroger', 'harris teeter', 'fred meyer', 'fry\'s', 'ralphs', 'king soopers', 'smiths', 'city market',
        'dillons', 'mariano\'s', 'pick-n-save', 'copps', 'metro market', 'qfc',
        'leadernet', 'medicine shop', 'cardinal health', 'meijer', 'publix',
        # retail business
        'food lion', 'giant food', 'giant company', 'hannaford', 'stop & shop', 
        'rite aid', 'winn-dixie', 'harveys', 'fresco y mas',
        # topco associates - skipping acme (already covered) and ingles (confused with shingles)
        'topco', 'bashas', 'big-y', 'brookshire\'s', 'super one', 'coborn\'s', 'cash wise',
        'marketplace', 'giant eagle', 'hartig', 'king kullen', 'food city', 'raley\'s',
        'bel air', 'nob hill', 'save mart', 'spartan nash', 'price chopper', 'market 32',
        'tops friendly', 'shop rite', 'wegmans', 'weis', 
        'walgreens', 'duane reade', 'walmart'
    ]

    variants = []
    # 1. add spaces for '-' and '\''
    for p in pharmacies:
        p = p.replace('\'', ' ')  # \' is replaced with space in NormQuery
        variants.append(p)
        if '-' in p:
            variants.append(p.replace('-', ' '))
    
    # 2. try different tokens for '&', 'n', and 'and'
    new_variants = []
    for v in variants:
        tokens = v.split()
        if '&' in tokens:  # & is replaced with space in NormQuery
            index = tokens.index('&')
            tokens[index] = 'n'
            new_variants.append(' '.join(tokens))  # variant with '&' replaced by 'n'
            tokens[index] = 'and'
            new_variants.append(' '.join(tokens))  # variant with '&' replaced by 'and'
        elif 'n' in tokens:
            new_variants.append(v)  # original
            index = tokens.index('n')
            tokens[index] = 'and'
            new_variants.append(' '.join(tokens))  # variant with 'n' replaced by 'and'
        elif 'and' in tokens:
            new_variants.append(v)  # original
            index = tokens.index('and')
            tokens[index] = 'n'
            new_variants.append(' '.join(tokens))  # variant with 'and' replaced by 'n'
        else:
            new_variants.append(v)  # original
    variants = new_variants.copy()
    
    # 3. try with spaces removed
    new_variants = []
    for v in variants:
        new_variants.append(v)  # original
        if len(v.split()) > 1:
            new_variants.append(''.join(v.split()))  # variant without spaces
    assert len(set(new_variants)) == len(new_variants)  # should all be unique
    return new_variants

PHARMACY_VARIANTS = get_all_pharmacy_variants()

def about_vaccine(norm_q):
    """
    Whether the query is unambiguously about the COVID-19 vaccine.
    """
    covid_names = ['covid', 'coronavirus']
    has_covid = any([k in norm_q for k in covid_names])
    vaccine_names = ['vaccin', 'vacin', 'vax', 'dose', 'shot', 'booster',
                     'johnson', 'pfizer', 'moderna']
    has_vaccine = any([k in norm_q for k in vaccine_names])
    if has_covid and has_vaccine:
        return True
    return False

def wants_vaccine(norm_q):
    """
    Return whether the query unambiguously expresses intent to book a COVID-19 vaccine.
    """
    if not type(norm_q) in {str, np.str_}:
        print('Warning: query %s is not str' % norm_q)
        return False
    if not about_vaccine(norm_q):
        return False 
    
    # eg, [get covid vaccine], [find a covid vaccine], [book me a covid 19 vaccine in santa clara]
    regex = r'(get|schedule|find|book)( me)?( a)? (covid|coronavirus)(-19| 19|19)? (vaccine|shot)'
    m = re.match(regex, norm_q)
    if m is not None:
        return True 
    
    # check for pharmacies
    if any([p in norm_q for p in PHARMACY_VARIANTS]):
        return True
    
    # check for other keywords
    keywords = [
            'near me', 'location', 'locator', 'locate', 'appointment', 'registration',
            'register', 'finder', 'sign-up', 'sign up', 'walk-in', 'walk in', 
            'where to get', 'where can i get'
    ]
    if any([k in norm_q for k in keywords]):
        return True
    return False


def is_vaccine_intent_url(url):
    """
    Check whether URL matches known regexes for vaccine intent.
    """
    # eg, https://app.acuityscheduling.com/schedule.php?owner=21488344
    m = re.match(r'https?:\/\/app\.acuityscheduling\.com\/schedule\.php\?owner=', url)
    if m is not None:
        return True
    
    # eg, https://corporate.walmart.com/media-library/document/covid-19-vaccine-locations-walmart-and-sams-clubs-alabama-2-9-21/_proxyDocument?id=00000177-87d4-d0dc-af7f-ffff0bae0000
    m = re.match(r'https?:\/\/corporate\.walmart\.com\/media-library\/document\/covid-19-vaccine-locations-walmart-and-sams-clubs-.+\/_proxyDocument\?id=', url)
    if m is not None:
        return True
    
    # eg, https://states.aarp.org/new-hampshire/covid-19-vaccine-distribution
    m = re.match(r'https?:\/\/states\.aarp\.org\/.+\/covid-19-vaccine-distribution', url)
    if m is not None: 
        return True
    
    # eg, https://www.cvs.com/store-locator/cvs-pharmacy-locations/covid-vaccine/Montana/Billings
    # eg, https://www.cvs.com/store-locator/albuquerque-nm-pharmacies/9640-menaul-blvd-ne-albuquerque-nm-87112/storeid=8915/covid-19-vaccine
    m = re.match(r'https?:\/\/www\.cvs\.com\/store-locator\/.*covid(-19)?-vaccine', url)
    if m is not None:
        return True
    
    # eg, https://www.hy-vee.com/my-pharmacy/covid-vaccine?fbclid=IwAR1zZCzvSLSqp_i70pKh3mLWYQOaj6grGbCrQ2LzjZMGJCx58lrVu1GhQWc
    m = re.match(r'https?:\/\/www\.hy-vee\.com\/my-pharmacy\/covid-vaccine\?fbclid=', url)
    if m is not None:
        return True
    
    # eg, https://www.solvhealth.com/nm/c/albuquerque-nm-srv-covid-vaccine
    m = re.match(r'https?:\/\/www\.solvhealth\.com\/.+\/c\/.+-srv-covid-vaccine', url)
    if m is not None:
        return True
    return False


def is_not_vaccine_intent_url(url):
    """
    Check whether URL matches known regexes for _not_ being vaccine intent.
    These are mostly generic pharmacy and store pages that are not COVID-19-specific.
    """
    # need to check this since some URLs will match both
    if is_vaccine_intent_url(url):
        return False
    
    # eg, https://local.albertsons.com/ar/texarkana/3710-state-line-ave.html  
    m = re.match(r'https?:\/\/local\.(albertsons|jewelosco|safeway|shaws)\.com\/', url)
    if m is not None:
        return True
    
    # eg, https://local.pharmacy.albertsons.com/mt/kalispell/900-w-idaho-st.html 
    # eg, https://local.pharmacy.jewelosco.com/ia/clinton/1309-n-2nd-st.html  
    # eg, https://local.pharmacy.safeway.com/hi/honolulu/1234-s-beretania-st.html  
    m = re.match(r'https?:\/\/local\.pharmacy\.(albertsons|jewelosco|safeway|shaws)\.com\/', url)
    if m is not None:
        return True
    
    # eg, https://pharmacy.stopandshop.com/ma/swampscott/450-paradise-road  
    m = re.match(r'https?:\/\/pharmacy\.stopandshop.com\/', url)
    if m is not None:
        return True
    
    # eg, https://www.cvs.com/store-locator/cvs-pharmacy-address/2420+N+Reserve+St-Missoula-MT-59808/storeid=16320  
    # eg, https://www.cvs.com/store-locator/antioch-il-pharmacies/983-il-route-59-antioch-il-60002/storeid=8926?WT.mc_id=LS_BING_8926  
    m = re.match(r'https?:\/\/www\.cvs\.com\/store-locator\/', url)
    if m is not None:
        return True
    
    # https://www.cvs.com/minuteclinic/clinic-locator/ca/burbank/specialty/covid-19-testing-9795.html
    m = re.match(r'https?:\/\/www\.cvs\.com\/minuteclinic\/clinic-locator\/', url)
    if m is not None:
        return True
    
    # http://www.kingsoopers.com/stores/details/620/00044?cid=loc_62000044PP_bg
    # http://www.smithsfoodanddrug.com/stores/details/706/00180?cid=loc_70600180PP_bg
    m = re.match(r'https?:\/\/www\.(kingsoopers|smithsfoodanddrug)\.com\/stores\/details\/', url)
    if m is not None:
        return True
    
    # https://www.goodrx.com/pharmacy-near-me/albertsons-osco/wy/cody/1825-17th-st/1962686410 
    # https://www.goodrx.com/pharmacy-near-me/target-cvs/il/streamwood/1001-sutton-rd/1215959929
    # https://www.goodrx.com/pharmacy-near-me/walmart/tx/schertz/6102-fm-3009/1740207802
    m = re.match(r'https?:\/\/www\.goodrx\.com\/pharmacy-near-me\/', url)
    if m is not None:
        return True
    
    # https://www.costco.com/warehouse-locations/billings-mt-69.html
    m = re.match(r'https?:\/\/www\.costco\.com\/warehouse-locations\/', url)
    if m is not None:
        return True
    
    # https://www.riteaid.com/locations/ca/riverside/6150-van-buren-boulevard.html
    m = re.match(r'https?:\/\/www\.riteaid\.com\/locations\/', url)
    if m is not None:
        return True
    
    # https://www.walgreens.com/locator/walgreens-1224+s+tamiami+trl-sarasota-fl-34239/id=3264  
    m = re.match(r'https?:\/\/www\.walgreens\.com\/locator\/', url)
    if m is not None:
        return True
    
    # https://www.walmart.com/store/1797-thomaston-me
    m = re.match(r'https?:\/\/www\.walmart\.com\/store\/', url)
    if m is not None:
        return True
    return False


def about_vaccine_requirement(norm_q):
    """
    Simple heuristic to check if query is about vaccine requirements.
    """
    if not type(norm_q) in {str, np.str_}:
        print('Warning: query %s is not str' % norm_q)
        return False
    if 'vaccin' in norm_q:
        keywords = ['mandate', 'proof', 'pass', 'require', 'card', 'record']
        if any([k in norm_q for k in keywords]):
            return True
    return False

def about_covid_stats(norm_q):
    """
    Simple heuristic to check if query is about COVID stats.
    """
    if not type(norm_q) in {str, np.str_}:
        print('Warning: query %s is not str' % norm_q)
        return False   
    covid_names = ['covid', 'coronavirus']
    has_covid = any([k in norm_q for k in covid_names])
    stats = ['cases', 'deaths', 'hospitalizations', 'stats', 'statistics']
    has_stats = any([k in norm_q for k in stats])
    if has_covid and has_stats:
        return True
    return False

def about_vaccine_safety(norm_q):
    """
    Simple heuristic to check if query is about vaccine safety.
    """
    if not type(norm_q) in {str, np.str_}:
        print('Warning: query %s is not str' % norm_q)
        return False   
    if 'vaccin' in norm_q:
        keywords = ['side effects', 'deaths', 'risks']
        if any([k in norm_q for k in keywords]):
            return True
    return False

def about_vaccine_incentives(norm_q):
    """
    Simple heuristic to check if query is about vaccine incentives.
    """
    if not type(norm_q) in {str, np.str_}:
        print('Warning: query %s is not str' % norm_q)
        return False   
    if 'vaccin' in norm_q:
        keywords = ['lottery', 'winner', 'sweepstakes', 'sweep stakes', 'incentives']
        if any([k in norm_q for k in keywords]):
            return True
    return False
    
def not_about_vaccine(norm_q):
    """
    Simple heuristic to check if query is _not_ about vaccine (tests, cases, deaths).
    """
    if not type(norm_q) in {str, np.str_}:
        print('Warning: query %s is not str' % norm_q)
        return False
    keywords = {'test', 'case', 'death'}
    if any([k in norm_q for k in keywords]):
        return True
    return False


##########################################################################
# Functions to analyze vaccine intent results from Bing.
##########################################################################
def load_AMT_labeled_urls(only_nonnan=True):
    """
    Load saved labels from AMT. If only_nonnan, drop the nan labels (indicating inconclusive).
    """
    with open(os.path.join(PATH_TO_ANNOTATIONS, 'gnn_url_labels.pkl'), 'rb') as f:
        urls, labels = pickle.load(f)
    if only_nonnan:
        urls, labels = urls[~np.isnan(labels)], labels[~np.isnan(labels)]
    url2label = dict(zip(urls, labels))
    return url2label


def load_positive_urls():
    """
    Load positive URLs from AMT and GNN.
    """
    amt_url2label = load_AMT_labeled_urls()
    amt_pos_urls = np.array([u for u,l in amt_url2label.items() if l == 1])
    gnn_pos_urls = pd.read_csv(os.path.join(PATH_TO_ANNOTATIONS, 'gnn_expanded_urls.csv')).url.values
    assert np.sum(np.isin(amt_pos_urls, gnn_pos_urls)) == 0  # there should be no overlap
    return amt_pos_urls, gnn_pos_urls


def merge_user_df_and_zcta_df(user_df, zcta_df, zip2zcta):
    """
    Merge info from user_df and zcta_df, compute prop_vaccine_intent per ZCTA.
    
    user_df has one row per vaccine intent user with the date and location of 
    their first vaccine intent.
    zcta_df has one row per ZCTA with its estimated number of Bing users and 
    total population size from Census.
    """
    mode_prop = user_df['MaxZipCount'] / user_df['QueryCountWithZip']
    level_bools = (user_df['QueryCountWithZip'] >= 10) & (mode_prop >= 0.25)
    print('Can assign %d (%.2f%%) of vaccine intent users to Zip' % 
          (level_bools.sum(), 100. * level_bools.sum() / len(user_df)))
    zip_user_df = user_df[level_bools].copy()
    zip_user_df['ZCTA_from_zip'] = np.array([zip2zcta.get(z, np.nan) for z in zip_user_df.MaxZip])
    # num users per ZCTA who have expressed vaccine intent
    num_vaccine_intent = zip_user_df.groupby('ZCTA_from_zip').size().rename('num_vaccine_intent')
    zcta_df_extended = zcta_df.merge(num_vaccine_intent, how='left', left_index=True, right_index=True)
    zcta_df_extended = zcta_df_extended.fillna(0)
    
    zcta_df_extended['prop_vaccine_intent'] = zcta_df_extended.num_vaccine_intent / zcta_df_extended.bing_count
    zcta_props = zcta_df_extended.prop_vaccine_intent
    se = np.sqrt(zcta_props * (1-zcta_props) / zcta_df_extended.bing_count)
    zcta_df_extended['prop_vaccine_intent_lower'] = zcta_props - (1.96 * se)
    zcta_df_extended['prop_vaccine_intent_upper'] = zcta_props + (1.96 * se)
    return zcta_df_extended


def merge_user_df_and_county_df(user_df, county_df):
    """
    Merge info from user_df and county_df, compute prop_vaccine_intent per county.
    
    user_df has one row per vaccine intent user with the date and location of 
    their first vaccine intent.
    county_df has one row per county with its estimated number of Bing users and 
    total population size from Census.
    """
    mode_prop = user_df['MaxCountyCount'] / user_df['QueryCountWithCounty']
    level_bools = (user_df['QueryCountWithCounty'] >= 10) & (mode_prop >= 0.25)
    print('Can assign %d (%.2f%%) of vaccine intent users to County' % 
          (level_bools.sum(), 100. * level_bools.sum() / len(user_df)))
    county_user_df = user_df[level_bools]
    # num users per county who have expressed vaccine intent
    num_vaccine_intent = county_user_df.groupby('MaxCounty').size().rename('num_vaccine_intent')
    county_df_extended = county_df.merge(num_vaccine_intent, how='left', left_index=True, right_index=True)
    county_df_extended = county_df_extended.fillna(0)
    
    county_df_extended['prop_vaccine_intent'] = county_df_extended.num_vaccine_intent / county_df_extended.bing_count
    county_props = county_df_extended.prop_vaccine_intent
    se = np.sqrt(county_props * (1-county_props) / county_df_extended.bing_count)
    county_df_extended['prop_vaccine_intent_lower'] = county_props - (1.96 * se)
    county_df_extended['prop_vaccine_intent_upper'] = county_props + (1.96 * se)
    return county_df_extended


def compare_cdc_and_vaccine_intent_props(cdc_df, vi_prop_df=None, region2zctas=None, merged_zcta_df=None, 
                                         weight_by_coverage=True, make_plot=True, mode='state', region2weight=None, 
                                         min_completeness=80):
    """
    Get cumulative CDC vaccine rate and vaccine intent rate per region (state or county).
    If vi_prop_df is provided, vaccine intent proportions are already precomputed.
    If not, compute proportions based on weighted average over ZCTAs.
    """
    assert mode in {'state', 'county'}
    cdc_subdf = cdc_df[cdc_df.Date == datetime.datetime(2021, 8, 31)]  # cumulative counts up to end of Aug
    if mode == 'state':
        cdc_region2prop = dict(zip(cdc_subdf.Location, cdc_subdf.Series_Complete_Pop_Pct))
    else:
        cdc_region2prop = dict(zip(cdc_subdf.FIPS, cdc_subdf.Series_Complete_Pop_Pct))
    kept_regions = []
    cdc_prop = []
    vaccine_intent_prop = []
    if vi_prop_df is not None:
        assert 'prop_vaccine_intent' in vi_prop_df.columns
        regions = vi_prop_df.index
        for region, vi_prop in zip(vi_prop_df.index, vi_prop_df.prop_vaccine_intent):
            if region in cdc_region2prop:
                kept_regions.append(region)
                cdc_prop.append(cdc_region2prop[region]/100)  # want this as proportion, not percentage
                vaccine_intent_prop.append(vi_prop)
    else:
        assert region2zctas is not None and merged_zcta_df is not None
        regions = sorted(region2zctas.keys())
        for region in regions:
            if region in cdc_region2prop:
                kept_regions.append(region)
                cdc_prop.append(cdc_region2prop[region]/100)  # want this as proportion, not percentage
                vi_prop = get_cumulative_vaccine_intent(region2zctas[region], merged_zcta_df, weight_by_coverage=weight_by_coverage)
                vaccine_intent_prop.append(vi_prop)
    x, y = np.array(cdc_prop), np.array(vaccine_intent_prop)
    
    isnan = np.isnan(x)
    print('CDC proportions missing for %d regions' % (len(regions)-len(x)+np.sum(isnan)))
    if mode == 'county':
        completeness = cdc_subdf.set_index('FIPS').loc[kept_regions].Completeness_pct.values
        # following KFF analysis using CDC county data
        # https://www.kff.org/coronavirus-covid-19/issue-brief/vaccination-is-local-covid-19-vaccination-rates-vary-by-county-and-key-characteristics/
        low_completeness = completeness < min_completeness       
        print('CDC completeness below %d%% for %d counties' % (min_completeness, np.sum(low_completeness)))
    else: 
        low_completeness = np.zeros(len(x)).astype(bool)
    to_keep = ~(isnan | low_completeness)
    x, y = x[to_keep], y[to_keep]
    very_low = x < 0.01
    if np.sum(very_low) > 0:
        print('Warning: %d regions have CDC proportion under 1%%' % np.sum(very_low))
    
    print('Keeping %d regions' % len(x))
    ratio = y / x
    print('Avg ratio of vaccine intent / CDC rate: %.3f' % np.mean(ratio))
    if region2weight is None:
        w = None
    else:
        w = np.array([region2weight[r] for r in kept_regions])[to_keep]
    r, r_lower, r_upper = get_pearsonr_with_95_cis(x, y, w=w)
    print('Correlation: r=%.3f (%.3f, %.3f)' % (r, r_lower, r_upper))
    mdl = LinearRegression().fit(x.reshape(-1, 1), y, w)
    slope = mdl.coef_[0]
    intercept = mdl.intercept_
    r2 = mdl.score(x.reshape(-1, 1), y, w)
    assert np.isclose(r * r, r2) # R^2 from regression should be square of Pearson correlation
    print('Linear regression: slope=%.3f, intercept=%.3f' % (slope, intercept))
        
    if make_plot:
        fig, ax = plt.subplots(figsize=(8,8))
        if region2weight is None:
            ax.scatter(x, y, alpha=0.5)
        else:
            ax.scatter(x, y, s=0.1*w, alpha=0.5)
        ax.plot(x, slope*x+intercept, color='black')
        ax.grid(alpha=0.3)
        ax.set_xlabel(f'Prop. {mode} population vaccinated (CDC)', fontsize=14)
        ax.set_ylabel(f'Prop. {mode} users with vaccine intent', fontsize=14)
        ax.tick_params(which='both', labelsize=12)
        if mode == 'state':  # label if we are at state-level; too many counties to label
            texts = []
            for i, s in enumerate(regions):
                texts.append(plt.text(x[i], y[i], s, fontsize=12))
            adjust_text(texts)
    return np.array(kept_regions)[to_keep], x, y
    
    
def get_cumulative_vaccine_intent(zctas, merged_zcta_df, weight_by_coverage=True, demo_props=None, normalize=True,
                                  num_col='num_vaccine_intent'):
    """
    Compute cumulative proportion of users with vaccine intent for a set of ZCTAs (eg, for a state or county).
    """
    if demo_props is not None:
        assert ((demo_props >= 0) & (demo_props <= 1)).all()
        assert len(demo_props) == len(zctas)
    kept_zcta_df = merged_zcta_df.loc[zctas]
    if weight_by_coverage:
        est_per_zcta = kept_zcta_df[num_col].values / kept_zcta_df.bing_count.values
        num_vaccine_intent = kept_zcta_df.pop_size.values * est_per_zcta
        overall_pop = kept_zcta_df.pop_size.values
    else:
        num_vaccine_intent = kept_zcta_df.num_vaccine_intent.values
        overall_pop = kept_zcta_df.pop_size.values
    if demo_props is not None: # scale by proportion of ZCTA belonging to demographic
        num_vaccine_intent *= demo_props
        overall_pop *= demo_props
    num = np.sum(num_vaccine_intent)
    denom = np.sum(overall_pop)
    if normalize:
        return num/denom
    return num


def get_cumulative_prop_vaccine_intent_with_ci(zctas, merged_zcta_df):
    """
    Compute cumulative proportion of users with vaccine intent for a set of ZCTAs (eg, for a state or county),
    with 95% confidence interval. Assumes that ZCTA weighting is used.
    """
    kept_zcta_df = merged_zcta_df.loc[zctas]
    zcta_props = kept_zcta_df['num_vaccine_intent'].values / kept_zcta_df.bing_count.values
    num_vaccine_intent = kept_zcta_df.pop_size.values * zcta_props
    overall_pop = kept_zcta_df.pop_size.values
    prop_vi = np.sum(num_vaccine_intent) / np.sum(overall_pop)
    
    # approx following https://online.stat.psu.edu/stat506/lesson/6/6.3
    # ignore finite proportion correction factor since pop_size >> bing_count
    zcta_variances = zcta_props * (1-zcta_props) / kept_zcta_df.bing_count.values
    total_var = np.sum((overall_pop**2) * zcta_variances)
    total_var /= np.sum(overall_pop)**2
    total_se = np.sqrt(total_var)
    prop_vi_lower = prop_vi-(1.96*total_se)
    prop_vi_upper = prop_vi+(1.96*total_se)
    return prop_vi, prop_vi_lower, prop_vi_upper
    
    
def get_daily_vaccine_intent_over_time(zctas, datetimes, zip_user_df, merged_zcta_df, weight_by_coverage=True):
    """
    Compute daily proportion of new vaccine intent users for a set of ZCTAs (eg, for a state or county).
    """
    kept_user_df = zip_user_df[zip_user_df.ZCTA_from_zip.isin(zctas) & zip_user_df.datetime.isin(datetimes)]
    if weight_by_coverage:
        # make ZCTA x date matrix of num vaccine intent
        # total prop vaccine intent per day should be weighted sum over rows (ZCTA weight is constant over time)
        zctas_repeated = np.repeat(zctas, len(datetimes))
        dates_repeated = np.tile(datetimes, len(zctas))
        zcta_date_df = pd.DataFrame({'ZCTA_from_zip': zctas_repeated, 'datetime': dates_repeated}).set_index(['ZCTA_from_zip', 'datetime'])
        zcta_date_sums = kept_user_df.groupby(['ZCTA_from_zip', 'datetime']).size().rename('num_vaccine_intent')  # num t0s per ZCTA and date
        zcta_date_df = zcta_date_df.merge(zcta_date_sums, how='left', left_index=True, right_index=True).fillna(0)
        mat = zcta_date_df.values.reshape((len(zctas), len(datetimes)))
        weights = merged_zcta_df.loc[zctas].pop_size.values / merged_zcta_df.loc[zctas].bing_count.values
        nums = mat.T @ weights
        denom = merged_zcta_df.loc[zctas].pop_size.sum()
    else:
        date_sums = kept_user_df.groupby('datetime').size()  # num t0s per date
        nums = date_sums.loc[datetimes].values
        denom = merged_zcta_df.loc[zctas].bing_count.sum()
    return nums/denom
    
    
def get_daily_first_dose_over_time(cdc_state_df, state_abbr, datetimes):
    """
    Compute daily proportion of population getting first dose based on CDC data.
    """
    cdc_in_state = cdc_state_df[cdc_state_df.Location == state_abbr].sort_values('Date')
    cum_curve = cdc_in_state.Administered_Dose1_Pop_Pct.values / 100
    daily_curve = get_new_counts_from_cumulative(cum_curve)
    date2count = dict(zip(cdc_in_state.Date, daily_curve))
    counts = [date2count.get(d, 0) for d in datetimes]
    return counts

    
def compare_vaccine_intent_props_and_demographics(merged_zcta_df, zcta_df, zcta_cols, metric='corr',
                                                  verbose=True):
    """
    Measure relationship between ZCTA-level proportion with vaccine intent and ZCTA demographics.
    If metric is 'corr', use Pearson correlation.
    If metric is 'quartile', compute percent difference in top quartile vs bottom quartile.
    If metric is 'decile', compute percent difference in top decile vs bottom decile.
    If metric is 'percent_demo', estimate the percentage of the demographic that has been vaccinated (this 
        only applies if demographic variable is a percent of ZCTA population).
    """
    assert metric in ['corr', 'decile', 'quartile', 'percent_demo']
    zctas = merged_zcta_df.index.values
    x = merged_zcta_df.prop_vaccine_intent.values
    pop_sizes = merged_zcta_df.pop_size.values
    
    results = []
    for col in zcta_cols:
        y = zcta_df.loc[zctas][col].values
        isnan = np.isnan(y) # only demographic should be nan
        if metric == 'corr':
            w = np.sqrt(pop_sizes[~isnan])  # weight by sqrt population
            r, lower, upper = get_pearsonr_with_95_cis(x[~isnan], y[~isnan], w=w)
            if verbose: 
                print(col, 'r = %.4f (%.4f, %.4f)' % (r, lower, upper))
            results.append({'demo': col, 'r': r, 'lower': lower, 'upper': upper})
        elif metric in ['decile', 'quartile']:
            # note: use all ZCTAs to get cutoffs, not just ZCTAs that we kept
            if metric == 'decile':
                bottom_cutoff = zcta_df[col].quantile(0.1)
                top_cutoff = zcta_df[col].quantile(0.9)
            else:
                bottom_cutoff = zcta_df[col].quantile(0.25)
                top_cutoff = zcta_df[col].quantile(0.75)
            in_bottom = y <= bottom_cutoff
            bottom_mean = np.sum(pop_sizes[in_bottom] * x[in_bottom]) / np.sum(pop_sizes[in_bottom])  # weighted average
            in_top = y >= top_cutoff
            top_mean = np.sum(pop_sizes[in_top] * x[in_top]) / np.sum(pop_sizes[in_top])
            r = 100 * ((top_mean / bottom_mean) - 1)
            if verbose:
                print(col, 'bottom %s: %d, top %s: %d, r = %.2f' % (metric, np.sum(in_bottom), metric, np.sum(in_top), r))
            results.append({'demo': col, 'bottom_mean': bottom_mean, 'bottom_n': np.sum(in_bottom),
                            'top_mean': top_mean, 'top_n': np.sum(in_bottom), 'r': r})
        else:
            if 'percent' in col or 'prop' in col:
                overall_prop = get_cumulative_vaccine_intent(zctas, merged_zcta_df, weight_by_coverage=True)
                if 'percent' in col:
                    y /= 100
                demo_prop = get_cumulative_vaccine_intent(zctas[~isnan], merged_zcta_df, weight_by_coverage=True, demo_props=y[~isnan])
                r = 100 * ((demo_prop / overall_prop) - 1)  # percent diff in demo to overall rate
            else:
                r = np.nan  # can only compute for proportion-type variables
            if verbose:
                print(col, 'r = %.2f' % r)
            results.append({'demo': col, 'r': r})
    return pd.DataFrame(results, columns=list(results[-1].keys())).set_index('demo')
    
    
def get_bootstrapped_demographic_trends(merged_zcta_df, zcta_df, zcta_cols, metric='quartile', 
                                        sample_zctas=True, sample_vi_rates=True, num_trials=100):
    """
    Compute bootstrapped 95% CIs for measure of demographic trends.
    """
    df = compare_vaccine_intent_props_and_demographics(merged_zcta_df, zcta_df, zcta_cols, metric=metric,
                                                       verbose=False)
    results_df = {'demo': zcta_cols, 'point_est': df.r.values}
    for t in range(num_trials):
        if t%100 == 0: 
            print(t)
        col = 'trial%d' % t
        # sample ZCTAs with replacement
        if sample_zctas:
            sample_df = merged_zcta_df.sample(n=len(merged_zcta_df), replace=True, random_state=t)
        else:
            sample_df = merged_zcta_df.copy()
        # sample each ZCTA's proportion
        if sample_vi_rates:
            np.random.seed(t)
            bing_counts = sample_df.bing_count.values
            prop_vi = sample_df.prop_vaccine_intent.values
            assert np.isclose(sample_df.num_vaccine_intent.values / bing_counts, prop_vi).all()
            bing_counts = np.round(bing_counts).astype(int)  # round to integer so we can sample from Binomial
            sample_prop_vi = np.random.binomial(bing_counts, prop_vi) / bing_counts
            sample_df['prop_vaccine_intent'] = sample_prop_vi
        df = compare_vaccine_intent_props_and_demographics(sample_df, zcta_df, zcta_cols, metric=metric,
                                                           verbose=False)
        results_df[col] = df.r.values
    results_df = pd.DataFrame(results_df).set_index('demo')
    sample_cols = ['trial%d' % t for t in range(num_trials)]
    results_df['lower'] = results_df[sample_cols].quantile(0.025, axis=1)
    results_df['upper'] = results_df[sample_cols].quantile(0.975, axis=1)
    return results_df


def plot_vaccine_intent_demographic_trends(results_df, height_col, metric='corr', 
                                           ax=None, color='tab:blue', sort_bars=True):
    """
    Plot demographic trends as a bar graph. Adds error bars if 'lower' and 'upper' are specified.
    """
    if sort_bars:
        results_df = results_df.sort_values(height_col, ascending=False)
    zcta_cols = results_df.index
    x_pos = np.arange(len(zcta_cols))
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 4))
    bar_heights = results_df[height_col].values
    if 'lower' in results_df.columns and 'upper' in results_df.columns:
        yerr = [bar_heights-results_df.lower.values, results_df.upper.values-bar_heights]
        ax.bar(x_pos, bar_heights, color=color, align='center', yerr=yerr, ecolor='black')
    else:
        ax.bar(x_pos, bar_heights, color=color, align='center')
    if metric == 'corr':
        title = 'Correlation btwn ZCTA vaccine intent and demographic'
    elif metric == 'decile':
        title = 'Top vs. bottom decile: difference in vaccine intent'
    elif metric == 'quartile':
        title = 'Top vs. bottom quartile: difference in vaccine intent'
    else:
        title = 'Subpopulation vaccine intent rate: difference compared to overall rate'
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x_pos)
    labels = [get_print_name_for_col(c) for c in zcta_cols]
    ax.set_xticklabels(labels, rotation=75, fontsize=14)
    ax.grid(alpha=0.3)
    ax.tick_params(axis='y', labelsize=12)
    return ax
 

def get_quartile_vaccine_intent_over_time(merged_zcta_df, zcta_df, zip_user_df, col, datetimes,
                                          verbose=True):
    """
    Split ZCTAs into bottom/top quartile based on demographic col, then get vaccine intent over time
    per quartile.
    """
    zctas = merged_zcta_df.index.values
    y = zcta_df.loc[zctas][col].values
    bottom_zctas = zctas[y <= zcta_df[col].quantile(.25)]  # note: use all ZCTAs to get cutoffs, not just ZCTAs that we kept
    top_zctas = zctas[y >= zcta_df[col].quantile(.75)]
    results = {'date': datetimes}
    for group, zctas in zip(['top', 'bottom'], [top_zctas, bottom_zctas]):
        cumul = get_cumulative_vaccine_intent(zctas, merged_zcta_df)
        if verbose:
            print('%s, N=%d, cumulative prop: %.4f' % (group, len(zctas), cumul))
        daily = get_daily_vaccine_intent_over_time(zctas, datetimes, zip_user_df, merged_zcta_df)
        assert np.isclose(cumul, np.sum(daily))
        smoothed_daily = smooth_new_counts(daily)
        results[group] = smoothed_daily
    return pd.DataFrame(results).set_index('date')

    
def plot_quartile_comparison_over_time(results_df, col, datetimes, num_trials=1, axes=None):
    """
    Plot the vaccine intent curve over time for each quartile, then plot
    the percent difference.
    """
    if axes is None:
        fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
        fig.subplots_adjust(hspace=0.1)
    ax = axes[0]
    label = get_print_name_for_col(col)
    ax.set_title('Comparing quartiles: %s' % label, fontsize=12)
    for group, color in zip(['top', 'bottom'], ['tab:blue', 'tab:orange']):
        ax.plot_date(datetimes, results_df[group], fmt='-', marker=None, color=color,
                     label='%s quartile' % group)
        if num_trials > 1:
            group_cols = ['%s_trial%d' % (group, t) for t in range(num_trials)]
            lower = results_df[group_cols].quantile(.025, axis=1)
            upper = results_df[group_cols].quantile(.975, axis=1)
            ax.fill_between(datetimes, lower, upper, alpha=0.5, color=color)
    ax.legend(fontsize=11)
    ax.set_ylabel('Prop. users showing\nfirst vaccine intent', fontsize=12)
    ax.grid(alpha=0.3)

    ax = axes[1]
    percent_diff = 100 * (results_df['top']/results_df['bottom']-1)
    ax.plot_date(datetimes, percent_diff, fmt='-', marker=None, color='tab:blue')
    ax.set_ylabel('Percent difference,\ntop vs bottom', fontsize=12)
    if num_trials > 1:
        top_cols = ['top_trial%d' % t for t in range(num_trials)]
        bottom_cols = ['bottom_trial%d' % t for t in range(num_trials)]
        percent_diffs = 100 * (results_df[top_cols]/results_df[bottom_cols]-1)  # num dates x num trials
        lower = percent_diffs.quantile(.025, axis=1)
        upper = percent_diffs.quantile(.975, axis=1)
        ax.fill_between(datetimes, lower, upper, alpha=0.5, color='tab:blue')
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m'))
    ax.set_xlabel('Month', fontsize=12)
    return axes

    
def get_breakdown_over_demographics(groups, num_col, zctas, merged_zcta_df, zcta_df):
    """
    Get breakdown of overall group (eg, holdouts, users with vaccine intent) by demographic.
    """
    demo_cols = []
    for group in groups:
        if f'percent_{group}' in zcta_df.columns:
            demo_cols.append(f'percent_{group}')
        elif f'prop_{group}' in zcta_df.columns:
            demo_cols.append(f'prop_{group}')
        else:
            raise Exception('Cannot find percentage or proportion col for', group)
    kept_zctas = zcta_df.loc[zctas].dropna(subset=demo_cols).index
    print('Keeping %d out of %d ZCTAs with demographic info for all groups' % (len(kept_zctas), len(zctas)))
        
    pop_sizes = zcta_df.loc[kept_zctas].total_pop
    num_overall = get_cumulative_vaccine_intent(kept_zctas, merged_zcta_df, weight_by_coverage=True, 
                                                normalize=False, num_col=num_col)
    print('Overall count = %d' % num_overall)
    breakdown = {}
    for group in groups:
        if f'percent_{group}' in zcta_df.columns:
            demo_vals = zcta_df.loc[kept_zctas][f'percent_{group}'] / 100
        else:
            demo_vals = zcta_df.loc[kept_zctas][f'prop_{group}']
        prop_pop = np.sum(demo_vals * pop_sizes) / np.sum(pop_sizes)
        num_group = get_cumulative_vaccine_intent(kept_zctas, merged_zcta_df, weight_by_coverage=True, 
                                                  demo_props=demo_vals.values, normalize=False, num_col=num_col)
        prop_overall = num_group / num_overall
        print('%s: prop population = %.3f, count = %d, prop overall count = %.3f, ratio of props = %.3f' % (
            group, prop_pop, num_group, prop_overall, prop_overall / prop_pop))
        breakdown[group] = prop_overall
    return breakdown 


##########################################################################
# Functions for holdout vs nonholdout analysis.
##########################################################################
def predict_holdout_vs_nonholdout_from_demographics(zip_user_df, exog_cols, mode='corr', state=None, state2code=None, 
                                                    make_plot=True, ax=None):
    """
    Use demographic features in exog_cols to predict holdout vs non-holdout (unmatched).
    """
    assert mode in {'corr', 'reg'}
    if state is not None:
        code = state2code[state]
        subdf = zip_user_df[zip_user_df.state_code_from_zip == code].dropna(subset=exog_cols)
    else:
        subdf = zip_user_df.dropna(subset=exog_cols)
        state = 'United States'
    holdouts = subdf[subdf.is_holdout]
    nonholdouts = subdf[subdf.is_nonholdout]
    print('Num holdout: %d, num nonholdout: %d' % (len(holdouts), len(nonholdouts)))
    y = np.concatenate([np.zeros(len(nonholdouts)), np.ones(len(holdouts))])
    x = pd.concat([nonholdouts, holdouts])[exog_cols]
    x = x[exog_cols]
    assert len(y) == len(x)
    if mode == 'corr':
        bar_heights, yerr_lower, yerr_upper = [], [], []
        for col in exog_cols:
            r, lower, upper = get_pearsonr_with_95_cis(y, x[col].values)
            print(col, 'r = %.4f (%.4f, %.4f)' % (r, lower, upper))
            bar_heights.append(r)
            yerr_lower.append(r-lower)
            yerr_upper.append(upper-r)
            pb_r, _ = pointbiserialr(y, x[col].values)  # correlation between binary and continuous variable
            assert np.isclose(pb_r, r)
    else:
        x = x.apply(zscore)
        x['const'] = 1.0
        mdl = sm.Logit(y, x)
        res = mdl.fit()
        print('Log likelihood: %.4f' % res.llf)
        bar_heights = res.params.loc[exog_cols]
        yerr_lower = bar_heights - res.conf_int()[0].loc[exog_cols]
        yerr_upper = res.conf_int()[1].loc[exog_cols] - bar_heights
    
    if make_plot:
        x_pos = np.arange(len(exog_cols))
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))
        ax.set_title(state, fontsize=14)
        ax.bar(x_pos, bar_heights, yerr=[yerr_lower, yerr_upper], align='center', ecolor='black', capsize=5)
        if mode == 'corr':
            ax.set_ylabel('Correlation between holdout\nand demographic', fontsize=12)
        else:
            ax.set_ylabel('Logistic regression coef.\n(z-scored exogenous variables)', fontsize=12)
        ax.set_xticks(x_pos)
        labels = [get_print_name_for_col(s) for s in exog_cols]
        ax.set_xticklabels(labels, rotation=75, fontsize=12)
        ax.grid(alpha=0.3)
        return ax, y, x
    return y, x

def make_holdout_nonholdout_matched_pairs(holdout_df, nonholdout_df, tol=10):
    """
    Construct bipartite graph between holdouts and non-holdouts. An edge represents a valid match: same county and
    AvgQueryCount is within tol of each other. Run Hopcroft-Karp matching algorithm to find maximal matching.
    """
    ts = time.time()
    G = nx.Graph()
    seen_states = set()
    for county, holdout_county_df in holdout_df.groupby('MaxCounty'):
        state = get_state_code_from_fips(county)
        if state not in seen_states:
            print('Starting state', state)
            seen_states.add(state)
        nonholdout_county_df = nonholdout_df[nonholdout_df.MaxCounty == county]
        if len(nonholdout_county_df) > 0:
            nonholdout_qcs = nonholdout_county_df.AvgQueryCount
            for _, row in holdout_county_df.iterrows():
                min_qc = row['AvgQueryCount'] - tol
                max_qc = row['AvgQueryCount'] + tol
                valid_nonholdouts = nonholdout_county_df[(nonholdout_qcs >= min_qc) & (nonholdout_qcs <= max_qc)]
                if len(valid_nonholdouts) > 0:
                    src_nodes = [row['ClientId']] * len(valid_nonholdouts)
                    tgt_nodes = valid_nonholdouts.ClientId.values
                    weights = valid_nonholdouts.AvgQueryCount.values - row['AvgQueryCount']
                    edges = zip(src_nodes, tgt_nodes, weights)
                    G.add_weighted_edges_from(edges)
    assert bipartite.is_bipartite(G)
    print('Finished making bipartite graph [time=%.3fs]' % (time.time()-ts))
    
    ts = time.time()
    matches = {}
    for c in nx.connected_components(G):
        c_matches = bipartite.hopcroft_karp_matching(G.subgraph(c))
        matches = {**matches, **c_matches}
    matched_holdouts = []
    matched_nonholdouts = []
    for h in holdout_df.ClientId.values:
        if h in matches:
            matched_holdouts.append(h)
            matched_nonholdouts.append(matches[h])
    print('Matched %d holdouts (%.2f%%) [time=%.3fs]' % (len(matched_holdouts), 
           100. * len(matched_holdouts) / len(holdout_df), time.time()-ts))
    return G, matched_holdouts, matched_nonholdouts


def get_proportion_untrusted_per_user(cid_domain_df, domain2rating, user_min_clicks=5):
    """
    Get proportion of news that is untrusted per user.
    """
    assert user_min_clicks >= 1
    trusted_cols = [d for d in cid_domain_df.columns if domain2rating.get(d, 'None') == 'T']
    untrusted_cols = [d for d in cid_domain_df.columns if domain2rating.get(d, 'None') == 'N']
    print('Found %d trusted and %d untrusted domains' % (len(trusted_cols), len(untrusted_cols)))
    name2props = {}
    for name, lbl in zip(['holdout', 'nonholdout'], [1, 0]):
        subdf = cid_domain_df[cid_domain_df.label == lbl]
        untrusted_clicks = subdf[untrusted_cols].sum(axis=1).values
        trusted_clicks = subdf[trusted_cols].sum(axis=1).values
        denom = subdf.num_clicks.values
        prop_untrusted = untrusted_clicks / np.clip(denom, 1e-5, None)  # clip so we don't have divide by 0 errors; we will drop anyway
        to_drop = denom < user_min_clicks
        print('%ss: keeping %d users (%.2f%%) with at least %d clicks' % (
            name.capitalize(), np.sum(~to_drop), 100. * np.sum(~to_drop) / len(to_drop), user_min_clicks))
        prop_untrusted[to_drop] = np.nan
        print('Prop untrusted mean = %.5f, median = %.3f' % (np.nanmean(prop_untrusted), np.nanmedian(prop_untrusted)))
        name2props[name] = prop_untrusted
    holdout_prop_untrusted, nonholdout_prop_untrusted = name2props['holdout'], name2props['nonholdout']
    holdout_mean, nonholdout_mean = np.nanmean(holdout_prop_untrusted), np.nanmean(nonholdout_prop_untrusted)
    print('Ratio of holdout to non-holdout mean: %.5f' % (holdout_mean / nonholdout_mean))
    return holdout_prop_untrusted, nonholdout_prop_untrusted


def compute_holdout_vs_nonholdout_domain_risk_ratio(cid_domain_df, domain2score, domain2rating, smoothing=1e-5):
    """
    Compute holdout vs nonholdout risk ratio per domain. Add Newsguard scores and ratings to dataframe.
    """
    scored_domains = sorted(domain2score.keys())
    holdout_subdf = cid_domain_df[cid_domain_df.label == 1]
    domain_num_clicks = holdout_subdf[scored_domains].values + smoothing
    overall_num_clicks = holdout_subdf.num_clicks.values + (smoothing * len(scored_domains))
    holdout_prop_clicks = (domain_num_clicks.T / overall_num_clicks).T  # (num_holdouts, num_domains)
    holdout_mean_prop_clicks = np.mean(holdout_prop_clicks, axis=0)

    nonholdout_subdf = cid_domain_df[cid_domain_df.label == 0]
    domain_num_clicks = nonholdout_subdf[scored_domains].values + smoothing
    overall_num_clicks = nonholdout_subdf.num_clicks.values + (smoothing * len(scored_domains))
    nonholdout_prop_clicks = (domain_num_clicks.T / overall_num_clicks).T 
    nonholdout_mean_prop_clicks = np.mean(nonholdout_prop_clicks, axis=0)

    risk_ratio = holdout_mean_prop_clicks / nonholdout_mean_prop_clicks
    domain_df = pd.DataFrame({'domain': scored_domains, 
                              'holdout_prop_clicks': holdout_mean_prop_clicks,
                              'nonholdout_prop_clicks': nonholdout_mean_prop_clicks,
                              'avg_prop_clicks': (holdout_mean_prop_clicks+nonholdout_mean_prop_clicks)/2,
                              'risk_ratio': risk_ratio,
                              'newsguard_score': [domain2score[d] for d in scored_domains],
                              'newsguard_rating': [domain2rating[d] for d in scored_domains],
                             }).sort_values('risk_ratio', ascending=False)
    return domain_df


def plot_holdout_vs_nonholdout_domain_risk_ratio(domain_df, yticks=None, min_ratio=None, max_ratio=None,
                                                 to_label=None):
    """
    Plot Newsguard score vs domain risk ratio (log).
    """
    domain_df_complete = domain_df.dropna(subset=['risk_ratio', 'avg_prop_clicks', 'newsguard_score'])
    if min_ratio is not None:
        domain_df_complete = domain_df_complete[domain_df_complete.risk_ratio >= min_ratio]
    if max_ratio is not None:
        domain_df_complete = domain_df_complete[domain_df_complete.risk_ratio <= max_ratio]
    print('Visualizing %d domains' % len(domain_df_complete))
    
    fig, ax = plt.subplots(figsize=(6,6))
    x = domain_df_complete.newsguard_score.values
    y = np.log(domain_df_complete.risk_ratio.values)
    s = np.sqrt(domain_df_complete.avg_prop_clicks) * 1e3
    ax.scatter(x, y, s=s, c=y, cmap='plasma', alpha=0.5)

    z = np.polyfit(x, y, 2, w=s)  # weighted best fit polynomial
    p = np.poly1d(z)
    xp = np.linspace(0, 100, 101)
    ax.plot(xp, p(xp), color='black', alpha=0.8)

    ax.grid(alpha=0.2)
    ax.set_title('News consumption of holdouts vs early adopters\nApr-Jun 2021', fontsize=14)
    ax.set_xlabel('Score from Newsguard', fontsize=14)
    ax.set_ylabel('Ratio of click probs. (within news clicks)', fontsize=14)
    if yticks is not None:
        ytick_labels = ['%sx' % t for t in yticks]
        ax.set_yticks(np.log(yticks), labels=ytick_labels)
    ax.tick_params(labelsize=12)
    if to_label is not None:
        kept_df = domain_df_complete[domain_df_complete.domain.isin(to_label)]
        print('Found %d domains to label' % len(kept_df))
        for _, row in kept_df.iterrows():
            x_pt = row.newsguard_score
            y_pt = np.log(row.risk_ratio)
            ax.annotate(row.domain.split('.')[0], (x_pt, y_pt), fontsize=14)
    return ax


def get_clicks_for_date_range(group, date_range, only_vaccine_or_covid=True, all_pos_urls=None):
    """
    Load clicks for holdouts or non-holdouts in this date range.
    """
    assert group in {'holdout', 'nonholdout'}
    if all_pos_urls is None:
        amt_pos_urls, gnn_pos_urls = load_positive_urls()
        all_pos_urls = np.concatenate([amt_pos_urls, gnn_pos_urls])
    fn = os.path.join(PATH_TO_BING_DATA, '%s_clicks_%s.tsv' % (group, date_range))
    df = pd.read_csv(fn, delimiter='\t', header=0, quoting=csv.QUOTE_NONE, on_bad_lines='skip')
    df = df.dropna(subset=['ClientId', 'ClickUrl'])[['ClientId', 'ClickUrl', 'Request_RequestTime']]
    print('original length', len(df))
    df = df[df.ClickUrl.apply(lambda x: not skip_url_label(x))]  # drop internal, drop ads
    print('after dropping internal and ads', len(df))
    df['is_pos_url'] = df.ClickUrl.isin(all_pos_urls) | df.ClickUrl.apply(is_vaccine_intent_url)
    df['NormalizedUrl'] = df.ClickUrl.apply(normalize_url)
    keywords = ['vaccin', 'vax']
    df['about_vaccine'] = df.NormalizedUrl.apply(lambda x: any([k in x for k in keywords]))
    keywords = ['covid', 'coronavirus']
    df['about_covid'] = df.NormalizedUrl.apply(lambda x: any([k in x for k in keywords]))
    print(df[['is_pos_url', 'about_vaccine', 'about_covid']].mean())
    
    if only_vaccine_or_covid:
        to_keep = df.about_vaccine | df.about_covid
        print('only about vaccine and/or covid', to_keep.sum())
        return df[to_keep]
    return df


##########################################################################
# Functions for constructing and analyzing ontology.
##########################################################################
def do_community_detection_on_subgraph(G, url_list, urls_to_group, min_component_size=10, 
                                       mode='flat', mod_res=1):
    """
    Do Louvain community detection on induced subgraph, given URLs of interest (eg, all vaccine-related URLs).
    """
    assert G.shape == (len(url_list), len(url_list))
    url_df = pd.DataFrame({'url': url_list})
    url_df['keep_url'] = url_df.url.isin(urls_to_group)
    print('Found %d out of %d URLs in graph' % (url_df.keep_url.sum(), len(urls_to_group)))
    to_keep = url_df.keep_url.values
    subgraph = G[to_keep, :][:, to_keep]
    print('Original subgraph: shape = %s, num edges = %d' % (subgraph.shape, subgraph.count_nonzero()))

    row, col = subgraph.nonzero()
    edge_weights = subgraph.data
    subG = nx.Graph()
    subG.add_weighted_edges_from(zip(row, col, edge_weights), weight='weight')
    url_df_kept = url_df[url_df.keep_url]
    attr_dict = {i:u for i, u in enumerate(url_df_kept.url.values)}
    nx.set_node_attributes(subG, attr_dict, 'url')

    # remove self-loops
    selfloops = nx.selfloop_edges(subG)
    subG.remove_edges_from(nx.selfloop_edges(subG))
    assert len(list(nx.selfloop_edges(subG))) == 0
    # remove components that are too small
    components = nx.connected_components(subG)
    nodes_to_keep = set()
    for c in components:
        if len(c) >= min_component_size:
            nodes_to_keep = nodes_to_keep.union(c)
    print('Keeping %d out of %d nodes in component of at least size %d' % (len(nodes_to_keep), 
                 subG.number_of_nodes(), min_component_size))
    subG = nx.subgraph(subG, nodes_to_keep)
    
    # do louvain community detection on induced subgraph
    if mode == 'flat':  
        comms = nx_comm.louvain_communities(subG, seed=0, weight='weight', resolution=mod_res)
        all_edge_weights = nx.get_edge_attributes(subG, 'weight')
        m = np.sum(list(all_edge_weights.values()))
        qcs = np.array([community_modularity(subG, s, m, resolution=mod_res) for s in comms])
        assert np.isclose(nx_comm.modularity(subG, comms, weight='weight', resolution=mod_res), np.sum(qcs))
        print('Total modularity: %.4f [resolution = %s]' % (np.sum(qcs), mod_res))
        print('Number of communities:', len(comms))
        return subG, comms, qcs
    else:  # hierarchical
        comms = nx_comm.louvain_partitions(subG, seed=0, weight='weight', resolution=mod_res)
        return subG, comms
        
    
def community_modularity(G, s, m, resolution=1):
    """
    Compute modularity of a single community. 
    https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.community.quality.modularity.html
    """
    commG = nx.subgraph(G, s)
    edge_weights = nx.get_edge_attributes(commG, 'weight')
    s_in = np.sum(list(edge_weights.values())) / m
    s_out = 0
    for n in s:
        s_out += G.degree(n, weight='weight')
    s_out = (s_out / (2*m)) ** 2
    s_out *= resolution
    qc = s_in - s_out
    return qc


def run_community_detection_experiment(clicks_df, graph_name, urls_to_group=None, mod_res=1, min_users=5):
    """
    Do community detection on vaccine URLs.
    """
    num_users = clicks_df.groupby('NormalizedUrl')['ClientId'].nunique()
    if urls_to_group is None:
        urls_to_group = num_users.index[(num_users >= min_users)]
        print('Keeping %d out of %d URLs with at least %d users' % (len(urls_to_group), len(num_users), min_users))
    else:
        print('Grouping %d URLs' % len(urls_to_group))
    num_clicks = clicks_df.groupby('NormalizedUrl').size()

    with open(os.path.join(PATH_TO_PROCESSED_DATA, '%s_nodes.pkl' % graph_name), 'rb') as f:
        url_list = pickle.load(f)
    G = sparse.load_npz(os.path.join(PATH_TO_PROCESSED_DATA, '%s.npz' % graph_name))
    subG, comms, qcs = do_community_detection_on_subgraph(G, url_list, urls_to_group, mod_res=mod_res)
    order = np.argsort(-1 * qcs)  # order communities by modularity
    results = []
    for rank, ci in enumerate(order):  # by community index
        s = list(comms[ci])  # set of nodes
        commG = nx.subgraph(subG, s)
        pr = nx.pagerank(commG)  
        urls = [commG.nodes[n]['url'] for n in s]
        urls_sorted = num_clicks.loc[urls].sort_values(ascending=False) # order nodes by num clicks
        total_clicks = urls_sorted.sum()
        cumul_clicks = np.cumsum(urls_sorted.values/total_clicks)
        url2node = {commG.nodes[n]['url']:n for n in s}
        for ui, url in enumerate(urls_sorted.index.values):
            row = {'topic': rank, 'topic_size': len(s), 'topic_clicks': total_clicks, 'topic_mod': qcs[ci], 
                   'url': url, 'prop_clicks': num_clicks.loc[url]/total_clicks, 'cumul_clicks': cumul_clicks[ui],
                   'pr_centrality': pr[url2node[url]], 'num_users': num_users.loc[url]}
            results.append(row)
    columns = list(results[-1].keys())
    result_df = pd.DataFrame(results, columns=columns)
    return result_df
    

def print_samples_from_communities(result_fn, min_topic_size=100, sample_size=30, seed=0):
    """
    Print results from community detection.
    """
    result_df = pd.read_csv(os.path.join(PATH_TO_PROCESSED_DATA, result_fn))
    total_num_topics = len(result_df.topic.unique())
    print('Found %d topics and %d URLs in total' % (total_num_topics, len(result_df)))
    num_topics = 0
    num_urls = 0
    for topic, subdf in result_df.groupby('topic'):
        if len(subdf) >= min_topic_size:
            num_topics += 1
            num_urls += len(subdf)
    print('Has at least %d URLs: num topics = %d (%.2f%%), num URLs = %d (%.2f%%)' % (
        min_topic_size, num_topics, 100. * num_topics / total_num_topics, num_urls, 100. * num_urls / len(result_df)))
    print('Random seed for sampling from topics: %d' % seed)
    
    np.random.seed(seed)
    for topic, subdf in result_df.groupby('topic'):
        if len(subdf) >= min_topic_size:
            print('\nTopic %d: size = %d, modularity = %.4f' % (topic, len(subdf), subdf.iloc[0]['topic_mod']))
            idx = np.arange(len(subdf))
            if len(idx) <= sample_size:  # topic is small enough that we can print all
                sample = idx
            else:
                sample = sorted(np.random.choice(idx, size=sample_size, replace=False))
            for i in sample:
                print(i, subdf.iloc[i]['url'])
            

def print_top_queries_per_community(result_fn, graph_name, min_topic_size=100, top_n=10, groupby_cols=None):
    """
    Print top queries per topic.
    """
    with open(os.path.join(PATH_TO_PROCESSED_DATA, '%s_nodes.pkl' % graph_name), 'rb') as f:
        query_list, url_list = pickle.load(f)
    G = sparse.load_npz(os.path.join(PATH_TO_PROCESSED_DATA, '%s.npz' % graph_name))
    print('Loaded graph: %d queries, %d URLs' % (len(query_list), len(url_list)))
    url2idx = {u:i for i,u in enumerate(url_list)}
    
    result_df = pd.read_csv(os.path.join(PATH_TO_PROCESSED_DATA, result_fn))
    if groupby_cols is None:
        groupby_cols = ['topic']
    for topic, subdf in result_df.groupby(groupby_cols):
        if len(subdf) >= min_topic_size:
            print('\nTopic %s: size = %d, modularity = %.4f' % (topic, len(subdf), subdf.iloc[0]['topic_mod']))
            urls = subdf.url.values
            url_idx = [url2idx[u] for u in urls]
            vec = np.zeros(len(url_list))
            vec[url_idx] = 1
            query_sums = G @ vec  # total number of times query resulted in click on topic
            print('Total number of clicks on topic:', np.sum(query_sums))
            top_queries = np.argsort(-1 * query_sums)[:top_n]
            for qi, i in enumerate(top_queries):
                print('%d. [%s], %d' % (qi+1, query_list[i], query_sums[i]))
    

def load_topics_and_descriptions(result_fn, description_fn, min_topic_size=100):
    """
    Load community detection results and topic descriptions.
    """
    result_df = pd.read_csv(os.path.join(PATH_TO_PROCESSED_DATA, result_fn))
    description_df = pd.read_excel(os.path.join(PATH_TO_PROCESSED_DATA, description_fn))
    if 'REDO' in result_fn:
        assert 'orig_topic' in result_df.columns and 'Orig Topic' in description_df.columns
        # overwrite topic with <original topic>-<topic>
        result_df['topic'] = ['%d-%d' % (o, t) for o, t in zip(result_df.orig_topic, result_df.topic)]
        description_df['Topic'] = ['%d-%d' % (o, t) for o, t in zip(description_df['Orig Topic'], description_df['Topic'])]
    deduped = result_df.drop_duplicates(subset=['topic']).set_index(['topic'])
    topic_summaries = deduped[['topic_size', 'topic_clicks']]
    topic_df = pd.merge(topic_summaries, description_df, how='left', left_index=True, right_on='Topic')
    topic_df = topic_df.set_index('Topic')
    keep = topic_df.topic_size >= min_topic_size
    assert topic_df[keep].Description.isnull().sum() == 0 
    topic_df = topic_df[keep]
    return result_df, topic_df


def construct_final_files_for_ontology(topic_df, result_df, num_clicks, min_users=10):
    """
    Construct hierarchical ontology.
    
    subcategory -> top category
    [subcat], [num URLs], [num clicks], [top cat]
    
    cluster -> subcategory
    [cluster ID], [num URLs], [num clicks], [subcat1], [subcat2]
    
    URL -> cluster
    [URL], [num clicks], [prop clicks], [num users], [cluster ID]
    """
    kept_topics = topic_df[topic_df.Label1 != 'MISC'].copy()
    kept_urls = result_df[result_df.topic.isin(kept_topics.index)].copy()
    print('Keeping %d non-misc topics -> %d URLs' % (len(kept_topics), len(kept_urls)))
    
    # sort by Label1, Label2, then descending in size
    kept_topics['neg_topic_size'] = -kept_topics.topic_size
    kept_topics = kept_topics.sort_values(['Label1', 'Label2', 'neg_topic_size'])
    ordered_topics = kept_topics.index.values
    topic2cluster = {t:i for i, t in enumerate(ordered_topics)}
    kept_topics['cluster_id'] = np.arange(len(kept_topics))
    kept_topics = kept_topics.reset_index()
    col_names = {'topic_size': 'num_URLs', 'topic_clicks': 'num_clicks', 'Label1': 'subcat1',
                 'Label2': 'subcat2'}
    kept_cols = ['cluster_id'] + list(col_names.keys())
    kept_topics = kept_topics[kept_cols].rename(columns=col_names)
    
    kept_urls['cluster_id'] = kept_urls.topic.apply(lambda x: topic2cluster[x])
    kept_urls['num_clicks'] = num_clicks.loc[kept_urls.url.values].values
    assert (kept_urls.num_clicks >= kept_urls.num_users).all()
    kept_urls['neg_num_clicks'] = -kept_urls.num_clicks
    # sort by cluster ID, then descending in num clicks
    kept_urls = kept_urls.sort_values(['cluster_id', 'neg_num_clicks'])
    kept_urls = kept_urls[kept_urls.num_users >= min_users].reset_index()
    print('Keeping %d URLs with at least %d users' % (len(kept_urls), min_users))
    kept_cols = ['url', 'num_clicks', 'prop_clicks', 'num_users', 'cluster_id']
    kept_urls = kept_urls[kept_cols].rename(columns={'url': 'URL'})
    
    subcats = set(kept_topics.subcat1.dropna().unique()).union(set(kept_topics.subcat2.dropna().unique()))
    print('Found %d subcategories' % len(subcats))
    subcat_order = sorted(list(subcats))
    subcat_df = []
    for s in subcat_order:
        subcat_topics = kept_topics[(kept_topics.subcat1 == s) | (kept_topics.subcat2 == s)]
        subcat_df.append({'subcat': s, 'num_URLs': subcat_topics.num_URLs.sum(),
                          'num_clicks': subcat_topics.num_clicks.sum(),
                          'topcat': ABBR2CAT[s[0]]})
    subcat_df = pd.DataFrame(subcat_df, columns=list(subcat_df[-1].keys()))

    return subcat_df, ordered_topics, kept_topics, kept_urls

    
def compute_topic_ratios(labeled_df, cols, state=None, zctas=None, label_col='label', 
                         neg_col=None, weight_by_coverage=True, verbose=True):
    """
    Compute (weighted) topic probabilities per class, then computes ratios between positive 
    and negative class.
    """
    assert label_col in labeled_df.columns
    kept_df = labeled_df.copy()
    if state is not None:
        kept_df = kept_df[kept_df.MaxState == state]
    if zctas is not None:
        kept_df = kept_df[kept_df.ZCTA.isin(zctas)]
    if verbose:
        print(kept_df[label_col].value_counts())
        
    positive_class = kept_df[kept_df[label_col] == 1]  # typically holdouts
    if weight_by_coverage:  # weight by user's ZCTA inverse coverage
        weights = positive_class.inverse_coverage.values  # num_holdouts
        col_vals = positive_class[cols].values  # num_holdouts x num_cols
        nonnan_vals = (~np.isnan(col_vals)).astype(int)
        weighted_val_sums = np.nansum(col_vals * weights[:, np.newaxis], axis=0)  # multiply each col by weights
        weight_sums = np.sum(nonnan_vals * weights[:, np.newaxis], axis=0)
        positive_means = weighted_val_sums / weight_sums
    else:
        positive_means = positive_class[cols].mean().values
        
    if neg_col is None:
        negative_class = kept_df[kept_df[label_col] == 0]  # typically early adopters
    else:
        negative_class = kept_df[kept_df[neg_col] == 1]
    if weight_by_coverage:  # weight by user's ZCTA inverse coverage
        weights = negative_class.inverse_coverage.values
        col_vals = negative_class[cols].values
        nonnan_vals = (~np.isnan(col_vals)).astype(int)
        weighted_val_sums = np.nansum(col_vals * weights[:, np.newaxis], axis=0)  # multiply each col by weights
        weight_sums = np.sum(nonnan_vals * weights[:, np.newaxis], axis=0)
        negative_means = weighted_val_sums / weight_sums
    else:
        negative_means = negative_class[cols].mean().values
    
    results = pd.DataFrame({'category': cols, 'positive_class_mean': positive_means, 
                            'negative_class_mean': negative_means,
                            'avg_mean': (positive_means+negative_means)/2, 
                            'ratio': positive_means/negative_means}).set_index('category')
    if not weight_by_coverage:
        # if we're not weighting by coverage, then this is a simple risk ratio
        # and we can calculate CIs
        N1 = len(positive_class)
        N0 = len(negative_class)
        lower, upper = [], []
        for rr, p1, p0 in zip(results.ratio, results.positive_class_mean, results.negative_class_mean):
            _, lower_rr, upper_rr = get_cis_for_risk_ratio(p1, N1, p0, N0)
            assert np.isclose(rr, _), 'old ratio: %.3f, new ratio: %.3f' % (rr, _)
            lower.append(lower_rr)
            upper.append(upper_rr)
        results['lower'] = lower
        results['upper'] = upper
    return results.sort_values('ratio', ascending=False)


def get_bootstrapped_topic_ratios(labeled_df, cols, kwargs, num_trials=100):
    """
    Repeatedly samples rows with replacement, then recomputes topic ratios.
    """
    kwargs['verbose'] = False        
    ratios = compute_topic_ratios(labeled_df, cols, **kwargs)
    ratios = ratios.ratio.rename('point_est')  # get point estimate
    for i in range(num_trials):
        if i % 10 == 0: print(i)
        # sample rows with replacement
        labeled_df_i = labeled_df.sample(n=len(labeled_df), replace=True, random_state=i)  
        ratios_i = compute_topic_ratios(labeled_df_i, cols, **kwargs)
        ratios = pd.merge(ratios, ratios_i.ratio.rename('trial%d'%i), 
                          how='inner', left_index=True, right_index=True)
    bootstrap_cols = ['trial%d' % i for i in range(num_trials)]
    ratios['lower'] = np.percentile(ratios[bootstrap_cols].values, 2.5, axis=1)
    ratios['upper'] = np.percentile(ratios[bootstrap_cols].values, 97.5, axis=1)
    return ratios


def plot_ratios_across_topics(ratios, topic_order, xticks, with_err=True, colors=None):
    """
    Plots log ratios across topics. If with_err, then lower and upper bounds are included.
    """
    fig, ax = plt.subplots(figsize=(6,10))
    if with_err:
        log_ratio = np.log(ratios.loc[topic_order].point_est.values)
        lower = np.log(ratios.loc[topic_order].lower.values)
        upper = np.log(ratios.loc[topic_order].upper.values)
        err = [log_ratio-lower, upper-log_ratio]
    else:
        log_ratio = np.log(ratios.loc[topic_order].ratio.values)
    if with_err:
        ax.barh(topic_order, log_ratio, color=colors, align='center', xerr=err, ecolor='black')
    else:
        ax.barh(topic_order, log_ratio, color=colors, align='center')
    ax.set_xlabel('Ratio of click probs. (within vaccine clicks)', fontsize=14)
    ax.grid(alpha=0.2)
    xtick_labels = ['%sx' % t for t in xticks]
    ax.set_xticks(np.log(xticks), labels=xtick_labels)
    ax.tick_params(labelsize=12)
    return ax
    
    
def get_category_proportions_over_time(all_clicks_df, cat_cols, group='holdout'):
    """
    Returns category proportions over time.
    """
    assert group in {'holdout', 'nonholdout', 'all'}
    dates_to_plot = []
    cat_props = None
    end_of_week = all_clicks_df.datetime.min() + datetime.timedelta(days=6)
    while end_of_week <= all_clicks_df.datetime.max():
        dates_to_plot.append(end_of_week)
        week_dates = get_datetimes_in_range(end_of_week + datetime.timedelta(-6), end_of_week)
        clicks_in_week = all_clicks_df[all_clicks_df.datetime.isin(week_dates)]
        if group == 'holdout':
            clicks_in_week = clicks_in_week[clicks_in_week.label == 1]
        elif group == 'nonholdout':
            clicks_in_week = clicks_in_week[clicks_in_week.label == 0]
        week_props = clicks_in_week[cat_cols].mean().rename(end_of_week.strftime('%Y-%m-%d'))
        if cat_props is None:
            cat_props = week_props.copy()
        else:
            cat_props = pd.merge(cat_props, week_props, left_index=True, right_index=True, how='outer')
        end_of_week = end_of_week + datetime.timedelta(days=1)
    return dates_to_plot, cat_props


def plot_category_proportions_over_time(dates_to_plot, cat_props, cat2color=None, ax=None, 
                                        cat2abbr=None, legend_title=None):
    """
    Visualizes category proportions over time as a stacked proportions plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 3.5))
    base = np.zeros(len(dates_to_plot))
    for cat, props in cat_props.iterrows():
        print('%s: mean prop = %.5f' % (cat, props.mean()))
        label = cat if cat2abbr is None else '%s: %s' % (cat2abbr[cat], cat)
        if cat2color is not None:
            ax.fill_between(dates_to_plot, base, base+props.values, label=label, color=cat2color[cat])
        else:
            ax.fill_between(dates_to_plot, base, base+props.values, label=label)
        ax.plot(dates_to_plot, base+props.values, color='black', linewidth=0.2)  # draw boundaries
        base = base + props.values

    ax.set_ylabel('Prop. vaccine clicks', fontsize=14)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], bbox_to_anchor=(1.05, 1.0), loc='upper left', fontsize=12,
              title=legend_title, title_fontsize=13)
    ax.set_xlabel('Month', fontsize=14)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.grid(alpha=0.2)
    return ax


def add_time_to_vi_to_clicks_df(clicks_df, all_vi_df, min_datetime, max_datetime, 
                                days_before=14, days_after=7, verbose=True):
    """
    Adds columns to clicks_df indicating time before/after vaccine intent. 
    For each day from min_datetime to max_datetime: 
        find all users who expressed vaccine intent on that day,
        label clicks within the week (3 days before/after),
        label clicks per day from -days_before to days_after.
    """
    assert 'datetime' in clicks_df.columns and 'datetime' in all_vi_df.columns
    clicks_df['in_vi_range'] = np.zeros(len(clicks_df)).astype(bool)
    clicks_df['in_manual_vi_range'] = np.zeros(len(clicks_df)).astype(bool)
    days_in_range_indices = range(-days_before, days_after+1)
    for i in days_in_range_indices:
        clicks_df['is_t%s' % i] = np.zeros(len(clicks_df)).astype(bool)
        
    for t in get_datetimes_in_range(min_datetime, max_datetime):
        if verbose:
            print(t)
        # users who showed vaccine intent on day t
        users_on_day = all_vi_df[all_vi_df.datetime == t].ClientId.values
        if len(users_on_day) == 0:
            print('Warning: found 0 users with vaccine intent on %s' % t.strftime('%Y-%m-%d'))
        days_in_week = get_datetimes_in_range(t+datetime.timedelta(days=-3), 
                                              t+datetime.timedelta(days=3))
        # click's user is within 3 days of vaccine intent
        in_vi_range_t = clicks_df.ClientId.isin(users_on_day) & clicks_df.datetime.isin(days_in_week)  
        clicks_df['in_vi_range'] = clicks_df.in_vi_range | in_vi_range_t
        # click's user is within 3 days of non-GNN vaccine intent
        users_on_day = all_vi_df[(all_vi_df.datetime == t) & (all_vi_df.UrlLabelSource != 2)].ClientId.values
        in_manual_vi_range_t = clicks_df.ClientId.isin(users_on_day) & clicks_df.datetime.isin(days_in_week)  
        clicks_df['in_manual_vi_range'] = clicks_df.in_manual_vi_range | in_manual_vi_range_t

        days_in_range = get_datetimes_in_range(t+datetime.timedelta(days=-days_before), 
                                               t+datetime.timedelta(days=days_after))    
        for i, day in zip(days_in_range_indices, days_in_range):
            # click is t+i days away from manual VI
            matches_day = clicks_df.ClientId.isin(users_on_day) & (clicks_df.datetime == day)
            col = 'is_t%s' % i
            # this should be our first time seeing this day as t+i for this specific t
            assert (~clicks_df[col][matches_day]).all()
            clicks_df[col] = clicks_df[col] | matches_day

    cols_to_check = ['in_vi_range', 'in_manual_vi_range'] + ['is_t%s' % i for i in days_in_range_indices]
    print(clicks_df[cols_to_check].sum())
    return clicks_df

def plot_subcat_dynamics(t2ratios, subcat, days_in_range_indices, ax=None, yticks=None,
                         plot_cis=True):
    """
    Plot click ratios over range of days before/after vaccine intent.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6,4))
    line, lower, upper = [], [], []
    for t in days_in_range_indices:
        row = t2ratios[t].loc[subcat]
        if 'ratio' in row:
            line.append(np.log(row.ratio))
        else:
            assert 'point_est' in row
            line.append(np.log(row.point_est))
        if plot_cis:
            if np.isclose(row.lower, 0):
                print('Warning: found 0 in lower')
                lower.append(np.nan)
            else:
                lower.append(np.log(row.lower))
            upper.append(np.log(row.upper))
    # use most generous lower bound to fill in missing lower
    line, lower, upper = np.array(line), np.array(lower), np.array(upper)
    max_diff = np.nanmax(line-lower)
    lower[np.isnan(lower)] = (line-max_diff)[np.isnan(lower)]
    
    ax.plot(days_in_range_indices, line, color='tab:blue')
    if plot_cis:
        ax.fill_between(days_in_range_indices, lower, upper, alpha=0.5, color='tab:blue')
    ax.set_xlim(days_in_range_indices[0], days_in_range_indices[-1])
    ax.hlines([0], days_in_range_indices[0], days_in_range_indices[-1], color='grey', alpha=0.5)
    if yticks is not None:
        ytick_labels = ['%sx' % t for t in yticks]
        ax.set_yticks(np.log(yticks), labels=ytick_labels)
    ax.tick_params(labelsize=12)
    ax.grid(alpha=0.2)
    return ax

    
##########################################################################
# Functions from supplementary experiments.
##########################################################################
def compare_query_timeseries_bing_vs_google(query, datetimes, all_vi_queries, 
                                            make_plot=True, smooth_days=7, ax=None, set_labels=True,
                                            verbose=True):
    """
    Compare normalized search interest over time, Bing vs. Google.
    """
    assert smooth_days >= 1
    # From Google: 
    # Numbers represent search interest relative to the highest point on the chart for the given region and time.
    # A value of 100 is the peak popularity for the term. A value of 50 means that the term is half as popular. 
    # A score of 0 means there was not enough data for this term.
    google_df = pd.read_csv('../google_search_trends/%s.csv' % query.replace(' ', '_'), header=1)
    expected_col = '%s: (United States)' % query
    assert expected_col in google_df.columns
    google_df['Day'] = google_df.Day.apply(lambda s: datetime.datetime.strptime(s, '%Y-%m-%d'))
    google_df = google_df.set_index('Day')
    google_vec = google_df.loc[datetimes][expected_col].values
    google_vec[google_vec.astype('str') == '<1'] = 0
    google_vec = google_vec.astype(int)
    google_vec = smooth_new_counts(google_vec, interval=smooth_days)[smooth_days-1:]
    google_vec = 100 * google_vec / np.max(google_vec)

    query_df = all_vi_queries[all_vi_queries.NormQuery == query]
    if verbose: 
        print('num Bing instances of [%s]:' % query, len(query_df))
    date_counts = query_df.groupby('datetime').size()
    date2count = dict(zip(date_counts.index, date_counts.values))
    bing_vec = np.array([date2count.get(d, 0) for d in datetimes])
    bing_vec = smooth_new_counts(bing_vec, interval=smooth_days)[smooth_days-1:]
    bing_vec = 100 * bing_vec / np.max(bing_vec)  # normalize so that max is 100
    
    r, r_lower, r_upper = get_pearsonr_with_95_cis(google_vec, bing_vec)
    if verbose:
        print('Pearson correlation: r=%.4f (%.4f, %.4f)' % (r, r_lower, r_upper))
    if make_plot:
        if ax is None:
            fig, ax = plt.subplots(figsize=(6,4))
        ax.plot(datetimes[smooth_days-1:], google_vec, label='Google')
        ax.plot(datetimes[smooth_days-1:], bing_vec, label='Bing')
        if set_labels:
            ax.legend()
            ax.set_xlabel('Month', fontsize=12)
            ax.set_ylabel('Query freq', fontsize=12)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m'))
        ax.grid(alpha=0.2)
        ax.set_title('[%s]\nr=%.2f' % (query, r), fontsize=12)
    return r, r_lower, r_upper
    
def compare_query_over_states_bing_vs_google(query, states, all_vi_queries, bing_state_totals, 
                                         make_plot=True, state2abbr=None, ax=None, set_labels=True,
                                         verbose=True):
    """
    Compare normalized search interest across states, Bing vs. Google.
    """
    # From Google:
    # Values are calculated on a scale from 0 to 100, where 100 is the location with the most popularity 
    # as a fraction of total searches in that location, a value of 50 indicates a location which is half 
    # as popular. A value of 0 indicates a location where there was not enough data for this term.
    google_df = pd.read_csv('../google_search_trends/%s_geomap.csv' % query.replace(' ', '_'), header=1)
    cols = google_df.columns
    assert cols[1].startswith('%s:' % query)
    google_df = google_df.set_index('Region')
    google_vec = google_df.loc[states][cols[1]].values
    google_vec[google_vec.astype('str') == '<1'] = 0
    google_vec = google_vec.astype(float)
    
    query_df = all_vi_queries[all_vi_queries.NormQuery == query]
    state_query_counts = query_df.DataSource_State.value_counts()
    missing_states = [s for s in states if s not in state_query_counts.index]
    if len(missing_states) > 0:  # add query count of 0 for missing states
        missing_df = pd.Series([0]*len(missing_states), index=missing_states)
        state_query_counts = pd.concat([state_query_counts, missing_df])
    bing_vec = state_query_counts.loc[states] / bing_state_totals.loc[states]
    bing_vec = 100 * bing_vec / np.max(bing_vec)  # normalize so that max is 100
    
    isnan = np.isnan(google_vec) | np.isnan(bing_vec)
    print('[%s]: computing over %d states' % (query, len(states) - np.sum(isnan)))
    google_vec, bing_vec = google_vec[~isnan], bing_vec[~isnan]
    r, r_lower, r_upper = get_pearsonr_with_95_cis(google_vec, bing_vec)
    if verbose: 
        print('num Bing instances of [%s]:' % query, len(query_df))
        print('Pearson correlation: r=%.4f (%.4f, %.4f)' % (r, r_lower, r_upper))
    if make_plot:
        if ax is None:
            fig, ax = plt.subplots(figsize=(6,6))
        ax.scatter(google_vec, bing_vec, alpha=0.5)
        if state2abbr is not None:
            for s, g, b in zip(states[~isnan], google_vec, bing_vec):
                if (g >= np.percentile(google_vec, 90) or b >= np.percentile(bing_vec, 90)) and g > 10 and b > 10:  # label the top states
                    ax.annotate(state2abbr[s], (g, b))
        if set_labels:
            ax.set_xlabel('Google, query freq', fontsize=12)
            ax.set_ylabel('Bing, query freq', fontsize=12)
        ax.grid(alpha=0.2)
        ax.set_title('[%s]\nr=%.2f' % (query, r), fontsize=12)
    return r, r_lower, r_upper

def get_ratios_per_category_and_domain(clicks_df, all_cats, verbose=True):
    """
    Get ratio of category_prop / overall_prop per domain and category.
    """
    overall_props = (clicks_df.domain.value_counts() / len(clicks_df)).rename('overall_prop')
    if verbose:
        print('Keeping %d domains' % len(overall_props))
    ratios = overall_props.copy()
    for c in all_cats:
        cat_clicks = clicks_df[clicks_df[c] == 1]
        cat_props = (cat_clicks.domain.value_counts() / len(cat_clicks)).rename('%s_prop' % c)
        ratios = pd.merge(ratios, cat_props, how='left', left_index=True, right_index=True).fillna(0)
        ratios[c] = ratios['%s_prop' % c] / ratios.overall_prop
    return ratios

def plot_top_domains_per_category(all_ratios, cat_order, axes, cmap, domain2score, 
                                  num_trials=100, num_to_show=10, min_prop=1e-2, xticks=None):
    """
    Plot top news domains per category.
    """
    row_size = len(axes[-1])
    axes = axes.flatten()
    assert len(cat_order) <= len(axes)
    if len(cat_order) < len(axes):
        print('Warning: only %d categories for %d axes' % (len(cat_order), len(axes)))
    for i, (c, ax) in enumerate(list(zip(cat_order, axes))):
        ax.set_title(c, fontsize=12)
        if '%s_prop' % c in all_ratios.columns:
            domains_to_keep = all_ratios[all_ratios['%s_prop' % c] >= min_prop].index  # keep domains that account for at least 0.1% of clicks
        else:
            domains_to_keep = all_ratios.index
        print(c, len(domains_to_keep))
        point_est = all_ratios[c].loc[domains_to_keep]  # point estimate of ratios using all data
        log_ratios_to_plot = np.log(point_est.sort_values(ascending=False).head(num_to_show).sort_values())
        ratio_samples = all_ratios[['%s_%d' % (c, t) for t in range(num_trials)]]  # results from bootstrapped samples
        lower = ratio_samples.quantile(0.025, axis=1)
        log_lower = np.log(lower.loc[log_ratios_to_plot.index])
        upper = ratio_samples.quantile(0.975, axis=1)
        log_upper = np.log(upper.loc[log_ratios_to_plot.index])
        xerr = [log_ratios_to_plot-log_lower, log_upper-log_ratios_to_plot]
        colors = [cmap(domain2score[d]/100) for d in log_ratios_to_plot.index]
        domain_labels = [l[:20] for l in log_ratios_to_plot.index]  # truncate if long
        ax.barh(domain_labels, log_ratios_to_plot.values, color=colors, align='center', xerr=xerr, ecolor='black')
        ax.set_xlim(0, np.log(32))
        if i >= (len(axes)-row_size):  # last row
            if xticks is None:
                xticks = np.array([1, 4, 16])
            xtick_labels = ['%dx' % t for t in xticks]
            ax.set_xticks(np.log(xticks), labels=xtick_labels)
            ax.set_xlabel('Ratio to overall click props.', fontsize=12)
        ax.tick_params(labelsize=11)
        ax.grid(alpha=0.3)


if __name__ == '__main__':
    result_fn = 'matched_users_vaccine_topics_04_08_r17.csv'
    print_samples_from_communities(result_fn, min_topic_size=100, seed=2)
    # print_top_queries_per_community(result_fn, 'matched_users_qu_graph_04_08', min_topic_size=100, top_n=10)
