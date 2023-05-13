import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from constants_and_utils import *

PATH_TO_VACCINE_COUNTY_DATA = '../data/COVID-19_Vaccinations_in_the_United_States_County.csv'
PATH_TO_VACCINE_STATE_DATA = '../data/COVID-19_Vaccinations_in_the_United_States_Jurisdiction.csv'
PATH_TO_MANDATE_DATA = '../data/State-Level_Vaccine_Mandates_-_All.csv'
PATH_TO_PASSPORT_DATA = '../data/vaccine_passports.csv'


def convert_possible_string_to_float(string):
    """
    Helper function to convert string with possible commas to float.
    """
    if type(string) is float:
        return string 
    try:
        return float(string.replace(',', ''))
    except:
        return np.nan   


def convert_datestring_to_datetime(string):
    """
    Helper function to convert string to datetime object.
    """
    try:
        return datetime.datetime.strptime(string.split()[0], '%m/%d/%Y')
    except:
        return np.nan 


def load_vaccine_county_data():
    """
    Load county-level vaccine data from CDC (https://data.cdc.gov/Vaccinations/COVID-19-Vaccinations-in-the-United-States-County/8xkx-amqh).
    """
    df = pd.read_csv(PATH_TO_VACCINE_COUNTY_DATA, low_memory=False)
    df['FIPS'] = pd.to_numeric(df.FIPS, errors='coerce')
    df = df.dropna(subset=['FIPS'])
    df['Date'] = df.Date.apply(convert_datestring_to_datetime)
    return df   


def load_vaccine_state_data():
    """
    Load state-level vaccine data from CDC (https://data.cdc.gov/Vaccinations/COVID-19-Vaccinations-in-the-United-States-Jurisdi/unsk-b7fc).
    """
    df = pd.read_csv(PATH_TO_VACCINE_STATE_DATA, delimiter=';', low_memory=False)
    df['Date'] = df.Date.apply(convert_datestring_to_datetime)
    for col in df.columns:  # convert all numerical columns to ints
        if col not in ['Date', 'MMWR_week', 'Location']:
            df[col] = df[col].apply(convert_possible_string_to_float)
    return df 


def load_mandate_data():
    """
    Load state-level mandate data from CDC (https://data.cdc.gov/Policy-Surveillance/State-Level-Vaccine-Mandates-All/kw6u-z8u2).
    """
    df = pd.read_csv(PATH_TO_MANDATE_DATA).fillna('None')
    for col in ['date_signed', 'effective_date']:
        df[col] = df[col].apply(convert_datestring_to_datetime)
    return df 


def load_passport_data():
    """
    Load self-coded data about vaccine passports.
    """
    df = pd.read_csv(PATH_TO_PASSPORT_DATA)
    for col in ['date_announced', 'date_effective']:
        df[col] = df[col].apply(convert_datestring_to_datetime)
    return df


def get_percent_change_in_new_counts(ts, interval=14):
    """
    Get trend: compare this past week, ie, sum(day t-6 to t), to 
    previous week, ie, sum(day t-13 to day t-7).
    """
    trend = np.ones(len(ts)) * np.nan 
    for i in range((interval*2)-1, len(ts)):
        num = np.sum(ts[i-interval+1:i+1])  # last week
        denom = np.sum(ts[i-(interval*2)+1:i-interval+1])  # previous week
        trend[i] = 100. * (num-denom) / denom
    return trend 


def get_vaccine_timeseries_for_state(vaccine_df, state_abbr, mode='smoothed daily', 
                                     df_type='state', normalize=False):
    """
    Get total number of people with at least one dose, completely vaccinated,
    and completely vaccinated + boosted over time for a given state.
    Args:
        vaccine_df: pandas DataFrame loaded from PATH_TO_VACCINE_DATA
        state_abbr: two-letter abbreviation for state
        daily: whether to return daily *new* vaccinations, instead of cumulative
    Returns:
        dts: datetimes in chronological order
        one_dose_ts: timeseries corresponding to these dates, for at least one dose
        complete_ts: timeseries corresponding to these dates, for completely vaccinaed
        booster_ts: timeseries corresponding to these dates, for completely vaccinated + boosted
    """
    assert mode in ['cumulative', 'daily', 'smoothed daily', 'percent change in daily']

    if df_type == 'state':  # dataframe loaded by load_vaccine_state_data 
        state_df = vaccine_df[vaccine_df.Location == state_abbr].sort_values('Date')
        dts = state_df.Date.values
        if normalize:
            one_dose_ts = state_df['Administered_Dose1_Pop_Pct'].values / 100.
            complete_ts = state_df['Series_Complete_Pop_Pct'].values / 100.
            booster_ts = state_df['Additional_Doses_Vax_Pct'].values / 100.
        else:
            one_dose_ts = state_df['Administered_Dose1_Recip'].values
            complete_ts = state_df['Series_Complete_Yes'].values
            booster_ts = state_df['Additional_Doses'].values
    
    else:  # dataframe loaded by load_vaccine_county_data
        state_df = vaccine_df[vaccine_df.Recip_State == state_abbr]
        date_sums = state_df.groupby('Date').sum()  # sum over counties per day
        dts = date_sums.index.values  # groupby puts in chronological order
        one_dose_ts = date_sums['Administered_Dose1_Recip'].values
        complete_ts = date_sums['Series_Complete_Yes'].values
        booster_ts = date_sums['Booster_Doses'].values
        if normalize:
            county_pops = state_df.groupby('Recip_County').mean()['Census2019'].values  # mean pop over days per county; there should only be one pop
            total_pop = np.nansum(county_pops)
            one_dose_ts /= total_pop 
            complete_ts /= total_pop 
            booster_ts /= total_pop 

    if 'daily' in mode:
        one_dose_ts = get_new_counts_from_cumulative(one_dose_ts)
        complete_ts = get_new_counts_from_cumulative(complete_ts)
        booster_ts = get_new_counts_from_cumulative(booster_ts)
        if mode == 'smoothed daily':
            one_dose_ts = smooth_new_counts(one_dose_ts)
            complete_ts = smooth_new_counts(complete_ts)
            booster_ts = smooth_new_counts(booster_ts)
        elif mode.startswith('percent change'):
            one_dose_ts = get_percent_change_in_new_counts(one_dose_ts)
            complete_ts = get_percent_change_in_new_counts(complete_ts)
            booster_ts = get_percent_change_in_new_counts(booster_ts)
    return dts, one_dose_ts, complete_ts, booster_ts


def plot_vaccine_timeseries_for_state(vaccine_df, state_abbr, ax=None, 
                                      mode='smoothed daily', df_type='state', 
                                      to_plot=['at least 1 dose', 'complete', 'boosted']):
    """
    Plot proportion of people with at least one dose, completely vaccinated, and completely
    vaccinated + boosted ovr time for a given state.
    Args:
        vaccine_df: pandas DataFrame loaded from PATH_TO_VACCINE_DATA
        state_abbr: two-letter abbreviation for state
        ax (optional): matplotlib axes for plotting
    Returns:
        ax: matplotlib axes with visualizations added
    """
    dts, one_dose_ts, complete_ts, booster_ts = get_vaccine_timeseries_for_state(vaccine_df, state_abbr, 
                                                    mode=mode, df_type=df_type)
    labels = ['at least 1 dose', 'complete', 'boosted']
    timeseries = [one_dose_ts, complete_ts, booster_ts]
    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 5))
    for lbl, ts in zip(labels, timeseries):
        if lbl in to_plot:
            ax.plot_date(dts, ts, linestyle='-', marker=None, label=lbl)
    ax.set_ylabel('%s vaccinations' % mode, fontsize=12)
    ax.grid(alpha=0.3)

    if mode in ['daily', 'smoothed daily']:
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(0, ymax)
    if mode.startswith('percent change'):
        ax.set_ylim(-100, 100)
        xmin, xmax = ax.get_xlim()
        ax.hlines([0], xmin, xmax, color='grey', alpha=0.7)
    return ax


def get_vaccine_timeseries_for_counties_in_state(vaccine_df, state_abbr, normalize=False):
    """
    Get total number / proportion of people with at least one dose, completely vaccinated,
    and completely vaccinated + boosted over time for each county (with nonzero population)
    in a state.
    Args:
        vaccine_df: pandas DataFrame loaded from PATH_TO_VACCINE_DATA
        state_abbr: two-letter abbreviation for state
        normalize: whether to return the proportion of people, instead of number 
    Returns:
        counties: list of county names
        all_dts: list per county of datetimes in chronological order
        all_one_dose: list per county of timeseries for at least one dose
        all_complete: list per county of timeseries for completely vaccinaed
        all_booster: list per county of timeseries for completely vaccinated + boosted
    """
    counties = []
    all_dts = []
    all_one_dose = []
    all_complete = []
    all_booster = []
    state_df = vaccine_df[vaccine_df.Recip_State == state_abbr]
    for cty, subdf in state_df.groupby('Recip_County'):
        county_pop = subdf.iloc[0]['Census2019']
        if not np.isnan(county_pop) and county_pop > 0:  # skip "Unknown" county with nan or zero population
            dts, one_dose, complete, booster = get_vaccine_timeseries_for_state(subdf, state_abbr, normalize=normalize)
            counties.append(cty)
            all_dts.append(dts)  # note: available dates can be different per county 
            all_one_dose.append(one_dose)
            all_complete.append(complete)
            all_booster.append(booster)
    return counties, all_dts, all_one_dose, all_complete, all_booster 


def get_mandate_ranges_for_state(mandate_df, state):
    """
    Get dictionary of mandate type (mandated group, booster or not) mapped to
    list of mandate ranges (from date signed to date effective) for this state.
    Args:
        mandate_df: pandas DataFrame loaded from PATH_TO_MANDATE_DATA
        state: name of state
    Returns:
        mandate2dates: a dict that maps mandate type to list of mandate ranges
    """
    state_df = mandate_df[mandate_df.state == state].sort_values(by='date_signed')
    mandate2ranges = {}
    for group, subdf in state_df.groupby('vaccination_mandate_group'):
        # there can be multiple mandates issued for the same mandate type, eg, if the date effective
        # has been extended
        mandate2ranges[group] = list(zip(subdf.date_signed, subdf.effective_date))
    return mandate2ranges


def plot_mandates_for_state(mandate_df, state, ax=None, only_plot_first=False):
    """
    Plot mandate ranges for healthcare worker and government worker mandates 
    for this state.
    Args:
        mandate_df: pandas DataFrame loaded from PATH_TO_MANDATE_DATA
        state: name of state
        ax (optional): matplotlib axes for plotting
    Returns:
        ax: matplotlib axes with visualizations added
    """
    mandate2ranges = get_mandate_ranges_for_state(mandate_df, state)
    groups = ['Healthcare workers', 'Government workers']
    colors = ['red', 'purple']
    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 5))
    ymin, ymax = ax.get_ylim()
    for g, c in zip(groups, colors):
        if g in mandate2ranges:
            ranges = mandate2ranges[g]
            if only_plot_first:
                ax.vlines([ranges[0][0]], ymin, ymax, color=c, label=g)
            else:
                kwargs = {'alpha': 0.2, 'color': c, 'label': g}
                for start, end in ranges:
                    ax.axvspan(start, end, **kwargs)
                    kwargs['label'] = None  # change to None after first range so we don't label multiple times
    return ax


def plot_vaccines_and_mandates_for_state(vaccine_df, state_abbr, mandate_df, state):
    """
    Plot vaccine timeseries and mandate ranges on the same axes for a given state.
    Args:
        vaccine_df: pandas DataFrame loaded from PATH_TO_VACCINE_DATA
        state_abbr: two-letter abbreviation for state
        mandate_df: pandas DataFrame loaded from PATH_TO_MANDATE_DATA
        state: name of state
    Returns:
        None
    """
    ax = plot_vaccine_timeseries_for_state(vaccine_df, state_abbr, mode='smoothed daily',
                                           to_plot=['at least 1 dose'])
    ax = plot_mandates_for_state(mandate_df, state, ax=ax, only_plot_first=True)
    ax.set_title(state, fontsize=14)
    ax.legend(fontsize=12)
    return ax
