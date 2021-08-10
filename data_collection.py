#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 14:00:13 2021

@author: JoshM

The functions in this file collect data from the World Bank API:
https://datahelpdesk.worldbank.org/knowledgebase/topics/125589-developer-information

The World Bank has a large database of statistics called "World Development 
Indicators". These are measures such as GDP or population for each country in
the world. A human-readable list of indicators can be found at
https://data.worldbank.org/indicator

Each indicator has a code such as 'NY.GDP.MKTP.CD' which is shown by the urls 
in the indicator list and are used by the API. Thus these codes are needed when
specifying which data to collect. The data is also stored using the codes as
column headers for ease of manipulation and only converted into the descriptive
labels when loaded by analysis.py.
"""

import requests
from requests_cache import CachedSession
import os.path
import pandas as pd
from time import gmtime, sleep

# Delay between subsequent non-cached requests in seconds
DELAY = .75
# Filenames and locations for data storage
FOLDER_DIR = './data/'
COUNTRY_INDEX_DIR = FOLDER_DIR + 'countries.csv'
FULL_DATA_DIR = FOLDER_DIR + 'values.csv'
# Cache namespace and backend type
CACHE = 'WB_cache'
CACHE_BACKEND = 'sqlite'
# Indicator codes and names. The keys function as a list of which data to
# collect and are the primary use of this dict. The value mappings simply
# allow renaming of fields with human-readable labels.
# The API indicator codes must always be used as keys; values can be any
# descriptive names.
INDICATORS = {'NY.GDP.MKTP.CD':'GDP (current US\$)',
              'NY.GDP.MKTP.KD.ZG':'GDP growth (annual \%)',
              'NY.GDP.PCAP.CD':'GDP per capita (current US\$)',
              'NY.GDP.PCAP.KD.ZG':'GDP per capita growth (annual \%)',
              'SP.POP.TOTL':'Population, total',
              'SP.POP.GROW':'Population growth (annual \%)',
              'SP.URB.TOTL.IN.ZS':'Urban population (\% of total population)',
              'FP.CPI.TOTL.ZG':'Inflation, consumer prices (annual \%)',
              'EG.USE.ELEC.KH.PC':'Electric power consumption (kWh per capita)'
              }

def load_data(force_reload=False):
    """Loads the data matrix as a dataframe from the stored file, generating
    the file first if necessary.

    Parameters
    ----------
    force_reload : bool, optional
        If True, always generates the data files from scratch, otherwise reads
        existing files. The default is False.

    Returns
    -------
    DataFrame
        The dataframe representing the data matrix stored at FULL_DATA_DIR.

    """
    
    if (not os.path.exists(FULL_DATA_DIR)) or force_reload:
        generate_data(force_reload=force_reload)
    return pd.read_csv(FULL_DATA_DIR)

def generate_data(force_reload=False):
    """Generates the data matrix consisting of basic country information and
    the indicators listed in the INDICATORS constant and stores it in 
    FULL_DATA_DIR in csv format.
    
    A country index file is generated at the path given by COUNTRY_INDEX_DIR 
    in csv format and contains the fields:
        'ID': 3 letter code used to index countries in the World Bank
            API.
        'Name': The name of the country.
        'Region': 3 letter code for the region in which the country is located.
        'Income Level': The World Bank classification for level of income 
            of a country.
        'Lending Type': The type of lending for which a country is 
            classified by World Bank.
    
    For each country in the index, values are collected for each indicator 
    listed in the INDICATORS constant. This data contains the fields:
        'ID': The 3 letter code used for indexing countries in the World Bank
            API
        {Indicator ID: Indicator Value} for each indicator
        
    The data from the country index is merged with the data for the indicators
    using country IDs as the key. This full set of data is then stored at the 
    path given by FULL_DATA_DIR in csv format.

    Parameters
    ----------
    force_reload : bool, optional
        If True, always generates the country index from scratch; otherwise 
        uses existing country index file, if one exists.
        The default is False.

    Returns
    -------
    None.

    """
    
    if (not os.path.exists(COUNTRY_INDEX_DIR)) or force_reload:
        if not os.path.exists(FOLDER_DIR):
            os.mkdir(FOLDER_DIR)
        # Populate country list
        country_df = build_country_df()
        # Write file
        country_df.to_csv(COUNTRY_INDEX_DIR, index=False)
    else:
        # Read file
        country_df = pd.read_csv(COUNTRY_INDEX_DIR)
    # Generate indicator dataframe
    indicator_df = build_indicator_df(country_df['ID'], INDICATORS.keys())
    # Merge dataframes
    full_df = pd.merge(country_df, indicator_df, on='ID')
    # Write file
    full_df.to_csv(FULL_DATA_DIR, index=False)
    print("Data successfully stored in {0}".format(FULL_DATA_DIR))
    return

#TODO: Switch over storage to SQL
#TODO: Support repairing partially generated sets (missing rows)
#TODO: Support adding indicators to existing data (add more columns)

def try_request(url, params=None, session=None, num_tries=5, delay=DELAY):
    """Makes a GET request to the specified url until a successful response 
    code (200) is returned. Attempts up to num_tries times with specified delay
    between attempts. Only considers an unsuccessful http response as a failure
    , successfully returned error messages from an API will not trigger request
    retries.

    Parameters
    ----------
    url : str
        The url to which the the request is sent.
    params : dict of form {param_name:param_value}, optional
        The parameters to include in the request.
    session : Session or CachedSession, optional
        The session or cached session with which to make the request. If none
        is provided, a non-sessioned request is made.
    num_tries : int, optional
        The number of times to try the request before giving up.
        The default is 5.
    delay: float, optional
        The time in seconds to wait between retrying requests so as to not
        overwhelm an API. The default is defined by the modules DELAY constant.

    Returns
    -------
    Response or None
        The response provided by the server, if successful, or None if a
        successful response could not be retrieved.

    """
    # Define get function as either using a session or not
    if session is None:
            session = requests
    r = session.get(url, params=params)
    if r.status_code == 200:
        # Success
        return r
    else:
        # Output error message
        print("Error: {0}: {1} from URL: {2}".format(
            r.status_code, r.reason, r.url))
    i = 1
    while i < num_tries:
        sleep(delay)
        print("Retrying...{0}".format(i))
        r = session.get(url, params=params)
        if r.status_code == 200:
            # Only print success message if first attempt was unsuccessful
            print("Success!")
            return r
        i += 1
    print("Failed after {0} Attempts!".format(i))
    return None

def build_country_df():
    """Retrieves the list of countries from the World Bank API and constructs
    a dataframe of basic information about them. This is primarily used as an 
    index for collecting more detailed information through the indicator API.

    Returns
    -------
    DataFrame
        The dataframe fields are:
            'ID': 3 letter code used to index countries in the World Bank
                API.
            'Name': The name of the country.
            'Region': 3 letter code for the region in which the country is
                located.
            'Income Level': The World Bank classification for level of income 
                for a country.
            'Lending Type': The type of lending for which a country is 
                classified by World Bank.

    """
    print("Building country list...")
    session = CachedSession(CACHE, backend=CACHE_BACKEND)
    params = {'format':'json',
              'per_page':500}
    url = 'http://api.worldbank.org/v2/country'
    
    r = try_request(url,params=params,session=session)
    if r is None:
        # Unsuccessful http response
        print("Country list could not be retrieved.\nAborting...")
        return
    j = r.json()
    
    data = list_country_info(j)
    # Should always fit on one page but just in case, iterate over pages if
    # there are multiple
    if j[0]['pages'] > 1:
        for i in range(2,j[0]['pages']+1):
            params['page'] = i
            r = try_request(url, params=params, session=session)
            data.extend(list_country_info(r.json()))
    print("Country list built with {0} rows.".format(len(data)))
    return pd.DataFrame(data)
            
def list_country_info(obj):
    """Produces a list of extracted info from a single page of JSON data from
    a country list query. Filters the None values returned from data about
    aggregates listed on the page.

    Parameters
    ----------
    obj : dict
        Dict representing JSON data from country list query.

    Returns
    -------
    list of dicts
        List of the extracted info for each country on the page.

    """
    
    # Extract info and filter None values
    return [i for i in (extract_country_info(c) for c in obj[1])
            if i is not None]

def extract_country_info(obj):
    """Extracts basic information about a country from its entry in the
    JSON data from a country list query.

    Parameters
    ----------
    obj : dict
        Dict representing JSON of a single entry from a country list query.

    Returns
    -------
    dict or None
        Dict of country info with keys:
            ID, Name, Region, Income Level, Lending Type.
        Returns None if the entry is for an Aggregate (such as a region)

    """
    
    # Not interested in aggregates
    if obj['region']['value'] == 'Aggregates':
        return
    # Extract fields
    info = {'ID':obj['id'],
            'Name':obj['name'],
            'Region':obj['region']['value'],
            'Income Level':obj['incomeLevel']['value'],
            'Lending Type':obj['lendingType']['value']}
    return info

def _print_age_warning(obj):
    """Prints warning about the date that certain data was collected.

    Parameters
    ----------
    obj : dict
        Dict representing JSON of an entry from an indicator query.

    Returns
    -------
    None.

    """
    
    s = "Warning: {0} data for {1} is from {2}"
    print(s.format(obj['indicator']['value'],
             obj['country']['value'],
             obj['date']))
    return

def extract_indicator_values(obj, warning_age = 10):
    """Extracts values of indicators from JSON of an indicator query for a
    specific country.

    Parameters
    ----------
    obj : dict
        Dict representing JSON data from an indicator query for a country.
    warning_age : int, optional
        The minimum age for which a warning will be printed to inform the user
        about old, possibly obselete data. The default is 10.

    Returns
    -------
    dict
        Dict of indicator values with form {indicator_id:indicator_value}.

    """
    
    data = {}
    for i in obj[1]:
        if gmtime().tm_year - int(i['date']) > warning_age:
            # Data is more than warning_age years old
            _print_age_warning(i)
            #TODO possibly: keep list of aged fields for doing something with
        # Data order not guaranteed by API, index by key
        data[i['indicator']['id']] = i['value']
    return data

def build_indicator_df(country_ids, indicator_ids):
    """Retrieves the values of the given indicators for each of the given
    countries and constructs a dataframe of the information. The country and
    indicator IDs must match those used by the World Bank API. ISO2 codes for
    country IDs are also supported.

    Parameters
    ----------
    country_ids : array-like
        List of ID or ISO2 codes of countries for which information is to be
        collected.
    indicator_ids : array-like
        List of indicator codes for which values are to be collected.

    Returns
    -------
    DataFrame
        The dataframe containing the retrieved information. The fields are:
            'ID': The 3 letter code used for indexing countries in the World
            Bank API
            {Indicator ID: Indicator Value} for each indicator
        Missing entries may be present in cases where no data was returned for
        a specific indicator for a specific country.

    """
    print("Collecting indicator data. This may take several minutes...")
    session = CachedSession(CACHE, backend=CACHE_BACKEND)
    params = {'format':'json',
              'per_page':60, # Maximum of 60 indicators allowed per request
              'mrnev':1, # Return only most recent non-empty value
              'source':2} # Use values from World Development Indicators table
    indicator_str = ';'.join(indicator_ids)
    url = 'http://api.worldbank.org/v2/country/{0}/indicator/'+indicator_str
    #TODO: rewrite to use country/all
    data = []
    for i in country_ids:
        # Request indicator data on country
        r = try_request(url.format(i), params=params, session=session)
        if r is None:
            # Unsuccessful response
            print("Skipping...")
            sleep(DELAY)
            continue
        try:
            d = {'ID':i, **extract_indicator_values(r.json())}
            data.append(d)
        except Exception as e:
            # Rrror code from API or empty response
            print(type(e).__name__+':', e,
                  'while processing JSON from {0}\nSkipping...'.format(r.url))
        
        if not getattr(r, 'from_cache', False):
            # Only wait if response was not from cache
            sleep(DELAY)
    print("Indicator data collected with {0} rows.\n {1} rows missing data!".format(len(data), len(country_ids)-len(data)))
    return pd.DataFrame(data)

