# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 14:00:13 2021

@author: JoshM
"""

from requests_cache import CachedSession
import os.path
import pandas as pd
from time import gmtime, sleep

#delay between subsequent non-cached requests in seconds
DELAY = 0.75
#filenames and locations for data storage
FOLDER_DIR = './data/'
COUNTRY_INDEX_DIR = FOLDER_DIR + 'countries.csv'
FULL_DATA_DIR = FOLDER_DIR + 'values.csv'
#cache name and backend
CACHE = 'WB_cache'
CACHE_BACKEND = 'sqlite'
#indicator codes and names
INDICATORS = {'NY.GDP.MKTP.CD':'GDP (current US$)',
              'NY.GDP.MKTP.KD.ZG':'GDP growth (annual %)',
              'NY.GDP.PCAP.CD':'GDP per capita (current US$)',
              'NY.GDP.PCAP.KD.ZG':'GDP per capita growth (annual %)',
              'SP.POP.TOTL':'Population, total',
              'SP.POP.GROW':'Population growth (annual %)',
              'SP.URB.TOTL.IN.ZS':'Urban population (% of total population)',
              'FP.CPI.TOTL.ZG':'Inflation, consumer prices (annual %)',
              'EG.USE.ELEC.KH.PC':'Electric power consumption (kWh per capita)'
              }

def try_request(session, url, params, num_tries=5):
    #TODO possibly: support non-session requests
    r = session.get(url, params=params)
    if r.status_code == 200:
        return r
    else:
        print("Error: {0}: {1} from URL: {2}".format(
            r.status_code, r.reason, r.url))
    i = 1
    while i < num_tries:
        sleep(DELAY)
        print("Retrying...{0}".format(i))
        r = session.get(url, params=params)
        if r.status_code == 200:
            print("Success!")
            return r
        i += 1
    print("Failed after {0} Attempts!".format(i))
    return None

def load_data(force_reload=False):
    if (not os.path.exists(FULL_DATA_DIR)) or force_reload:
        generate_data(force_reload=force_reload)
    return pd.read_csv(FULL_DATA_DIR)

def generate_data(force_reload=False):
    if (not os.path.exists(COUNTRY_INDEX_DIR)) or force_reload:
        #populate country list
        country_df = build_country_df()
        #write file
        country_df.to_csv(COUNTRY_INDEX_DIR, index=False)
    else:
        country_df = pd.read_csv(COUNTRY_INDEX_DIR)
    #generate
    indicator_df = build_indicator_df(country_df['ID'], INDICATORS.keys())
    full_df = pd.merge(country_df, indicator_df, on='ID')
    full_df.to_csv(FULL_DATA_DIR, index=False)
    print("Data successfully stored in {0}".format(FULL_DATA_DIR))
    return

#TODO: complete partially generated sets (missing rows)

def build_country_df():
    session = CachedSession(CACHE, backend=CACHE_BACKEND)
    params = {'format':'json',
              'per_page':500}
    url = 'http://api.worldbank.org/v2/country'
    
    r = try_request(session,url,params)
    if r is None:
        print("Country list could not be retrieved.\nAborting...")
        return
    j = r.json()
    
    data = list_country_info(j)
    #should always fit on one page but just in case
    if j[0]['pages'] > 1:
        for i in range(2,j[0]['pages']+1):
            params['page'] = i
            r = session.get(url, params=params)
            data.extend(list_country_info(r.json()))
    #TODO: abstract df creation
    return pd.DataFrame(data, columns=['ID','Name','Region','Income','Lending'])
            
def list_country_info(obj):
    #extract info and filter None values
    return [i for i in (extract_country_info(c) for c in obj[1])
            if i is not None]

def extract_country_info(obj):
    #not interested in aggregates
    if obj['region']['value'] == 'Aggregates':
        return
    #extract fields #TODO: rework as dict for consistency
    l = [obj['id'], #code
         obj['name'], #name
         obj['region']['id'], #region
         obj['incomeLevel']['id'], #income level
         obj['lendingType']['id']] #lending type
    return l

def _print_age_warning(obj):
    s = "Warning: {0} data for {1} is from {2}"
    print(s.format(obj['indicator']['value'],
             obj['country']['value'],
             obj['date']))
    return

def extract_indicator_values(obj, warning_age = 10):
    #store country id for later use as join key
    #data = {'ID':obj[1][0]['country']['id']}
    data = {}
    for i in obj[1]:
        if gmtime().tm_year - int(i['date']) > warning_age:
            #data is more than warning_age years old
            _print_age_warning(i)
            #TODO possibly: keep list of aged fields for addressing
        #data order not guaranteed, index by key
        data[i['indicator']['id']] = i['value']
    return data

def build_indicator_df(country_ids, indicator_ids):
    session = CachedSession(CACHE, backend=CACHE_BACKEND)
    params = {'format':'json',
              'per_page':50,
              'mrnev':1,
              'source':2}
    indicator_str = ';'.join(indicator_ids)
    url = 'http://api.worldbank.org/v2/country/{0}/indicator/'+indicator_str
    
    data = []
    for i in country_ids:
        #request indicator data on country
        r = try_request(session, url.format(i), params=params)
        if r is None:
            print("Skipping...")
            sleep(DELAY)
            continue
        try:
            d = {'ID':i, **extract_indicator_values(r.json())}
        except Exception as e:
            print(type(e).__name__+':', e,
                  'while processing JSON from {0}\nSkipping...'.format(r.url))
        data.append(d)
        if not getattr(r, 'from_cache', False):
            sleep(DELAY)
    return pd.DataFrame(data)
