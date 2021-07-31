# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 14:00:13 2021

@author: JoshM
"""

import requests
from requests_cache import CachedSession
import os
import pandas as pd


def print_warning(r, message=None):
    print("Warning: {0}:{1} from URL {2}".format(r.status_code,
                                                 r.reason,
                                                 r.url))
    if message:
        print(message)

def build_country_index():
    session = CachedSession('WB_cache')
    params = {'format':'json',
              'per_page':500}
    url = 'http://api.worldbank.org/v2/country'
    
    r = session.get(url, params=params)
    if r.status_code != 200:
        print_warning(r, "Aborting...")
        return
    j = r.json()
    
    data = list_country_info(j)
    #should always fit on one page but just in case
    if j[0]['pages'] > 1:
        for i in range(2,j[0]['pages']+1):
            params['page'] = i
            r = session.get(url, params=params)
            data.extend(list_country_info(r.json()))
    return pd.DataFrame(data, columns=['ID','Name','Region','Income','Lending'])
            
def list_country_info(obj):
    #extract info and filter None values
    return [i for i in (extract_country_info(c) for c in obj[1]) if i is not None]

def extract_country_info(obj):
    #not interested in aggregates
    if obj['region']['value'] == 'Aggregates':
        return None
    #extract fields
    l = [obj['id'], #code
         obj['name'], #name
         obj['region']['id'], #region
         obj['incomeLevel']['id'], #income level
         obj['lendingType']['id']] #lending type
    return l

#TODO: file management
#TODO: build indicator list
#TODO: extract indicator values for countries