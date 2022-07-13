from contextlib import closing
from requests import get
from bs4 import BeautifulSoup
import pandas as pd
import time
import numpy as np

# the list of IE number we want to screen
def ierange():
    return np.arange(20, 40.1, 0.1)

# connect the web server
def getresponse(url):
    try:
        with closing(get(url, stream=True)) as resp:
            content_type = resp.headers['Content-Type'].lower()
            if resp.status_code == 200 and content_type is not None and content_type.find('html') > -1:
                return resp.content
            else:
                return None
    except:
        print('Page Not Found')
        return None

# retrieve the webpage and parse the web by bs4
def getwebpage(ie):
    url = 'https://webbook.nist.gov/cgi/cbook.cgi?Value={}&VType=IE&Formula=&AllowExtra=on&Units=SI'.format(ie)
    resp_html = getresponse(url)
    try:
        html = BeautifulSoup(resp_html, 'html.parser')
        main = html.find('main', id='main')
    except:
        main = None
    return main

# extract CAS Name, Link and IE value from the parsed html
def getval(main):
    if main is not None:
        ie_dict = {'CAS Name': [],
                   'CAS Link': [],
                   'IE / eV': []
                   }
        ie_records = main.find_all('strong')
        for record in ie_records:
            ie_val = record.text.strip()
            try:
                ie_dict['IE / eV'].append(ie_val)
            except:
                ie_dict['IE / eV'].append('-1 eV')
        compound_records = main.find_all('a')
        for compound in compound_records:
            cas_name = compound.text
            cas_link = compound['href']
            try:
                ie_dict['CAS Name'].append(cas_name)
                ie_dict['CAS Link'].append(cas_link)
            except:
                ie_dict['CAS Name'].append('Name Not Found')
                ie_dict['CAS Link'].append('Link Not Found')
    return ie_dict


# define the function to excecute the scrapper with previously defined tools
def nistspider(ierange):
    df = pd.DataFrame(columns=['CAS Name', 'CAS Link', 'IE / eV'])
    byhand_dict = {'Bad IE': []}
    start = time.time()
    for ie in ierange:
        main = getwebpage(ie)
        # use sleep to avoid ip ban
        time.sleep(np.random.randint(0, 1))
        if 'No Matching' in str(main.find_all('h1')):
            print('There is no compounds having IE of %f'%ie)
            continue
        if 'Search Results' in str(main.find_all('h1')):
            ie_dict = getval(main)
            df_temp = pd.DataFrame(ie_dict)
            df = pd.concat([df, df_temp], axis=0)
            continue
        else:
            # This part is set to handle page that directly links to a single molecule but not a page with IE number
            byhand_dict['Bad IE'].append(ie)
    df_byhand = pd.DataFrame(byhand_dict)
    print(time.time() - start)
    return df, df_byhand


table, table_byhand = nistspider(ierange())
table.to_csv(r'output_csv_path', index=False)
table_byhand.to_csv(r'output_bad_struc_csv_path', index=False)