from contextlib import closing
from requests import get
from bs4 import BeautifulSoup
import pandas as pd
import time
from numpy import random
import re

# get the list of CAS identity from the csv
def get_cas_txt(path):
    with open(path, 'r') as f:
        cas_list = []
        lines = f.readlines()
        for line in lines:
            if 'html' in line:
                continue
            cas = line.split()[0]
            cas_list.append(cas)
    return cas_list

# retrieve the CAS number by canonical representation
def get_cas_csv(path):
    cas_list = []
    df = pd.read_csv(path)
    cas_col = df['CAS Link']
    for cas in cas_col:
        cas_list.append(re.findall('ID=C(.+)&Units=SI', cas)[0])
    return cas_list

# connect the web server
def get_response(url):
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

# locate and return the table and main section in the webpage
def get_webpage(CAS):
    url = 'https://webbook.nist.gov/cgi/cbook.cgi?ID=C{}&Mask=20#Ion-Energetics'.format(CAS)
    resp_html = get_response(url)
    try:
        html = BeautifulSoup(resp_html, 'html.parser')
        table = html.find('table', attrs={'aria-label': 'One dimensional data'})
        main = html.find('main', id='main')
    except:
        main = None
        table = None
    return table, main

# extract the InChI representation of molecules
def get_val(CAS):
    table, main = get_webpage(CAS)
    if table is not None and main is not None:
        row = table.find_all('td', class_='right-nowrap')
        compound_name = main.find_all('h1', id='Top')[0].text
        _inchi = main.find_all('span')[0].text
        stdinchikey = main.find_all('span')[1].text

        if 'InChI' not in _inchi:
            _inchi = 'No-InChi'
            print(compound_name, ' have no InChi whose cas is ', CAS)
        try:
            ie_split = row[0].text.split()
            ie_val =  float(ie_split[0])
            ie_std = float(ie_split[2])
        except:
            ie_val,ie_std = None, None
            compound_name, stdinchikey, _inchi = None, None, None
    else:
        ie_val,ie_std = None, None
        compound_name, stdinchikey, _inchi = None, None, None
    return ie_val, ie_std, compound_name, stdinchikey, _inchi

# define the function to excecute the scrapper with previously defined tools
def nist_spider(path):
    start = time.time()
    bad_cas = {'Bad CAS': []}

    data = {'Name': [],
            'InChi': [],
            'InChiKey': [],
            'Ionisation Energy / eV': [],
            'Standard Deviation': []}
    for CAS in get_cas_csv(path):
        ie_val, ie_std, compound_name, stdinchikey, _inchi = get_val(CAS)
        if ie_val is None or compound_name is None:
            continue
        if ie_val > 100:
            print('Be Careful with molecule', CAS, '.')
            bad_cas['Bad CAS'].append(CAS)
        data['Name'].append(compound_name)
        data['InChi'].append(_inchi)
        data['InChiKey'].append(stdinchikey)
        data['Ionisation Energy / eV'].append(ie_val)
        data['Standard Deviation'].append(ie_std)
        # rest the spider in case of IP ban
        time.sleep(random.randint(0, 1))
    df = pd.DataFrame(data)
    df.to_csv(r'output_csv_path', index=False)
    pd.DataFrame(bad_cas).to_csv(r'output_bad_struc_csv_path', index=False)
    print(time.time() - start)


nist_spider(r'input_csv_path')

