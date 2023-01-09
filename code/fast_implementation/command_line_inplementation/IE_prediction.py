# Scikit-Learn
from sklearn.preprocessing import quantile_transform

# Data Structure and Math
import pandas as pd
import numpy as np

# Descriptor Calculators
import padelpy
from padelpy import padeldescriptor
from rdkit import Chem
import rdkit.Chem.Descriptors as Descriptors
from mordred import Calculator, descriptors

# Model IO
from joblib import load

import warnings
warnings.simplefilter('ignore', UserWarning)

data = pd.read_csv('https://raw.githubusercontent.com/REMUU/IonEner-Pred/main/datasets/full_set/nist_organic_full_set.csv')

# get name of descriptors
des_cols = data.iloc[:,5:].columns

new_records = []
count_input = 0

# check the mw and element types of the molecule
def checkAD(smi):
    appDomain = ['H', 
                'B', 'C', 'N', 'O','F',
                'Si', 'P', 'S', 'Cl',
                'Ge', 'As', 'Se', 'Br',
                'I',]

    atomInAppDomain = []
    atomNotInAppDomain = []
    molNotInAppDomain = []

    mol = Chem.MolFromSmiles(smi)
    mw = Descriptors.ExactMolWt(Chem.MolFromSmiles(smi))
    if mw < 13. or mw > 671.:
        molNotInAppDomain.append(smi)

    for atom in mol.GetAtoms():
        at = atom.GetSymbol()
        if at in appDomain and at not in atomInAppDomain:
            atomInAppDomain.append(at)
            continue
        if at not in appDomain and at not in atomNotInAppDomain:
            atomNotInAppDomain.append(at)

    aInA = ''
    for i in atomInAppDomain:
        if aInA == '':
            aInA = i
            continue
        aInA = aInA + ', ' + i

    aNotInA = ''
    for i in atomNotInAppDomain:
        if aNotInA == '':
            aNotInA = i
            continue
        aNotInA = aNotInA + ', ' + i

    if atomNotInAppDomain != []:
        print('\n\n\n')
        print('!!!!!!!!!!!!!!!!!!!!!!WARNING!!!!!!!!!!!!!!!!!!!!!!')
        print("Following elements are not presented in the training of models. Our parameters may generate very different results as the user's expectation. Please Use with Cautions.")
        print(aNotInA)
        print('\n\n\n')
    elif molNotInAppDomain != []:
        print('\n\n\n')
        print('!!!!!!!!!!!!!!!!!!!!!!WARNING!!!!!!!!!!!!!!!!!!!!!!')
        print("This molecule have the relative molecular weight out of the coverage of the dataset. Our parameters may generate very different results as the user's expectation. Please Use with Cautions.")
        print('\n\n\n')
    
while len(new_records) < 2:
    count_input += 1
    print('\n')
    smi_input = input('Please type in your SMILES {}: '.format(count_input))
    checkAD(smi_input)
    print('\n')
    rad_input = input('Please state if the molecule is radical (1) or non-radical (0): ')
    new_records.append([smi_input, rad_input])

while len(new_records) >= 2:
    print('\n')
    entry_input = input('Do you want to add new SMILES? (Y or N): ')
    if entry_input == 'N' or entry_input == 'n':
        break
    count_input += 1
    print('\n')
    smi_input = input('Please type in your SMILES {}: '.format(count_input))
    checkAD(smi_input)
    print('\n')
    rad_input = input('Please state if the molecule is radical (1) or non-radical (0): ')
    new_records.append([smi_input, rad_input])

# calculate descriptors
calc = Calculator(descriptors, ignore_3D=True)
padeldescriptor(descriptortypes='./descriptors.xml')
df_pred = pd.DataFrame()

for rec in new_records:
    smi = rec[0]
    rad = rec[1]
    try:
        mol = Chem.MolFromSmiles(smi)
    except:
        print('SMILES: {} is invalid, please check your input.')
    # calculate and merge descriptors
    mordred_descriptors = calc(mol)
    mordred_df = pd.DataFrame([float(i[1]) for i in mordred_descriptors.items()], index=[str(i[0]) for i in mordred_descriptors.items()]).T
    maccs_pubchem_descriptors = padelpy.from_smiles(smi, fingerprints=True, descriptors=False, output_csv=None)
    maccs_pubchem_df = pd.DataFrame([[float(i[1]) for i in maccs_pubchem_descriptors.items()]], columns=[i[0] for i in maccs_pubchem_descriptors.items()])
    descriptor_df = pd.concat([pd.DataFrame([[smi,rad]],columns=['SMILES','BinaryRadical']),mordred_df,maccs_pubchem_df],axis=1)
    df_pred = pd.concat([df_pred,descriptor_df], axis=0)

X_test = quantile_transform(df_pred[des_cols])

ie_preds = []
fold_counts = [i+1 for i in range(10)]

for f in fold_counts:
    reg = load('{}_fold_svr.joblib'.format(f))
    ie_preds.append(reg.predict(X_test))

ie_preds_avg = list(np.average(ie_preds,axis=0))

output_dict = {'SMILES':[], 'Avg IE / eV':[]}

for i in range(len(ie_preds_avg)):
    output_dict['SMILES'].append(df_pred.iloc[i,0])
    output_dict['Avg IE / eV'].append(round(ie_preds_avg[i],2))

output_df = pd.DataFrame(output_dict)
print('\n\n\n')
print('Prediction Result is as Following:')
print(output_df)