# import required libraries
from rdkit import Chem
from mordred import Calculator, descriptors
import pandas as pd
import numpy as np

# read SMILES from the csv
df_input = pd.read_csv(r'input_csv_path')
# ignpore 3D descriptors during the calculation
calc = Calculator(descriptors, ignore_3D=True)
df_output = pd.DataFrame()
# iterate through all recored to get their descriptors
for i in range(len(df_input['SMILES'])): 
    # read molecules' information from the input csv file
    name = df_input.iloc[i][0]
    inchi = df_input.iloc[i][1]
    link = df_input.iloc[i][2]
    smi = df_input.iloc[i][3]
    ie = df_input.iloc[i][4]
    # read the SMILES if the structure is not valid then let mol be None
    try:
        mol = Chem.MolFromSmiles(smi)
    except:
        mol = None
    if mol is None:
        continue
    descriptor_dict = calc(mol)
    # organize the structure of Dataframe
    right = pd.DataFrame(descriptor_dict, index=descriptor_dict.keys()).T
    left = pd.DataFrame(np.array([[name], [inchi], [link], [smi], [ie]]).T,
                        columns=['CAS Name', 'InChI', 'CAS Link', 'smiles', 'IE'])
    df_output = pd.concat([df_output, pd.concat([left, right], axis=1)], axis=0)
# write the output csv
df_output.to_csv(r'output_csv_path', index=False)