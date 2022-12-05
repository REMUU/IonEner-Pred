# import required libraries
import pandas as pd
from cirpy import resolve
from rdkit import Chem

# import raw data and define the data structure for output
df = pd.read_csv(r'input_csv_path')
bad_mol = pd.DataFrame(columns=['Bad Molecule Name'])
dataset = pd.DataFrame(columns=['CAS Name', 'InChI', 'CAS Link', 'smiles', 'IE / eV'])

# iteration on try to get SMILES from InChI and the Name of molecules
for i in range(len(df)):
    name = df.iloc[i][0]
    inchi = df.iloc[i][1]
    link = df.iloc[i][2]
    ie = df.iloc[i][3]
    # try to use InChI to get SMILES
    if inchi is not None:
        try:
            smiles1 = Chem.MolToSmiles(Chem.MolFromInchi(inchi))
            smiles2 = resolve(name, 'smiles')
        except:
            smiles1 = None
            smiles2 = None
            smiles = None
        smiles = None
        if smiles1 is not None:
            smiles = smiles1
        elif smiles2 is not None and smiles1 is None:
            smiles = smiles2
        if smiles is None:
            smiles = None
            bad_mol = pd.concat([bad_mol, pd.DataFrame({'Bad Molecule Name': [name]})], axis=0, sort=False)
    # try to use the name of molecule to get SMILES when the InChI is failed
    else:
        try:
            smiles = resolve(name, 'smiles')
        except:
            smiles = None
            bad_mol = pd.concat([bad_mol, pd.DataFrame({'Bad Molecule Name': [name]})], axis=0, sort=False)
    # update the dataset Dataframe by concat Dataframe of the molecule and the dataset
    temp_df = pd.DataFrame({'CAS Name': [name],
                            'InChI': [inchi],
                            'CAS Link': [link],
                            'smiles': [smiles],
                            'IE / eV': [ie]})
    dataset = pd.concat([dataset, temp_df], axis=0, sort=False)

# write the dataset and wrong molecules into respective CSV files
dataset.to_excel(r'output_excel_path', index=False)
bad_mol.to_csv(r'output_bad_mol_csv_path', index=False)