import rdkit.Chem as Chem
import pandas as pd
import numpy as np

'''
Modified version of the code created by Fine et. al. to assign functional groups

https://github.com/chopralab/candiy_spectrum

'''

input_file_path = '/path to input file/'
output_file_path = '/path to output file/'



func_grp_smarts = {'alkane':'[CX4;H0,H1,H2,H4]','methyl':'[CH3]','alkene':'[CX3]=[CX3]','alkyne':'[CX2]#C',
                   'alcohols':'[#6][OX2H]','amines':'[NX3;H2,H1;!$(NC=O)]', 'nitriles':'[NX1]#[CX2]', 
                   'aromatics':'[$([cX3](:*):*),$([cX2+](:*):*)]','alkyl halides':'[#6][F,Cl,Br,I]', 
                   'esters':'[#6][CX3](=O)[OX2H0][#6]', 'ketones':'[#6][CX3](=O)[#6]','aldehydes':'[CX3H1](=O)[#6]', 
                   'carboxylic acids':'[CX3](=O)[OX2H1]', 'ether': '[OD2]([#6])[#6]','acyl halides':'[CX3](=[OX1])[F,Cl,Br,I]',
                   'amides':'[NX3][CX3](=[OX1])[#6]','nitro':'[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]'}


def identify_functional_groups(smiles):
    '''Identify the presence of functional groups present in molecule 
       denoted by inchi
    Args:
        root: (string) path to spectra data
        files: (list) jdx files present in root
        save_path: (string) path to store csv file
        bins: (np.array) used for standardizing
        is_mass: (bool) whether data being parsed is Mass or IR
    Returns:
        mol_func_groups: (list) contains binary values of functional groups presence
                          None if inchi to molecule conversion returns warning or error
    '''
    
    try:
        #Convert inchi to molecule
        mol = Chem.MolFromSmiles(smiles)#, treatWarningAsError=True)

        mol_func_grps = []

        #populate the list with binary values
        for _, func_struct in func_grp_structs.items():
            struct_matches = mol.GetSubstructMatches(func_struct)
            contains_func_grp = int(len(struct_matches)>0)
            mol_func_grps.append(contains_func_grp)
        return mol_func_grps
    except:

        return None

def save_target_to_csv(smiles_df, save_path, func_grp_structs):
    '''Save the target dataframe as csv to path
    Args:
        smiles_df: (pd.DataFrame) contains CAS and Inchi of molecules
        save_path: (string) path to store csv file
    Returns:
        None
    '''
    column_names = list(func_grp_structs.keys())
    column_names = ['id', 'cano_smi', 'smiles', 'spectrum'] + column_names
    target_df = pd.DataFrame(index = smiles_df.index, columns = column_names)

    #Iterate the rows, don't use df.apply since a list is being returned.
    for ind, (_, row) in enumerate(smiles_df.iterrows()):

        labels = identify_functional_groups(row['cano_smi'])
        new_row = [ind, row['cano_smi'], row['smiles'], row['spectrum']] +labels
        target_df.iloc[ind, :] = new_row

    target_df.dropna(inplace = True)
    target_df.to_pickle(save_path)

df = np.load(input_file_path, allow_pickle=True)

func_grp_structs = {func_name : Chem.MolFromSmarts(func_smarts) for func_name, func_smarts in func_grp_smarts.items()}

save_target_to_csv(df, output_file_path, func_grp_structs)

