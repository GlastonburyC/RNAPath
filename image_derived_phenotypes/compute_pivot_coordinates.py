import numpy as np
import pandas as pd
from math import exp
from scipy.stats import gmean
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--tissue_name", type=str, default=None)
parser.add_argument("--idps_dir", type=str, default=None)
args = parser.parse_args()

def orthonormal(i, D):
    return np.sqrt((D - i) / (D - i + 1))

def orthogonal(i, D):
    return 1

norm_functions = {
    "orthonormal": orthonormal,
    "orthogonal": orthogonal
}

'''
    Function to compute pivot coordinates from a compositional dataset
'''
def pivotCoord(x, fast=False, method = "pivot", base = exp(1), norm = "orthonormal"):
    if x.shape[1] < 2:
        raise ValueError("data must be of dimension greater equal 2")
    if np.any(x < 0):
        raise ValueError("negative values not allowed")

    D = x.shape[1]
    x_ilr = np.empty((x.shape[0], x.shape[1] - 1))
    for i in range(0, x_ilr.shape[1]):
        norm_func = norm_functions.get(norm, orthonormal)
        gm = x.iloc[:,(i+1):D].apply(gmean, axis=1)
        x_ilr[:,i] = norm_func(i+1, D) * np.log(gm/(x.iloc[:,i]))  
    
    x_ilr = pd.DataFrame(x_ilr)
    
    return -x_ilr

# Open compositional dataset
df = pd.read_csv(f'{args.idps_dir}/{args.tissue_name}_compositional.csv', index_col=0)
# set values < 1e-3 to 1e-3 to avoid zeros.
df[df < 1e-3] = 1e-3
copy = df.copy()
out_df = df.copy()

# iteration over df columns
for col in df.columns.tolist():
    df = copy
    # move current column to the first position
    df.insert(0, col, df.pop(col))
    # compute pivot coordinates
    df_pivot = pivotCoord(df).set_index(df.index)
    # store pivot values in the output dataframe for the current idp
    out_df[col] = df_pivot[0]

# check if infinite rows are present
inf_rows = out_df.index[np.isinf(out_df).any(1)]

# store pivot coordinates csv
out_df.drop(inf_rows).to_csv(f'{args.idps_dir}/{args.tissue_name}_pivot.csv')
