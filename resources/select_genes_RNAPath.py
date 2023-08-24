import pandas as pd
import argparse

rnaseq = pd.read_csv('./rnaseq_complete.csv', low_memory=False)

parser = argparse.ArgumentParser()
parser.add_argument("--tissue_code", type=str, default=None)
args = parser.parse_args()



# select the slides belonging to the tissue
tis_slides = pd.read_csv(f'../splits/HE2RNA_100{args.tissue_code}/splits_0_bool.csv')['Unnamed: 0']

# reduce the rnaseq dataset to just contain the samples from the tissue of interest
df_red = rnaseq.loc[rnaseq['slides'].str.contains('|'.join(tis_slides.str[:-1])), :]

# get genes having TPM > 10 in at least 5% of samples of the selected tissue
idx_genes = (df_red.drop('slides', axis=1).astype(float) > 10).sum(0) > (df_red.shape[0] * 0.05)


# store genes into files
with open(f'../resources/gene_set_{args.tissue_code}.txt', 'w+') as f:
    f.write('gene_id')
    f.write(' ')
    f.write('gene_desc')
    f.write('\n')
    for idx,row in rnaseq.loc[:, idx_genes[idx_genes].index].iloc[0:2].T.iterrows():
        f.write(row[0])
        f.write(' ')
        f.write(row[1])
        f.write('\n')