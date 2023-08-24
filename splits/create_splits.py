import pandas as pd
import numpy as np
import math
import os

tissue = 'Heart'
tissue_code = 'HEA' # code to identify the slipt folder specific to the tissue


df = pd.read_csv('../resources/HE2RNA_dataset.csv')
spl_bool = pd.read_csv('./RNAPath_main/splits_0_bool.csv')
df.loc[df['tissue'].isin([tissue])]

slides_list = []
for slidename in df.loc[df['tissue'].isin([tissue])].slide_id.tolist():
    slides_list.append(slidename)

print(f'Total slides from {tissue}: {len(slides_list)}')

# Drop from the main csv the indexes of the slides from other tissues
drop_indexes = []
for idx, row in spl_bool.iterrows():
    idx, slide_id = idx, row[0]
    if slide_id not in slides_list:
        drop_indexes.append(idx)

os.makedirs(f'./RNAPath_{tissue_code}', exist_ok=True)
spl_bool.drop(drop_indexes, axis=0).reset_index(drop=True).to_csv(f'./RNAPath_{tissue_code}/splits_bool.csv', index=False)


spl_df = pd.read_csv('./RNAPath_main/splits_0.csv')
train_drop = []
val_drop = []
test_drop = []
for idx, row in spl_df.iterrows():
    sl_tr, sl_val, sl_te = row[1], row[2], row[3]
    if (df[df['slide_id'] == sl_tr].tissue.values[0] not in [tissue]) or sl_tr not in slides_list:
        train_drop.append(idx)
    if (not sl_val != sl_val and df[df['slide_id'] == sl_val].tissue.values[0] not in [tissue]) or sl_val not in slides_list:
        val_drop.append(idx)
    if (not sl_te != sl_te and df[df['slide_id'] == sl_te].tissue.values[0] not in [tissue]) or sl_te not in slides_list:
        test_drop.append(idx)

tiss_train = spl_df['train'].drop(train_drop).reset_index(drop=True)
tiss_val = spl_df['val'].drop(val_drop).reset_index(drop=True)
tiss_test = spl_df['test'].drop(test_drop).reset_index(drop=True)

pd.concat((tiss_train, tiss_val, tiss_test), axis=1).to_csv(f'./RNAPath_{tissue_code}/splits.csv')