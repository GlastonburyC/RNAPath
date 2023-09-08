import pandas as pd
import yaml
import argparse
import pandas as pd
import tensorqtl
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
import os

def get_first_two_fields(s):
    fields = s.split('-')
    return '-'.join(fields[:2])

parser = argparse.ArgumentParser()
parser.add_argument("--tissue_name", type=str, default=None)
parser.add_argument("--gtex_normalized_expression_bed_file", type=str, default=None)
parser.add_argument("--gtex_subject_phenotypes_file", type=str, default=None)
parser.add_argument("--gtex_covariates_file", type=str, default=None)
parser.add_argument("--idps_format", choices=['binary', 'compositional', 'pivot'], default='pivot)
args = parser.parse_args()

f = open("../clusters.yaml", "r")
doc = yaml.load(f, Loader=yaml.FullLoader)
classes = doc[args.tissue_name]['classes']

expression_df, gene_pos_df = tensorqtl.read_phenotype_bed(args.normalized_expression_bed_file)


# WSI Samples Dataset
df_img = pd.read_csv('../resources/slides_dataset.csv')
df_img = df_img[df_img.tissue == args.tissue_name]
df_img = df_img[~df_img.slide_id.str[:-1].duplicated()]
df_img = df_img.sort_values(by='slide_id')


# Subject phenotypes (to recover age, bmi, isc, sex)
sub_pheno = pd.read_csv(args.gtex_subject_phenotypes_file, sep = '\t', header=None)
sub_pheno.columns = sub_pheno.iloc[0]

clinical_info_df = pd.DataFrame()

# GTEx covariates
cov = pd.read_csv(args.gtex_covariates_file, sep ='\t')
col_names = cov.ID
cov = cov.T.iloc[1:]
cov.columns = col_names

subjects = cov.index.tolist()

# Open image derived phenotypes (idps) file
if args.idps_format == 'pivot': 
    idps_df = pd.read_csv(f'./IDPs/{args.tissue_name}_pivot.csv', index_col = 0)
elif args.idps_format == 'binary':
    idps_df = pd.read_csv(f'./IDPs/{args.tissue_name}_binary.csv', index_col = 0)
else:
    idps_df = pd.read_csv(f'./IDPs/{args.tissue_name}_compositional.csv', index_col = 0)


idps_df = idps_df.sort_index()
idps_df['case_id'] = idps_df.index
# get case id by removing the last field in the GTEx ID
idps_df.case_id = idps_df.case_id.apply(get_first_two_fields)
# reduce idps dataframe just to the samples having gene expression
idps_df = idps_df[idps_df.case_id.isin(expression_df.columns)]
idps_df = idps_df.drop('case_id', axis='columns')

# IDPs names
targets = idps_df.columns.tolist()

rnaseq = pd.read_csv('./RNA-SEQ-Analysis/GTEx_RNASeQ_tailed.gct', sep = '\t')

for i in range(idps_df.shape[0]):

    # subject id
    ID = idps_df.iloc[i].name.rsplit('-', 1)[0]

    if (ID not in subjects):
        continue
    #sex
    sex = float(sub_pheno[sub_pheno.SUBJID == ID].SEX.values.item())
    #age
    age = float(sub_pheno[sub_pheno.SUBJID == ID].AGE.values.item())
    #bmi
    bmi = float(sub_pheno[sub_pheno.SUBJID == ID].BMI.values.item())
    #ischemic time
    isc = float(sub_pheno[sub_pheno.SUBJID == ID].TRISCHD.values.item())
    #PC1
    pc1= float(cov.loc[ID]['PC1'])
    #PC2
    pc2= float(cov.loc[ID]['PC2'])
    #PC3
    pc3= float(cov.loc[ID]['PC3'])
    #PC4
    pc4= float(cov.loc[ID]['PC4'])
    #PC5
    pc5= float(cov.loc[ID]['PC5'])
    keys = []
    values = []

    # retrieve IDPs of the current subject
    for target in targets:
        keys.append(target)
        idp_val = idps_df[idps_df.index ==  idps_df.iloc[i].name][target].values.item()
        values.append(idp_val)
    
    idp_dict = {k: v for k, v in zip(keys, values)}

    main_dict = {'ID': ID, 'sex': sex, 'age': age, 'bmi':bmi, 'isc': isc, 'pc1': pc1, 'pc2': pc2, 'pc3': pc3, 'pc4': pc4, 'pc5': pc5}
    # append idps values to the dictionary
    main_dict.update(idp_dict)
    clinical_info_df = clinical_info_df.append(main_dict, ignore_index=True)


# This dataframe now contains both image derived phenotypes and clinical info, including
# the covariates we need for the linear model
if args.idps_format == 'pivot':
    clinical_info_df.to_csv(f'./IDPs/{args.tissue_name}_with_clinical_info_pivot.csv', index=False)
if args.idps_format == 'binary':
    clinical_info_df.to_csv(f'./IDPs/{args.tissue_name}_with_clinical_info_binary.csv', index=False)
else:
    clinical_info_df.to_csv(f'./IDPs/{args.tissue_name}_with_clinical_info_compositional.csv', index=False)

clinical_info_df = clinical_info_df.set_index('ID', drop=True)

# list of genes
genes = expression_df.index.tolist()
descriptions = rnaseq.loc[rnaseq.Name.isin(genes)]['Description'].tolist()


# for each image derived phenotype
for target in targets:
    print(target, flush=True)
    # p values dataframe
    summary_df = pd.DataFrame()

    # Initialize an empty list to store p-values
    p_values = []
    # Initialize an empty list to store parameters
    params = []
    # Initialize an empty list to store standard errors
    standard_errors = []

    for gene in genes:


        # Subset the data for the current gene
        data = pd.merge(clinical_info_df, expression_df.T[gene], left_index = True, right_index = True)
        data['expression'] = data[gene]
        del data[gene]

        # Build the linear model
        model = smf.ols(f"expression ~ 1 + bmi + sex + age + isc + pc1 + pc2 + pc3 + pc4 + pc5 + {target}", data=data)
        results = model.fit()
        
        # Append pvalue, parameters and standard errors to their corresponding lists
        p_values.append(results.pvalues[target])
        params.append(results.params[target])
        standard_errors.append(results.bse[target])
    
    # p-values correction
    rejected, p_values_fdr, _, _ = multipletests(p_values, alpha=0.01, method='fdr_bh')

    summary_df['gene_id'] = genes
    summary_df['gene_name'] = descriptions
    summary_df['pvalues'] = p_values_fdr
    summary_df['param'] = params
    summary_df['std_error'] = standard_errors

    
    if args.idps_format == 'pivot':
        os.makedirs(f'./DEA/{args.tissue_name}', exist_ok=True)
        summary_df.to_csv(f'./DEA/{args.tissue_name}/{target}_differential_expression.csv', index = False)

    elif args.idps_format == 'binary':
        os.makedirs(f'./DEA/{args.tissue_name}_binary', exist_ok=True)
        summary_df.to_csv(f'./DEA/{args.tissue_name}_binary/{target}_differential_expression.csv', index = False)
    else:
        os.makedirs(f'./DEA/{args.tissue_name}_compositional', exist_ok=True)
        summary_df.to_csv(f'./DEA/{args.tissue_name}_compositional/{target}_differential_expression.csv', index = False)
