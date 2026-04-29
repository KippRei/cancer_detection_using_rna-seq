import pandas as pd

# Read with semicolon separator
data = pd.read_csv('./dataset/LUSCexpfile.csv', sep=';', index_col=0, low_memory=False)

labels = data.iloc[0] # normal or tumor
gene_expr   = data.iloc[1:] # rna-seq/gene expression values

# Transpose so rows=patients, columns=genes (standard ML format)
gene_expr_T = gene_expr.T.astype(float)
gene_expr_T['label'] = labels.values  # add the label column

print(gene_expr_T.shape)
print(gene_expr_T['label'].value_counts())