"Imports and cleans data."

#%%

import csv
from collections import namedtuple
import re

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#%%
# csv view
with open('compilado_financeiro.csv', 'r') as file:
    for num, row in enumerate(file):
        if num < 5:
            print(row)

#%%

# imporing data and verifying data formatting
HEADER = """carteira_2016 organizacao nome_projeto area valor_total
beneficiarios solicitado_iCSHG carga_horaria ativo_2014 ativo_2015 
pl_2014 pl_2015 receitas_2014 receitas_2015 despesas_2014
despesas_2015 rec_empresas_2015 rec_fundacao_instituto_2015
rec_pf_2015 rec_governo_2015 rec_outros_2015 d_projetos
d_gerais_e_admin d_folha_de_pagamento d_outros"""

Record = namedtuple('Record', HEADER)

REGEX_PATTERN = re.compile(r'^\d*(\.\d*)|(^\d*)', flags=re.IGNORECASE)

for record in map(Record._make, csv.reader(open("compilado_financeiro.csv", "r"), delimiter=",")):
    for name in Record._fields:
        value = getattr(record, name)
        if name not in ("organizacao", "nome_projeto", "area"):
            m = REGEX_PATTERN.fullmatch(value)
            if m is None and name != value:
                print(name)
                print(value)

# some "-" that should be considered missing
# some negative numbers ok

#%%
compfin = pd.read_csv(
    "compilado_financeiro.csv",
    sep=',')
print(compfin.head())

#%%
# verifico o dtype
print(compfin.dtypes)

#%%
# convertendo os campos com erro
compfin.carga_horaria = pd.to_numeric(
    compfin.carga_horaria, errors='coerce')

compfin.ativo_2015 = pd.to_numeric(
    compfin.ativo_2015, errors='coerce')

compfin.receitas_2014 = pd.to_numeric(
    compfin.receitas_2014, errors='coerce')

#%%
# field distributions
for field in compfin.columns:
    if compfin[field].dtype != 'object':
        x = compfin[field]
        x = x[~np.isnan(x)]
        plt.figure(figsize=(10, 5))
        plt.title("%s, %d"%(field, x.size))
        sns.distplot(x, kde=False, rug=False)

#%%
#carteira 2016 - 40% rejected and 30%+
print(compfin.carteira_2016.value_counts())
print(compfin.carteira_2016.value_counts()
      /compfin.carteira_2016.count())

#%%
# field boxplots
for field in compfin.columns:
    if compfin[field].dtype != 'object':
        x = compfin[field]
        x = x[~np.isnan(x)]
        plt.figure(figsize=(10, 5))
        plt.title("%s, %d"%(field, x.size))
        ax = sns.boxplot(x)
        ax = sns.swarmplot(x, color=".25")

#%%
# There is substancial fraction within lower ranges
print(compfin.beneficiarios.describe())

#%%
# calculated fields
compfin['per_capita'] = compfin.valor_total/compfin.beneficiarios
compfin['beneficiarios_iCSHG'] = compfin.solicitado_iCSHG/compfin.per_capita
compfin['per_capita_por_hora'] = compfin.per_capita/compfin.carga_horaria

print("per capita")
print(compfin.per_capita.describe())
print()
print("beneficiarios iCSHG")
print(compfin.beneficiarios_iCSHG.describe())
print("per capita por hora")
print(compfin.beneficiarios_iCSHG.describe())

#%%
# indicators
print(compfin.isnull().sum())

#1. at least one value of detailed revenue or expense 2015
temp_sum = compfin[[
    'rec_empresas_2015',
    'rec_fundacao_instituto_2015',
    'rec_pf_2015', 'rec_governo_2015',
    'rec_outros_2015', 'd_projetos',
    'd_gerais_e_admin', 'd_folha_de_pagamento',
    'd_outros']].isnull()
compfin['has_detailed_info'] = ~temp_sum.all(axis=1)
compfin['has_detailed_info'].fillna(False)


print()
print(compfin['has_detailed_info'].value_counts())


#2. at least one consolidated value 2014 and 2015

temp_sum = compfin[['ativo_2014', 'ativo_2015',
                    'pl_2014', 'pl_2015',
                    'receitas_2014', 'receitas_2015',
                    'despesas_2014', 'despesas_2015']].isnull()
compfin['has_consolidated_info'] = ~temp_sum.all(axis=1)
compfin['has_consolidated_info'].fillna(False)

del temp_sum
print()
print(compfin['has_consolidated_info'].value_counts())

#%%
# verify that the above fields are coherent with indicators above
print(pd.pivot_table(compfin,
                     columns=['has_consolidated_info',
                              'has_detailed_info'],
                     values=['valor_total',
                             'beneficiarios',
                             'solicitado_iCSHG',
                             'carga_horaria'],
                     fill_value=0,
                     margins=False,
                     aggfunc=np.sum))

#%%
# verify that the above fields are coherent with indicators above
print(pd.pivot_table(compfin,
                     columns=['has_consolidated_info',
                              'has_detailed_info'],
                     values=['area'],
                     fill_value=1,
                     margins=False,
                     aggfunc='count'))

#%%
#% store dataframe
compfin.to_pickle('compfin.pkl')
