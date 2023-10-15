# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
/kaggle/input/bbva-data-challenge-2023/digital.csv
/kaggle/input/bbva-data-challenge-2023/Diccionario de datos.xlsx
/kaggle/input/bbva-data-challenge-2023/archive/sample_submission.csv
/kaggle/input/bbva-data-challenge-2023/archive/balances.csv
/kaggle/input/bbva-data-challenge-2023/archive/universe_train.csv
/kaggle/input/bbva-data-challenge-2023/archive/movements.csv
/kaggle/input/bbva-data-challenge-2023/archive/liabilities.csv
/kaggle/input/bbva-data-challenge-2023/archive/universe_test.csv
/kaggle/input/bbva-data-challenge-2023/archive/customers.csv
import seaborn as sns
universe_train = pd.read_csv('/kaggle/input/bbva-data-challenge-2023/archive/universe_train.csv')
universe_test  = pd.read_csv('/kaggle/input/bbva-data-challenge-2023/archive/universe_test.csv')
universe_train.head()
sns.countplot(data = universe_train, x = 'attrition')
universe_test.head()
def varCustomer():
    # Leer la base
    df_customers = pd.read_csv('/kaggle/input/bbva-data-challenge-2023/archive/customers.csv')
    
    # Columnas binarias
    for col_ in ['product_1', 'product_2', 'product_3', 'product_4', 'ofert_1', 'ofert_2', 'ofert_3']:
        df_customers[col_] = df_customers[col_].apply(lambda x: 1 if x=='Yes' else 0)
        
    # Generar dummies
    df_customers = pd.get_dummies(df_customers, columns = ['type_job', 'bureau_risk'], dtype=int)
    
    return df_customers
df_customers = varCustomer()
df_customers.head()
df_balance = pd.read_csv('/kaggle/input/bbva-data-challenge-2023/archive/balances.csv')
df_balance.head()
def varRCCPivot(df_balance, historia, pivot, cal = 'mean'):
    """
    df_balance: base de RCC
    historia: cantidad de meses para la historia (máximo 12)
    pivot: columna para el pivot
    cal: cálculo a realizar a las variables numéricas
    """
    # Pivot
    dfRCCVar = pd.pivot_table(df_balance[df_balance['month'] >= 12 - historia + 1], 
                                   index=['ID', 'period'], 
                                   columns=[pivot], 
                                   values=['balance_amount', 'days_default'], 
                                   aggfunc=cal, 
                                   fill_value=0)
    # Renombrar
    dfRCCVar.columns = [x[1]+'_' + x[0] + '_' +cal + '_' + str(historia) for x in dfRCCVar.columns]
    dfRCCVar.reset_index(inplace = True)
    
    return dfRCCVar
df_liabilities = pd.read_csv('/kaggle/input/bbva-data-challenge-2023/archive/liabilities.csv')
def varLiabilities(df_liabilities, historia, dict_cal):
    """
    df_liabilities: base de ahorros
    historia: cantidad de meses para la historia (máximo 12)
    dict_cal: diccionario con los cálculos a realizar a las variables numéricas
    """
    
    dfVarLiabilities = df_liabilities[df_liabilities['month'] >= 12 - historia + 1].groupby(['ID', 'period'])\
                        .agg(
                            dict_cal
                        )

    dfVarLiabilities.columns = [ 'pas_'+ x[1]+'_' + x[0] + '_' + str(historia) for x in dfVarLiabilities.columns]

    dfVarLiabilities.reset_index(inplace = True)
    
    return dfVarLiabilities
liabilities_dict = {
                        'product_1': ['min', 'mean', 'max'],
                        'product_2': ['min', 'mean', 'max'],
                        'month': ['count']
                    }
df_movements = pd.read_csv('/kaggle/input/bbva-data-challenge-2023/archive/movements.csv')
df_movements.head()
def varMovements(df_movements, historia, dict_cal):
    """
    df_movements: base de compras
    historia: cantidad de meses para la historia (máximo 12)
    dict_cal: diccionario con los cálculos a realizar a las variables numéricas
    """
    dfVarMovements = df_movements[df_movements['month'] >= 12 - historia + 1].groupby(['ID', 'period'])\
                        .agg(
                            dict_cal
                        )

    dfVarMovements.columns = [ 'mov_'+ x[1]+'_' + x[0] + '_' + str(historia) for x in dfVarMovements.columns]

    dfVarMovements.reset_index(inplace = True)
    
    return dfVarMovements
dic_movements = {
                    'type_1': ['min', 'mean', 'max', 'sum'],
                    'type_2': ['min', 'mean', 'max', 'sum'],
                    'type_3': ['min', 'mean', 'max', 'sum'],
                    'type_4': ['min', 'mean', 'max', 'sum'],
                    'month': ['count']
                }
dfVarMovements12 = varMovements(df_movements, 12, dic_movements)
dfVarMovements12.head()
df_digital = pd.read_csv('/kaggle/input/bbva-data-challenge-2023/digital.csv')
df_digital.head()
def varDigital(df_digital, historia, dict_cal):
    """
    df_digital: base digital
    historia: cantidad de meses para la historia (máximo 12)
    dict_cal: diccionario con los cálculos a realizar a las variables numéricas
    """
    dfVarDigital = df_digital[df_digital['month'] >= 12 - historia + 1].groupby(['ID', 'period'])\
                        .agg(
                            dict_cal
                        )

    dfVarDigital.columns = [ 'dig_'+ x[1]+'_' + x[0] + '_' + str(historia) for x in dfVarDigital.columns]

    dfVarDigital.reset_index(inplace = True)
    
    return dfVarDigital
dic_digital = {
                'dig_1': ['min', 'mean', 'max', 'sum'],
                'dig_2': ['min', 'mean', 'max', 'sum'],
                'dig_3': ['min', 'mean', 'max', 'sum'],
                'dig_4': ['min', 'mean', 'max', 'sum'],
                'dig_5': ['min', 'mean', 'max', 'sum'],
                'dig_6': ['min', 'mean', 'max', 'sum'],
                'dig_7': ['min', 'mean', 'max', 'sum'],
                'dig_8': ['min', 'mean', 'max', 'sum'],
                'dig_9': ['min', 'mean', 'max', 'sum'],
                'dig_10': ['min', 'mean', 'max', 'sum'],
                'dig_11': ['min', 'mean', 'max', 'sum'],
                'month': ['count']
            }
def getMatriz(universo = 'train'):
    
    if universo == 'train':
        df_universo = pd.read_csv('/kaggle/input/bbva-data-challenge-2023/archive/universe_train.csv')
    else:
        df_universo = pd.read_csv('/kaggle/input/bbva-data-challenge-2023/archive/universe_test.csv')
        
    # Consolidar
    
    # Customer
    consolidado = pd.merge(df_universo, varCustomer(), how='left', on = ['ID'])
    
    # RCC
    consolidado = pd.merge(consolidado, varRCCPivot(df_balance, 12, 'type', cal = 'mean'), how='left', on = ['ID', 'period'])
    consolidado = pd.merge(consolidado, varRCCPivot(df_balance, 12, 'product', cal = 'mean'), how='left', on = ['ID', 'period'])
    consolidado = pd.merge(consolidado, varRCCPivot(df_balance, 6, 'entity', cal = 'mean'), how='left', on = ['ID', 'period'])
    ### pueden añadir más variables
    
    # Liabilities
    consolidado = pd.merge(consolidado, varLiabilities(df_liabilities, 6, liabilities_dict), how='left', on = ['ID', 'period'])
    
    # Movements
    consolidado = pd.merge(consolidado, varMovements(df_movements, 12, dic_movements), how='left', on = ['ID', 'period'])
    
    # Digital
    consolidado = pd.merge(consolidado, varDigital(df_digital, 12, dic_digital), how='left', on = ['ID', 'period'])
    
    return consolidado
matrix_train = getMatriz(universo = 'train')
matrix_submit = getMatriz(universo = 'test')
matrix_train.head()
from sklearn.model_selection import train_test_split
columnas_eliminar = ['ID', 'attrition', 'period']
X = matrix_train.drop(columnas_eliminar, axis=1).copy()
y = matrix_train['attrition'].copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=11, stratify=y)
import lightgbm as lgb
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
plt.figure(figsize=(40,20))
train_data_lgb = lgb.Dataset(X_train, label=y_train)
test_data_lgb = lgb.Dataset(X_test, label=y_test)
params_k = {
            'boosting_type': 'gbdt',
            'objective': 'multiclass',
            'num_class': 6,
            'subsample': 0.95,
            'learning_rate': 0.01,
            'max_depth': 5,
            'num_leaves': 100,
            'feature_fraction': 0.9,
            #'max_bin': 100,
            'is_unbalance': True,
            #'boost_from_average': False,
            "random_seed":42,
            'verbose': -1
}
model = lgb.train(params_k,
                       train_data_lgb,
                       valid_sets=[test_data_lgb, train_data_lgb],
                       num_boost_round=1000,
                       early_stopping_rounds=100, verbose_eval=50
                 )
ax = lgb.plot_importance(model, max_num_features=20)
plt.show()
prediccion_lgb_test = model.predict(X_test, num_iteration=model.best_iteration)
prediccion_lgb_train = model.predict(X_train, num_iteration=model.best_iteration)
print('train', f1_score(y_train, np.argmax(prediccion_lgb_train, axis=1), average='macro'))
print('test ', f1_score(y_test, np.argmax(prediccion_lgb_test, axis=1), average='macro'))
prediccion_lgb_submit = model.predict(matrix_submit[X_train.columns], num_iteration=model.best_iteration)
submit = matrix_submit[['period', 'ID']].copy()
submit['target'] = np.argmax(prediccion_lgb_submit, axis=1)
submit.head()
submit.to_csv('submit_benchmark.csv', index=False)

