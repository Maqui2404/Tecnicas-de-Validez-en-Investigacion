import pandas as pd
import os
from factor_analyzer import FactorAnalyzer, calculate_kmo
from scipy.stats import bartlett
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Nombre del archivo
file_name = "Employee Attrition.csv"

# Verificar si el archivo existe en el directorio actual
if not os.path.exists(file_name):
    print(f"Archivo '{file_name}' no encontrado en el directorio actual.")
else:
    # Cargar el conjunto de datos
    df = pd.read_csv(file_name)

    # Verificar las columnas disponibles en el archivo
    print(df.columns)

    # Seleccionar las columnas relevantes para el análisis
    data = df[['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours', 
               'time_spend_company', 'Work_accident', 'promotion_last_5years']]

    # Imputar valores faltantes con la mediana de cada columna
    data.fillna(data.median(), inplace=True)

    # Normalizar los datos
    data = (data - data.mean()) / data.std()

    # Verificar si aún hay valores NaN o infinitos
    print(data.isna().sum())
    print(np.isinf(data).sum())

    # Cálculo de la medida KMO y prueba de Esfericidad de Bartlett
    kmo_all, kmo_model = calculate_kmo(data)
    print(f'KMO: {kmo_model:.2f}')

    chi2, p_value = bartlett(*data.values.T)
    print(f'Chi2: {chi2:.2f}, p-value: {p_value:.2f}')

    # Análisis Factorial Exploratorio (EFA)
    fa = FactorAnalyzer(n_factors=3, rotation='varimax')
    fa.fit(data)

    # Cargas factoriales
    loadings = fa.loadings_
    print('Cargas factoriales:')
    print(loadings)

    # Simulación de datos para CFA
    data['Factor1'] = np.dot(data.iloc[:, :3], loadings[:3, 0])
    data['Factor2'] = np.dot(data.iloc[:, 3:5], loadings[3:5, 1])
    data['Factor3'] = np.dot(data.iloc[:, 5:7], loadings[5:7, 2])

    # Modelo CFA
    model = ols('Factor1 ~ satisfaction_level + last_evaluation + number_project', data=data).fit()
    print(model.summary())

    # Índices de ajuste para CFA
    def calc_fit_indices(model):
        chi2 = model.ssr / model.df_resid
        rmsea = np.sqrt(chi2 / (model.nobs - model.df_model - 1))
        return {
            'Chi2': chi2,
            'RMSEA': rmsea,
            'CFI': sm.regression.linear_model.OLS(data['Factor1'], sm.add_constant(data.iloc[:, :5])).fit().rsquared
        }

    fit_indices = calc_fit_indices(model)
    print('Índices de ajuste:')
    print(fit_indices)
