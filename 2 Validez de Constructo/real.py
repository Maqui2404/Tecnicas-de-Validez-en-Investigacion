import pandas as pd
from sklearn.preprocessing import StandardScaler
from factor_analyzer import FactorAnalyzer, calculate_kmo, calculate_bartlett_sphericity
import statsmodels.api as sm
import numpy as np

# Cargar los datos
df = pd.read_csv('Employee Attrition.csv')

# Seleccionar columnas numéricas
numeric_columns = [
    'satisfaction_level', 'last_evaluation', 'number_project', 
    'average_montly_hours', 'time_spend_company', 'Work_accident', 
    'promotion_last_5years'
]

# Imputar valores faltantes con la media
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

# Estandarizar los datos
scaler = StandardScaler()
df_numeric = scaler.fit_transform(df[numeric_columns])

# Calcular KMO y prueba de esfericidad de Bartlett
kmo_all, kmo_model = calculate_kmo(df_numeric)
bartlett_chi_square, bartlett_p_value = calculate_bartlett_sphericity(df_numeric)

print(f"KMO Model: {kmo_model}")
print(f"Bartlett's Test: Chi-Square = {bartlett_chi_square}, p-value = {bartlett_p_value}")

# Realizar Análisis Factorial Exploratorio (EFA) con diferentes números de factores
for n_factors in range(1, 4):
    fa = FactorAnalyzer(n_factors=n_factors, rotation='varimax')
    fa.fit(df_numeric)
    
    # Verificar la varianza explicada por cada factor
    variance = fa.get_factor_variance()
    print(f"Number of Factors: {n_factors}")
    print(f"Factor Variance: {variance}")
    
    # Obtener las cargas factoriales
    loadings = fa.loadings_
    print(f"Factor Loadings for {n_factors} factors:\n{loadings}\n")

# Realizar Análisis Factorial Confirmatorio (CFA) utilizando statsmodels
X = df_numeric[:, 1:]
y = df_numeric[:, 0]
model = sm.OLS(y, X)
results = model.fit()

print(f"CFA Results: {results.summary()}")

# Validación de constructo completa
# Se recomienda además revisar la adecuación de los modelos ajustados y los índices de ajuste como CFI, TLI, RMSEA, entre otros.

# NOTA: Este código es un ejemplo básico. Para una validación de constructo completa,
# se recomienda el uso de bibliotecas especializadas como 'lavaan' en R o paquetes de análisis SEM en Python.
