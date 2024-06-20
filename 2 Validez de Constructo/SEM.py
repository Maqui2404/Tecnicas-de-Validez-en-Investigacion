import pandas as pd
from sklearn.preprocessing import StandardScaler
from semopy import Model, Optimizer
from semopy.inspector import inspect

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

# Convertir a DataFrame para semopy
df_numeric = pd.DataFrame(df_numeric, columns=numeric_columns)

# Definir el modelo SEM (ajustar según tu estructura teórica)
model_desc = """
# Definición de variables latentes
Satisfaction =~ satisfaction_level + last_evaluation + number_project
Engagement =~ average_montly_hours + time_spend_company + Work_accident + promotion_last_5years

# Correlaciones entre factores
Satisfaction ~~ Engagement
"""

# Crear y ajustar el modelo SEM
mod = Model(model_desc)
opt = Optimizer(mod)
opt.optimize(df_numeric)

# Inspeccionar los resultados
estimates = inspect(mod)
print(estimates)

# Evaluar índices de ajuste
from semopy.report import Report
rep = Report(mod)
rep.save('semopy_report.html')
