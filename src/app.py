import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pickle
from pickle import dump

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

# EDA

## Importando el dataset
total_data=pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/decision-tree-project-tutorial/main/diabetes.csv")

total_data.to_csv("../data/raw/total_data.csv", index = False)

## Exploración y limpieza de datos

### Análisis de variables univariante
fig, axis = plt.subplots(4, 5, figsize=(18, 8), gridspec_kw={"height_ratios": [6, 1] * 2})

sns.histplot(ax=axis[0, 0], data=total_data, x="Pregnancies")
sns.boxplot(ax=axis[1, 0], data=total_data, x="Pregnancies")

sns.histplot(ax=axis[0, 1], data=total_data, x="Glucose")
sns.boxplot(ax=axis[1, 1], data=total_data, x="Glucose")

sns.histplot(ax=axis[0, 2], data=total_data, x="BloodPressure")
sns.boxplot(ax=axis[1, 2], data=total_data, x="BloodPressure")

sns.histplot(ax=axis[0, 3], data=total_data, x="SkinThickness")
sns.boxplot(ax=axis[1, 3], data=total_data, x="SkinThickness")

sns.histplot(ax=axis[0, 4], data=total_data, x="Insulin")
sns.boxplot(ax=axis[1, 4], data=total_data, x="Insulin")

sns.histplot(ax=axis[2, 0], data=total_data, x="BMI")
sns.boxplot(ax=axis[3, 0], data=total_data, x="BMI")

sns.histplot(ax=axis[2, 1], data=total_data, x="DiabetesPedigreeFunction")
sns.boxplot(ax=axis[3, 1], data=total_data, x="DiabetesPedigreeFunction")

sns.histplot(ax=axis[2, 2], data=total_data, x="Age")
sns.boxplot(ax=axis[3, 2], data=total_data, x="Age")

sns.histplot(ax=axis[2, 3], data=total_data, x="Outcome")
sns.boxplot(ax=axis[3, 3], data=total_data, x="Outcome")

fig.delaxes(axis[2,4])
fig.delaxes(axis[3,4])

plt.tight_layout()

plt.show()

#### Imputando valores verosímiles cuando tenemos 0 en ciertas columnas

cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

def impute_zeros_by_median_per_outcome(dataset, columns):
    df_copy = dataset.copy()

    df_copy[columns] = df_copy[columns].astype(float)

    for col in columns:
        for outcome_val in [0, 1]:
            median_val = df_copy[(df_copy['Outcome'] == outcome_val) & (df_copy[col] != 0)][col].median()
            mask = (df_copy['Outcome'] == outcome_val) & (df_copy[col] == 0)
            df_copy.loc[mask, col] = median_val
    
    return df_copy

total_data_processed = impute_zeros_by_median_per_outcome(total_data, cols_with_zeros)

### Análisis de variables multivariante

#### Correlaciones

fig, axis = plt.subplots(figsize = (10, 6))

sns.heatmap(total_data_processed[["Pregnancies", "Glucose" , "BloodPressure" , "SkinThickness" , "Insulin","BMI","DiabetesPedigreeFunction","Age","Outcome"]].corr(), annot = True, fmt = ".2f")

plt.tight_layout()

plt.show()

#### Parallel plot

plt.figure(figsize=(12, 6))

pd.plotting.parallel_coordinates(total_data, "Outcome", color = ("#E58139", "#39E581", "#8139E5"))

plt.show()

#### Guardando el dataset modificado
total_data_processed.to_csv("../data/raw/total_data_processed.csv", index = False)

## Ingeniería de características

### División outliers
total_data_con_outliers = total_data_processed.copy()
total_data_sin_outliers = total_data_processed.copy()

def replace_outliers_from_column(column, df):
  column_stats = df[column].describe()
  column_iqr = column_stats["75%"] - column_stats["25%"]
  upper_limit = column_stats["75%"] + 1.5 * column_iqr
  lower_limit = column_stats["25%"] - 1.5 * column_iqr

  if lower_limit < 0:
     lower_limit=float(df[column].min())

  # Remove upper outliers
  df[column] = df[column].apply(lambda x: x if (x <= upper_limit) else upper_limit)
  # Remove lower outliers
  df[column] = df[column].apply(lambda x: x if (x >= lower_limit) else lower_limit)
  return df.copy(), [lower_limit, upper_limit]

outliers_dict = {}

columns_to_process = [col for col in total_data_sin_outliers.columns if col != 'Outcome']

for column in columns_to_process:
  total_data_sin_outliers, limits_list = replace_outliers_from_column(column, total_data_sin_outliers)
  outliers_dict[column] = limits_list

with open("../models/outliers_replacement.json", "w") as f:
    json.dump(outliers_dict, f)

### Escalado de valores: no es necesario normalizar ni min-max para este modelo

X_con_outliers = total_data_con_outliers.drop("Outcome", axis = 1)
X_sin_outliers = total_data_sin_outliers.drop("Outcome", axis = 1)
y = total_data_con_outliers["Outcome"]

X_train_con_outliers, X_test_con_outliers, y_train, y_test = train_test_split(X_con_outliers, y, test_size = 0.2, random_state = 42)
X_train_sin_outliers, X_test_sin_outliers = train_test_split(X_sin_outliers, test_size = 0.2, random_state = 42)

X_train_con_outliers.to_excel("../data/processed/X_train_con_outliers.xlsx", index = False)
X_train_sin_outliers.to_excel("../data/processed/X_train_sin_outliers.xlsx", index = False)
X_test_con_outliers.to_excel("../data/processed/X_test_con_outliers.xlsx", index = False)
X_test_sin_outliers.to_excel("../data/processed/X_test_sin_outliers.xlsx", index = False)
y_train.to_excel("../data/processed/y_train.xlsx", index = False)
y_test.to_excel("../data/processed/y_test.xlsx", index = False)

## Selección de características

### Con outliers
selection_model_con_outliers = SelectKBest(k = 5)
selection_model_con_outliers.fit(X_train_con_outliers, y_train)

selected_columns = X_train_con_outliers.columns[selection_model_con_outliers.get_support()]

X_train_con_outliers_sel = pd.DataFrame(
    selection_model_con_outliers.transform(X_train_con_outliers),
    columns=selected_columns
)

X_test_con_outliers_sel = pd.DataFrame(
    selection_model_con_outliers.transform(X_test_con_outliers),
    columns=selected_columns
)

with open("../models/feature_selection_con_outliers_k_5.json", "w") as f:
    json.dump(X_train_con_outliers_sel.columns.tolist(), f)

X_train_con_outliers_sel["Outcome"] = y_train.values
X_test_con_outliers_sel["Outcome"] = y_test.values

X_train_con_outliers_sel.to_csv("../data/processed/clean_train_con_outliers.csv", index=False)
X_test_con_outliers_sel.to_csv("../data/processed/clean_test_con_outliers.csv", index=False)

### Sin outliers

selection_model_sin_outliers = SelectKBest(k = 5)
selection_model_sin_outliers.fit(X_train_sin_outliers, y_train)

selected_columns = X_train_sin_outliers.columns[selection_model_sin_outliers.get_support()]

X_train_sin_outliers_sel = pd.DataFrame(
    selection_model_sin_outliers.transform(X_train_sin_outliers),
    columns=selected_columns
)

X_test_sin_outliers_sel = pd.DataFrame(
    selection_model_sin_outliers.transform(X_test_sin_outliers),
    columns=selected_columns
)

with open("../models/feature_selection_sin_outliers_k_5.json", "w") as f:
    json.dump(X_train_sin_outliers_sel.columns.tolist(), f)

X_train_sin_outliers_sel["Outcome"] = y_train.values
X_test_sin_outliers_sel["Outcome"] = y_test.values

X_train_sin_outliers_sel.to_csv("../data/processed/clean_train_sin_outliers.csv", index=False)
X_test_sin_outliers_sel.to_csv("../data/processed/clean_test_sin_outliers.csv", index=False)

# Machine Learning: Decision Tree de clasificación

## Sin feature selection
BASE_PATH = "../data/processed"
TRAIN_PATHS = [
    "X_train_con_outliers.xlsx",
    "X_train_sin_outliers.xlsx",
]
TRAIN_DATASETS = []
for path in TRAIN_PATHS:
    TRAIN_DATASETS.append(
        pd.read_excel(f"{BASE_PATH}/{path}")
    )

TEST_PATHS = [
    "X_test_con_outliers.xlsx",
    "X_test_sin_outliers.xlsx",
]
TEST_DATASETS = []
for path in TEST_PATHS:
    TEST_DATASETS.append(
        pd.read_excel(f"{BASE_PATH}/{path}")
    )

y_train = pd.read_excel(f"{BASE_PATH}/y_train.xlsx")
y_test = pd.read_excel(f"{BASE_PATH}/y_test.xlsx")

results = []
models=[]

for index, dataset in enumerate(TRAIN_DATASETS):
    model = DecisionTreeClassifier(random_state=42)
    model.fit(dataset, y_train)
    models.append(model)
    
    y_pred_train = model.predict(dataset)
    y_pred_test = model.predict(TEST_DATASETS[index])

    results.append(
        {
            "train": accuracy_score(y_train, y_pred_train),
            "test": accuracy_score(y_test, y_pred_test)
        }
    )

## Con feature selection

train_data_con_outliers = pd.read_csv("../data/processed/clean_train_con_outliers.csv")
test_data_con_outliers = pd.read_csv("../data/processed/clean_test_con_outliers.csv")

X_train_con_outliers = train_data_con_outliers.drop(["Outcome"], axis = 1)
y_train_con_outliers = train_data_con_outliers["Outcome"]
X_test_con_outliers = test_data_con_outliers.drop(["Outcome"], axis = 1)
y_test_con_outliers = test_data_con_outliers["Outcome"]

model = DecisionTreeClassifier(random_state = 42)
model.fit(X_train_con_outliers, y_train_con_outliers)

fig = plt.figure(figsize=(15,15))

tree.plot_tree(model, feature_names = list(X_train_con_outliers.columns), class_names = ["0", "1"], filled = True)

plt.show()

y_pred_train = model.predict(X_train_con_outliers)
y_pred_test = model.predict(X_test_con_outliers)

print(f"Train: {accuracy_score(y_train_con_outliers, y_pred_train)}")
print(f"Test: {accuracy_score(y_test_con_outliers, y_pred_test)}")

## Hiperparametrización

param_grid = {
    'max_depth': [3, 5, 7, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 5, 10],
    'criterion': ['gini', 'entropy']
}

grid = GridSearchCV(model, param_grid, scoring = "accuracy", cv = 5)

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

grid.fit(X_train_con_outliers, y_train_con_outliers)

print(f"Mejores hiperparámetros: {grid.best_params_}")

model = DecisionTreeClassifier(criterion = "entropy", max_depth = 5, min_samples_leaf = 4, min_samples_split = 2, random_state = 42)
model.fit(X_train_con_outliers, y_train_con_outliers)

y_pred_train = model.predict(X_train_con_outliers)
y_pred_test = model.predict(X_test_con_outliers)

results.append(
        {
            "train": accuracy_score(y_test_con_outliers, y_pred_test),
            "test": accuracy_score(y_train_con_outliers, y_pred_train),
            "best_params": grid.best_params_
        }
)

## Guardado de modelo y resultados

dump(model, open("../models/decision_tree_classifier_42.sav", "wb"))
with open("../models/final_results.json", "w") as f:
    json.dump(results, f, indent=4)
