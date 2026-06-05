# librerie per la gestione del dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# metriche per valutare le prestazioni dei modelli
# successivamente faccio un confronto tra le prestazioni dei vari modelli e scelgo quello più accurato

# algoritmi di classificazione che uso nella mia tesi
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

# caricamento del dataset
dataset = pd.read_csv("training.csv")


print(dataset.shape[0], "righe e ",dataset.shape[1], "colonne") 
print("\nColonne presenti nel dataset:")
print(dataset.columns.tolist())
print("\nTipi di dato delle colonne:")
print(dataset.dtypes)
print("\nValori mancanti per colonna:")
print(dataset.isnull().sum())

# capisco come è distribuito il target (se è bilanciato o sbilanciato)
# ad esempio se ci sono tante vittorie e pochissime sconfitte il target potrebbe essere sbilanciato
print("\nDistribuzione del target:")
print(dataset["target_result"].value_counts())

# separazione tra feature e target

target_column = "target_result"

Y = dataset[target_column]

X = dataset.drop(columns=[target_column, "year"])

# tengo solo le colonne numeriche (so che ci stanno solo colonne numeriche ma dovrei farlo a priori come se non lo sapessi)
X = X.select_dtypes(include=["int64", "float64"])

print("X è per i dati mentre Y è per le etichette")
print("X contiene", X.shape[0]," righe e ",X.shape[1], "feature")
print("Y contiene", Y.shape[0], "etichette")

# Divisione temporale del dataset
# Training: Mondiali dal 2002 al 2018
# Test: Mondiale 2022
# utilizzo questa divisione per far "studiare" il modello dai mondiali 2002,2006,2010,2014,2018 e poi testarlo sul 2022 per vedere quanto è accurato
# una volta fatto ciò poi andrò nel 2026 a fare le previsioni per il mondiale che ancora si deve giocare


train_set = dataset["year"] < 2022
test_set = dataset["year"] == 2022

train_x = X[train_set]
train_y = Y[train_set]

test_x = X[test_set]
test_y = Y[test_set]

print("Training set contiene", train_x.shape[0], "righe e", train_x.shape[1], "feature")
print("Test set contiene", test_x.shape[0], "righe e", test_x.shape[1], "feature")
print("Etichette training:", train_y.shape[0])
print("Etichette test:", test_y.shape[0])


# ho preparato il train e il test
# ho diviso dati e etichette sia per train che per test
# ora parto con gli algoritmi. Il primo algoritmo è KNN

# Ricerca del miglior valore di n_neighbors per KNN
# La scelta viene fatta usando solo i Mondiali dal 2002 al 2018

# prima di tutto vado a vedere qual è il miglior valore di k (iperparametro) per fare la KNN
valori_k = [1, 3, 5, 7, 9, 11, 15]
anni_validation = [2006, 2010, 2014, 2018]

risultati_knn = {}

#cross validation

for k in valori_k:
    accuracy_k = []

    for anno in anni_validation:
        train_temporaneo = dataset["year"] < anno
        validation_temporaneo = dataset["year"] == anno

        train_x_temp = X[train_temporaneo]
        train_y_temp = Y[train_temporaneo]

        validation_x_temp = X[validation_temporaneo]
        validation_y_temp = Y[validation_temporaneo]

        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(train_x_temp, train_y_temp)

        predizioni = knn.predict(validation_x_temp)

        accuracy = accuracy_score(validation_y_temp, predizioni)
        accuracy_k.append(accuracy)

    media_accuracy = np.mean(accuracy_k)
    risultati_knn[k] = media_accuracy

    print("K =", k, "Accuracy media =", media_accuracy)

miglior_k = max(risultati_knn, key=risultati_knn.get)

print("\nMiglior valore di K:", miglior_k)
print("Accuracy media migliore:", risultati_knn[miglior_k])

knn = KNeighborsClassifier(n_neighbors=miglior_k)
knn.fit(train_x, train_y)
pred_knn = knn.predict(test_x)
print("\nRisultati KNN")
print("Accuracy:", accuracy_score(test_y, pred_knn))
print("Matrice di confusione:")
print(confusion_matrix(test_y, pred_knn))
print("Report classificazione:")
print(classification_report(test_y, pred_knn))

# Random Forest

# Ricerca dei migliori iperparametri per Random Forest
# La scelta viene fatta usando solo i Mondiali dal 2002 al 2018

parametri_random_forest = [
    {"n_estimators": 50, "max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1},
    {"n_estimators": 100, "max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1},
    {"n_estimators": 200, "max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1},

    {"n_estimators": 100, "max_depth": 3, "min_samples_split": 2, "min_samples_leaf": 1},
    {"n_estimators": 100, "max_depth": 5, "min_samples_split": 2, "min_samples_leaf": 1},
    {"n_estimators": 100, "max_depth": 8, "min_samples_split": 2, "min_samples_leaf": 1},

    {"n_estimators": 100, "max_depth": 5, "min_samples_split": 5, "min_samples_leaf": 1},
    {"n_estimators": 100, "max_depth": 5, "min_samples_split": 10, "min_samples_leaf": 1},

    {"n_estimators": 100, "max_depth": 5, "min_samples_split": 2, "min_samples_leaf": 2},
    {"n_estimators": 100, "max_depth": 5, "min_samples_split": 2, "min_samples_leaf": 4},
]


risultati_random_forest = []

for parametri in parametri_random_forest:
    accuracy_parametri = []

    for anno in anni_validation:
        train_temporaneo = dataset["year"] < anno
        validation_temporaneo = dataset["year"] == anno

        train_x_temp = X[train_temporaneo]
        train_y_temp = Y[train_temporaneo]

        validation_x_temp = X[validation_temporaneo]
        validation_y_temp = Y[validation_temporaneo]

        random_forest = RandomForestClassifier(
            n_estimators=parametri["n_estimators"],
            max_depth=parametri["max_depth"],
            min_samples_split=parametri["min_samples_split"],
            min_samples_leaf=parametri["min_samples_leaf"],
            random_state=42
        )

        # addestro il modello e poi faccio la predizione sul validation di quello specifico ciclo del for
        random_forest.fit(train_x_temp, train_y_temp)

        predizioni = random_forest.predict(validation_x_temp)

        #trovo l'accuracy per ogni validation diverso e le aggiungo all'array delle accuracy
        accuracy = accuracy_score(validation_y_temp, predizioni)
        accuracy_parametri.append(accuracy)

    media_accuracy = np.mean(accuracy_parametri)

    risultati_random_forest.append((parametri, media_accuracy))

    print(parametri, "Accuracy media =", media_accuracy)

def prendi_accuracy(elemento):
    return elemento[1]

migliori_parametri_rf, migliore_accuracy_rf = max(
    risultati_random_forest,
    key=prendi_accuracy
)

print("\nMigliori parametri Random Forest:", migliori_parametri_rf)
print("Accuracy media migliore Random Forest:", migliore_accuracy_rf)


# Random Forest finale con i migliori iperparametri sul mondiale 2022

random_forest = RandomForestClassifier(
    n_estimators=migliori_parametri_rf["n_estimators"],
    max_depth=migliori_parametri_rf["max_depth"],
    min_samples_split=migliori_parametri_rf["min_samples_split"],
    min_samples_leaf=migliori_parametri_rf["min_samples_leaf"],
    random_state=42
)

random_forest.fit(train_x, train_y)

pred_random_forest = random_forest.predict(test_x)

print("\nRisultati Random Forest sul test set 2022")
print("Accuracy:", accuracy_score(test_y, pred_random_forest))
print("Matrice di confusione:")
print(confusion_matrix(test_y, pred_random_forest))
print("Report classificazione:")
print(classification_report(test_y, pred_random_forest, zero_division=0)) #zero division 0 è per evitare un warning se un modello non predice mai una classe


# Random Forest bilanciata con gli stessi iperparametri

random_forest_balanced = RandomForestClassifier(
    n_estimators=migliori_parametri_rf["n_estimators"],
    max_depth=migliori_parametri_rf["max_depth"],
    min_samples_split=migliori_parametri_rf["min_samples_split"],
    min_samples_leaf=migliori_parametri_rf["min_samples_leaf"],
    class_weight="balanced",
    random_state=42
)

random_forest_balanced.fit(train_x, train_y)

pred_random_forest_balanced = random_forest_balanced.predict(test_x)

print("\nRisultati Random Forest bilanciata sul test set 2022")
print("Accuracy:", accuracy_score(test_y, pred_random_forest_balanced))
print("Matrice di confusione:")
print(confusion_matrix(test_y, pred_random_forest_balanced))
print("Report classificazione:")
print(classification_report(test_y, pred_random_forest_balanced, zero_division=0))

# ============================================================
# STANDARDSCALER
# Il fit viene fatto SOLO sul training set per evitare data leakage.
# Lo stesso scaler viene poi applicato al test set con .transform().
# ============================================================

scaler = StandardScaler()
train_x_scaled = scaler.fit_transform(train_x)   # fit + transform sul train
test_x_scaled  = scaler.transform(test_x)         # solo transform sul test

# Riconvertiamo in DataFrame per mantenere i nomi delle colonne (utile per debug)
train_x_scaled = pd.DataFrame(train_x_scaled, columns=train_x.columns, index=train_x.index)
test_x_scaled  = pd.DataFrame(test_x_scaled,  columns=test_x.columns,  index=test_x.index)

# ============================================================
# SMOTE — sovra-campionamento della classe minoritaria (pareggi)
# SMOTE va applicato SOLO sul training set, mai sul test.
# Genera esempi sintetici interpolando tra campioni esistenti della
# classe minoritaria, in modo da bilanciare le tre classi.
# ============================================================

print("\nDistribuzione classi nel training set PRIMA di SMOTE:")
print(train_y.value_counts())

smote = SMOTE(random_state=42)
train_x_smote, train_y_smote = smote.fit_resample(train_x_scaled, train_y)

print("\nDistribuzione classi nel training set DOPO SMOTE:")
print(pd.Series(train_y_smote).value_counts())

# ============================================================
# KNN CON STANDARDSCALER + SMOTE
# ============================================================

print("\n" + "="*60)
print("KNN con StandardScaler + SMOTE")
print("="*60)

valori_k = [1, 3, 5, 7, 9, 11, 15]
risultati_knn = {}

for k in valori_k:
    accuracy_k = []

    for anno in anni_validation:
        train_temp_mask = dataset["year"] < anno
        val_temp_mask   = dataset["year"] == anno

        train_x_temp = X[train_temp_mask]
        train_y_temp = Y[train_temp_mask]
        val_x_temp   = X[val_temp_mask]
        val_y_temp   = Y[val_temp_mask]

        # Scaling: fit solo sul train temporaneo
        scaler_temp = StandardScaler()
        train_x_temp_scaled = scaler_temp.fit_transform(train_x_temp)
        val_x_temp_scaled   = scaler_temp.transform(val_x_temp)

        # SMOTE solo sul train temporaneo
        smote_temp = SMOTE(random_state=42)
        train_x_temp_smote, train_y_temp_smote = smote_temp.fit_resample(
            train_x_temp_scaled, train_y_temp
        )

        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(train_x_temp_smote, train_y_temp_smote)

        predizioni = knn.predict(val_x_temp_scaled)
        accuracy_k.append(accuracy_score(val_y_temp, predizioni))

    media_accuracy = np.mean(accuracy_k)
    risultati_knn[k] = media_accuracy
    print(f"K = {k}  Accuracy media = {media_accuracy:.4f}")

miglior_k = max(risultati_knn, key=risultati_knn.get)
print(f"\nMiglior valore di K: {miglior_k}")
print(f"Accuracy media migliore: {risultati_knn[miglior_k]:.4f}")

# Modello finale KNN
knn_finale = KNeighborsClassifier(n_neighbors=miglior_k)
knn_finale.fit(train_x_smote, train_y_smote)
pred_knn = knn_finale.predict(test_x_scaled)

print("\nRisultati KNN sul test set 2022")
print("Accuracy:", accuracy_score(test_y, pred_knn))
print("Matrice di confusione:")
print(confusion_matrix(test_y, pred_knn))
print("Report classificazione:")
print(classification_report(test_y, pred_knn, zero_division=0))

# ============================================================
# RANDOM FOREST CON STANDARDSCALER + SMOTE
# (RF non richiede scaling ma lo applichiamo per coerenza;
#  la differenza principale qui è SMOTE sui pareggi)
# ============================================================

print("\n" + "="*60)
print("Random Forest con StandardScaler + SMOTE")
print("="*60)

parametri_random_forest = [
    {"n_estimators": 50,  "max_depth": None, "min_samples_split": 2,  "min_samples_leaf": 1},
    {"n_estimators": 100, "max_depth": None, "min_samples_split": 2,  "min_samples_leaf": 1},
    {"n_estimators": 200, "max_depth": None, "min_samples_split": 2,  "min_samples_leaf": 1},
    {"n_estimators": 100, "max_depth": 3,    "min_samples_split": 2,  "min_samples_leaf": 1},
    {"n_estimators": 100, "max_depth": 5,    "min_samples_split": 2,  "min_samples_leaf": 1},
    {"n_estimators": 100, "max_depth": 8,    "min_samples_split": 2,  "min_samples_leaf": 1},
    {"n_estimators": 100, "max_depth": 5,    "min_samples_split": 5,  "min_samples_leaf": 1},
    {"n_estimators": 100, "max_depth": 5,    "min_samples_split": 10, "min_samples_leaf": 1},
    {"n_estimators": 100, "max_depth": 5,    "min_samples_split": 2,  "min_samples_leaf": 2},
    {"n_estimators": 100, "max_depth": 5,    "min_samples_split": 2,  "min_samples_leaf": 4},
]

risultati_random_forest = []

for parametri in parametri_random_forest:
    accuracy_parametri = []

    for anno in anni_validation:
        train_temp_mask = dataset["year"] < anno
        val_temp_mask   = dataset["year"] == anno

        train_x_temp = X[train_temp_mask]
        train_y_temp = Y[train_temp_mask]
        val_x_temp   = X[val_temp_mask]
        val_y_temp   = Y[val_temp_mask]

        scaler_temp = StandardScaler()
        train_x_temp_scaled = scaler_temp.fit_transform(train_x_temp)
        val_x_temp_scaled   = scaler_temp.transform(val_x_temp)

        smote_temp = SMOTE(random_state=42)
        train_x_temp_smote, train_y_temp_smote = smote_temp.fit_resample(
            train_x_temp_scaled, train_y_temp
        )

        rf = RandomForestClassifier(
            n_estimators=parametri["n_estimators"],
            max_depth=parametri["max_depth"],
            min_samples_split=parametri["min_samples_split"],
            min_samples_leaf=parametri["min_samples_leaf"],
            random_state=42
        )
        rf.fit(train_x_temp_smote, train_y_temp_smote)

        predizioni = rf.predict(val_x_temp_scaled)
        accuracy_parametri.append(accuracy_score(val_y_temp, predizioni))

    media_accuracy = np.mean(accuracy_parametri)
    risultati_random_forest.append((parametri, media_accuracy))
    print(parametri, f"Accuracy media = {media_accuracy:.4f}")

migliori_parametri_rf, migliore_accuracy_rf = max(
    risultati_random_forest, key=lambda x: x[1]
)
print(f"\nMigliori parametri Random Forest: {migliori_parametri_rf}")
print(f"Accuracy media migliore: {migliore_accuracy_rf:.4f}")

# Modello finale Random Forest
rf_finale = RandomForestClassifier(
    n_estimators=migliori_parametri_rf["n_estimators"],
    max_depth=migliori_parametri_rf["max_depth"],
    min_samples_split=migliori_parametri_rf["min_samples_split"],
    min_samples_leaf=migliori_parametri_rf["min_samples_leaf"],
    random_state=42
)
rf_finale.fit(train_x_smote, train_y_smote)
pred_rf = rf_finale.predict(test_x_scaled)

print("\nRisultati Random Forest sul test set 2022")
print("Accuracy:", accuracy_score(test_y, pred_rf))
print("Matrice di confusione:")
print(confusion_matrix(test_y, pred_rf))
print("Report classificazione:")
print(classification_report(test_y, pred_rf, zero_division=0))

# ============================================================
# SVM CON STANDARDSCALER + SMOTE
# SVM è il modello che beneficia di più dello scaling.
# ============================================================

print("\n" + "="*60)
print("SVM con StandardScaler + SMOTE")
print("="*60)

parametri_svm = [
    {"C": 0.1,  "kernel": "rbf",    "gamma": "scale"},
    {"C": 1.0,  "kernel": "rbf",    "gamma": "scale"},
    {"C": 10.0, "kernel": "rbf",    "gamma": "scale"},
    {"C": 1.0,  "kernel": "rbf",    "gamma": "auto"},
    {"C": 10.0, "kernel": "rbf",    "gamma": "auto"},
    {"C": 1.0,  "kernel": "linear", "gamma": "scale"},
    {"C": 10.0, "kernel": "linear", "gamma": "scale"},
]

risultati_svm = []

for parametri in parametri_svm:
    accuracy_parametri = []

    for anno in anni_validation:
        train_temp_mask = dataset["year"] < anno
        val_temp_mask   = dataset["year"] == anno

        train_x_temp = X[train_temp_mask]
        train_y_temp = Y[train_temp_mask]
        val_x_temp   = X[val_temp_mask]
        val_y_temp   = Y[val_temp_mask]

        scaler_temp = StandardScaler()
        train_x_temp_scaled = scaler_temp.fit_transform(train_x_temp)
        val_x_temp_scaled   = scaler_temp.transform(val_x_temp)

        smote_temp = SMOTE(random_state=42)
        train_x_temp_smote, train_y_temp_smote = smote_temp.fit_resample(
            train_x_temp_scaled, train_y_temp
        )

        svm = SVC(
            C=parametri["C"],
            kernel=parametri["kernel"],
            gamma=parametri["gamma"],
            random_state=42
        )
        svm.fit(train_x_temp_smote, train_y_temp_smote)

        predizioni = svm.predict(val_x_temp_scaled)
        accuracy_parametri.append(accuracy_score(val_y_temp, predizioni))

    media_accuracy = np.mean(accuracy_parametri)
    risultati_svm.append((parametri, media_accuracy))
    print(parametri, f"Accuracy media = {media_accuracy:.4f}")
migliori_parametri_svm, migliore_accuracy_svm = max(
    risultati_svm, key=lambda x: x[1]
)
print(f"\nMigliori parametri SVM: {migliori_parametri_svm}")
print(f"Accuracy media migliore: {migliore_accuracy_svm:.4f}")
svm_finale = SVC(
    C=migliori_parametri_svm["C"],
    kernel=migliori_parametri_svm["kernel"],
    gamma=migliori_parametri_svm["gamma"],
    random_state=42
)
svm_finale.fit(train_x_smote, train_y_smote)
pred_svm = svm_finale.predict(test_x_scaled)

print("\nRisultati SVM sul test set 2022")
print("Accuracy:", accuracy_score(test_y, pred_svm))
print("Matrice di confusione:")
print(confusion_matrix(test_y, pred_svm))
print("Report classificazione:")
print(classification_report(test_y, pred_svm, zero_division=0))

# ============================================================
# VARIANTI RANDOM FOREST
# Entrambe le varianti usano i migliori iperparametri RF già trovati.
#
# Variante 1: SMOTE parziale (bilancia i pareggi solo all'80%)
# Variante 2: SMOTE completo + class_weight="balanced"
# ============================================================

# ============================================================
# VARIANTE 1 — SMOTE PARZIALE
# Invece di portare tutte le classi a 139 campioni,
# portiamo solo i pareggi (classe 1) a ~110, cioè circa l'80%
# della classe più numerosa. Questo bilancia senza
# "esagerare" con i campioni sintetici.
# ============================================================

print("\n" + "="*60)
print("VARIANTE 1 — Random Forest con SMOTE parziale")
print("(pareggi portati a 110 invece di 139)")
print("="*60)

risultati_v1 = []

for parametri in parametri_random_forest:
    accuracy_parametri = []

    for anno in anni_validation:
        train_temp_mask = dataset["year"] < anno
        val_temp_mask   = dataset["year"] == anno

        train_x_temp = X[train_temp_mask]
        train_y_temp = Y[train_temp_mask]
        val_x_temp   = X[val_temp_mask]
        val_y_temp   = Y[val_temp_mask]

        scaler_temp = StandardScaler()
        train_x_temp_scaled = scaler_temp.fit_transform(train_x_temp)
        val_x_temp_scaled   = scaler_temp.transform(val_x_temp)

        # Calcolo il target per SMOTE parziale in modo dinamico:
        # porta la classe 1 all'80% della classe maggioritaria (classe 0)
        n_classe_0    = (train_y_temp == 0).sum()
        target_classe1 = max(int(n_classe_0 * 0.80), (train_y_temp == 1).sum() + 1)

        smote_parziale = SMOTE(
            sampling_strategy={1: target_classe1},
            random_state=42
        )

        try:
            train_x_temp_smote, train_y_temp_smote = smote_parziale.fit_resample(
                train_x_temp_scaled, train_y_temp
            )
        except ValueError:
            # fallback: SMOTE standard se il dataset temporaneo è troppo piccolo
            smote_parziale = SMOTE(random_state=42)
            train_x_temp_smote, train_y_temp_smote = smote_parziale.fit_resample(
                train_x_temp_scaled, train_y_temp
            )

        rf = RandomForestClassifier(
            n_estimators=parametri["n_estimators"],
            max_depth=parametri["max_depth"],
            min_samples_split=parametri["min_samples_split"],
            min_samples_leaf=parametri["min_samples_leaf"],
            random_state=42
        )
        rf.fit(train_x_temp_smote, train_y_temp_smote)

        predizioni = rf.predict(val_x_temp_scaled)
        accuracy_parametri.append(accuracy_score(val_y_temp, predizioni))

    media_accuracy = np.mean(accuracy_parametri)
    risultati_v1.append((parametri, media_accuracy))
    print(parametri, f"Accuracy media = {media_accuracy:.4f}")

migliori_parametri_v1, migliore_accuracy_v1 = max(risultati_v1, key=lambda x: x[1])
print(f"\nMigliori parametri Variante 1: {migliori_parametri_v1}")
print(f"Accuracy media migliore: {migliore_accuracy_v1:.4f}")

# Modello finale Variante 1
n_classe_0_finale    = (train_y == 0).sum()
target_classe1_finale = max(int(n_classe_0_finale * 0.80), (train_y == 1).sum() + 1)

smote_parziale_finale = SMOTE(
    sampling_strategy={1: target_classe1_finale},
    random_state=42
)
train_x_v1, train_y_v1 = smote_parziale_finale.fit_resample(train_x_scaled, train_y)

print(f"\nDistribuzione classi dopo SMOTE parziale:")
print(pd.Series(train_y_v1).value_counts())

rf_v1 = RandomForestClassifier(
    n_estimators=migliori_parametri_v1["n_estimators"],
    max_depth=migliori_parametri_v1["max_depth"],
    min_samples_split=migliori_parametri_v1["min_samples_split"],
    min_samples_leaf=migliori_parametri_v1["min_samples_leaf"],
    random_state=42
)
rf_v1.fit(train_x_v1, train_y_v1)
pred_v1 = rf_v1.predict(test_x_scaled)

print("\nRisultati Variante 1 sul test set 2022")
print("Accuracy:", accuracy_score(test_y, pred_v1))
print("Matrice di confusione:")
print(confusion_matrix(test_y, pred_v1))
print("Report classificazione:")
print(classification_report(test_y, pred_v1, zero_division=0))

# ============================================================
# FEATURE IMPORTANCE — RANDOM FOREST CON SMOTE PARZIALE
# ============================================================

feature_importance = pd.DataFrame({
    "feature": train_x.columns,
    "importance": rf_v1.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="importance",
    ascending=False
)

print("\nFeature importance Random Forest con SMOTE parziale:")
print(feature_importance.head(15))

top_features = feature_importance.head(15)

plt.figure(figsize=(10, 6))
plt.barh(top_features["feature"], top_features["importance"])
plt.gca().invert_yaxis()
plt.xlabel("Importanza")
plt.title("Feature importance - Random Forest con SMOTE parziale")
plt.tight_layout()
plt.savefig("feature_importance_rf.png", dpi=300, bbox_inches="tight")
plt.show()
# ============================================================
# VARIANTE 2 — SMOTE COMPLETO + class_weight="balanced"
# Doppio meccanismo: SMOTE bilancia i dati di input,
# class_weight="balanced" fa sì che RF penalizzi di più
# gli errori sulle classi meno frequenti durante il training.
# ============================================================

print("\n" + "="*60)
print("VARIANTE 2 — Random Forest con SMOTE completo + class_weight='balanced'")
print("="*60)

risultati_v2 = []

for parametri in parametri_random_forest:
    accuracy_parametri = []

    for anno in anni_validation:
        train_temp_mask = dataset["year"] < anno
        val_temp_mask   = dataset["year"] == anno

        train_x_temp = X[train_temp_mask]
        train_y_temp = Y[train_temp_mask]
        val_x_temp   = X[val_temp_mask]
        val_y_temp   = Y[val_temp_mask]

        scaler_temp = StandardScaler()
        train_x_temp_scaled = scaler_temp.fit_transform(train_x_temp)
        val_x_temp_scaled   = scaler_temp.transform(val_x_temp)

        smote_temp = SMOTE(random_state=42)
        train_x_temp_smote, train_y_temp_smote = smote_temp.fit_resample(
            train_x_temp_scaled, train_y_temp
        )

        # class_weight="balanced" ricalcola automaticamente i pesi
        # in base alla frequenza delle classi nel training set
        rf = RandomForestClassifier(
            n_estimators=parametri["n_estimators"],
            max_depth=parametri["max_depth"],
            min_samples_split=parametri["min_samples_split"],
            min_samples_leaf=parametri["min_samples_leaf"],
            class_weight="balanced",
            random_state=42
        )
        rf.fit(train_x_temp_smote, train_y_temp_smote)

        predizioni = rf.predict(val_x_temp_scaled)
        accuracy_parametri.append(accuracy_score(val_y_temp, predizioni))

    media_accuracy = np.mean(accuracy_parametri)
    risultati_v2.append((parametri, media_accuracy))
    print(parametri, f"Accuracy media = {media_accuracy:.4f}")

migliori_parametri_v2, migliore_accuracy_v2 = max(risultati_v2, key=lambda x: x[1])
print(f"\nMigliori parametri Variante 2: {migliori_parametri_v2}")
print(f"Accuracy media migliore: {migliore_accuracy_v2:.4f}")

# Modello finale Variante 2
smote_v2 = SMOTE(random_state=42)
train_x_v2, train_y_v2 = smote_v2.fit_resample(train_x_scaled, train_y)

rf_v2 = RandomForestClassifier(
    n_estimators=migliori_parametri_v2["n_estimators"],
    max_depth=migliori_parametri_v2["max_depth"],
    min_samples_split=migliori_parametri_v2["min_samples_split"],
    min_samples_leaf=migliori_parametri_v2["min_samples_leaf"],
    class_weight="balanced",
    random_state=42
)
rf_v2.fit(train_x_v2, train_y_v2)
pred_v2 = rf_v2.predict(test_x_scaled)

print("\nRisultati Variante 2 sul test set 2022")
print("Accuracy:", accuracy_score(test_y, pred_v2))
print("Matrice di confusione:")
print(confusion_matrix(test_y, pred_v2))
print("Report classificazione:")
print(classification_report(test_y, pred_v2, zero_division=0))

# ============================================================
# PARTE 1 — XGBOOST
# XGBoost è un algoritmo boosting: costruisce alberi in sequenza,
# ognuno corregge gli errori del precedente. Tende a superare
# Random Forest su dataset piccoli e sbilanciati.
# ============================================================

print("\n" + "="*60)
print("XGBOOST con StandardScaler + SMOTE parziale")
print("="*60)

# XGBoost vuole le etichette come 0,1,2 (già così nel nostro caso)
# ma usiamo LabelEncoder per sicurezza
le = LabelEncoder()
train_y_enc = le.fit_transform(train_y)
test_y_enc  = le.transform(test_y)


parametri_xgb = [
    {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1,  "subsample": 1.0},
    {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.1,  "subsample": 1.0},
    {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.1,  "subsample": 1.0},
    {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.05, "subsample": 1.0},
    {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1,  "subsample": 0.8},
    {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.8},
    {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.1,  "subsample": 0.8},
    {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.8},
]

risultati_xgb = []

for parametri in parametri_xgb:
    accuracy_parametri = []

    for anno in anni_validation:
        train_temp_mask = dataset["year"] < anno
        val_temp_mask   = dataset["year"] == anno

        train_x_temp = X[train_temp_mask]
        train_y_temp = Y[train_temp_mask]
        val_x_temp   = X[val_temp_mask]
        val_y_temp   = Y[val_temp_mask]

        # Encoding etichette per XGBoost
        train_y_temp_enc = le.transform(train_y_temp)
        val_y_temp_enc   = le.transform(val_y_temp)

        scaler_temp = StandardScaler()
        train_x_temp_scaled = scaler_temp.fit_transform(train_x_temp)
        val_x_temp_scaled   = scaler_temp.transform(val_x_temp)

        # SMOTE parziale — pareggi all'80% della classe maggioritaria
        n_classe_0     = (train_y_temp == 0).sum()
        target_classe1 = max(int(n_classe_0 * 0.80), (train_y_temp == 1).sum() + 1)

        try:
            smote_temp = SMOTE(sampling_strategy={1: target_classe1}, random_state=42)
            train_x_temp_smote, train_y_temp_smote = smote_temp.fit_resample(
                train_x_temp_scaled, train_y_temp_enc
            )
        except ValueError:
            smote_temp = SMOTE(random_state=42)
            train_x_temp_smote, train_y_temp_smote = smote_temp.fit_resample(
                train_x_temp_scaled, train_y_temp_enc
            )

        xgb = XGBClassifier(
            n_estimators=parametri["n_estimators"],
            max_depth=parametri["max_depth"],
            learning_rate=parametri["learning_rate"],
            subsample=parametri["subsample"],
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=42,
            verbosity=0
        )
        xgb.fit(train_x_temp_smote, train_y_temp_smote)

        predizioni = xgb.predict(val_x_temp_scaled)
        accuracy_parametri.append(accuracy_score(val_y_temp_enc, predizioni))

    media_accuracy = np.mean(accuracy_parametri)
    risultati_xgb.append((parametri, media_accuracy))
    print(parametri, f"Accuracy media = {media_accuracy:.4f}")

migliori_parametri_xgb, migliore_accuracy_xgb = max(risultati_xgb, key=lambda x: x[1])
print(f"\nMigliori parametri XGBoost: {migliori_parametri_xgb}")
print(f"Accuracy media migliore: {migliore_accuracy_xgb:.4f}")

# SMOTE parziale finale per XGBoost
n_classe_0_finale    = (train_y == 0).sum()
target_classe1_finale = max(int(n_classe_0_finale * 0.80), (train_y == 1).sum() + 1)

smote_xgb = SMOTE(sampling_strategy={1: target_classe1_finale}, random_state=42)
train_x_xgb, train_y_xgb = smote_xgb.fit_resample(train_x_scaled, train_y_enc)

xgb_finale = XGBClassifier(
    n_estimators=migliori_parametri_xgb["n_estimators"],
    max_depth=migliori_parametri_xgb["max_depth"],
    learning_rate=migliori_parametri_xgb["learning_rate"],
    subsample=migliori_parametri_xgb["subsample"],
    use_label_encoder=False,
    eval_metric="mlogloss",
    random_state=42,
    verbosity=0
)
xgb_finale.fit(train_x_xgb, train_y_xgb)
pred_xgb = xgb_finale.predict(test_x_scaled)

# Riconverti le predizioni alle etichette originali (0,1,2)
pred_xgb_orig = le.inverse_transform(pred_xgb)

print("\nRisultati XGBoost sul test set 2022")
print("Accuracy:", accuracy_score(test_y, pred_xgb_orig))
print("Matrice di confusione:")
print(confusion_matrix(test_y, pred_xgb_orig))
print("Report classificazione:")
print(classification_report(test_y, pred_xgb_orig, zero_division=0))


# ============================================================
# RIEPILOGO FINALE DI TUTTI I MODELLI
# ============================================================

print("\n" + "="*60)
print("RIEPILOGO COMPLETO AGGIORNATO (test set 2022)")
print("="*60)

tutti_i_modelli = {
    f"KNN (k={miglior_k})":              pred_knn,
    "RF originale":                     pred_random_forest,
    "RF bilanciata":                    pred_random_forest_balanced,
    "RF + SMOTE completo":              pred_rf,
    "RF + SMOTE parziale (V1)":         pred_v1,
    "RF + SMOTE + balanced (V2)":       pred_v2,
    "SVM":                              pred_svm,
    "XGBoost + SMOTE parziale":         pred_xgb_orig,
}
# ============================================================
# CREAZIONE TABELLA RIEPILOGO MODELLI PER LA TESI
# ============================================================

tabella_metriche_modelli = []

for nome, pred in tutti_i_modelli.items():
    acc = accuracy_score(test_y, pred)
    f1_mac = f1_score(test_y, pred, average="macro", zero_division=0)
    f1_c1 = f1_score(test_y, pred, labels=[1], average="macro", zero_division=0)

    tabella_metriche_modelli.append([
        nome,
        round(acc, 4),
        round(f1_mac, 4),
        round(f1_c1, 4)
    ])

tabella_metriche_modelli = pd.DataFrame(
    tabella_metriche_modelli,
    columns=["Modello", "Accuracy", "F1 macro", "F1 classe 1"]
)

print("\nTabella metriche modelli:")
print(tabella_metriche_modelli)

tabella_metriche_modelli.to_csv("tabella_metriche_modelli_2022.csv", index=False)
dati_matrice_confusione = pd.DataFrame({
    "classe_reale": test_y.values,
    "classe_predetta": pred_v1
})

dati_matrice_confusione.to_csv("dati_matrice_confusione_v1.csv", index=False)

print(f"\n{'Modello':<35} {'Accuracy':>10} {'F1 macro':>10} {'F1 classe 1':>12}")
print("-" * 70)

for nome, pred in tutti_i_modelli.items():
    acc    = accuracy_score(test_y, pred)
    f1_mac = f1_score(test_y, pred, average="macro", zero_division=0)
    f1_c1  = f1_score(test_y, pred, labels=[1], average="macro", zero_division=0)
    print(f"{nome:<35} {acc:>10.4f} {f1_mac:>10.4f} {f1_c1:>12.4f}")


# scelgo il modello con accuracy più alta sul test set 2022

modelli_da_confrontare = {
    "KNN": pred_knn,
    "Random Forest originale": pred_random_forest,
    "Random Forest bilanciata": pred_random_forest_balanced,
    "Random Forest SMOTE completo": pred_rf,
    "SVM": pred_svm,
    "Random Forest SMOTE parziale": pred_v1,
    "Random Forest SMOTE balanced": pred_v2
}

# uso anche XGBoost, quindi aggiungo anche lui al confronto
if "pred_xgb_orig" in globals():
    modelli_da_confrontare["XGBoost"] = pred_xgb_orig

accuracy_modelli = {}

for nome_modello in modelli_da_confrontare:
    predizioni = modelli_da_confrontare[nome_modello]
    accuracy = accuracy_score(test_y, predizioni)

    accuracy_modelli[nome_modello] = accuracy

miglior_modello = max(accuracy_modelli, key=accuracy_modelli.get)

print("\nMiglior modello per accuracy sul test set 2022:", miglior_modello)
print("Accuracy migliore:", accuracy_modelli[miglior_modello])

pred_miglior_modello = modelli_da_confrontare[miglior_modello]

# scelgo anche il modello con F1 macro più alta sul test set 2022

f1_macro_modelli = {}

for nome_modello in modelli_da_confrontare:
    predizioni = modelli_da_confrontare[nome_modello]
    f1_macro = f1_score(test_y, predizioni, average="macro", zero_division=0)

    f1_macro_modelli[nome_modello] = f1_macro

miglior_modello_f1_macro = max(f1_macro_modelli, key=f1_macro_modelli.get)

print("\nMiglior modello per F1 macro sul test set 2022:", miglior_modello_f1_macro)
print("F1 macro migliore:", f1_macro_modelli[miglior_modello_f1_macro])

pred_miglior_modello_f1_macro = modelli_da_confrontare[miglior_modello_f1_macro]


# Valutazione del passaggio del girone sul Mondiale 2022

partite_info = pd.read_csv("predizione_2022.csv", sep=";")

partite_2022 = partite_info[partite_info["year"] == 2022].reset_index(drop=True)
# resetto l'indice perchè altrimenti Pandas manterrebbe gli indici originali


gironi_2022 = {
    "A": ["Qatar", "Ecuador", "Senegal", "Netherlands"],
    "B": ["England", "Iran", "United States", "Wales"],
    "C": ["Argentina", "Saudi Arabia", "Mexico", "Poland"],
    "D": ["France", "Australia", "Denmark", "Tunisia"],
    "E": ["Spain", "Costa Rica", "Germany", "Japan"],
    "F": ["Belgium", "Canada", "Morocco", "Croatia"],
    "G": ["Brazil", "Serbia", "Switzerland", "Cameroon"],
    "H": ["Portugal", "Ghana", "Uruguay", "South Korea"]
}

qualificate_reali = [
    "Netherlands", "Senegal",
    "England", "United States",
    "Argentina", "Poland",
    "France", "Australia",
    "Japan", "Spain",
    "Morocco", "Croatia",
    "Brazil", "Switzerland",
    "Portugal", "South Korea"
]
# ============================================================
# VALUTAZIONE PASSAGGIO GIRONI 2022
# Confronto tra Random Forest originale e Random Forest + SMOTE parziale
# ============================================================


def valuta_passaggio_gironi_2022(predizioni_modello, nome_modello, nome_file_csv, stampa_classifica=False):

    partite_gironi = partite_2022.head(48).copy()
    partite_gironi["predizione"] = predizioni_modello[:48]

    classifica = {}

    # creo la classifica iniziale dei gironi mettendo tutte le squadre a 0 punti
    for girone in gironi_2022:
        for squadra in gironi_2022[girone]:
            classifica[squadra] = {
                "girone": girone,
                "squadra": squadra,
                "punti": 0,
                "vittorie": 0
            }

    for i in range(len(partite_gironi)):  # ciclo per ogni partita
        home = partite_gironi.loc[i, "home_team"]
        away = partite_gironi.loc[i, "away_team"]
        predizione = partite_gironi.loc[i, "predizione"]

        if predizione == 0:
            classifica[home]["punti"] = classifica[home]["punti"] + 3
            classifica[home]["vittorie"] = classifica[home]["vittorie"] + 1

        elif predizione == 1:
            classifica[home]["punti"] = classifica[home]["punti"] + 1
            classifica[away]["punti"] = classifica[away]["punti"] + 1

        elif predizione == 2:
            classifica[away]["punti"] = classifica[away]["punti"] + 3
            classifica[away]["vittorie"] = classifica[away]["vittorie"] + 1

    classifica = pd.DataFrame(classifica.values())
    risultati = []

    # ordino i gironi
    for girone in gironi_2022:
        classifica_girone = classifica[classifica["girone"] == girone].copy()

        classifica_girone = classifica_girone.sort_values(
            by=["punti", "vittorie"],
            ascending=False
        ).reset_index(drop=True)

        # creo la colonna
        classifica_girone["posizione"] = classifica_girone.index + 1

        risultati.append(classifica_girone)

    risultati = pd.concat(risultati, ignore_index=True)

    risultati["esito_predetto"] = "NON PASSA"
    risultati.loc[risultati["posizione"] <= 2, "esito_predetto"] = "PASSA"

    risultati["esito_reale"] = "NON PASSA"
    risultati.loc[risultati["squadra"].isin(qualificate_reali), "esito_reale"] = "PASSA"

    risultati["corretto"] = risultati["esito_predetto"] == risultati["esito_reale"]

    accuracy_passaggio = risultati["corretto"].mean()
    squadre_corrette = risultati["corretto"].sum()
    totale_squadre = risultati.shape[0]

    if stampa_classifica:
        print("\n" + "="*80)
        print("Risultato passaggio gironi 2022 usando:", nome_modello)
        print("="*80)
        print(risultati[["girone", "squadra", "punti", "posizione", "esito_predetto", "esito_reale", "corretto"]])

    print("\nRisultato sintetico usando:", nome_modello)
    print("Accuracy sul passaggio del girone:", accuracy_passaggio)
    print("Squadre corrette:", squadre_corrette, "su", totale_squadre)

    risultati.to_csv(nome_file_csv, index=False)

    return risultati, accuracy_passaggio, squadre_corrette, totale_squadre


# ============================================================
# CONFRONTO 2022: modello accuracy vs modello F1 macro
# ============================================================

risultati_accuracy, accuracy_passaggio_accuracy, corrette_accuracy, totale_accuracy = valuta_passaggio_gironi_2022(
    pred_random_forest,
    "Random Forest originale",
    "valutazione_passaggio_2022_accuracy.csv",
    stampa_classifica=True
)

risultati_f1_macro, accuracy_passaggio_f1_macro, corrette_f1_macro, totale_f1_macro = valuta_passaggio_gironi_2022(
    pred_v1,
    "Random Forest SMOTE parziale",
    "valutazione_passaggio_2022_f1_macro.csv",
    stampa_classifica=True
)


# ============================================================
# TABELLA RIASSUNTIVA PER LA DISCUSSIONE
# ============================================================

tabella_confronto_2022 = pd.DataFrame({
    "Criterio": [
        "Modello scelto per accuracy",
        "Modello scelto per F1 macro"
    ],
    "Modello": [
        "Random Forest originale",
        "Random Forest SMOTE parziale"
    ],
    "Accuracy partite": [
        accuracy_score(test_y, pred_random_forest),
        accuracy_score(test_y, pred_v1)
    ],
    "F1 macro": [
        f1_score(test_y, pred_random_forest, average="macro", zero_division=0),
        f1_score(test_y, pred_v1, average="macro", zero_division=0)
    ],
    "F1 classe 1": [
        f1_score(test_y, pred_random_forest, labels=[1], average="macro", zero_division=0),
        f1_score(test_y, pred_v1, labels=[1], average="macro", zero_division=0)
    ],
    "Squadre corrette 2022": [
        str(corrette_accuracy) + "/" + str(totale_accuracy),
        str(corrette_f1_macro) + "/" + str(totale_f1_macro)
    ],
    "Accuracy passaggio gironi": [
        accuracy_passaggio_accuracy,
        accuracy_passaggio_f1_macro
    ]
})

print("\n" + "="*80)
print("CONFRONTO PASSAGGIO GIRONI 2022")
print("="*80)
print(tabella_confronto_2022)

# ============================================================
# PARTE 2 — PREVISIONI MONDIALE 2026
#
# Per fare previsioni sul 2026 ho bisogno di un file CSV
# con le stesse 77 feature usate nel training, una riga
# per ogni partita da predire.
#
# STRUTTURA DEL FILE prediction_2026.csv:
#   - stesse colonne di training.csv ESCLUSE "year" e "target_result"
#   - una colonna aggiuntiva "match_id" (es. "BRA_vs_ARG")
#     per identificare la partita nell'output
# ============================================================

print("\n" + "="*60)
print("PARTE 2 — STRUTTURA FILE PER PREVISIONI 2026")
print("="*60)

# Stampa la lista delle feature necessarie nel CSV 2026
feature_necessarie = list(X.columns)
print(f"\nIl file prediction_2026.csv deve avere {len(feature_necessarie)} colonne feature")
print("più una colonna 'match_id' per identificare la partita.\n")

# ============================================================
# PREPARAZIONE PARTITE MONDIALE 2026
# ============================================================

# Carico il file con le squadre del 2026
squadre_2026 = pd.read_csv("test.csv", sep=";")

# Dizionario dei gironi 2026
# I nomi devono essere scritti come compaiono nel file test.csv

gironi_2026 = {
    "A": ["Mexico", "South Africa", "South Korea", "Czech Republic"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curacao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"]
}

# Controllo che tutte le squadre dei gironi siano presenti nel file test.csv

squadre_presenti = squadre_2026["team"].tolist()

for girone in gironi_2026:
    for squadra in gironi_2026[girone]:
        if squadra not in squadre_presenti:
            print("ATTENZIONE: squadra non trovata nel dataset:", squadra)

# Creo tutte le partite dei gironi
# Ogni girone da 4 squadre ha 6 partite

partite_2026 = []
for girone in gironi_2026:
    squadre = gironi_2026[girone]
    for i in range(len(squadre)):
        for j in range(i + 1, len(squadre)):
            home_team = squadre[i]
            away_team = squadre[j]
            partita = {
                "girone": girone,
                "home_team": home_team,
                "away_team": away_team,
                "match_id": home_team + "_vs_" + away_team
            }
            partite_2026.append(partita)
partite_2026 = pd.DataFrame(partite_2026)
print("\nPartite gironi 2026 create:")
print(partite_2026)
partite_2026.to_csv("partite_gironi_2026.csv", index=False)

# ho creato il file con le partite dei gironi 2026.
# ora vado a creare un file csv per le predizioni che deve avere le stesse features di quello sulle predizioni 2022

# ============================================================
# CREAZIONE DEL DATASET prediction_2026.csv
# ============================================================

# Prendo le colonne usate nel training, escludendo year e target_result
feature_training = X.columns.tolist()

# Carico le partite storiche per calcolare gli scontri diretti
partite_storiche = pd.read_csv("predizione_2022.csv", sep=";")

# Tengo solo le partite già giocate
partite_storiche = partite_storiche[partite_storiche["year"] < 2026].copy()

def calcola_h2h(home_team, away_team, partite_storiche):
    
    scontri_diretti = partite_storiche[
        (
            (partite_storiche["home_team"] == home_team) &
            (partite_storiche["away_team"] == away_team)
        )
        |
        (
            (partite_storiche["home_team"] == away_team) &
            (partite_storiche["away_team"] == home_team)
        )
    ].copy()

    h2h_matches_before = len(scontri_diretti)

    h2h_home_team_wins_before = 0
    h2h_away_team_wins_before = 0
    h2h_draws_before = 0

    h2h_home_team_goals_before = 0
    h2h_away_team_goals_before = 0

    for i in range(len(scontri_diretti)):

        squadra_home_storica = scontri_diretti.iloc[i]["home_team"]

        gol_home_storici = scontri_diretti.iloc[i]["home_score"]
        gol_away_storici = scontri_diretti.iloc[i]["away_score"]

        # Ricalcolo i gol dal punto di vista della partita 2026
        if squadra_home_storica == home_team:
            gol_home_2026 = gol_home_storici
            gol_away_2026 = gol_away_storici
        else:
            gol_home_2026 = gol_away_storici
            gol_away_2026 = gol_home_storici

        h2h_home_team_goals_before = h2h_home_team_goals_before + gol_home_2026
        h2h_away_team_goals_before = h2h_away_team_goals_before + gol_away_2026

        if gol_home_2026 > gol_away_2026:
            h2h_home_team_wins_before = h2h_home_team_wins_before + 1
        elif gol_home_2026 < gol_away_2026:
            h2h_away_team_wins_before = h2h_away_team_wins_before + 1
        else:
            h2h_draws_before = h2h_draws_before + 1

    if h2h_matches_before > 0:
        h2h_home_win_rate_before = h2h_home_team_wins_before / h2h_matches_before
        h2h_away_win_rate_before = h2h_away_team_wins_before / h2h_matches_before
        h2h_draw_rate_before = h2h_draws_before / h2h_matches_before

        h2h_avg_goals_home_team_before = h2h_home_team_goals_before / h2h_matches_before
        h2h_avg_goals_away_team_before = h2h_away_team_goals_before / h2h_matches_before
    else:
        h2h_home_win_rate_before = 0
        h2h_away_win_rate_before = 0
        h2h_draw_rate_before = 0

        h2h_avg_goals_home_team_before = 0
        h2h_avg_goals_away_team_before = 0

    h2h_goal_diff_before = h2h_home_team_goals_before - h2h_away_team_goals_before

    return {
        "h2h_matches_before": h2h_matches_before,
        "h2h_home_team_wins_before": h2h_home_team_wins_before,
        "h2h_away_team_wins_before": h2h_away_team_wins_before,
        "h2h_draws_before": h2h_draws_before,
        "h2h_home_team_goals_before": h2h_home_team_goals_before,
        "h2h_away_team_goals_before": h2h_away_team_goals_before,
        "h2h_goal_diff_before": h2h_goal_diff_before,
        "h2h_home_win_rate_before": h2h_home_win_rate_before,
        "h2h_away_win_rate_before": h2h_away_win_rate_before,
        "h2h_draw_rate_before": h2h_draw_rate_before,
        "h2h_avg_goals_home_team_before": h2h_avg_goals_home_team_before,
        "h2h_avg_goals_away_team_before": h2h_avg_goals_away_team_before
    }



righe_prediction_2026 = []

for i in range(len(partite_2026)):
    girone = partite_2026.loc[i, "girone"]
    home_team = partite_2026.loc[i, "home_team"]
    away_team = partite_2026.loc[i, "away_team"]
    match_id = partite_2026.loc[i, "match_id"]

    dati_home = squadre_2026[squadre_2026["team"] == home_team].iloc[0]
    dati_away = squadre_2026[squadre_2026["team"] == away_team].iloc[0]

    nuova_riga = {}

    nuova_riga["girone"] = girone
    nuova_riga["home_team"] = home_team
    nuova_riga["away_team"] = away_team
    nuova_riga["match_id"] = match_id

    colonne_base = [
        "is_host",
        "goals_scored_last_4y",
        "goals_received_last_4y",
        "wins_last_4y",
        "losses_last_4y",
        "draws_last_4y",
        "world_cup_titles_before",
        "squad_total_market_value_eur",
        "fifa_rank_pre_tournament",
        "fifa_points_pre_tournament",
        "squad_avg_age",
        "world_cup_participations_before",
        "groups_passed_before",
        "round16_before",
        "quarterfinals_before",
        "semifinals_before",
        "finals_before"
    ]

    for colonna in colonne_base:
        nuova_riga["home_" + colonna] = dati_home[colonna]
        nuova_riga["away_" + colonna] = dati_away[colonna]
        nuova_riga[colonna + "_diff"] = dati_home[colonna] - dati_away[colonna]

    # Calcolo le feature head-to-head usando le partite storiche disponibili
    h2h = calcola_h2h(home_team, away_team, partite_storiche)

    for colonna in h2h:
        nuova_riga[colonna] = h2h[colonna]

    # Gestione campo neutro
    # Se una delle due squadre è paese ospitante, considero la partita non neutrale
    if dati_home["is_host"] == 1 or dati_away["is_host"] == 1:
        nuova_riga["neutral_False"] = 1
        nuova_riga["neutral_True"] = 0
    else:
        nuova_riga["neutral_False"] = 0
        nuova_riga["neutral_True"] = 1

    # One-hot encoding dei continenti
    continenti = ["Africa", "Asia", "Europe", "North America", "Oceania", "South America"]

    for continente in continenti:
        nuova_riga["home_continent_" + continente] = 0
        nuova_riga["away_continent_" + continente] = 0

    nuova_riga["home_continent_" + dati_home["continent"]] = 1
    nuova_riga["away_continent_" + dati_away["continent"]] = 1

    righe_prediction_2026.append(nuova_riga)

prediction_2026 = pd.DataFrame(righe_prediction_2026)

# Aggiungo eventuali colonne mancanti rispetto al training
for colonna in feature_training:
    if colonna not in prediction_2026.columns:
        prediction_2026[colonna] = 0

# Riordino le colonne: prima info partita, poi feature nello stesso ordine del training
prediction_2026 = prediction_2026[
    ["girone", "home_team", "away_team", "match_id"] + feature_training
]

prediction_2026.to_csv("prediction_2026.csv", index=False)


# ============================================================
# SCELTA DEL MODELLO DA USARE PER IL 2026
# ============================================================
# Per la previsione del Mondiale 2026 scelgo la Random Forest con SMOTE parziale all'80%,
# perché ha mostrato il miglior compromesso tra accuracy, F1 macro e valutazione del passaggio del turno nel 2022.

modello_2026 = rf_v1
miglior_modello_2026 = "Random Forest SMOTE parziale 80%"

print("\nModello usato per il Mondiale 2026:", miglior_modello_2026)


# ============================================================
# PREDIZIONE PARTITE MONDIALE 2026
# ============================================================

# Uso direttamente il dataset prediction_2026 già creato sopra.
# Prendo solo le feature usate nel training.

X_2026 = prediction_2026[feature_training]

# Alcuni modelli sono stati addestrati su dati scalati.
# Quindi, se il modello migliore è uno di quelli con StandardScaler,
# devo scalare anche i dati del 2026 con lo stesso scaler usato sul training.

X_2026_scalato = scaler.transform(X_2026)
X_2026_scalato = pd.DataFrame(X_2026_scalato, columns=X_2026.columns)

predizioni_2026 = modello_2026.predict(X_2026_scalato)

prediction_2026["predizione"] = predizioni_2026

print("\nPredizioni partite 2026:")
print(prediction_2026[["girone", "home_team", "away_team", "predizione"]])

# ============================================================
# SIMULAZIONE GIRONI MONDIALE 2026
# ============================================================

classifica_2026 = {}

for girone in gironi_2026:
    for squadra in gironi_2026[girone]:
        classifica_2026[squadra] = {
            "girone": girone,
            "squadra": squadra,
            "punti": 0,
            "vittorie": 0
        }

for i in range(len(prediction_2026)):
    home = prediction_2026.loc[i, "home_team"]
    away = prediction_2026.loc[i, "away_team"]
    predizione = prediction_2026.loc[i, "predizione"]

    if predizione == 0:
        classifica_2026[home]["punti"] = classifica_2026[home]["punti"] + 3
        classifica_2026[home]["vittorie"] = classifica_2026[home]["vittorie"] + 1

    elif predizione == 1:
        classifica_2026[home]["punti"] = classifica_2026[home]["punti"] + 1
        classifica_2026[away]["punti"] = classifica_2026[away]["punti"] + 1

    elif predizione == 2:
        classifica_2026[away]["punti"] = classifica_2026[away]["punti"] + 3
        classifica_2026[away]["vittorie"] = classifica_2026[away]["vittorie"] + 1

classifica_2026 = pd.DataFrame(classifica_2026.values())

risultati_2026 = []

for girone in gironi_2026:
    classifica_girone = classifica_2026[classifica_2026["girone"] == girone].copy()

    classifica_girone = classifica_girone.sort_values(
        by=["punti", "vittorie"],
        ascending=False
    ).reset_index(drop=True)

    classifica_girone["posizione"] = classifica_girone.index + 1

    risultati_2026.append(classifica_girone)

risultati_2026 = pd.concat(risultati_2026, ignore_index=True)

risultati_2026["esito_predetto"] = "NON PASSA"

risultati_2026.loc[
    risultati_2026["posizione"] <= 2,
    "esito_predetto"
] = "PASSA"

terze_classificate = risultati_2026[risultati_2026["posizione"] == 3].copy()

migliori_terze = terze_classificate.sort_values(
    by=["punti", "vittorie"],
    ascending=False
).head(8)

risultati_2026.loc[
    risultati_2026["squadra"].isin(migliori_terze["squadra"]),
    "esito_predetto"
] = "PASSA"

print("\nClassifiche simulate Mondiale 2026:")
print(risultati_2026[["girone", "squadra", "punti", "vittorie", "posizione", "esito_predetto"]])

print("\nSquadre previste come qualificate al turno successivo:")
print(risultati_2026[risultati_2026["esito_predetto"] == "PASSA"][["girone", "squadra", "punti", "posizione"]])

print("\nSquadre previste come eliminate:")
print(risultati_2026[risultati_2026["esito_predetto"] == "NON PASSA"][["girone", "squadra", "punti", "posizione"]])

risultati_2026.to_csv("simulazione_gironi_2026.csv", index=False)
