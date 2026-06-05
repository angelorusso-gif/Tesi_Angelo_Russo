import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

dataset = pd.read_csv("training.csv")

dataset_train = dataset[dataset["year"] < 2022]

classi = ["Vittoria casa", "Pareggio", "Vittoria ospite"]

distribuzione_originale = dataset_train["target_result"].value_counts().sort_index().values

distribuzione_smote_totale = [139, 139, 139]
distribuzione_smote_parziale = [139, 111, 108]

x = np.arange(len(classi))
larghezza = 0.34

plt.figure(figsize=(10, 6))
ax = plt.gca()
ax.set_facecolor("#FAFAFA")

barre1 = plt.bar(
    x - larghezza / 2,
    distribuzione_originale,
    width=larghezza,
    label="Training set originale",
    color="#2F6DAE",
    edgecolor="white",
    linewidth=1.5
)

barre2 = plt.bar(
    x + larghezza / 2,
    distribuzione_smote_totale,
    width=larghezza,
    label="Dopo SMOTE totale",
    color="#59A14F",
    edgecolor="white",
    linewidth=1.5
)

plt.title("Distribuzione delle classi nel training set prima e dopo SMOTE totale", fontsize=16, fontweight="bold", pad=18)
plt.xlabel("Classe", fontsize=13, labelpad=10)
plt.ylabel("Numero di campioni", fontsize=13, labelpad=10)
plt.xticks(x, classi, fontsize=12)
plt.yticks(fontsize=11)
plt.grid(axis="y", linestyle="--", alpha=0.25)
plt.gca().set_axisbelow(True)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#BBBBBB")
ax.spines["bottom"].set_color("#BBBBBB")

plt.legend(fontsize=11, frameon=True, facecolor="white", edgecolor="#DDDDDD", loc="upper right")

for barre in [barre1, barre2]:
    for barra in barre:
        altezza = barra.get_height()
        plt.text(
            barra.get_x() + barra.get_width() / 2,
            altezza + 3,
            str(int(altezza)),
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold"
        )

plt.ylim(0, max(max(distribuzione_originale), max(distribuzione_smote_totale)) + 30)
plt.tight_layout()
plt.savefig("grafico_distribuzione_target_smote_totale.png", dpi=300, bbox_inches="tight")
plt.show()


plt.figure(figsize=(10, 6))
ax = plt.gca()
ax.set_facecolor("#FAFAFA")

barre1 = plt.bar(
    x - larghezza / 2,
    distribuzione_originale,
    width=larghezza,
    label="Training set originale",
    color="#2F6DAE",
    edgecolor="white",
    linewidth=1.5
)

barre2 = plt.bar(
    x + larghezza / 2,
    distribuzione_smote_parziale,
    width=larghezza,
    label="Dopo SMOTE parziale",
    color="#F28E2B",
    edgecolor="white",
    linewidth=1.5
)

plt.title("Distribuzione delle classi nel training set prima e dopo SMOTE parziale", fontsize=16, fontweight="bold", pad=18)
plt.xlabel("Classe", fontsize=13, labelpad=10)
plt.ylabel("Numero di campioni", fontsize=13, labelpad=10)
plt.xticks(x, classi, fontsize=12)
plt.yticks(fontsize=11)
plt.grid(axis="y", linestyle="--", alpha=0.25)
plt.gca().set_axisbelow(True)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#BBBBBB")
ax.spines["bottom"].set_color("#BBBBBB")

plt.legend(fontsize=11, frameon=True, facecolor="white", edgecolor="#DDDDDD", loc="upper right")

for barre in [barre1, barre2]:
    for barra in barre:
        altezza = barra.get_height()
        plt.text(
            barra.get_x() + barra.get_width() / 2,
            altezza + 3,
            str(int(altezza)),
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold"
        )

plt.ylim(0, max(max(distribuzione_originale), max(distribuzione_smote_parziale)) + 30)
plt.tight_layout()
plt.savefig("grafico_distribuzione_target_smote_parziale.png", dpi=300, bbox_inches="tight")
plt.show()

#---------------------------------------------------------------------
# GRAFICO : METRICHE DI VALUTAZIONE DEI MODELLI
#---------------------------------------------------------------------


def crea_heatmap(modelli, metriche, valori, titolo, nome_file, cmap="Blues"):
    fig, ax = plt.subplots(figsize=(9, 5.5))

    immagine = ax.imshow(valori, cmap=cmap, vmin=0, vmax=0.70)

    ax.set_xticks(np.arange(len(metriche)))
    ax.set_yticks(np.arange(len(modelli)))

    ax.set_xticklabels(metriche, fontsize=12)
    ax.set_yticklabels(modelli, fontsize=11)

    for i in range(len(modelli)):
        for j in range(len(metriche)):
            valore = valori[i, j]
            colore_testo = "white" if valore > 0.38 else "black"
            ax.text(
                j,
                i,
                f"{valore:.4f}",
                ha="center",
                va="center",
                color=colore_testo,
                fontsize=11,
                fontweight="bold"
            )

    ax.set_title(titolo, fontsize=15, fontweight="bold", pad=18)
    ax.set_xlabel("Metriche di valutazione", fontsize=12, labelpad=10)
    ax.set_ylabel("Modelli", fontsize=12, labelpad=10)

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_xticks(np.arange(len(metriche) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(modelli) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    cbar = plt.colorbar(immagine, ax=ax)
    cbar.set_label("Valore della metrica", fontsize=11)

    plt.tight_layout()
    plt.savefig(nome_file, dpi=300, bbox_inches="tight")
    plt.show()


metriche = ["Accuracy", "F1 macro", "F1 classe 1"]

modelli_pre_smote = [
    "KNN",
    "SVM",
    "Random Forest"
]

valori_pre_smote = np.array([
    [0.3750, 0.3670, 0.2778],
    [0.4219, 0.3722, 0.2000],
    [0.5781, 0.4326, 0.0000]
])

crea_heatmap(
    modelli_pre_smote,
    metriche,
    valori_pre_smote,
    "Confronto delle metriche prima di SMOTE",
    "grafico_metriche_pre_smote.png",
    cmap="Blues"
)


modelli_post_smote = [
    "RF bilanciata",
    "RF + SMOTE completo",
    "RF + SMOTE 80%",
    "RF + SMOTE + balanced",
    "XGBoost + SMOTE"
]

valori_post_smote = np.array([
    [0.5469, 0.4196, 0.0000],
    [0.5156, 0.3954, 0.0000],
    [0.5938, 0.4782, 0.1250],
    [0.5156, 0.3954, 0.0000],
    [0.4844, 0.4084, 0.0870]
])

crea_heatmap(
    modelli_post_smote,
    metriche,
    valori_post_smote,
    "Confronto delle metriche dopo SMOTE",
    "grafico_metriche_post_smote.png",
    cmap="Oranges"
)

#------------------------------------------------
# GRAFICO CONFRONTO RF E RF SMOTE
#------------------------------------------------

tabella = pd.read_csv("confronto_passaggio_gironi_2022.csv")

# Arrotondo le colonne numeriche
for colonna in tabella.columns:
    if tabella[colonna].dtype == "float64":
        tabella[colonna] = tabella[colonna].round(4)

fig, ax = plt.subplots(figsize=(13, 3.5))
ax.axis("off")

tabella_grafica = ax.table(
    cellText=tabella.values,
    colLabels=tabella.columns,
    cellLoc="center",
    loc="center"
)

tabella_grafica.auto_set_font_size(False)
tabella_grafica.set_fontsize(10)
tabella_grafica.scale(1.2, 1.8)

# Evidenzio l'intestazione
for colonna in range(len(tabella.columns)):
    cella = tabella_grafica[0, colonna]
    cella.set_text_props(weight="bold")
    cella.set_facecolor("#d9eaf7")

# Alterno leggermente le righe
for riga in range(1, len(tabella) + 1):
    for colonna in range(len(tabella.columns)):
        cella = tabella_grafica[riga, colonna]
        if riga % 2 == 0:
            cella.set_facecolor("#f2f2f2")

plt.tight_layout()
plt.savefig("tabella_confronto_passaggio_2022.png", dpi=300, bbox_inches="tight")
plt.show()

#--------------------------------------------------
# GRAFICO CONFRONTO MODELLI
#--------------------------------------------------

tabella = pd.read_csv("tabella_metriche_modelli_2022.csv")

fig, ax = plt.subplots(figsize=(12, 4.8))
ax.axis("off")

tabella_plot = ax.table(
    cellText=tabella.values,
    colLabels=tabella.columns,
    cellLoc="center",
    colLoc="center",
    loc="center"
)

tabella_plot.auto_set_font_size(False)
tabella_plot.set_fontsize(10)
tabella_plot.scale(1, 1.7)

for (riga, colonna), cella in tabella_plot.get_celld().items():
    if riga == 0:
        cella.set_text_props(weight="bold")
        cella.set_facecolor("#d9eaf7")
    else:
        if riga % 2 == 0:
            cella.set_facecolor("#f2f2f2")
        else:
            cella.set_facecolor("#ffffff")

    cella.set_edgecolor("#444444")

plt.tight_layout()
plt.savefig("tabella_metriche_modelli_2022.png", dpi=300, bbox_inches="tight")
plt.show()
