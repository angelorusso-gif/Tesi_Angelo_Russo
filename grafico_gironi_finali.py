import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("simulazione_gironi_2026.csv")
df.columns = df.columns.str.strip()

if "girone" not in df.columns:
    raise KeyError("Manca la colonna 'girone'")

if "squadra" not in df.columns:
    if "team" in df.columns:
        df = df.rename(columns={"team": "squadra"})
    else:
        raise KeyError("Manca la colonna 'squadra'")

if "posizione" not in df.columns:
    if "Pos." in df.columns:
        df = df.rename(columns={"Pos.": "posizione"})
    elif "pos" in df.columns:
        df = df.rename(columns={"pos": "posizione"})
    else:
        raise KeyError("Manca la colonna 'posizione'")

if "punti" not in df.columns:
    raise KeyError("Manca la colonna 'punti'")

if "vittorie" not in df.columns:
    raise KeyError("Manca la colonna 'vittorie'")

if "risultato_predetto" in df.columns:
    col_esito = "risultato_predetto"
elif "esito_predetto" in df.columns:
    col_esito = "esito_predetto"
elif "esito" in df.columns:
    col_esito = "esito"
else:
    raise KeyError("Manca una colonna tra 'risultato_predetto', 'esito_predetto', 'esito'")

df[col_esito] = df[col_esito].astype(str).str.upper().str.strip()

def abbrevia(nome, max_len=14):
    nome = str(nome)
    if len(nome) <= max_len:
        return nome
    return nome[:max_len-3] + "..."

def crea_tabella_girone(ax, dati_girone, nome_girone):
    ax.axis("off")

    dati_girone = dati_girone.sort_values("posizione").copy()

    cell_text = []
    cell_colours = []

    for _, r in dati_girone.iterrows():
        esito = str(r[col_esito]).upper().strip()

        cell_text.append([
            int(r["posizione"]),
            abbrevia(r["squadra"]),
            int(r["punti"]),
            int(r["vittorie"]),
            esito
        ])

        if esito == "PASSA":
            row_color = ["#dfeeda"] * 5
        else:
            row_color = ["#f4d6d6"] * 5

        cell_colours.append(row_color)

    table = ax.table(
        cellText=cell_text,
        colLabels=["Pos.", "Squadra", "Punti", "Vittorie", "Esito"],
        cellColours=cell_colours,
        colColours=["#1f5a91"] * 5,
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.10, 0.34, 0.14, 0.18, 0.24]
    )

    table.scale(1, 1.6)
    table.auto_set_font_size(False)
    table.set_fontsize(8)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#444444")
        cell.set_linewidth(0.8)

        if row == 0:
            cell.set_text_props(color="white", weight="bold", fontsize=8)

    ax.set_title(f"Girone {nome_girone}", fontsize=12, fontweight="bold", pad=10)

def crea_blocco(lista_gironi, titolo, nome_file):
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8))
    axes = axes.flatten()

    for ax, girone in zip(axes, lista_gironi):
        dati = df[df["girone"] == girone]
        crea_tabella_girone(ax, dati, girone)

    fig.suptitle(titolo, fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(nome_file, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Salvata: {nome_file}")

crea_blocco(["A", "B", "C", "D"], "Risultati finali della simulazione - Gironi A-D", "gironi_2026_blocco1.jpg")
crea_blocco(["E", "F", "G", "H"], "Risultati finali della simulazione - Gironi E-H", "gironi_2026_blocco2.jpg")
crea_blocco(["I", "J", "K", "L"], "Risultati finali della simulazione - Gironi I-L", "gironi_2026_blocco3.jpg")

print("FINE. Le 3 immagini sono state create.")
