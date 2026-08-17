"""Bureau d'Analyse Terrestre - analyse des relevés de la sonde Klaxo-3."""

import csv
import os
import re
import urllib.request
from datetime import datetime

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_URL = (
    "https://raw.githubusercontent.com/planetsig/ufo-reports/master/"
    "csv-data/ufo-complete-geocoded-time-standardized.csv"
)
DATA_FILE = "releves_klaxo3.csv"

COLONNES = [
    "datetime",
    "city",
    "state",
    "country",
    "shape",
    "duration_seconds",
    "duration_hours_min",
    "comments",
    "date_posted",
    "latitude",
    "longitude",
]
NB_CHAMPS_ATTENDU = len(COLONNES)


def telecharger_donnees():
    if os.path.exists(DATA_FILE):
        return
    print(f"Téléchargement de {DATA_FILE}...")
    urllib.request.urlretrieve(DATA_URL, DATA_FILE)


def phase1_ouvrir_la_caisse():
    print("=== Phase 1 : ouvrir la caisse ===")

    with open(DATA_FILE, newline="", encoding="utf-8") as f:
        nb_lignes_fichier = sum(1 for _ in f)

    with open(DATA_FILE, newline="", encoding="utf-8") as f:
        toutes_les_lignes = list(csv.reader(f))

    lignes_valides = [l for l in toutes_les_lignes if len(l) == NB_CHAMPS_ATTENDU]
    lignes_ecartees = [l for l in toutes_les_lignes if len(l) != NB_CHAMPS_ATTENDU]

    print(f"Lignes dans le fichier       : {nb_lignes_fichier}")
    print(f"Lignes chargées ({NB_CHAMPS_ATTENDU} champs) : {len(lignes_valides)}")
    print(f"Lignes écartées (≠{NB_CHAMPS_ATTENDU} champs) : {len(lignes_ecartees)}")

    if lignes_ecartees:
        exemple = lignes_ecartees[0]
        print(
            f"\nExemple de ligne écartée ({len(exemple)} champs au lieu de {NB_CHAMPS_ATTENDU}):"
        )
        print(exemple)

    return lignes_valides, lignes_ecartees


def convertir_nombre(releves, champ):
    """Convertit champ en float sur place. Ne supprime aucune ligne : les
    valeurs qui résistent deviennent None. Retourne (nb_echecs, exemples)."""
    exemples = []
    nb_echecs = 0
    for r in releves:
        brut = r[champ]
        try:
            r[champ] = float(brut)
        except ValueError:
            nb_echecs += 1
            if len(exemples) < 5:
                exemples.append(brut)
            r[champ] = None
    return nb_echecs, exemples


def convertir_date(releves, champ, fmt):
    """Convertit champ en datetime sur place, même logique que convertir_nombre."""
    exemples = []
    nb_echecs = 0
    for r in releves:
        brut = r[champ]
        try:
            r[champ] = datetime.strptime(brut, fmt)
        except ValueError:
            nb_echecs += 1
            if len(exemples) < 5:
                exemples.append(brut)
            r[champ] = None
    return nb_echecs, exemples


def phase2_rien_nest_du_bon_type(lignes_valides):
    print("\n=== Phase 2 : rien n'est du bon type ===")

    releves = [dict(zip(COLONNES, l)) for l in lignes_valides]

    conversions = [
        ("duration_seconds", convertir_nombre, ()),
        ("latitude", convertir_nombre, ()),
        ("longitude", convertir_nombre, ()),
        ("datetime", convertir_date, ("%m/%d/%Y %H:%M",)),
        ("date_posted", convertir_date, ("%m/%d/%Y",)),
    ]

    for champ, fonction, args in conversions:
        nb_echecs, exemples = fonction(releves, champ, *args)
        print(f"{champ}: {nb_echecs} valeur(s) non convertie(s) — exemples : {exemples}")

    return releves


def phase3_trier_les_canulars(releves):
    print("\n=== Phase 3 : le Conseil veut trier les canulars ===")

    for r in releves:
        r["canular"] = bool(re.search(r"hoax", r["comments"], re.IGNORECASE))

    nb_canulars = sum(r["canular"] for r in releves)
    proportion = 100 * nb_canulars / len(releves)

    print("Règle : un relevé est un canular si le mot 'hoax' apparaît dans comments.")
    print(f"Canulars détectés : {nb_canulars} ({proportion:.2f} % de {len(releves)})")

    return releves


def entrainer_et_evaluer(df, colonnes_cat, colonnes_num, colonne_texte, random_state=42):
    """Entraîne une régression logistique sur les colonnes données et rend
    (nb de relevés de test, rappel, précision, exactitude), calculés sur un
    jeu de test jamais vu à l'entraînement."""
    colonnes = colonnes_cat + colonnes_num + ([colonne_texte] if colonne_texte else [])
    X = df[colonnes].copy()
    for c in colonnes_num:
        X[c] = X[c].fillna(X[c].median())
    for c in colonnes_cat:
        X[c] = X[c].fillna("inconnu")
    if colonne_texte:
        X[colonne_texte] = X[colonne_texte].fillna("")
    y = df["canular"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    transformateurs = [("cat", OneHotEncoder(handle_unknown="ignore"), colonnes_cat)]
    if colonne_texte:
        transformateurs.append(
            ("texte", TfidfVectorizer(max_features=3000, stop_words="english"), colonne_texte)
        )

    modele = Pipeline(
        [
            ("pretraitement", ColumnTransformer(transformateurs, remainder="passthrough")),
            ("classifieur", LogisticRegression(max_iter=5000, class_weight="balanced")),
        ]
    )
    modele.fit(X_train, y_train)
    y_pred = modele.predict(X_test)

    rappel = recall_score(y_test, y_pred, zero_division=0)
    precision = precision_score(y_test, y_pred, zero_division=0)
    exactitude = accuracy_score(y_test, y_pred)
    return len(X_test), rappel, precision, exactitude


def phase4_premier_verdict(releves):
    print("\n=== Phase 4 : le premier verdict ===")

    df = pd.DataFrame(releves)
    colonnes_cat = ["shape", "state", "country"]
    colonnes_num = ["duration_seconds", "latitude", "longitude"]
    colonne_texte = "comments"

    n_test, rappel, precision, _ = entrainer_et_evaluer(df, colonnes_cat, colonnes_num, colonne_texte)

    print(f"Évalué sur {n_test} relevés jamais vus à l'entraînement (20 % du jeu, stratifié).")
    print(f"Rappel    : {100 * rappel:.1f} / 100 canulars réellement présents attrapés")
    print(f"Précision : {100 * precision:.1f} / 100 relevés signalés qui sont vraiment des canulars")

    return df


def phase5_le_conseil_ne_vous_croit_pas(df):
    print("\n=== Phase 5 : le Conseil ne vous croit pas ===")

    avant = entrainer_et_evaluer(
        df, ["shape", "state", "country"], ["duration_seconds", "latitude", "longitude"], "comments"
    )

    apres = entrainer_et_evaluer(
        df, ["shape", "state", "country"], ["duration_seconds", "latitude", "longitude"], None
    )

    print("Avant (avec comments) :")
    print(f"  Rappel    : {100 * avant[1]:.1f} / 100")
    print(f"  Précision : {100 * avant[2]:.1f} / 100")
    print("Après (sans comments) :")
    print(f"  Rappel    : {100 * apres[1]:.1f} / 100")
    print(f"  Précision : {100 * apres[2]:.1f} / 100")

    return avant, apres


def phase6_le_modele_le_plus_bete(df, apres):
    print("\n=== Phase 6 : le modèle le plus bête du Bureau ===")

    y = df["canular"]
    _, y_test = train_test_split(y, test_size=0.2, random_state=42, stratify=y)
    exactitude_stagiaire = accuracy_score(y_test, [False] * len(y_test))
    exactitude_modele = apres[3]

    print(f"Exactitude du stagiaire (toujours 'pas canular') : {100 * exactitude_stagiaire:.1f} %")
    print(f"Exactitude de notre modèle (sans comments)        : {100 * exactitude_modele:.1f} %")


if __name__ == "__main__":
    telecharger_donnees()
    lignes_valides, lignes_ecartees = phase1_ouvrir_la_caisse()
    releves = phase2_rien_nest_du_bon_type(lignes_valides)
    releves = phase3_trier_les_canulars(releves)
    df = phase4_premier_verdict(releves)
    avant, apres = phase5_le_conseil_ne_vous_croit_pas(df)
    phase6_le_modele_le_plus_bete(df, apres)
