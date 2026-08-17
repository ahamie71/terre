"""Bureau d'Analyse Terrestre - analyse des relevés de la sonde Klaxo-3."""

import csv
import os
import re
import urllib.request
from datetime import datetime

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


if __name__ == "__main__":
    telecharger_donnees()
    lignes_valides, lignes_ecartees = phase1_ouvrir_la_caisse()
    releves = phase2_rien_nest_du_bon_type(lignes_valides)
    releves = phase3_trier_les_canulars(releves)
