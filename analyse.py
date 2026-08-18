"""Bureau d'Analyse Terrestre - analyse des relevés de la sonde Klaxo-3."""

import csv
import os
import re
import urllib.request
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

warnings.filterwarnings("ignore", category=ConvergenceWarning)

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
    for r in releves:
        r["_datetime_brut"] = r["datetime"]

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


def entrainer_et_evaluer(df, colonnes_cat, colonnes_num, colonne_texte, random_state=42, decoupe=None):
    """Entraîne une régression logistique sur les colonnes données et rend
    (nb de relevés de test, rappel, précision, exactitude), calculés sur un
    jeu de test jamais vu à l'entraînement.

    decoupe, si fourni, est un couple (idx_train, idx_test) tout fait — sinon
    une découpe aléatoire stratifiée est utilisée. Attention : les moyennes/
    médianes ci-dessous sont encore calculées sur tout df, pas juste sur
    l'apprentissage — c'est la fuite que la phase 10 va corriger."""
    colonnes = colonnes_cat + colonnes_num + ([colonne_texte] if colonne_texte else [])
    X = df[colonnes].copy()
    for c in colonnes_num:
        X[c] = X[c].fillna(X[c].median())
    for c in colonnes_cat:
        # les trous de ces colonnes sont des chaînes vides dans le fichier, pas des NaN
        X[c] = X[c].replace("", "inconnu").fillna("inconnu")
    if colonne_texte:
        X[colonne_texte] = X[colonne_texte].fillna("")
    y = df["canular"]

    if decoupe is not None:
        idx_train, idx_test = decoupe
        X_train, X_test = X.loc[idx_train], X.loc[idx_test]
        y_train, y_test = y.loc[idx_train], y.loc[idx_test]
    else:
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
            ("classifieur", LogisticRegression(max_iter=5000, class_weight="balanced", random_state=42)),
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


COLONNES_CAT_HONNETES = ["shape", "state", "country"]
COLONNES_NUM_HONNETES = ["duration_seconds", "latitude", "longitude"]


def reparer_heure_24h(df):
    """datetime garde 1220 valeurs à None depuis la phase 2 (heure '24:00',
    invalide). On en a besoin ici pour grouper/trier par date, donc on les
    répare : '24:00' = minuit le jour suivant. Ne change rien au compte déjà
    publié en phase 2, juste une réparation ciblée sur ce même bug."""
    manquants = df["datetime"].isna()
    brut = df.loc[manquants, "_datetime_brut"]
    jours = pd.to_datetime(brut.str.replace(" 24:00", "", regex=False), format="%m/%d/%Y")
    df.loc[manquants, "datetime"] = jours + pd.Timedelta(days=1)
    return int(manquants.sum())


def phase7_plusieurs_temoins_un_seul_evenement(df):
    print("\n=== Phase 7 : plusieurs témoins, un seul événement ===")

    n_repares = reparer_heure_24h(df)
    print(f"(heures '24:00' réparées pour pouvoir grouper par jour : {n_repares})")

    df["jour"] = df["datetime"].dt.date
    df["groupe_evenement"] = df.groupby(["jour", "city", "state"], dropna=False).ngroup()

    tailles = df.groupby("groupe_evenement").size()
    multi = tailles[tailles > 1]
    gid_max = multi.idxmax()
    plus_gros = df.loc[df["groupe_evenement"] == gid_max, ["jour", "city", "state"]].iloc[0]

    print(f"Événements avec plus d'un témoin : {len(multi)}")
    print(
        f"Témoins pour le plus gros : {multi.max()} "
        f"({plus_gros['jour']}, {plus_gros['city']}, {plus_gros['state']})"
    )

    idx_train_hier, idx_test_hier = train_test_split(
        df.index, test_size=0.2, random_state=42, stratify=df["canular"]
    )
    train_set, test_set = set(idx_train_hier), set(idx_test_hier)
    a_cheval = sum(
        len(idxs)
        for _, sous in df.groupby("groupe_evenement")
        for idxs in [set(sous.index)]
        if idxs & train_set and idxs & test_set
    )
    print(f"Relevés à cheval sur les deux côtés dans la découpe d'hier : {a_cheval}")

    # témoignages recopiés mot pour mot (on ignore les textes très courts,
    # trop génériques — "Fireball", "UFO" — qui se répètent par hasard)
    c = df["comments"].fillna("")
    doublons = c[(c != "") & (c.str.len() >= 40)].value_counts()
    doublons = doublons[doublons > 1]
    print(f"Témoignages recopiés à l'identique (≥40 caractères) : {doublons.sum()} lignes, {len(doublons)} textes")

    groupe_final = df["groupe_evenement"].to_numpy().copy()
    for texte in doublons.index:
        positions = df.index.get_indexer(df.index[df["comments"] == texte])
        gids = np.unique(groupe_final[positions])
        cible = gids[0]
        for g in gids[1:]:
            groupe_final[groupe_final == g] = cible
    df["groupe_final"] = groupe_final

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    idx_train_pos, idx_test_pos = next(gss.split(df, groups=df["groupe_final"]))
    idx_train, idx_test = df.index[idx_train_pos], df.index[idx_test_pos]

    avant = entrainer_et_evaluer(
        df, COLONNES_CAT_HONNETES, COLONNES_NUM_HONNETES, None, decoupe=(idx_train_hier, idx_test_hier)
    )
    apres = entrainer_et_evaluer(
        df, COLONNES_CAT_HONNETES, COLONNES_NUM_HONNETES, None, decoupe=(idx_train, idx_test)
    )

    print("Phase 4 (modèle honnête, sans comments) avant / après la découpe groupée :")
    print(f"  Rappel    : {100 * avant[1]:.1f} → {100 * apres[1]:.1f} / 100")
    print(f"  Précision : {100 * avant[2]:.1f} → {100 * apres[2]:.1f} / 100")

    exemple = df[df["groupe_evenement"] == gid_max][["datetime", "city", "state", "shape", "canular"]]
    print(f"\nExemple, un événement complet ({len(exemple)} témoins) :")
    print(exemple.to_string())

    return df, (idx_train, idx_test)


def phase8_ordre_des_choses(df):
    print("\n=== Phase 8 : l'ordre des choses ===")

    print(
        "Coupure sur datetime (l'instant où le témoin lève les yeux), pas date_posted "
        "(quand le Bureau reçoit/traite le dossier) : 8836 relevés ont plus de 10 ans "
        "d'écart entre les deux, donc date_posted mélangerait des vieux dossiers traités "
        "tard avec des dossiers récents traités vite. datetime seul respecte l'ordre réel."
    )

    # on regroupe par événement (jour+ville+state, phase 7) pour ne pas couper un
    # événement en deux, mais PAS par témoignage recopié : ces doublons peuvent être
    # à des décennies d'écart et casseraient l'ordre chronologique. Sans conséquence
    # ici puisque comments est déjà hors du modèle depuis la phase 5.
    date_du_groupe = df.groupby("groupe_evenement")["datetime"].min().sort_values()
    taille_du_groupe = df["groupe_evenement"].value_counts()

    objectif_train = 0.8 * len(df)
    cumul = 0
    groupes_train = []
    for g in date_du_groupe.index:
        if cumul >= objectif_train:
            break
        groupes_train.append(g)
        cumul += taille_du_groupe[g]
    groupes_train = set(groupes_train)

    idx_train = df.index[df["groupe_evenement"].isin(groupes_train)]
    idx_test = df.index[~df["groupe_evenement"].isin(groupes_train)]

    date_coupure = df.loc[idx_train, "datetime"].max()
    proportion_train = 100 * df.loc[idx_train, "canular"].mean()
    proportion_test = 100 * df.loc[idx_test, "canular"].mean()

    print(f"Date de coupure : {date_coupure.date()} (train = avant, test = à partir de là)")
    print(f"Relevés : {len(idx_train)} en apprentissage, {len(idx_test)} en test")
    print(f"Proportion de canulars — train : {proportion_train:.2f} % | test : {proportion_test:.2f} %")

    _, rappel, precision, _ = entrainer_et_evaluer(
        df, COLONNES_CAT_HONNETES, COLONNES_NUM_HONNETES, None, decoupe=(idx_train, idx_test)
    )
    print(f"Rappel : {100 * rappel:.1f} / 100  —  Précision : {100 * precision:.1f} / 100")

    return df, (idx_train, idx_test)


def est_troue(df, colonne):
    return df[colonne].isna() | (df[colonne] == "")


def phase9_les_cases_vides(df):
    print("\n=== Phase 9 : les cases vides ===")

    trous = {c: est_troue(df, c).sum() for c in df.columns if c not in ("canular",)}
    trois_plus_trouees = sorted(trous, key=trous.get, reverse=True)[:3]
    print(f"Trois colonnes les plus trouées : {trois_plus_trouees}")

    for col in trois_plus_trouees:
        t = est_troue(df, col)
        p_troue = 100 * df.loc[t, "canular"].mean()
        p_plein = 100 * df.loc[~t, "canular"].mean()
        print(
            f"  {col} : {t.sum()} trous — canular si troué {p_troue:.2f} % "
            f"vs canular si rempli {p_plein:.2f} %"
        )

    print(
        "Traitement retenu : les trous des colonnes catégorielles (country, state) partent "
        "dans leur propre catégorie 'inconnu' plutôt que d'être fondus dans une valeur "
        "existante — le one-hot lui donne sa colonne à elle, donc le modèle voit toujours "
        "qu'il y avait un trou à cet endroit, et peut apprendre que ça corrèle avec canular. "
        "duration_hours_min n'est pas encore une feature du modèle ; le même principe "
        "s'appliquera quand on la travaillera en phase 11."
    )

    return df


def _vide_en_nan(X):
    return X.replace("", np.nan)


def construire_pipeline(colonnes_cat, colonnes_num, colonne_texte):
    """Pipeline complet : rien n'est appris (médiane, catégories, vocabulaire)
    avant .fit(), donc rien n'est appris ailleurs que sur l'apprentissage."""
    transformateurs = []
    if colonnes_cat:
        pipeline_cat = Pipeline(
            [
                ("vide_en_nan", FunctionTransformer(_vide_en_nan, feature_names_out="one-to-one")),
                ("imputer", SimpleImputer(strategy="constant", fill_value="inconnu")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformateurs.append(("cat", pipeline_cat, colonnes_cat))
    if colonnes_num:
        transformateurs.append(("num", SimpleImputer(strategy="median"), colonnes_num))
    if colonne_texte:
        transformateurs.append(
            ("texte", TfidfVectorizer(max_features=3000, stop_words="english"), colonne_texte)
        )

    return Pipeline(
        [
            ("pretraitement", ColumnTransformer(transformateurs, remainder="drop")),
            ("classifieur", LogisticRegression(max_iter=5000, class_weight="balanced", random_state=42)),
        ]
    )


def entrainer_pipeline(df, colonnes_cat, colonnes_num, colonne_texte, idx_train, idx_test):
    colonnes = colonnes_cat + colonnes_num + ([colonne_texte] if colonne_texte else [])
    X = df[colonnes]
    y = df["canular"]

    pipeline = construire_pipeline(colonnes_cat, colonnes_num, colonne_texte)
    pipeline.fit(X.loc[idx_train], y.loc[idx_train])
    y_pred = pipeline.predict(X.loc[idx_test])

    rappel = recall_score(y.loc[idx_test], y_pred, zero_division=0)
    precision = precision_score(y.loc[idx_test], y_pred, zero_division=0)
    exactitude = accuracy_score(y.loc[idx_test], y_pred)
    return len(idx_test), rappel, precision, exactitude, pipeline


def phase10_chaine_de_traitement(df, idx_train, idx_test):
    print("\n=== Phase 10 : la chaîne de traitement du Bureau ===")

    proportion_train = 100 * df.loc[idx_train, "canular"].mean()
    proportion_test = 100 * df.loc[idx_test, "canular"].mean()
    print(f"Proportion de canulars — apprentissage : {proportion_train:.2f} % | test : {proportion_test:.2f} %")
    print("(déjà rendu en phase 8 : ni l'un ni l'autre n'est proche de 0, la découpe n'est pas dégénérée)")

    avant = entrainer_et_evaluer(df, COLONNES_CAT_HONNETES, COLONNES_NUM_HONNETES, None, decoupe=(idx_train, idx_test))
    _, rappel, precision, _, pipeline = entrainer_pipeline(
        df, COLONNES_CAT_HONNETES, COLONNES_NUM_HONNETES, None, idx_train, idx_test
    )

    print("Rappel / précision, fillna manuel sur tout df → pipeline appris sur l'apprentissage seul :")
    print(f"  Rappel    : {100 * avant[1]:.1f} → {100 * rappel:.1f} / 100")
    print(f"  Précision : {100 * avant[2]:.1f} → {100 * precision:.1f} / 100")

    nouveau = pd.DataFrame(
        [
            {
                "shape": "triangle",
                "state": "il",
                "country": "us",
                "duration_seconds": 300.0,
                "latitude": 41.57,
                "longitude": -87.78,
            }
        ]
    )
    prediction = pipeline.predict(nouveau)[0]
    print(f"Relevé inventé à la main → chaîne complète en un seul appel → prédiction : {'canular' if prediction else 'pas canular'}")

    return df, pipeline


PLAFOND_DUREE_S = 365 * 86400  # au-delà, une durée n'est plus crédible pour une observation


def _en_secondes(valeur, unite):
    if unite.startswith("sec"):
        return valeur
    if unite.startswith("min"):
        return valeur * 60
    if unite.startswith("hour") or unite.startswith("hr") or unite == "h":
        return valeur * 3600
    if unite.startswith("day"):
        return valeur * 86400
    if unite.startswith("week"):
        return valeur * 604800
    return None


def parser_duree_texte(texte):
    """Essaie de lire une durée dans le texte libre du témoin (duration_hours_min)
    et la rend en secondes. Rend None si rien de reconnaissable."""
    if not texte:
        return None
    t = texte.strip().lower()

    m = re.fullmatch(r"(\d{1,2}):(\d{2}):(\d{2})", t)
    if m:
        h, mi, s = map(int, m.groups())
        return h * 3600 + mi * 60 + s
    m = re.fullmatch(r"(\d{1,2}):(\d{2})", t)
    if m:
        mi, s = map(int, m.groups())
        return mi * 60 + s

    m = re.search(r"(\d+)\s*min\w*\.?\s*(\d+)\s*sec", t)
    if m:
        return int(m.group(1)) * 60 + int(m.group(2))

    m = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)\s*(seconds?|secs?|minutes?|mins?|hours?|hrs?|days?|weeks?)\b",
        t,
    )
    if m:
        a, b, unite = float(m.group(1)), float(m.group(2)), m.group(3)
        return _en_secondes((a + b) / 2, unite)

    m = re.search(r"(\d+)\s*/\s*(\d+)\s*(seconds?|secs?|minutes?|mins?|hours?|hrs?)\b", t)
    if m:
        a, b, unite = m.groups()
        return _en_secondes(float(a) / float(b), unite)

    m = re.search(r"(\d+(?:\.\d+)?)\s*(seconds?|secs?|s\b|minutes?|mins?|m\b|hours?|hrs?|h\b|days?|weeks?)", t)
    if m:
        return _en_secondes(float(m.group(1)), m.group(2))

    return None


def reconcilier_duree(df):
    df["duree_texte"] = df["duration_hours_min"].apply(parser_duree_texte)

    d_sec = df["duration_seconds"]
    plausible = d_sec.where(d_sec.isna() | (d_sec <= PLAFOND_DUREE_S))
    n_implausible = int((d_sec.notna() & (d_sec > PLAFOND_DUREE_S)).sum())

    contradictions = (
        d_sec.notna()
        & (d_sec != 0)
        & (d_sec <= PLAFOND_DUREE_S)
        & df["duree_texte"].notna()
        & ((d_sec / df["duree_texte"] > 3) | (df["duree_texte"] / d_sec > 3))
    )

    duree_finale = plausible.where(plausible.notna() & (plausible != 0), df["duree_texte"])
    duree_finale = duree_finale.where(duree_finale.isna() | (duree_finale <= PLAFOND_DUREE_S))
    df["duree_s"] = duree_finale

    return n_implausible, int(contradictions.sum()), contradictions


def phase11_combien_de_temps_ca_a_dure(df, idx_train, idx_test):
    print("\n=== Phase 11 : combien de temps ça a duré ===")

    top3_avant = df["duration_seconds"].sort_values(ascending=False).head(3)
    print("Trois durées les plus longues, telles quelles :")
    print(df.loc[top3_avant.index, ["duration_seconds", "duration_hours_min"]].to_string())

    n_implausible, n_contradictions, contradictions = reconcilier_duree(df)

    print(
        f"\n{n_implausible} valeurs de duration_seconds au-delà d'un an (ex. « 31 years », "
        f"« 23000hrs ») jugées pas crédibles pour une observation — mises de côté plutôt que "
        f"gardées telles quelles (ça se voit tout de suite dans la médiane si on les garde)."
    )
    print(f"Colonnes qui se contredisent (facteur > 3 entre les deux) : {n_contradictions}")

    n_inutilisable = int(df["duree_s"].isna().sum())
    print(f"Relevés dont la durée reste inutilisable après traitement : {n_inutilisable}")
    print(f"Durée médiane : {df['duree_s'].median():.0f} s")
    print(f"Relevés annonçant plus d'une journée d'observation : {int((df['duree_s'] > 86400).sum())}")

    exemple = df[contradictions].iloc[0]
    print("\nExemple, les deux colonnes racontent deux histoires différentes :")
    print(exemple[["datetime", "city", "duration_seconds", "duration_hours_min", "duree_s"]].to_string())

    print(f"\nLignes avant : {len(df)} — lignes après : {len(df)} (aucune perdue)")

    _, rappel_avant, precision_avant, _, _ = entrainer_pipeline(
        df, COLONNES_CAT_HONNETES, ["duration_seconds", "latitude", "longitude"], None, idx_train, idx_test
    )
    _, rappel_apres, precision_apres, _, _ = entrainer_pipeline(
        df, COLONNES_CAT_HONNETES, ["duree_s", "latitude", "longitude"], None, idx_train, idx_test
    )
    print(f"\nRappel    : {100 * rappel_avant:.1f} → {100 * rappel_apres:.1f} / 100 (duration_seconds → duree_s)")
    print(f"Précision : {100 * precision_avant:.1f} → {100 * precision_apres:.1f} / 100")

    return df


class RegroupeurVillesRares(BaseEstimator, TransformerMixin):
    """Ne garde en catégorie propre que les villes vues au moins `seuil` fois
    dans l'apprentissage ; tout le reste (et toute ville jamais vue) devient
    'autre'. La liste des villes gardées n'est apprise qu'au .fit()."""

    def __init__(self, seuil=20):
        self.seuil = seuil

    def fit(self, X, y=None):
        comptes = pd.Series(np.ravel(X)).value_counts()
        self.villes_frequentes_ = set(comptes[comptes >= self.seuil].index)
        return self

    def transform(self, X):
        s = pd.Series(np.ravel(X))
        s = s.where(s.isin(self.villes_frequentes_), "autre")
        return s.to_numpy().reshape(-1, 1)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features if input_features is not None else ["city"])


class EncodeurHeureCyclique(BaseEstimator, TransformerMixin):
    """23h et 0h sont voisines dans le ciel ; sin/cos le disent au modèle,
    une seule colonne 0-23 ne le pouvait pas. Rien n'est appris des données."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        heures = pd.to_datetime(pd.Series(np.ravel(X))).dt.hour.to_numpy()
        radians = 2 * np.pi * heures / 24
        return np.column_stack([np.sin(radians), np.cos(radians)])

    def get_feature_names_out(self, input_features=None):
        return np.array(["heure_sin", "heure_cos"])


FUSIONS_FORMES = {"changed": "changing", "round": "circle"}


def _fusionner_formes(X):
    return X.replace(FUSIONS_FORMES)


def construire_pipeline_v2(seuil_villes=20):
    """Comme construire_pipeline, avec en plus la ville (regroupée, apprise
    sur train seul) et l'heure (encodage cyclique)."""
    pipeline_ville = Pipeline(
        [
            ("regroupeur", RegroupeurVillesRares(seuil=seuil_villes)),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    pipeline_forme = Pipeline(
        [
            ("fusion", FunctionTransformer(_fusionner_formes, feature_names_out="one-to-one")),
            ("vide_en_nan", FunctionTransformer(_vide_en_nan, feature_names_out="one-to-one")),
            ("imputer", SimpleImputer(strategy="constant", fill_value="inconnu")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    pipeline_cat = Pipeline(
        [
            ("vide_en_nan", FunctionTransformer(_vide_en_nan, feature_names_out="one-to-one")),
            ("imputer", SimpleImputer(strategy="constant", fill_value="inconnu")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    pretraitement = ColumnTransformer(
        [
            ("ville", pipeline_ville, ["city"]),
            ("forme", pipeline_forme, ["shape"]),
            ("cat", pipeline_cat, ["state", "country"]),
            ("heure", EncodeurHeureCyclique(), ["datetime"]),
            ("num", SimpleImputer(strategy="median"), ["duree_s", "latitude", "longitude"]),
        ],
        remainder="drop",
    )
    return Pipeline(
        [
            ("pretraitement", pretraitement),
            ("classifieur", LogisticRegression(max_iter=5000, class_weight="balanced", random_state=42)),
        ]
    )


def phase12_la_ville_et_lheure(df, idx_train, idx_test):
    print("\n=== Phase 12 : la ville et l'heure ===")

    n_villes_uniques = int((df["city"].value_counts() == 1).sum())
    print(f"Villes n'apparaissant qu'une fois dans toute la transmission : {n_villes_uniques}")

    colonnes_avant = ["shape", "state", "country", "duree_s", "latitude", "longitude"]
    pipeline_avant = construire_pipeline(COLONNES_CAT_HONNETES, ["duree_s", "latitude", "longitude"], None)
    pipeline_avant.fit(df.loc[idx_train, colonnes_avant], df.loc[idx_train, "canular"])
    largeur_avant = pipeline_avant.named_steps["pretraitement"].transform(df.loc[idx_test, colonnes_avant]).shape[1]

    pipeline_v2 = construire_pipeline_v2(seuil_villes=20)
    colonnes_v2 = ["city", "shape", "state", "country", "datetime", "duree_s", "latitude", "longitude"]
    pipeline_v2.fit(df.loc[idx_train, colonnes_v2], df.loc[idx_train, "canular"])
    largeur_apres = pipeline_v2.named_steps["pretraitement"].transform(df.loc[idx_test, colonnes_v2]).shape[1]

    n_villes_gardees = len(pipeline_v2.named_steps["pretraitement"].named_transformers_["ville"].named_steps["regroupeur"].villes_frequentes_)

    print(f"Colonnes du tableau : {largeur_avant} (sans ville/heure) → {largeur_apres} (avec)")
    print(f"Règle ville : gardée si vue ≥ 20 fois dans l'apprentissage ({n_villes_gardees} villes gardées), sinon 'autre'.")

    def point_heure(h):
        r = 2 * np.pi * h / 24
        return np.array([np.sin(r), np.cos(r)])

    d_23_0 = np.linalg.norm(point_heure(23) - point_heure(0))
    d_23_20 = np.linalg.norm(point_heure(23) - point_heure(20))
    print(f"Distance encodée 23h↔0h : {d_23_0:.3f}  —  23h↔20h : {d_23_20:.3f}")

    formes_avant = set(df["shape"]) - {""}
    formes_apres = {FUSIONS_FORMES.get(f, f) for f in formes_avant}
    print(f"Formes : {len(formes_avant)} → {len(formes_apres)} après fusion ({FUSIONS_FORMES})")

    y_pred = pipeline_v2.predict(df.loc[idx_test, colonnes_v2])
    y_test = df.loc[idx_test, "canular"]
    rappel = recall_score(y_test, y_pred, zero_division=0)
    precision = precision_score(y_test, y_pred, zero_division=0)
    print(f"Rappel : {100 * rappel:.1f} / 100  —  Précision : {100 * precision:.1f} / 100")

    return df, pipeline_v2


COLONNES_MODELE = ["city", "shape", "state", "country", "datetime", "duree_s", "latitude", "longitude"]

COUT_CANULAR_MANQUE = 30  # faux négatif : une équipe part travailler sur du vent
COUT_FAUSSE_ALERTE = 2  # faux positif : le Bureau perd une observation potentiellement utile


def facture(y_vrai, y_predit):
    y_vrai = np.asarray(y_vrai)
    y_predit = np.asarray(y_predit)
    faux_negatifs = int((y_vrai & ~y_predit).sum())
    faux_positifs = int((~y_vrai & y_predit).sum())
    return faux_negatifs * COUT_CANULAR_MANQUE + faux_positifs * COUT_FAUSSE_ALERTE


def phase13_la_facture_du_bureau(df, pipeline, idx_train, idx_test):
    print("\n=== Phase 13 : la facture du Bureau ===")

    probas = pipeline.predict_proba(df.loc[idx_test, COLONNES_MODELE])[:, 1]
    y_test = df.loc[idx_test, "canular"].to_numpy()

    seuils = np.round(np.arange(0.0, 1.001, 0.01), 2)
    factures = np.array([facture(y_test, probas >= s) for s in seuils])

    print("Facture selon la frontière (extrait tous les 0,05) :")
    for s, f in zip(seuils, factures):
        if round(s * 100) % 5 == 0:
            print(f"  {s:.2f} : {f} crédits")

    i_min = int(np.argmin(factures))
    seuil_optimal = float(seuils[i_min])
    facture_optimale = int(factures[i_min])
    facture_050 = int(factures[np.where(seuils == 0.5)[0][0]])

    vrais_positifs = np.array([(y_test & (probas >= s)).sum() for s in seuils])
    utiles = np.where(vrais_positifs > 0)[0]
    i_utile = utiles[np.argmin(factures[utiles])]
    seuil_utile = float(seuils[i_utile])
    facture_utile = int(factures[i_utile])

    print(f"\nOptimum strict : {seuil_optimal:.2f} → {facture_optimale} crédits", end="")
    if vrais_positifs[i_min] == 0:
        print(" (ça revient à ne jamais rien signaler)")
    else:
        print()
    print(
        f"Meilleure frontière qui attrape encore au moins un vrai canular : "
        f"{seuil_utile:.2f} → {facture_utile} crédits ({vrais_positifs[i_utile]} canulars attrapés)"
    )
    print(f"Frontière à 0.5 : {facture_050} crédits")
    print(f"Écart (retenue vs 0.5) : {facture_050 - facture_utile} crédits économisés")
    if vrais_positifs[i_min] == 0:
        print(
            "Retenue : la frontière utile, pas l'optimum strict — un système qui n'attrape jamais "
            "rien n'est pas une décision qu'on peut défendre, et l'écart de coût entre les deux "
            "est faible (quelques dizaines de crédits sur toute la partie test)."
        )

    return seuil_utile


def phase14_promesse_a_80_pourcent(df, pipeline, idx_train, idx_test):
    print("\n=== Phase 14 : une promesse à 80 % ===")

    X_train = df.loc[idx_train, COLONNES_MODELE]
    y_train = df.loc[idx_train, "canular"]
    X_test = df.loc[idx_test, COLONNES_MODELE]
    y_test = df.loc[idx_test, "canular"].to_numpy()

    def table_tranches(probas):
        tranches = pd.qcut(probas, 10, duplicates="drop")
        t = pd.DataFrame({"tranche": tranches, "proba": probas, "reel": y_test})
        return t.groupby("tranche", observed=True).agg(
            n=("reel", "size"), proba_annoncee=("proba", "mean"), proportion_reelle=("reel", "mean")
        )

    probas_brutes = pipeline.predict_proba(X_test)[:, 1]
    table_avant = table_tranches(probas_brutes)
    print("Avant calibration :")
    print(table_avant.to_string())

    sens = "trop confiant" if (table_avant["proba_annoncee"] > table_avant["proportion_reelle"]).all() else "trop prudent"
    print(f"\nLe système est {sens} : la probabilité annoncée dépasse largement ce qui se produit vraiment.")

    calibreur = CalibratedClassifierCV(construire_pipeline_v2(seuil_villes=20), method="isotonic", cv=3)
    calibreur.fit(X_train, y_train)
    probas_calibrees = calibreur.predict_proba(X_test)[:, 1]
    table_apres = table_tranches(probas_calibrees)
    print("\nAprès calibration (isotonic, apprise par validation croisée sur l'apprentissage) :")
    print(table_apres.to_string())

    return calibreur


def phase15_deux_analystes_deux_chiffres(df, pipeline, idx_test, n_reechantillons=1000):
    print("\n=== Phase 15 : deux analystes, deux chiffres ===")

    y_test = df.loc[idx_test, "canular"].to_numpy()
    y_pred = pipeline.predict(df.loc[idx_test, COLONNES_MODELE])
    n_canulars = int(y_test.sum())

    rng = np.random.default_rng(42)
    n = len(y_test)
    rappels, precisions = [], []
    for _ in range(n_reechantillons):
        tirage = rng.integers(0, n, n)
        rappels.append(recall_score(y_test[tirage], y_pred[tirage], zero_division=0))
        precisions.append(precision_score(y_test[tirage], y_pred[tirage], zero_division=0))
    rappels, precisions = np.array(rappels), np.array(precisions)

    r_bas, r_haut = np.percentile(rappels, [2.5, 97.5]) * 100
    p_bas, p_haut = np.percentile(precisions, [2.5, 97.5]) * 100

    print(f"Taille de la partie test : {n}")
    print(f"Canulars réellement présents dans le test : {n_canulars}")
    print(f"Rappel    : {100 * rappels.mean():.1f} % [{r_bas:.1f} — {r_haut:.1f}] sur {n_reechantillons} rééchantillonnages")
    print(f"Précision : {100 * precisions.mean():.1f} % [{p_bas:.1f} — {p_haut:.1f}]")
    print(
        f"Réponse au Conseil : avec seulement {n_canulars} canulars dans le test, la fourchette "
        f"du rappel fait à elle seule {r_haut - r_bas:.0f} points — 0,31 et 0,34 tiennent "
        f"largement dans ce bruit, les deux systèmes ne sont pas distinguables sur ce chiffre."
    )

    return rappels, precisions


def expliquer_ligne(pipeline, ligne_df, top=5):
    pretraitement = pipeline.named_steps["pretraitement"]
    clf = pipeline.named_steps["classifieur"]
    X_transforme = pretraitement.transform(ligne_df)
    if hasattr(X_transforme, "toarray"):
        X_transforme = X_transforme.toarray()
    noms = pretraitement.get_feature_names_out()
    contributions = X_transforme[0] * clf.coef_[0]
    ordre = np.argsort(-np.abs(contributions))
    resultat = [(noms[i].replace("__", " : "), contributions[i]) for i in ordre if abs(contributions[i]) > 1e-9]
    return resultat[:top]


def importance_permutation(pipeline, X_test, y_test, colonnes, n_repets=5, seed=42):
    rng = np.random.default_rng(seed)
    base = recall_score(y_test, pipeline.predict(X_test), zero_division=0)
    baisses = {}
    for col in colonnes:
        chutes = []
        for _ in range(n_repets):
            X_permute = X_test.copy()
            X_permute[col] = rng.permutation(X_permute[col].to_numpy())
            score = recall_score(y_test, pipeline.predict(X_permute), zero_division=0)
            chutes.append(base - score)
        baisses[col] = float(np.mean(chutes))
    return base, baisses


def phase16_trois_dossiers_sur_le_bureau(df, pipeline, idx_test, seuil):
    print("\n=== Phase 16 : trois dossiers sur le bureau ===")

    X_test = df.loc[idx_test, COLONNES_MODELE]
    y_test = df.loc[idx_test, "canular"]
    probas = pd.Series(pipeline.predict_proba(X_test)[:, 1], index=idx_test)
    signales = probas >= seuil

    idx_confiant = probas.idxmax()
    au_dessus = probas[signales]
    idx_limite = (au_dessus - seuil).idxmin() if len(au_dessus) else idx_confiant
    faux_negatifs = probas[y_test.astype(bool) & ~signales]
    idx_manque = faux_negatifs.idxmax() if len(faux_negatifs) else probas.index[0]

    for nom, idx in [
        ("Signalé avec forte confiance", idx_confiant),
        ("Tout juste au-dessus de la frontière", idx_limite),
        ("Canular laissé passer", idx_manque),
    ]:
        ligne = df.loc[[idx], COLONNES_MODELE]
        print(f"\n{nom} — relevé #{idx} ({df.loc[idx,'city']}, {df.loc[idx,'datetime']}) :")
        print(f"  Probabilité annoncée : {probas[idx]:.3f} — vraiment canular : {bool(y_test[idx])}")
        for feature, contrib in expliquer_ligne(pipeline, ligne):
            signe = "pousse vers canular" if contrib > 0 else "pousse vers pas-canular"
            print(f"    {feature} : {contrib:+.2f} ({signe})")

    base, baisses = importance_permutation(pipeline, X_test, y_test, COLONNES_MODELE)
    classement = sorted(baisses.items(), key=lambda kv: -kv[1])
    print(f"\nImportance globale (rappel de base {100*base:.1f}, chute en mélangeant chaque colonne) :")
    for col, chute in classement:
        print(f"  {col} : {100*chute:+.2f} points de rappel")

    return classement


def phase17_langle_mort_du_bureau(df, pipeline, idx_test, seuil):
    print("\n=== Phase 17 : l'angle mort du Bureau ===")

    proportion_us = 100 * (df["country"] == "us").mean()
    print(f"Relevés venant des États-Unis : {proportion_us:.1f} %")

    X_test = df.loc[idx_test, COLONNES_MODELE]
    y_test = df.loc[idx_test, "canular"]
    y_pred = pd.Series(pipeline.predict_proba(X_test)[:, 1] >= seuil, index=idx_test)
    zones = df.loc[idx_test, "country"].replace("", "inconnu")

    print(f"\n{'zone':12} {'n':>7} {'% canular':>10} {'rappel':>8} {'précision':>10}")
    for zone, n in zones.value_counts().items():
        m = (zones == zone).to_numpy()
        r = recall_score(y_test[m], y_pred[m], zero_division=0)
        p = precision_score(y_test[m], y_pred[m], zero_division=0)
        prop = 100 * y_test[m].mean()
        print(f"{zone:12} {n:>7} {prop:>9.2f}% {100*r:>7.1f}% {100*p:>9.1f}%")

    print(
        "\nDécision : une seule frontière pour tout le monde. Les zones hors US pèsent chacune "
        "quelques centaines à quelques milliers de relevés en test — d'après la phase 15, ça ne "
        "suffit pas pour caler une frontière séparée et fiable par zone."
    )


def phase18_la_transmission_darchive(df, pipeline, idx_train, idx_test):
    print("\n=== Phase 18 : la transmission d'archive ===")

    df["annee"] = df["datetime"].dt.year
    par_annee = df[df["annee"] >= 1990].groupby("annee")["canular"].agg(["mean", "size"])
    par_annee["proportion"] = 100 * par_annee["mean"]

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(par_annee.index, par_annee["proportion"], marker="o")
    ax.set_xlabel("Année")
    ax.set_ylabel("% de canulars")
    ax.set_title("Proportion de canulars par année (1990+, avant 1990 trop peu de relevés/an)")
    fig.tight_layout()
    fig.savefig("graphique_canulars_par_annee.png", dpi=120)
    plt.close(fig)
    print("Graphique enregistré : graphique_canulars_par_annee.png")
    print(par_annee[["size", "proportion"]].round(2).to_string())

    _, rappel_p8, precision_p8, _ = entrainer_et_evaluer(
        df, COLONNES_CAT_HONNETES, ["duration_seconds", "latitude", "longitude"], None, decoupe=(idx_train, idx_test)
    )
    y_test = df.loc[idx_test, "canular"]
    y_pred = pipeline.predict(df.loc[idx_test, COLONNES_MODELE])
    rappel_p12 = recall_score(y_test, y_pred, zero_division=0)
    precision_p12 = precision_score(y_test, y_pred, zero_division=0)

    print("\nÉpreuve (entraînement sur l'ancien, test sur le récent — même découpe que la phase 8) :")
    print(f"  Phase 8  (sans ville/heure) : rappel {100*rappel_p8:.1f} — précision {100*precision_p8:.1f}")
    print(f"  Phase 12 (modèle final)     : rappel {100*rappel_p12:.1f} — précision {100*precision_p12:.1f}")

    print(
        "\nSurveillance sans connaître la vérité :\n"
        "  1. Taux de signalement (% de relevés marqués canular par semaine).\n"
        "  2. Probabilité moyenne annoncée par semaine (après calibration, phase 14).\n"
        "  3. Taux de country/state manquants par semaine (dérive des données en entrée).\n"
        "Fréquence : hebdomadaire.\n"
        "Seuil d'alerte : écart de plus de 3 points de pourcentage vs la moyenne des 8 semaines "
        "précédentes sur l'indicateur 1 ou 3 → on rappelle les analystes."
    )


if __name__ == "__main__":
    telecharger_donnees()
    lignes_valides, lignes_ecartees = phase1_ouvrir_la_caisse()
    releves = phase2_rien_nest_du_bon_type(lignes_valides)
    releves = phase3_trier_les_canulars(releves)
    df = phase4_premier_verdict(releves)
    avant, apres = phase5_le_conseil_ne_vous_croit_pas(df)
    phase6_le_modele_le_plus_bete(df, apres)
    df, (idx_train, idx_test) = phase7_plusieurs_temoins_un_seul_evenement(df)
    df, (idx_train, idx_test) = phase8_ordre_des_choses(df)
    df = phase9_les_cases_vides(df)
    df, pipeline = phase10_chaine_de_traitement(df, idx_train, idx_test)
    df = phase11_combien_de_temps_ca_a_dure(df, idx_train, idx_test)
    df, pipeline = phase12_la_ville_et_lheure(df, idx_train, idx_test)
    seuil = phase13_la_facture_du_bureau(df, pipeline, idx_train, idx_test)
    calibreur = phase14_promesse_a_80_pourcent(df, pipeline, idx_train, idx_test)
    phase15_deux_analystes_deux_chiffres(df, pipeline, idx_test)
    phase16_trois_dossiers_sur_le_bureau(df, pipeline, idx_test, seuil)
    phase17_langle_mort_du_bureau(df, pipeline, idx_test, seuil)
    phase18_la_transmission_darchive(df, pipeline, idx_train, idx_test)
