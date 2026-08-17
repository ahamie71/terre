# RAPPORT.md — Bureau d'Analyse Terrestre

## Phase 1 : ouvrir la caisse

L'URL du sujet avait une coquille (tiret manquant), j'ai téléchargé le bon fichier depuis le repo.

- Lignes dans le fichier : 88 875
- Lignes chargées (11 champs) : 88 679
- Lignes écartées (≠ 11 champs) : 196

88 679 + 196 = 88 875, rien n'est perdu.

Les 196 lignes de côté ont 12 champs au lieu de 11 : une virgule en trop quelque part avant le
commentaire, qui décale tout le reste. Ça ne tombe pas toujours au même endroit dans la ligne,
donc impossible de deviner comment recoller les champs sans risquer de mettre une ville dans la
colonne pays. Je les mets de côté pour l'instant.

## Phase 2 : rien n'est du bon type

Conversion en nombres (duration_seconds, latitude, longitude) et en dates (datetime, date_posted).
Aucune ligne supprimée, les valeurs qui résistent deviennent `None`.

- duration_seconds : 5 échecs (2 vides, 3 avec un backtick collé, ex `2\``) — bug du service qui
  convertit le texte libre du témoin en secondes.
- latitude : 1 échec, `33q.200088` — une lettre parasite qui suffit à casser toute la colonne.
  Vient du géocodage/transmission.
- longitude : 0 échec.
- datetime : 1220 échecs, tous en `... 24:00` (ex `10/10/2005 24:00`) — heure invalide, bug du
  service de standardisation, pas le témoin.
- date_posted : 0 échec.

## Phase 3 : trier les canulars

Règle : un relevé est un canular si le mot "hoax" apparaît dans comments.
802 relevés marqués (0,90 % des 88 679).

Limite : la plupart de ces mentions ("possible hoax??", 674 sur 802) sont des notes NUFORC pas
sûres d'elles, pas des canulars confirmés. Et au moins une ligne dit littéralement "No Hoax" dans
le commentaire — la règle l'attrape quand même à tort, elle ne gère pas la négation.

## Phase 4 : le premier verdict

Régression logistique (shape, state, country, duration_seconds, latitude, longitude, comments en
TF-IDF). Évalué sur 17 736 relevés (20 % du jeu, jamais vus à l'entraînement).

- Rappel : 99,4 / 100 canulars réellement présents attrapés
- Précision : 99,4 / 100 signalés qui le sont vraiment

C'est presque trop beau — logique, comments contient littéralement le mot "hoax" qui a servi à
fabriquer l'étiquette en phase 3. À creuser en phase 5.

## Phase 5 : le Conseil ne vous croit pas

| Colonne | Qui écrit | Quand | Savait déjà si canular ? |
|---|---|---|---|
| shape | témoin | le soir même | Non |
| state | témoin | le soir même | Non |
| country | témoin | le soir même | Non |
| duration_seconds | service (calculé depuis le texte du témoin) | à la transmission | Non |
| latitude | service (géocodage) | à la transmission | Non |
| longitude | service (géocodage) | à la transmission | Non |
| comments | témoin, puis employé du Bureau (notes "((NUFORC Note...))") | soir même, puis des semaines plus tard | **Oui** |

comments sort du modèle : c'est là-dedans que l'employé note ses doutes sur un canular des
semaines après coup, et c'est littéralement le champ dont on a tiré l'étiquette en phase 3.

Avant / après (même modèle, sans comments) :
- Rappel : 99,4 → 50,6 / 100
- Précision : 99,4 → 1,1 / 100

L'écart s'explique : le premier chiffre trichait, il avait accès au mot qui a servi à fabriquer
l'étiquette elle-même. Sans lui, le modèle n'a plus que des colonnes que le témoin ou le service
remplissent avant de savoir quoi que ce soit sur un canular, et ça tombe autour du hasard.

## Phase 6 : le modèle le plus bête du Bureau

- Stagiaire ("jamais canular") : 99,1 % de bonnes réponses
- Notre modèle (sans comments) : 59,5 % de bonnes réponses

Le stagiaire gagne sur l'exactitude, et pourtant son système est inutile : il rate 100 % des
canulars puisqu'il n'en signale jamais. L'exactitude est trompeuse ici parce que les canulars sont
rares (0,9 %) — prédire "non" tout le temps suffit à avoir raison presque toujours sans rien
détecter. La mesure qui compte, c'est le rappel (et la précision) : c'est elle qui dit si le
système attrape vraiment des canulars, ce que l'exactitude ne dit pas.

# Le Conseil renvoie le rapport

## Phase 7 : plusieurs témoins, un seul événement

Un événement = même jour + même ville + même state (colonnes `datetime` tronqué au jour, `city`,
`state`). Pour grouper par jour il fallait d'abord réparer les 1220 heures `24:00` laissées à
`None` en phase 2 (24:00 = minuit le lendemain) — sans ça il manquait la date de 1220 lignes.

- Événements avec plus d'un témoin : 2399
- Le plus gros : 56 témoins, le 31 octobre 2004 à Tinley Park (IL)
- Relevés à cheval sur les deux côtés dans la découpe d'hier (aléatoire) : 2051 — soit le modèle
  apprenait une partie d'un événement et se faisait noter sur le reste du même événement.
- Témoignages recopiés mot pour mot (≥40 caractères, pour ignorer les phrases courtes du genre
  "Fireball" qui se répètent par coïncidence) : 139 lignes sur 62 textes distincts. Je les fonds
  dans le même groupe que l'événement concerné, pour la même raison : qu'ils partent ensemble.

Nouvelle découpe : tous les relevés d'un même groupe (événement + doublons fondus) partent du même
côté (`GroupShuffleSplit`).

Rappel / précision du modèle honnête (sans comments), avant → après cette découpe :
- Rappel : 50,6 → 51,9 / 100
- Précision : 1,1 → 1,1 / 100

Ça bouge à peine. Logique : la vraie fuite grave (comments) est déjà partie en phase 5. Ce qui
reste (shape, state, country, duration, lat/long) ne permet pas de "reconnaître" un événement par
cœur de la même manière — un témoin de Tinley Park et un autre écrivent des durées et des formes
différentes même pour le même objet dans le ciel.

## Phase 8 : l'ordre des choses

Coupure sur `datetime` (l'instant où le témoin lève les yeux), pas `date_posted` (quand le Bureau
reçoit/traite le dossier) : 8836 relevés ont plus de 10 ans d'écart entre les deux (jusqu'à
96 ans), donc `date_posted` mélangerait des dossiers anciens traités tard avec des dossiers
récents. Seul `datetime` respecte l'ordre réel des événements.

Découpe par événement (jour+ville+state de la phase 7, pas la fusion par témoignage recopié —
ces doublons peuvent être à des décennies d'écart et casseraient l'ordre chronologique ; sans
conséquence puisque comments est déjà hors du modèle).

- Date de coupure : 11 janvier 2012 (apprentissage = avant, test = à partir de là)
- Relevés : 70 944 en apprentissage, 17 735 en test
- Proportion de canulars — apprentissage : 0,94 % | test : 0,76 %
- Rappel : 51,9 / 100 — Précision : 1,1 / 100

Les deux proportions ne sont pas égales, et ça raconte quelque chose : le taux de canulars (tel
qu'on le mesure, via les notes NUFORC) grimpe fort entre 2005 et 2008 (jusqu'à 2,5 %/an) puis
redescend vers 2012-2013 (~0,5-0,6 %). Ce n'est probablement pas que les gens mentent plus ou
moins selon les années — c'est que les éditeurs NUFORC ont annoté plus ou moins activement
"possible hoax" selon les périodes. Notre étiquette de la phase 3 porte la trace des habitudes du
Bureau, pas seulement des faits.

## Phase 9 : les cases vides

Trois colonnes les plus trouées, canular selon trou ou pas :

| Colonne | Trous | % canular si troué | % canular si rempli |
|---|---|---|---|
| country | 12 365 | 1,16 % | 0,86 % |
| state | 7 409 | 1,30 % | 0,87 % |
| duration_hours_min | 3 017 | 2,35 % | 0,85 % |

Dans les trois cas, un trou est associé à plus de canulars que la moyenne (jusqu'à x2,8 pour
duration_hours_min). Le trou porte de l'info, donc pas question de le boucher sans laisser de
trace.

Traitement retenu : les trous de country et state (déjà utilisées par le modèle) partent dans leur
propre catégorie "inconnu" plutôt que d'être fondus avec une valeur existante — corrigé au passage
un bug silencieux où mon `fillna` ne visait que les `NaN` alors que ces trous sont des chaînes
vides dans le fichier, donc il ne faisait rien. Le one-hot donne sa colonne à "inconnu", le modèle
peut donc toujours distinguer un trou d'une vraie valeur et apprendre la corrélation avec canular.
duration_hours_min n'est pas encore une feature ; même principe prévu quand elle sera traitée en
phase 11.

## Phase 10 : la chaîne de traitement du Bureau

Jusqu'ici, `entrainer_et_evaluer` faisait `df[c].fillna(df[c].median())` sur tout le tableau
*avant* de couper train/test — la médiane (et le remplissage catégoriel) voyait donc une miette du
test. Nouveau `construire_pipeline` : tout ce qui s'apprend (médiane, catégories vues) est dans un
`sklearn.Pipeline`, appris uniquement par `.fit(X_train, ...)`.

Rappel / précision, découpe de la phase 8, avant → après :
- Rappel : 51,9 → 53,3 / 100
- Précision : 1,1 → 1,1 / 100

Ça bouge à peine, et c'est cohérent : à ce stade il y a très peu de valeurs manquantes numériques
(6 au total) et peu de catégories, donc la médiane "propre" et la médiane "sale" sont presque la
même valeur. Le vrai risque de fuite grossira en phase 12 avec l'encodage des villes (une liste
apprise depuis les données) — c'est pour ça qu'on corrige la mécanique maintenant, avant d'ajouter
des features qui ont vraiment besoin d'être apprises sur l'apprentissage seul.

Deuxième point : à 0,9 % de canulars une découpe malchanceuse pourrait donner un test presque vide
en canulars. Déjà vérifié en phase 8 : apprentissage 0,94 %, test 0,76 %, aucun des deux n'est
proche de zéro.

Démonstration : un relevé inventé à la main (`shape=triangle, state=il, country=us,
duration_seconds=300, latitude=41.57, longitude=-87.78`) traverse toute la chaîne en un seul appel
`pipeline.predict(...)` et ressort avec une prédiction, sans retaper une étape à la main.


