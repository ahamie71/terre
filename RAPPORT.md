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
- Précision : 98,8 / 100 signalés qui le sont vraiment

C'est presque trop beau — logique, comments contient littéralement le mot "hoax" qui a servi à
fabriquer l'étiquette en phase 3. À creuser en phase 5.


