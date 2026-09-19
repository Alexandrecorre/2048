# Chapitre 7 — Profondeur de recherche × qualité de l'évaluation

Produit par `python -m g2048 search-experiment results/search_experiment --seed 0`
([`python/g2048/search_experiment.py`](../python/g2048/search_experiment.py)),
qui compare deux évaluateurs avec un agent expectimax
([`crates/core/src/search.rs`](../crates/core/src/search.rs)) à profondeur
1 à 5 :

- **pauvre** : une seule piste (`empty_cells`), poids = 1.0 — quasiment la
  ligne de base "aucune piste" du chapitre 6.
- **riche** : poids appris par TD(0) sur les 6 pistes du chapitre 5
  (2000 parties d'entraînement).

Échelle réduite pour rester dans un temps raisonnable (le facteur de
branchement de l'expectimax croît vite avec la profondeur) : 20 parties
aux profondeurs 1-2, 15 à la profondeur 3, 6 à la profondeur 4, 3 à la
profondeur 5, plafond de 800 coups par partie.

## Résultats (score moyen)

| profondeur | pauvre (empty_cells) | riche (TD, 6 pistes) | écart pauvre/riche | ms/coup (pauvre) | ms/coup (riche) |
|-----------:|----------------------:|-----------------------:|--------------------:|------------------:|------------------:|
| 1          |                 3 931  |                 13 828  |               -72 % |             0.015 |             0.046 |
| 2          |                 7 898  |                 13 638  |               -42 % |             0.269 |             0.515 |
| 3          |                13 086  |                 14 016  |                -7 % |             1.652 |             2.919 |
| 4          |                12 983  |                 14 110  |                -8 % |             6.266 |            12.309 |
| 5          |                14 153  |                 14 127  |                +0 % |            36.045 |           329.549 |

## Réponse à la question de recherche du chapitre 0

**« Combien la recherche compense-t-elle une mauvaise évaluation ? »**
Beaucoup, et vite. À profondeur 1 (pas de recherche, chapitre 6), l'écart
est énorme (riche 3,5x meilleur que pauvre). Il se réduit drastiquement
dès la profondeur 3 (écart de 7%), et **s'efface complètement à
profondeur 5** : l'évaluateur pauvre avec 5 coups de recherche
(14 153) égale, voire dépasse légèrement, l'évaluateur riche
(14 127) — pour un coût cependant très différent : 36 ms/coup pour le
pauvre contre 330 ms/coup pour le riche à cette même profondeur (le
riche a une fonction d'évaluation plus coûteuse à calculer par nœud,
et son avantage informationnel diminue mais son coût de calcul reste
comparable par nœud, réévalué beaucoup plus souvent qu'à faible
profondeur).

**Lecture pratique** : à budget de calcul égal, mieux vaut investir dans
plus de profondeur de recherche que dans une évaluation plus riche, une
fois passé un seuil de profondeur (~3 ici). L'évaluation riche garde
l'avantage seulement quand le temps de calcul par coup est contraint
(profondeur 1-2, comme les agents à profondeur fixe du chapitre 6).

*Note méthodologique* : peu de parties aux profondeurs 4-5 (3-6 parties)
pour tenir dans un temps raisonnable — la tendance est nette mais les
valeurs à ces profondeurs ont une variance élevée ; un run à plus grande
échelle confirmerait la position exacte du point de croisement.

## Implémentation (élagage et table de transposition)

- **Élagage des nœuds de hasard** : au-delà de la dernière demi-couche,
  seule la tuile "2" (90% des cas) est explorée ; la tuile "4" n'est
  modélisée qu'à la demi-couche la plus proche de la feuille. Au plus
  4 cases vides sont développées par nœud de hasard (sous-ensemble
  représentatif plutôt que l'espérance exacte sur toutes les cases
  vides, qui ferait exploser le facteur de branchement).
- **Table de transposition** : cache (plateau, profondeur restante) ->
  valeur, réutilisé sur toute la recherche d'un coup (vidé à chaque
  nouveau coup pour borner la mémoire).

## Reproduire

```bash
python -m g2048 search-experiment results/search_experiment --seed 0
```
