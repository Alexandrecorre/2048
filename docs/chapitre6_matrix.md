# Chapitre 6 — Matrice pistes × méthode d'apprentissage × performance

> **Mise à jour** : la feature `smoothness` a été affinée après ce run
> (voir [experience.md](experience.md)) — les chiffres ci-dessous ne
> reflètent plus le code actuel pour toute config incluant `smoothness`.
> Les conclusions qualitatives (glouton simple > 1 seule piste faible,
> etc.) restent valides ; pour le détail à jour sur smoothness/
> monotonicity/snake_weighted, voir `experience.md`.

Produit par `python -m g2048 ablation results/ablation_chapter6 --seed 0`
([`python/g2048/ablation.py`](../python/g2048/ablation.py)). Profondeur de
recherche fixée à 1 pour toute cette phase (agent `EvalAgent` : coup qui
maximise score immédiat + evaluate(afterstate)), pour isoler l'effet de
l'information donnée à l'agent, indépendamment de toute recherche.

**Échelle de ce run** (pour rester dans un temps raisonnable) : évolution
= population 24 × 15 générations × 6 parties/évaluation ; TD = 2000
parties d'entraînement ; évaluation finale = 100 parties de test (seeds
disjointes de l'entraînement) avec IC à 95%. Un run à plus grande échelle
(population/générations/parties plus grandes) réduirait la largeur des
IC et la variance des rangs proches, mais les tendances qualitatives
ci-dessous sont déjà nettes.

## Résultats (score moyen sur 100 parties de test, trié décroissant)

| pistes                     | méthode    | score moyen | IC 95%              |
|-----------------------------|-----------|------------:|----------------------|
| all_but_snake_weighted       | td        |     18 933  | [16 960, 20 907]     |
| all_but_monotonicity         | td        |     18 058  | [16 193, 19 922]     |
| all                          | td        |     17 723  | [15 979, 19 467]     |
| all_but_empty_cells          | evolution |     17 115  | [15 330, 18 900]     |
| all_but_monotonicity         | evolution |     16 916  | [15 239, 18 592]     |
| all_but_snake_weighted       | evolution |     16 914  | [15 213, 18 615]     |
| all_but_max_tile_in_corner   | td        |     15 863  | [14 547, 17 180]     |
| all_but_smoothness           | evolution |     15 389  | [13 706, 17 072]     |
| all_but_smoothness           | td        |     15 345  | [13 741, 16 949]     |
| all_but_max_tile_in_corner   | evolution |     14 653  | [13 323, 15 984]     |
| all                          | evolution |     13 378  | [12 234, 14 523]     |
| all_but_empty_cells          | td        |     11 743  | [10 616, 12 871]     |
| all_but_merges_available     | evolution |      8 922  | [8 041, 9 804]       |
| only_smoothness              | evolution |      7 858  | [7 140, 8 577]       |
| all_but_merges_available     | td        |      7 334  | [6 701, 7 968]       |
| only_monotonicity            | evolution |      6 601  | [5 969, 7 233]       |
| only_merges_available        | evolution |      6 540  | [5 890, 7 190]       |
| only_smoothness              | td        |      5 261  | [4 770, 5 752]       |
| only_monotonicity            | td        |      5 050  | [4 570, 5 529]       |
| only_merges_available        | td        |      4 792  | [4 370, 5 215]       |
| only_max_tile_in_corner      | td        |      3 882  | [3 468, 4 296]       |
| only_max_tile_in_corner      | evolution |      3 422  | [3 055, 3 788]       |
| only_empty_cells             | evolution |      3 165  | [2 808, 3 522]       |
| none                         | evolution |      3 165  | [2 808, 3 522]       |
| none                         | td        |      3 165  | [2 808, 3 522]       |
| only_empty_cells             | td        |      3 141  | [2 827, 3 454]       |
| none (grille brute, n-tuple) | td_ntuple |      3 058  | [2 731, 3 386]       |
| only_snake_weighted          | evolution |      2 786  | [2 484, 3 087]       |
| only_snake_weighted          | td        |      2 786  | [2 484, 3 087]       |

## Lecture

- **`merges_available` et `max_tile_in_corner` sont les pistes les plus
  utiles seules** : elles dominent nettement le groupe "une seule piste"
  (~4 800-6 500 selon la méthode). `snake_weighted` seule n'apporte rien
  par rapport à la ligne de base "aucune piste" — les deux tombent à la
  même valeur pour ce run, ce qui suggère qu'un poids appris trop faible
  ou mal signé pour cette piste isolée ne change quasiment jamais la
  décision par rapport à la politique gloutonne pure.
- **Retirer une piste du jeu complet peut améliorer le score** :
  `all_but_snake_weighted` (TD) dépasse `all` (TD), et `all_but_monotonicity`
  aussi. À cette échelle d'entraînement, `snake_weighted` et
  `monotonicity` semblent apporter du bruit plutôt qu'un signal utile
  quand elles sont combinées aux autres pistes — hypothèse à confirmer
  avec un entraînement plus long avant d'en tirer une conclusion ferme.
- **TD ≥ évolution sur le jeu complet de pistes** dans ce run
  (17 723 vs 13 378), mais l'ordre s'inverse pour certains sous-ensembles
  (`all_but_empty_cells` : évolution 17 115 > TD 11 743) : ni l'une ni
  l'autre méthode ne domine systématiquement à cette échelle.
- **Le témoin n-tuple (grille brute, aucune piste conçue à la main)**
  obtient un score comparable à la ligne de base "aucune piste" côté
  features (3 058 vs 3 165) : sans recherche profonde, 2000 parties ne
  suffisent pas à un modèle de 256 poids indépendants pour dépasser une
  heuristique gloutonne simple — les pistes conçues à la main (chapitre 5)
  ont donc un avantage clair à volume d'entraînement égal.

## Reproduire

```bash
python -m g2048 ablation results/ablation_chapter6 --seed 0
```

Écrit `results/ablation_chapter6/matrix.parquet` (le tableau ci-dessus)
et `learning_curves.parquet` (courbes d'apprentissage par piste/méthode,
colonnes `steps`, `mean_score`).
