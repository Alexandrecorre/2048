# Chapitre 10 — Synthèse

Une page de conclusions par question de recherche du [cadrage](../README.md)
(chapitre 0), à partir des résultats produits aux chapitres 4, 6 et 7.

## 1. Quelles pistes améliorent la performance ?

Voir [chapitre6_matrix.md](chapitre6_matrix.md) pour le détail (protocole
d'ablation complet, IC 95%).

- **`merges_available` et `max_tile_in_corner` sont les pistes les plus
  utiles isolément** : ~4 800-6 500 de score moyen selon la méthode, loin
  devant `smoothness`/`monotonicity` seules (~5 000-7 900) et
  `snake_weighted` seule, qui n'apporte rien par rapport à la ligne de
  base "aucune piste" (2 786 vs 3 165-3 165).
- **Retirer une piste du jeu complet peut améliorer le score** :
  `all_but_snake_weighted` et `all_but_monotonicity` dépassent `all` en
  TD dans notre run. À ce volume d'entraînement, ces deux pistes semblent
  apporter plus de bruit que de signal une fois combinées aux autres —
  hypothèse à confirmer avec un entraînement plus long.
- **Conclusion pratique** : si on ne devait garder que 2 pistes,
  `merges_available` et `max_tile_in_corner` sont le meilleur choix
  coût/bénéfice observé ici.

## 2. Apprentissage par TD ou par évolution : lequel est le plus efficace ?

- **Aucune méthode ne domine systématiquement** à l'échelle testée
  (population 24 × 15 générations pour l'évolution, 2000 parties pour
  TD). Sur le jeu complet de pistes, TD gagne nettement (17 723 vs
  13 378) ; mais sur `all_but_empty_cells`, c'est l'évolution qui gagne
  largement (17 115 vs 11 743).
- **Le témoin n-tuple (grille brute, chapitre 6c)** obtient un score
  (3 058) comparable à la ligne de base "aucune piste" (3 165) : à ce
  volume d'entraînement (2000 parties, 256 poids indépendants), il n'a
  pas eu le temps d'extraire un signal utile de la grille brute — les
  pistes conçues à la main (chapitre 5) restent nettement plus
  efficaces à budget d'entraînement égal.
- **Conclusion pratique** : le choix de la méthode d'apprentissage
  compte moins, à ce stade, que le choix des pistes elles-mêmes et le
  volume d'entraînement. Un budget de calcul plus long profiterait
  probablement plus à TD (apprentissage incrémental, converge
  généralement plus lentement mais plus finement) qu'à l'évolution
  (déjà proche de son optimum avec une population de cette taille).

## 3. Combien la recherche compense-t-elle une mauvaise évaluation ?

Voir [chapitre7_search.md](chapitre7_search.md) pour le détail.

- **Beaucoup, et à partir d'une profondeur modeste.** À profondeur 1
  (pas de recherche), l'écart entre une évaluation pauvre et une
  évaluation riche est énorme (-72%). Il se réduit à -7% dès la
  profondeur 3, et **s'efface complètement à profondeur 5** (14 153 vs
  14 127 — l'évaluation pauvre égale, voire dépasse légèrement, la
  riche).
- **Le coût de calcul, lui, ne s'efface pas** : à profondeur 5,
  l'agent "pauvre" reste ~9x plus rapide par coup (36 ms) que le
  "riche" (330 ms), car l'évaluation riche est plus coûteuse à calculer
  et se réévalue à chaque nœud de la recherche.
- **Conclusion pratique** : à budget de calcul égal, investir dans la
  profondeur de recherche rapporte plus qu'investir dans une évaluation
  plus riche, une fois un seuil de profondeur dépassé (~3 ici). Une
  évaluation riche ne garde l'avantage que quand le temps par coup est
  contraint (profondeur 1-2, le régime des agents du chapitre 6 sans
  recherche).

## Vue d'ensemble

| Question | Réponse courte |
|---|---|
| Quelles pistes ? | `merges_available` et `max_tile_in_corner` dominent seules ; le jeu complet n'est pas toujours optimal (bruit de `snake_weighted`/`monotonicity`). |
| TD ou évolution ? | Aucun gagnant net à ce volume d'entraînement — dépend du sous-ensemble de pistes. |
| Recherche vs évaluation ? | La recherche compense presque totalement une évaluation pauvre dès la profondeur ~5, au prix d'un temps de calcul très supérieur. |

## Limites et suite possible

- Les runs des chapitres 6 et 7 sont volontairement réduits en échelle
  (population/générations/parties limitées) pour tenir dans une session
  de travail interactive — les tendances sont nettes mais les valeurs
  exactes (en particulier aux profondeurs 4-5, sur 3-6 parties) ont une
  variance élevée qu'un run plus long réduirait.
- Le chapitre 8 (Deep RL, optionnel) confirme l'attente du plan :
  apprendre et comparer, pas battre le chapitre 7 — un DQN simple reste
  loin derrière expectimax même à faible profondeur.
- Le chapitre 9 (dashboard + viewer) permet d'explorer visuellement
  chacun des résultats ci-dessus ; voir [ui/README.md](../ui/README.md).
