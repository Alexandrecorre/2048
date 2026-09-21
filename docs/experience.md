# Jeu de données d'expérience persistant

Suite à une discussion sur comment aider les modèles à apprendre de leurs
propres parties plutôt que de tout repartir à zéro à chaque run (et sur
une piste de feature affinée pour le chapitre 5) : deux changements.

## 1. Feature `smoothness` affinée (chapitre 5)

Remplace la pénalité linéaire par écart de valeur (qui traitait une case
vide voisine comme un "écart" à pénaliser) par une récompense qui :
- **ignore les paires impliquant une case vide** (avoir de l'espace libre
  est déjà récompensé par `empty_cells`, ça ne doit pas être compté
  comme une incohérence ici) ;
- **récompense la proximité de valeurs entre cases occupées voisines**
  (8 à côté de 16, lui-même à côté de 32), sans imposer d'ordre global
  figé comme un serpent — décroissance exponentielle avec l'écart
  d'exposant (1.0 si identique, 0.5 si écart de 1, 0.25 si écart de 2...).

**Attention à l'échelle d'entraînement** : un premier run d'ablation à
petite échelle (population 24 × 15 générations, 2000 parties TD) a
d'abord semblé montrer que cette nouvelle feature nuisait au jeu complet
de pistes. Un second run ~7-10x plus grand a confirmé que c'était un
artefact d'échelle : `all` redevient le meilleur ou quasi-meilleur
config, et un fait nouveau émerge — **retirer `monotonicity` devient
beaucoup plus coûteux avec la nouvelle smoothness** (score divisé par 3)
qu'avec l'ancienne. Interprétation : la cohérence locale (nouvelle
smoothness) et l'ordre global (monotonicity) se complètent — l'une sans
l'autre est nettement moins efficace que les deux ensemble.

`snake_weighted` reste exclue du jeu de pistes par défaut
(`python/g2048/features.py:DEFAULT_FEATURES`) : la retirer reste neutre
à légèrement positif, sans le coût observé pour `monotonicity`. Elle
reste testable individuellement (`ALL_FEATURES`) dans le protocole
d'ablation du chapitre 6.

## 2. Jeu de données d'expérience persistant

Objectif : permettre à un modèle (le DQN du chapitre 8 pour commencer)
de réutiliser l'historique de jeu déjà produit par **tous les agents**
plutôt que de repartir d'un buffer vide à chaque run — même les parties
d'un agent faible contiennent de l'information sur ce qui fonctionne ou
non.

Ne duplique pas le stockage existant : les replays (`seed` + coups,
chapitre 3) et `replay_trajectory` (chapitre 9, viewer) suffisent déjà à
reconstruire l'état complet à chaque pas. Le nouveau module
[`python/g2048/experience.py`](../python/g2048/experience.py) se
contente de sélectionner et matérialiser les transitions utiles.

- **Sélection** : les 10 meilleures parties (par score final) de chaque
  expérience, taguées par l'agent qui les a produites (`source_agent`).
- **Format** : un unique Parquet, `results/experience/dataset.parquet` —
  colonnes `source_agent`, `source_experiment`, `seed`, `step_index`,
  `board_before` (16 exposants), `chosen_direction`, `gained`,
  `score_after`, `legal` (4 booléens), `terminal`.
- **Mise à jour** : automatique à la fin de chaque `g2048 run` (append
  idempotent — relancer la même expérience remplace ses lignes plutôt
  que de les dupliquer). `g2048 build-experience [results_root]`
  reconstruit le dataset à partir des expériences déjà existantes.
- **Utilisation (DQN, chapitre 8)** : `DqnConfig(warm_start_from_experience=True)`
  (ou `g2048 train-dqn ... --warm-start`) précharge le buffer de replay
  avec ces transitions, encodées selon la représentation choisie
  (`onehot` ou `features`) via `deep_rl.load_experience_into_buffer`.
- **Généralisable** : le format (état/action/récompense taggé par
  source) ne dépend pas du DQN — il peut nourrir d'autres méthodes
  (pré-entraînement supervisé, analyse de ce qui distingue une bonne
  d'une mauvaise décision) sans rien construire de nouveau.

### Reproduire

```bash
g2048 build-experience                      # backfill depuis results/ existant
g2048 train-dqn results/dqn_warm --representation features --warm-start
```
