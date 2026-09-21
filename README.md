# 2048 — recherche & apprentissage

## Questions de recherche

1. Quelles pistes (features) améliorent la performance d'un agent 2048 ?
2. Apprentissage par TD ou par évolution : lequel est le plus efficace ?
3. Combien la recherche (profondeur d'expectimax) compense-t-elle une mauvaise évaluation ?

## Métriques

- Taux de réussite par palier (atteindre 512, 1024, 2048, 4096, ...).
- Survie : nombre de coups joués avant fin de partie.
- Score final.
- Temps par coup (ms).
- Efficacité d'apprentissage : performance en fonction du nombre de parties d'entraînement.

## Périmètre

**Dans le périmètre :**
- Moteur de jeu 2048 en Rust (bitboard), exposé à Python.
- Agents de référence (aléatoire, glouton, coin, Monte Carlo).
- Bibliothèque de features/pistes activables individuellement.
- Apprentissage des poids : évolution (CMA-ES/GA), TD learning linéaire, réseau n-tuple témoin.
- Recherche par expectimax à profondeur variable.
- Dashboard + viewer de replays.

**Hors périmètre (explicite) :**
- Deep RL (chapitre 8) : optionnel, non prioritaire, objectif de comparaison seulement — pas de recherche de performance record.
- Toute UI mobile/desktop native : le viewer/dashboard reste web (React + FastAPI ou Streamlit).
- Multi-joueurs, variantes du jeu (tailles de grille autres que 4×4, autres règles de fusion).
- Déploiement en production / service hébergé.

## Structure du dépôt

```
2048/
  crates/
    core/          # moteur de jeu Rust (bitboard, règles, RNG seedable)
    py-bindings/   # extension pyo3 exposée à Python
  python/
    g2048/         # package Python (CLI, harnais d'expérimentation)
  experiments/     # configs TOML d'expériences
  results/         # sorties d'expériences (Parquet, replays JSONL, métadonnées) — gitignored
  ui/              # dashboard + viewer (React/FastAPI ou Streamlit)
```

Voir [plan.txt](plan.txt) pour le détail des chapitres.

## Résultats et documentation

- [Chapitre 4 — Tableau de référence des agents](docs/chapitre4_reference.md)
- [Chapitre 6 — Matrice pistes × méthode d'apprentissage](docs/chapitre6_matrix.md)
- [Chapitre 7 — Profondeur de recherche × qualité de l'évaluation](docs/chapitre7_search.md)
- [Chapitre 10 — Synthèse (une réponse par question de recherche)](docs/chapitre10_synthese.md)
- [Jeu de données d'expérience persistant + smoothness affinée](docs/experience.md)
- [ui/README.md](ui/README.md) — lancer le dashboard et le viewer
