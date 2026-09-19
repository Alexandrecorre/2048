# Chapitre 4 — Tableau de référence des agents

Produit par `python -m g2048 analyze results/<agent>_<date>/ ...` (script
[`python/g2048/analysis.py`](../python/g2048/analysis.py)) à partir des
configs [`experiments/`](../experiments) : `baseline_random.toml`,
`greedy.toml`, `corner.toml`, `monte_carlo.toml`.

Toutes les IA futures (chapitres 5+) seront comparées à ces lignes.

| expérience   | parties | score moyen | survie moyenne (coups) | ms/coup moyen | tuile max atteinte | réussite ≥128 | ≥256 | ≥512 | ≥1024 | ≥2048 | ≥4096 |
|--------------|--------:|------------:|------------------------:|---------------:|--------------------:|---------------:|------:|------:|-------:|-------:|-------:|
| random       |     200 |       1 063 |                      116 |          0.003 |                  256 |          51.5% |   7.0% |   0.0% |   0.0% |   0.0% |   0.0% |
| glouton      |     200 |       3 005 |                      255 |          0.002 |                  512 |          98.0% |  60.5% |  11.5% |   0.0% |   0.0% |   0.0% |
| coin         |     200 |       2 302 |                      207 |          0.002 |                  512 |          86.5% |  45.0% |   3.5% |   0.0% |   0.0% |   0.0% |
| monte carlo  |      50 |      28 927 |                    1 499 |          3.874 |                4 096 |         100.0% | 100.0% | 100.0% |  96.0% |  68.0% |   8.0% |

Configs : `seed = 42`, `max_moves` 3000–5000 selon l'agent. Monte Carlo :
`mc_simulations = 50`, `mc_rollout_moves = 200` (rollouts aléatoires),
50 parties (plus coûteux : ~3.9 ms/coup contre ~0.002–0.003 ms pour les
agents sans recherche).

## Lecture rapide

- **Glouton > coin > aléatoire** sur toutes les métriques : maximiser le
  score immédiat est déjà une heuristique correcte à profondeur 0.
- **Monte Carlo domine largement** (seul agent à dépasser 1024 de façon
  fiable) mais au prix d'un temps par coup ~1500x plus élevé — point de
  départ pour la question de recherche du chapitre 7 (profondeur de
  recherche vs qualité de l'évaluation).
- Aucun agent de référence n'atteint 2048 de façon fiable : c'est attendu,
  ce sont des bornes basses pour les chapitres 5-7 (pistes + apprentissage
  + recherche).

Reproduire :

```bash
python -m g2048 run experiments/baseline_random.toml
python -m g2048 run experiments/greedy.toml
python -m g2048 run experiments/corner.toml
python -m g2048 run experiments/monte_carlo.toml
python -m g2048 analyze results/baseline_random_*/ results/greedy_*/ results/corner_*/ results/monte_carlo_*/
```
