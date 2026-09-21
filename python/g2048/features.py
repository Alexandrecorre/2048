"""Pistes (chapitre 5) partagées entre les scripts d'entraînement et
d'expérimentation, pour que ces listes ne divergent pas d'un fichier à
l'autre."""

# Toutes les pistes implémentées. Utilisé par le protocole d'ablation du
# chapitre 6, qui doit justement pouvoir tester chacune (y compris
# snake_weighted, exclue par défaut ailleurs — voir DEFAULT_FEATURES).
ALL_FEATURES = [
    "empty_cells",
    "monotonicity",
    "smoothness",
    "max_tile_in_corner",
    "merges_available",
    "snake_weighted",
]

# Jeu de pistes par défaut utilisé en dehors de l'ablation (agents,
# entraînement TD/évolution "riche", DQN). Exclut snake_weighted : le
# protocole d'ablation du chapitre 6 a montré que la contrainte de
# serpent nuit souvent une fois combinée aux autres pistes
# (all_but_snake_weighted battait le jeu complet en TD) — voir
# docs/chapitre6_matrix.md. Cette liste peut évoluer si une future
# ablation change la conclusion.
DEFAULT_FEATURES = [f for f in ALL_FEATURES if f != "snake_weighted"]
