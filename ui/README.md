# IHM (chapitre 9)

Dashboard + viewer de replays. Stack : API FastAPI (`backend/`, lit les
dossiers `results/` via DuckDB) + frontend React/Vite (`frontend/`).

## Lancer le backend

**Important** : utiliser l'environnement virtuel du projet (`.venv` à la
racine du dépôt), pas le Python global — installer dans le Python
global peut échouer (fichiers verrouillés dans `Scripts/`) ou mélanger
les dépendances avec d'autres projets.

Depuis la racine du dépôt (PowerShell) :

```powershell
.venv\Scripts\Activate.ps1   # crée le venv d'abord si besoin : python -m venv .venv
pip install -e ".[ui]"       # fastapi, uvicorn, duckdb
python -m uvicorn ui.backend.main:app --port 8000 --reload
```

Le prompt doit afficher `(.venv)` avant de lancer `pip install` — sinon
l'activation n'a pas fonctionné (vérifier qu'on est bien à la racine du
dépôt, là où se trouve le dossier `.venv`).

## Lancer le frontend

```bash
cd ui/frontend
npm install   # première fois seulement
npm run dev
```

Ouvrir http://localhost:5173. Le frontend interroge l'API sur
`http://127.0.0.1:8000` (CORS déjà configuré côté backend pour le port
Vite par défaut, 5173).

## Pages

- **Dashboard** (`/`) : sélection d'une expérience (chapitres 3-4),
  métriques et distribution des scores ; matrice et courbes
  d'apprentissage de l'ablation (chapitre 6) ; courbes profondeur ×
  qualité (chapitre 7) ; historiques DQN (chapitre 8), si ces données ont
  été générées au préalable (`g2048 run`, `g2048 ablation`,
  `g2048 search-experiment`, `g2048 train-dqn`).
- **Viewer** (`/viewer`) : rejoue un replay coup par coup — grille,
  coup choisi et score des 4 alternatives, valeurs des features
  (chapitre 5) à chaque étape.

Si une section du dashboard affiche « pas encore de résultats », c'est
que la commande CLI correspondante n'a pas encore été lancée — l'IHM lit
uniquement ce qui existe dans `results/`, elle ne lance rien elle-même.
