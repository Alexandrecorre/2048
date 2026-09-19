import { useEffect, useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api } from "../api";

const MILESTONES = [128, 256, 512, 1024, 2048, 4096];

function useAsync(fn, deps) {
  const [state, setState] = useState({ loading: true, data: null, error: null });
  useEffect(() => {
    let cancelled = false;
    setState({ loading: true, data: null, error: null });
    fn()
      .then((data) => !cancelled && setState({ loading: false, data, error: null }))
      .catch((error) => !cancelled && setState({ loading: false, data: null, error }));
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
  return state;
}

function histogram(scores, bins = 12) {
  if (scores.length === 0) return [];
  const max = Math.max(...scores);
  const width = Math.max(max / bins, 1);
  const counts = new Array(bins).fill(0);
  for (const s of scores) {
    const idx = Math.min(Math.floor(s / width), bins - 1);
    counts[idx] += 1;
  }
  return counts.map((count, i) => ({
    range: `${Math.round(i * width)}`,
    count,
  }));
}

function ExperimentSection({ name }) {
  const summary = useAsync(() => api.experimentSummary(name), [name]);
  const scores = useAsync(() => api.experimentScores(name), [name]);

  const scoreHistogram = useMemo(() => {
    if (!scores.data) return [];
    return histogram(scores.data.map((r) => r.score));
  }, [scores.data]);

  if (summary.loading || scores.loading) return <p>Chargement…</p>;
  if (summary.error) return <p className="error">{String(summary.error)}</p>;

  const s = summary.data;
  return (
    <div className="panel">
      <h3>Résumé — {name}</h3>
      <table className="metrics">
        <tbody>
          <tr>
            <td>Parties</td>
            <td>{s.num_games}</td>
          </tr>
          <tr>
            <td>Score moyen</td>
            <td>{s.mean_score.toFixed(0)}</td>
          </tr>
          <tr>
            <td>Survie moyenne (coups)</td>
            <td>{s.mean_survival_moves.toFixed(0)}</td>
          </tr>
          <tr>
            <td>ms / coup</td>
            <td>{s.mean_ms_per_move.toFixed(4)}</td>
          </tr>
          <tr>
            <td>Tuile max atteinte</td>
            <td>{s.max_tile_reached}</td>
          </tr>
          {MILESTONES.map((m) => (
            <tr key={m}>
              <td>Réussite ≥ {m}</td>
              <td>{(s[`success_rate_${m}`] * 100).toFixed(1)}%</td>
            </tr>
          ))}
        </tbody>
      </table>

      <h4>Distribution des scores</h4>
      <ResponsiveContainer width="100%" height={200}>
        <BarChart data={scoreHistogram}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="range" tick={{ fontSize: 11 }} />
          <YAxis allowDecimals={false} />
          <Tooltip />
          <Bar dataKey="count" fill="#f2b179" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function AblationSection() {
  const matrix = useAsync(() => api.ablationMatrix(), []);
  const curves = useAsync(() => api.ablationLearningCurves(), []);
  const [filterMethod, setFilterMethod] = useState("all");

  if (matrix.loading) return <p>Chargement…</p>;
  if (matrix.error) return <p className="muted">Chapitre 6 : pas encore de résultats d'ablation.</p>;

  const rows = [...matrix.data].sort((a, b) => b.mean_score - a.mean_score);
  const methods = [...new Set(rows.map((r) => r.method))];

  const filteredCurves =
    !curves.data
      ? []
      : curves.data.filter((r) => filterMethod === "all" || r.method === filterMethod);

  const bySeries = {};
  for (const point of filteredCurves) {
    const key = `${point.feature_set} (${point.method})`;
    bySeries[key] = bySeries[key] || [];
    bySeries[key].push(point);
  }

  return (
    <div className="panel">
      <h3>Chapitre 6 — Pistes × méthode d'apprentissage</h3>
      <div className="table-scroll">
        <table className="metrics">
          <thead>
            <tr>
              <th>Pistes</th>
              <th>Méthode</th>
              <th>Score moyen</th>
              <th>IC 95%</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={`${r.feature_set}-${r.method}`}>
                <td>{r.feature_set}</td>
                <td>{r.method}</td>
                <td>{r.mean_score.toFixed(0)}</td>
                <td>
                  [{r.ci_low_95.toFixed(0)}, {r.ci_high_95.toFixed(0)}]
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {curves.data && (
        <>
          <h4>Courbes d'apprentissage</h4>
          <label>
            Méthode :{" "}
            <select value={filterMethod} onChange={(e) => setFilterMethod(e.target.value)}>
              <option value="all">toutes</option>
              {methods.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          </label>
          <ResponsiveContainer width="100%" height={260}>
            <LineChart>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="steps" type="number" allowDuplicatedCategory={false} />
              <YAxis />
              <Tooltip />
              {Object.entries(bySeries).map(([key, points], i) => (
                <Line
                  key={key}
                  data={points}
                  dataKey="mean_score"
                  name={key}
                  dot={false}
                  stroke={`hsl(${(i * 47) % 360}, 65%, 45%)`}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </>
      )}
    </div>
  );
}

function SearchExperimentSection() {
  const result = useAsync(() => api.searchExperiment(), []);
  if (result.loading) return <p>Chargement…</p>;
  if (result.error) return <p className="muted">Chapitre 7 : pas encore de résultats.</p>;

  const byEval = {};
  for (const row of result.data) {
    byEval[row.evaluation] = byEval[row.evaluation] || [];
    byEval[row.evaluation].push(row);
  }

  return (
    <div className="panel">
      <h3>Chapitre 7 — Profondeur × qualité de l'évaluation</h3>
      <h4>Score moyen</h4>
      <ResponsiveContainer width="100%" height={240}>
        <LineChart>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="depth" type="number" allowDuplicatedCategory={false} />
          <YAxis />
          <Tooltip />
          <Legend />
          {Object.entries(byEval).map(([name, points], i) => (
            <Line
              key={name}
              data={points}
              dataKey="mean_score"
              name={name}
              stroke={i === 0 ? "#f67c5f" : "#3c3a32"}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
      <h4>Temps par coup (ms, échelle log)</h4>
      <ResponsiveContainer width="100%" height={240}>
        <LineChart>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="depth" type="number" allowDuplicatedCategory={false} />
          <YAxis scale="log" domain={["auto", "auto"]} />
          <Tooltip />
          <Legend />
          {Object.entries(byEval).map(([name, points], i) => (
            <Line
              key={name}
              data={points}
              dataKey="mean_ms_per_move"
              name={name}
              stroke={i === 0 ? "#f67c5f" : "#3c3a32"}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

function DqnSection() {
  const onehot = useAsync(() => api.dqnHistory("onehot"), []);
  const features = useAsync(() => api.dqnHistory("features"), []);

  const hasAny = onehot.data || features.data;
  if (onehot.loading || features.loading) return <p>Chargement…</p>;
  if (!hasAny) return <p className="muted">Chapitre 8 : pas encore d'entraînement DQN.</p>;

  return (
    <div className="panel">
      <h3>Chapitre 8 — DQN (optionnel)</h3>
      <ResponsiveContainer width="100%" height={240}>
        <LineChart>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="episode" type="number" allowDuplicatedCategory={false} />
          <YAxis />
          <Tooltip />
          <Legend />
          {onehot.data && (
            <Line data={onehot.data} dataKey="mean_score" name="grille brute (one-hot)" stroke="#f67c5f" />
          )}
          {features.data && (
            <Line data={features.data} dataKey="mean_score" name="features (chapitre 5)" stroke="#3c3a32" />
          )}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export default function Dashboard() {
  const experiments = useAsync(() => api.experiments(), []);
  const [selected, setSelected] = useState(null);

  useEffect(() => {
    if (experiments.data && experiments.data.length > 0 && !selected) {
      setSelected(experiments.data[0].dir);
    }
  }, [experiments.data, selected]);

  return (
    <div>
      <h2>Dashboard</h2>

      <div className="panel">
        <h3>Expériences (chapitres 3-4)</h3>
        {experiments.loading && <p>Chargement…</p>}
        {experiments.error && <p className="error">{String(experiments.error)}</p>}
        {experiments.data && (
          <select value={selected ?? ""} onChange={(e) => setSelected(e.target.value)}>
            {experiments.data.map((exp) => (
              <option key={exp.dir} value={exp.dir}>
                {exp.dir} ({exp.agent}, {exp.num_games} parties)
              </option>
            ))}
          </select>
        )}
      </div>

      {selected && <ExperimentSection name={selected} />}
      <AblationSection />
      <SearchExperimentSection />
      <DqnSection />
    </div>
  );
}
