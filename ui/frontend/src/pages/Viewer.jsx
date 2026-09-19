import { useEffect, useState } from "react";
import { api } from "../api";
import Grid from "../components/Grid";

const FEATURE_NAMES = [
  "empty_cells",
  "monotonicity",
  "smoothness",
  "max_tile_in_corner",
  "merges_available",
  "snake_weighted",
];

const DIRECTION_NAMES = ["gauche", "droite", "haut", "bas"];

export default function Viewer() {
  const [experiments, setExperiments] = useState([]);
  const [experiment, setExperiment] = useState(null);
  const [replays, setReplays] = useState([]);
  const [seed, setSeed] = useState(null);
  const [trajectory, setTrajectory] = useState(null);
  const [step, setStep] = useState(0);
  const [error, setError] = useState(null);

  useEffect(() => {
    api.experiments().then((exps) => {
      setExperiments(exps);
      if (exps.length > 0) setExperiment(exps[0].dir);
    });
  }, []);

  useEffect(() => {
    if (!experiment) return;
    setReplays([]);
    setSeed(null);
    setTrajectory(null);
    api
      .experimentReplays(experiment)
      .then((r) => {
        setReplays(r);
        if (r.length > 0) setSeed(r[0].seed);
      })
      .catch((e) => setError(String(e)));
  }, [experiment]);

  useEffect(() => {
    if (!experiment || seed === null) return;
    setTrajectory(null);
    setStep(0);
    api
      .experimentReplay(experiment, seed)
      .then(setTrajectory)
      .catch((e) => setError(String(e)));
  }, [experiment, seed]);

  const current = trajectory ? trajectory[step] : null;

  return (
    <div>
      <h2>Viewer</h2>

      <div className="panel controls">
        <label>
          Expérience :{" "}
          <select value={experiment ?? ""} onChange={(e) => setExperiment(e.target.value)}>
            {experiments.map((exp) => (
              <option key={exp.dir} value={exp.dir}>
                {exp.dir}
              </option>
            ))}
          </select>
        </label>

        <label>
          Partie (seed) :{" "}
          <select value={seed ?? ""} onChange={(e) => setSeed(Number(e.target.value))}>
            {replays.map((r) => (
              <option key={r.seed} value={r.seed}>
                seed {r.seed} ({r.num_moves} coups)
              </option>
            ))}
          </select>
        </label>
      </div>

      {error && <p className="error">{error}</p>}

      {current && (
        <div className="viewer-layout">
          <div className="panel">
            <Grid cells={current.board_before} />
            <div className="viewer-controls">
              <button disabled={step === 0} onClick={() => setStep((s) => s - 1)}>
                ← précédent
              </button>
              <span>
                coup {step + 1} / {trajectory.length}
              </span>
              <button
                disabled={step === trajectory.length - 1}
                onClick={() => setStep((s) => s + 1)}
              >
                suivant →
              </button>
            </div>
            <p>
              Coup joué : <strong>{DIRECTION_NAMES[current.chosen_direction]}</strong> (+
              {current.gained}) — score après : {current.score_after}
            </p>
          </div>

          <div className="panel">
            <h4>Alternatives (score immédiat)</h4>
            <table className="metrics">
              <tbody>
                {current.move_options.map((opt) => (
                  <tr
                    key={opt.direction}
                    className={opt.direction === current.chosen_direction ? "chosen" : ""}
                  >
                    <td>{DIRECTION_NAMES[opt.direction]}</td>
                    <td>{opt.legal ? `+${opt.gained}` : "illégal"}</td>
                  </tr>
                ))}
              </tbody>
            </table>

            <h4>Features (chapitre 5)</h4>
            <table className="metrics">
              <tbody>
                {FEATURE_NAMES.map((name, i) => (
                  <tr key={name}>
                    <td>{name}</td>
                    <td>{current.features[i].toFixed(3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
