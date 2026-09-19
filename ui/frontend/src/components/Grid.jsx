const TILE_COLORS = {
  0: "#cdc1b4",
  1: "#eee4da",
  2: "#ede0c8",
  3: "#f2b179",
  4: "#f59563",
  5: "#f67c5f",
  6: "#f65e3b",
  7: "#edcf72",
  8: "#edcc61",
  9: "#edc850",
  10: "#edc53f",
  11: "#edc22e",
  12: "#3c3a32",
  13: "#3c3a32",
  14: "#3c3a32",
  15: "#3c3a32",
};

function tileValue(exponent) {
  return exponent === 0 ? "" : String(2 ** exponent);
}

export default function Grid({ cells }) {
  return (
    <div className="grid">
      {cells.map((exponent, i) => (
        <div
          key={i}
          className="cell"
          style={{
            background: TILE_COLORS[exponent] ?? "#3c3a32",
            color: exponent <= 2 ? "#776e65" : "#f9f6f2",
          }}
        >
          {tileValue(exponent)}
        </div>
      ))}
    </div>
  );
}
