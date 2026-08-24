import { useEffect, useState } from "react";

function Matrix({ matrix, labels }) {
  if (!matrix?.length || !labels?.length) return null;
  
  // Ensure matrix is valid and has the right dimensions
  if (!Array.isArray(matrix) || !Array.isArray(matrix[0])) {
    return <p className="error">Invalid confusion matrix format</p>;
  }
  
  // Ensure matrix dimensions match labels
  if (matrix.length !== labels.length) {
    return <p className="error">Matrix dimensions ({matrix.length}x{matrix[0]?.length || 0}) don't match labels ({labels.length})</p>;
  }
  
  try {
    const max = Math.max(...matrix.flat(), 1);
    return (
      <div className="matrix-wrap">
        <table className="matrix">
          <thead>
            <tr>
              <th />
              {labels.map((label, idx) => (
                <th key={idx} title={label}>
                  {label.split("___")[1]?.replaceAll("_", " ") || label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {matrix.map((row, i) => {
              const label = labels[i] || `Class ${i}`;
              return (
                <tr key={`row-${i}`}>
                  <th>{label.split("___")[1]?.replaceAll("_", " ") || label}</th>
                  {row.map((cell, j) => (
                    <td
                      key={`cell-${i}-${j}`}
                      style={{ background: `rgba(184, 92, 32, ${0.08 + (cell / max) * 0.7})` }}
                    >
                      {cell}
                    </td>
                  ))}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    );
  } catch (error) {
    console.error("Matrix rendering error:", error);
    return <p className="error">Error rendering confusion matrix: {error.message}</p>;
  }
}

export default function Research() {
  const [data, setData] = useState(null);

  useEffect(() => {
    fetch("/api/research")
      .then((r) => r.json())
      .then(setData)
      .catch(() => setData({ error: true }));
  }, []);

  if (!data) return <main>Loading research artifacts…</main>;
  if (data.error) {
    return (
      <main>
        <h1>Research</h1>
        <p>Start the API on port 8000 to load experiment files from the backend.</p>
      </main>
    );
  }

  return (
    <main>
      <p className="kicker">Methods</p>
      <h1>Research design</h1>
      <blockquote>{data.research_question}</blockquote>
      <ol className="plan">
        {data.experiment_plan.map((step) => (
          <li key={step}>{step}</li>
        ))}
      </ol>
      {data.evaluations.length === 0 && (
        <p className="lede">
          No evaluation JSON yet. After training, `python -m ml.evaluate` writes metrics
          into <code>ml/results</code> and they appear here.
        </p>
      )}
      {data.evaluations.map((ev) => {
        const block = ev.held_out_test || ev;
        const labels = ev.class_ids || [];
        return (
          <section key={ev.checkpoint || ev.model_name} className="eval-card">
            <h2>{ev.model_name || "Model"}</h2>
            <p className="mono">
              acc {((block.accuracy || 0) * 100).toFixed(1)}% · macro-F1{" "}
              {((block.macro_f1 || 0) * 100).toFixed(1)}%
            </p>
            {ev.ood_phone && (
              <p>
                Phone-photo OOD accuracy: {((ev.ood_phone.accuracy || 0) * 100).toFixed(1)}%
                on {ev.ood_phone.n_images} images.
              </p>
            )}
            <Matrix matrix={block.confusion_matrix} labels={labels} />
          </section>
        );
      })}
    </main>
  );
}
