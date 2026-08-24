import { useEffect, useState } from "react";

export default function Identify() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const [health, setHealth] = useState(null);

  useEffect(() => {
    fetch("/api/health")
      .then((r) => r.json())
      .then(setHealth)
      .catch(() => setHealth({ ok: false }));
  }, []);

  function onFile(next) {
    setResult(null);
    setError("");
    setFile(next);
    setPreview(next ? URL.createObjectURL(next) : "");
  }

  async function onSubmit(event) {
    event.preventDefault();
    if (!file) return;
    setBusy(true);
    setError("");
    const body = new FormData();
    body.append("file", file);
    try {
      const response = await fetch("/api/predict", { method: "POST", body });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || "Prediction failed.");
      setResult(payload);
    } catch (err) {
      setError(err.message || "Could not reach the FruitGuard API.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="identify">
      <section>
        <p className="kicker">Field upload</p>
        <h1>Identify from a photo</h1>
        <p className="lede">
          The model returns ranked class hypotheses and a confidence score. It does not
          recommend sprays or treatments.
        </p>
        {health && (
          <p className={`pill ${health.model_available ? "ok" : "warn"}`}>
            {health.model_available
              ? `Loaded checkpoint: ${health.model_name}`
              : "No checkpoint yet — quality analysis still runs. Train a model to enable class predictions."}
          </p>
        )}
        <form onSubmit={onSubmit} className="uploader">
          <label className="drop">
            <input
              type="file"
              accept="image/*"
              onChange={(e) => onFile(e.target.files?.[0] || null)}
            />
            {preview ? (
              <img src={preview} alt="Selected upload" />
            ) : (
              <span>Drop a leaf or fruit image, or click to choose one.</span>
            )}
          </label>
          <button className="btn" disabled={!file || busy} type="submit">
            {busy ? "Analyzing…" : "Run identification"}
          </button>
        </form>
        {error && <p className="error">{error}</p>}
      </section>
      {result && (
        <section className="results">
          <h2>Result</h2>
          <p className="advice">{result.advice}</p>
          {result.top_k.map((item) => (
            <article key={item.id} className="pred">
              <header>
                <strong>{item.label}</strong>
                <span className="mono">{(item.confidence * 100).toFixed(1)}%</span>
              </header>
              <div className="bar">
                <i style={{ width: `${Math.max(item.confidence * 100, 3)}%` }} />
              </div>
              <p>{item.summary}</p>
              <ul>
                {item.look_for.map((hint) => (
                  <li key={hint}>{hint}</li>
                ))}
              </ul>
            </article>
          ))}
          <aside className="quality">
            <h3>Image conditions</h3>
            <dl>
              <div>
                <dt>Size</dt>
                <dd>
                  {result.quality.width}×{result.quality.height}
                </dd>
              </div>
              <div>
                <dt>Blur var.</dt>
                <dd>{result.quality.blur_var}</dd>
              </div>
              <div>
                <dt>Brightness</dt>
                <dd>{result.quality.brightness}</dd>
              </div>
              <div>
                <dt>Contrast</dt>
                <dd>{result.quality.contrast}</dd>
              </div>
            </dl>
            {result.quality.warnings.map((w) => (
              <p key={w} className="warn-line">
                {w}
              </p>
            ))}
          </aside>
          <p className="disclaimer">{result.disclaimer}</p>
        </section>
      )}
    </main>
  );
}
