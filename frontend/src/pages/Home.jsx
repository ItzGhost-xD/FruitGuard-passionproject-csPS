import { Link } from "react-router-dom";

export default function Home() {
  return (
    <main>
      <section className="hero">
        <p className="kicker">CS research · computer vision · four fruits</p>
        <h1>Can a model name a fruit disease from a messy phone photo?</h1>
        <p className="lede">
          FruitGuard asks a research question, not just a product one: how accurately can
          computer vision identify common fruit diseases and pests from user-submitted
          images under real-world lighting and background conditions?
        </p>
        <div className="cta-row">
          <Link className="btn" to="/identify">
            Upload a leaf or fruit photo
          </Link>
          <Link className="btn ghost" to="/research">
            See the experiment design
          </Link>
        </div>
      </section>
      <section className="grid-3">
        <article className="card">
          <h2>Narrow scope</h2>
          <p>
            Apple, grape, peach, and tomato. Healthy tissue plus two to three common
            conditions each — fourteen classes, not a hundred noisy labels.
          </p>
        </article>
        <article className="card">
          <h2>Compared models</h2>
          <p>
            Color-histogram SVM, a small CNN trained from scratch, then transfer learning
            with MobileNetV3, EfficientNet-B0, and ResNet-18.
          </p>
        </article>
        <article className="card">
          <h2>The hard test</h2>
          <p>
            Lab accuracy is not the claim. Phone photos taken outside the training set
            decide whether the system is actually useful.
          </p>
        </article>
      </section>
    </main>
  );
}
