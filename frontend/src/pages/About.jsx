import { useEffect, useState } from "react";

export default function About() {
  const [classes, setClasses] = useState([]);

  useEffect(() => {
    fetch("/api/taxonomy")
      .then((r) => r.json())
      .then((t) => setClasses(t.classes || []))
      .catch(() => setClasses([]));
  }, []);

  return (
    <main>
      <p className="kicker">Safety and limits</p>
      <h1>What this project will not do</h1>
      <ul className="limits">
        <li>It will not tell anyone which pesticide, dose, or spray schedule to use.</li>
        <li>It will not replace a plant pathologist, grower, or extension agent.</li>
        <li>
          It may confuse look-alike symptoms, mixed infections, nutrient issues, and
          insect damage that is not in the fourteen-class set.
        </li>
        <li>
          High accuracy on PlantVillage does not imply high accuracy on orchard phone
          photos. That gap is part of the research claim.
        </li>
      </ul>
      <h2>In-scope classes</h2>
      <div className="class-list">
        {classes.map((item) => (
          <article key={item.id}>
            <h3>{item.label}</h3>
            <p>{item.summary}</p>
          </article>
        ))}
      </div>
    </main>
  );
}
