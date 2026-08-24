import { NavLink, Route, Routes } from "react-router-dom";
import Home from "./pages/Home.jsx";
import Identify from "./pages/Identify.jsx";
import Research from "./pages/Research.jsx";
import About from "./pages/About.jsx";

export default function App() {
  return (
    <div className="shell">
      <header className="top">
        <NavLink to="/" className="brand">
          <span className="mark" aria-hidden="true" />
          FruitGuard
        </NavLink>
        <nav>
          <NavLink to="/identify">Identify</NavLink>
          <NavLink to="/research">Research</NavLink>
          <NavLink to="/about">Limits</NavLink>
        </nav>
      </header>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/identify" element={<Identify />} />
        <Route path="/research" element={<Research />} />
        <Route path="/about" element={<About />} />
      </Routes>
      <footer>
        Identification only. Not a diagnosis, and never a pesticide recommendation.
      </footer>
    </div>
  );
}
