import { NavLink, Route, Routes } from "react-router-dom";
import Dashboard from "./pages/Dashboard";
import Viewer from "./pages/Viewer";

export default function App() {
  return (
    <div className="app">
      <header>
        <h1>2048 — recherche &amp; apprentissage</h1>
        <nav>
          <NavLink to="/" end>
            Dashboard
          </NavLink>
          <NavLink to="/viewer">Viewer</NavLink>
        </nav>
      </header>
      <main>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/viewer" element={<Viewer />} />
        </Routes>
      </main>
    </div>
  );
}
