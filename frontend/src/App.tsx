import { useEffect, useState } from "react";
import { Route, Routes, useLocation } from "react-router-dom";
import { api } from "./api/client";
import type { Health } from "./api/types";
import { Footer } from "./components/Footer";
import { Header } from "./components/Header";
import { Notice } from "./components/ui";
import Explain from "./pages/Explain";
import Landing from "./pages/Landing";
import Methodology from "./pages/Methodology";
import Models from "./pages/Models";
import Results from "./pages/Results";
import Scanner from "./pages/Scanner";

function ScrollToTop() {
  const { pathname } = useLocation();
  useEffect(() => { window.scrollTo(0, 0); }, [pathname]);
  return null;
}

/** Shows backend problems (missing models, smoke-test artifacts) instead of hiding them. */
function StatusBanner() {
  const [h, setH] = useState<Health | null>(null);
  const [down, setDown] = useState(false);
  useEffect(() => { api.health().then(setH).catch(() => setDown(true)); }, []);
  if (down) return <div className="mx-auto max-w-[1200px] px-5 pt-3 md:px-8"><Notice tone="warn">The EcoSort backend is not reachable. Start it with <code>uvicorn app.main:app</code> (see the README). The pages still open, but classification needs it.</Notice></div>;
  if (!h) return null;
  return (
    <div className="mx-auto max-w-[1200px] space-y-2 px-5 pt-3 md:px-8">
      {h.status === "degraded" && <Notice tone="warn"><b>Backend running, but not fully ready.</b> {h.problems.join(" · ") || "Some models are missing."}</Notice>}
      {h.smoke_test && <Notice tone="warn"><b>Smoke-test models.</b> These artifacts come from a quick pipeline test; predictions and metrics are not valid results.</Notice>}
    </div>
  );
}

export default function App() {
  return (
    <>
      <a href="#main" className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-4 focus:z-50 focus:rounded-lg focus:bg-white focus:px-3 focus:py-2">Skip to content</a>
      <Header />
      <StatusBanner />
      <ScrollToTop />
      <div id="main">
        <Routes>
          <Route path="/" element={<Landing />} />
          <Route path="/scanner" element={<Scanner />} />
          <Route path="/results" element={<Results />} />
          <Route path="/models" element={<Models />} />
          <Route path="/explain" element={<Explain />} />
          <Route path="/methodology" element={<Methodology />} />
          <Route path="*" element={<main className="px-5 py-24 text-center"><h1 className="text-2xl font-semibold">Page not found</h1></main>} />
        </Routes>
      </div>
      <Footer />
    </>
  );
}
