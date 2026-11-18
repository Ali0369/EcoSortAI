import { useEffect, useState } from "react";
import { ArrowRight, ExternalLink } from "lucide-react";
import { api } from "../api/client";
import type { DatasetSummary } from "../api/types";
import { label } from "../lib/bins";
import { PageNote } from "../components/Footer";

export default function Methodology() {
  const [ds, setDs] = useState<DatasetSummary | null>(null);
  useEffect(() => { api.dataset().then(setDs).catch(() => setDs(null)); }, []);
  const rows = ds ? Object.entries(ds.class_counts).sort((a, b) => b[1] - a[1]) : [];
  const max = rows.length ? rows[0][1] : 1;
  const pipeline = [["Image input", "Resize · baked-in preprocessing", "224 × 224 RGB"], ["EfficientNetB0", "Pretrained feature extractor", "7 × 7 × 1280 map"], ["CBAM attention", "Channel then spatial", "What and where"], ["Classification", "GAP · Dense · Dropout · Softmax", "6 material categories"]];
  return (
    <main className="mx-auto max-w-[1200px] px-5 pb-8 pt-8 md:px-8">
      <div className="eyebrow">OUR METHODOLOGY</div>
      <h1 className="mt-3 max-w-2xl text-3xl font-semibold leading-tight tracking-tight md:text-[44px]">Built with purpose. Designed to be understood.</h1>
      <p className="mt-4 max-w-xl text-sm text-muted">EcoSort AI explores how computer vision can turn everyday uncertainty into more informed recycling decisions. Here’s the approach.</p>

      <div className="mt-8 grid gap-5 lg:grid-cols-[1fr_1fr]">
        <section className="card p-7"><div className="text-sm font-semibold">01 / The data behind the decision</div>
          <h2 className="mt-5 text-3xl font-semibold">{ds?.name ?? "TrashNet"}</h2>
          <p className="mt-1 text-xs text-muted">{ds ? `${ds.total_images.toLocaleString()} images · ${ds.classes.length} material categories` : "Dataset details load from the training artifacts."}</p>
          <p className="mt-4 text-sm text-muted">A reference image dataset of common waste objects. Class imbalance and controlled backgrounds make real-world generalisation an important open challenge.</p>
          <div className="mt-6 text-[10px] text-muted">SOURCE</div>
          <a className="mt-1 inline-flex items-center gap-1 text-xs font-medium hover:underline" href="https://github.com/garythung/trashnet" target="_blank" rel="noreferrer">Yang &amp; Thung · TrashNet <ExternalLink className="h-3 w-3" /></a>
          {ds && <p className="mt-2 text-[11px] text-muted">Licence: {ds.license}</p>}
        </section>
        <section className="card p-7"><h2 className="text-lg font-semibold">A closer look at the dataset</h2>
          {rows.length ? <ul className="mt-5 space-y-3">{rows.map(([k, v]) => <li key={k} className="flex items-center gap-3 text-xs"><span className="w-16">{label(k)}</span><div className="h-2 flex-1 rounded-full bg-sage"><div className="h-full rounded-full bg-forest" style={{ width: `${(v / max) * 100}%`, opacity: k === rows[0][0] ? 1 : 0.35 }} /></div><span className="w-10 text-right font-medium">{v}</span></li>)}</ul>
            : <p className="mt-5 text-sm text-muted">Class counts appear here once the backend has the exported artifacts.</p>}
          {ds && <p className="mt-5 text-[11px] text-muted">Split: {Object.entries(ds.splits).map(([s, v]) => `${s} ${v.n}`).join(" · ")} (stratified)</p>}
        </section>
      </div>

      <section className="mt-5 rounded-card bg-deep p-7 text-white">
        <div className="text-sm font-semibold">02 / Proposed hybrid architecture</div>
        <div className="mt-1 text-[11px] font-semibold text-lime">PROPOSED · SEE THE MODEL COMPARISON FOR MEASURED RESULTS</div>
        <div className="mt-6 grid gap-3 md:grid-cols-4">
          {pipeline.map(([t, d, s], i) => (
            <div key={t} className="relative rounded-2xl bg-white/10 p-4"><div className="text-sm font-semibold">{t}</div><div className="mt-2 text-xs text-white/70">{d}</div><div className="mt-1 text-[11px] text-white/45">{s}</div>
              {i < 3 && <ArrowRight className="absolute -right-3 top-1/2 hidden h-4 w-4 -translate-y-1/2 text-lime md:block" />}</div>
          ))}
        </div>
        <p className="mt-5 text-[11px] text-white/60">A custom CNN (E1) and plain EfficientNetB0 (E2) are trained as baselines under the same protocol as the proposed EfficientNetB0 + CBAM hybrid (E3). Predictions below a validation-calibrated confidence threshold are reported as uncertain.</p>
      </section>

      <div className="mt-5 grid gap-5 md:grid-cols-3">
        {[["03 / Evaluate honestly", "A stratified 70/15/15 split with exact duplicate files removed. Augmentation only on training data. Macro-averaged metrics, confusion matrices and a paired significance test (McNemar) between E2 and E3."],
          ["04 / Make AI legible", "Grad-CAM is a visual aid. We show uncertainty, compare models on the same image and acknowledge limitations."],
          ["05 / Encourage better habits", "Material-specific advice, never a promise of local recyclability."]].map(([t, d]) => (
          <section key={t} className="card p-6"><h2 className="text-sm font-semibold">{t}</h2><p className="mt-3 text-xs leading-relaxed text-muted">{d}</p></section>
        ))}
      </div>
      <PageNote>Limitations: small dataset, controlled backgrounds, near-duplicate photos are not removed (only exact duplicates), so leakage is possible, and results come from a single split.</PageNote>
    </main>
  );
}
