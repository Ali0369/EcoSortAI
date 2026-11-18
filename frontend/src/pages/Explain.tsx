import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { ArrowUpRight } from "lucide-react";
import { api, ApiError } from "../api/client";
import type { ExplainResponse } from "../api/types";
import { EmptyState, Notice, Spinner } from "../components/ui";
import { PageNote } from "../components/Footer";
import { label, pct } from "../lib/bins";
import { useScan } from "../state/ScanContext";

export default function Explain() {
  const { file, modelId, models } = useScan();
  const [res, setRes] = useState<ExplainResponse | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [view, setView] = useState<"heatmap" | "overlay">("heatmap");

  useEffect(() => {
    if (!file || !modelId) return;
    let cancelled = false;
    setBusy(true); setErr(null); setRes(null);
    api.explain(file, modelId).then((r) => { if (!cancelled) setRes(r); })
      .catch((e) => { if (!cancelled) setErr(e instanceof ApiError ? e.message : "Couldn't create the explanation."); })
      .finally(() => { if (!cancelled) setBusy(false); });
    return () => { cancelled = true; };
  }, [file, modelId]);

  if (!file) {
    return <main className="px-5 py-20"><EmptyState title="Nothing to explain yet" text="Classify a photo first, then come back to see which parts of it the model paid attention to."
      action={<Link to="/scanner" className="btn-primary">Open the scanner <ArrowUpRight className="h-4 w-4" /></Link>} /></main>;
  }
  const src = (b: string) => `data:image/png;base64,${b}`;
  const model = models.find((m) => m.id === modelId);
  return (
    <main className="mx-auto max-w-[1200px] px-5 pb-8 pt-8 md:px-8">
      <div className="eyebrow">EXPLAINABLE BY DESIGN</div>
      <div className="mt-3 flex flex-wrap items-start justify-between gap-4">
        <div>
          <h1 className="text-3xl font-semibold tracking-tight md:text-[40px]">See what caught the model’s eye.</h1>
          <p className="mt-2 text-sm text-muted">{res ? `A closer look at the visual features associated with a ${label(res.explained_class).toLowerCase()} prediction.` : "Grad-CAM highlights the regions that most influenced the prediction."}</p>
        </div>
        {res && <div className="rounded-2xl bg-sage px-5 py-3 text-sm font-medium">{label(res.explained_class)} · {pct(res.confidence)} confidence</div>}
      </div>
      <div className="mt-5"><Notice>Grad-CAM computed live from {model ? `${model.code} · ${model.name}` : "the selected model"} for your image. Change the model in the scanner to compare.</Notice></div>
      {err && <div className="mt-4"><Notice tone="error">{err}</Notice></div>}
      {busy && <div className="mt-10 flex justify-center"><Spinner label="Computing Grad-CAM" /></div>}

      {res && (
        <>
          <div className="mt-6 grid gap-5 md:grid-cols-2">
            <section className="card p-5"><div className="flex justify-between text-sm font-semibold"><h2>Original image</h2><span className="text-xs font-normal text-muted">{file.name}</span></div>
              <img src={src(res.original_png_b64)} alt="Original" className="mt-4 w-full rounded-2xl object-contain" /><p className="mt-3 text-[11px] text-muted">Input used by the classifier</p></section>
            <section className="card p-5">
              <div className="flex items-center justify-between"><h2 className="text-sm font-semibold">Grad-CAM {view}</h2>
                <div className="flex rounded-lg bg-sage p-0.5 text-xs" role="tablist">
                  {(["heatmap", "overlay"] as const).map((v) => <button key={v} role="tab" aria-selected={view === v} onClick={() => setView(v)} className={`rounded-md px-3 py-1 ${view === v ? "bg-white font-semibold shadow-sm" : "text-muted"}`}>{v === "heatmap" ? "Heatmap" : "Overlay"}</button>)}
                </div></div>
              <img src={src(view === "heatmap" ? res.heatmap_png_b64 : res.overlay_png_b64)} alt={`Grad-CAM ${view}`} className="mt-4 w-full rounded-2xl object-contain" />
              <div className="mt-3 flex items-center gap-3 text-[11px] text-muted"><span>Lower attention</span><div className="h-2 flex-1 rounded-full" style={{ background: "linear-gradient(90deg,#1f3fbf,#25b6d8,#7be05f,#f5e04a,#e8452c)" }} /><span>Higher attention</span></div>
            </section>
          </div>
          <div className="mt-5 grid gap-5 lg:grid-cols-[1.45fr_1fr]">
            <section className="rounded-card bg-deep p-7 text-white"><h2 className="text-2xl font-semibold">Patterns, not magic.</h2>
              <p className="mt-3 text-sm text-white/70">For this image the strongest attention is in the <b className="text-white">{res.peak_region}</b> area, and about <b className="text-white">{Math.round(res.high_attention_area_fraction * 100)}%</b> of the image is in the high-attention range. The map is computed on a {res.feature_map_size[1]}×{res.feature_map_size[0]} feature grid and scaled up, so it is coarse.</p></section>
            <section className="flex flex-col rounded-card bg-sage p-7"><h2 className="text-xl font-semibold">A window, not proof.</h2>
              <p className="mt-3 text-sm text-ink">A heatmap shows association, not a causal explanation. Backgrounds, lighting and lookalike materials can still mislead a model.</p>
              <Link to="/methodology" className="btn-ghost mt-auto self-start">Review model methodology</Link></section>
          </div>
        </>
      )}
      <div className="mt-8"><Link to="/results" className="text-xs text-muted hover:text-forest">← Return to your result</Link></div>
      <PageNote>Explanations are an aid to understanding the model, not a guarantee that it is right.</PageNote>
    </main>
  );
}
