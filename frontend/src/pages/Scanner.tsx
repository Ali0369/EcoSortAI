import { useEffect, useRef, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { ArrowRight, ArrowUpRight, Check, ChevronDown, Circle, Loader2, X } from "lucide-react";
import { api, ApiError } from "../api/client";
import { BinStation, type StationPhase } from "../components/BinStation";
import { Dropzone, validateFile } from "../components/Dropzone";
import { MiniBin } from "../components/IsoBin";
import { Bar, Notice } from "../components/ui";
import { PageNote } from "../components/Footer";
import { BINS, CARD_ORDER, STATION_ORDER, label, pct } from "../lib/bins";
import { useScan } from "../state/ScanContext";

type Phase = "idle" | "ready" | "analyzing" | "sorting" | "sorted" | "uncertain" | "error";
const MIN_ANALYZE_MS = 1400; // lets the "analyzing" state be readable; the prediction itself is never delayed or altered

export default function Scanner() {
  const nav = useNavigate();
  const { file, previewUrl, prediction, models, modelsError, modelId, setModelId, categories, disclaimer, setFile, setPrediction, reset } = useScan();
  const [phase, setPhase] = useState<Phase>(() => (prediction ? (prediction.is_uncertain ? "uncertain" : "sorted") : file ? "ready" : "idle"));
  const [error, setError] = useState<string | null>(null);
  const [pickError, setPickError] = useState<string | null>(null);
  const [step, setStep] = useState(0);
  const [selected, setSelected] = useState<string>("plastic");
  const [sampleBusy, setSampleBusy] = useState(false);
  const [hasSample, setHasSample] = useState(false);
  const reqId = useRef(0);
  const replaceInput = useRef<HTMLInputElement>(null);

  useEffect(() => { api.samples().then((s) => setHasSample(s.samples.some((x) => x.kind === "uncertain-candidate"))).catch(() => setHasSample(false)); }, []);
  useEffect(() => {
    if (phase !== "analyzing") return;
    setStep(0);
    const t1 = setTimeout(() => setStep(1), 450), t2 = setTimeout(() => setStep(2), 950);
    return () => { clearTimeout(t1); clearTimeout(t2); };
  }, [phase]);

  const accept = (f: File) => {
    const problem = validateFile(f);
    if (problem) { setPickError(problem); return; }
    setPickError(null); setError(null); setFile(f); setPhase("ready");
  };
  const remove = () => { reqId.current++; reset(); setPhase("idle"); setError(null); setPickError(null); };

  const classify = async () => {
    if (!file) return;
    const id = ++reqId.current;
    setPhase("analyzing"); setError(null);
    const started = Date.now();
    try {
      const p = await api.predict(file, modelId);
      const wait = MIN_ANALYZE_MS - (Date.now() - started);
      if (wait > 0) await new Promise((r) => setTimeout(r, wait));
      if (id !== reqId.current) return;
      setPrediction(p);
      setPhase(p.is_uncertain ? "uncertain" : "sorting");
    } catch (e) {
      if (id !== reqId.current) return;
      setError(e instanceof ApiError ? e.message : "Something went wrong. Please try again.");
      setPhase("error");
    }
  };

  const loadSample = async () => {
    setSampleBusy(true);
    try {
      const list = await api.samples();
      const s = list.samples.find((x) => x.kind === "uncertain-candidate");
      if (!s) return;
      const blob = await (await fetch(api.sampleUrl(s.file))).blob();
      accept(new File([blob], s.file, { type: blob.type || "image/jpeg" }));
    } catch { setPickError("Couldn't load the example image."); } finally { setSampleBusy(false); }
  };

  const stationView = phase === "analyzing" || phase === "sorting" || phase === "sorted" || phase === "uncertain";
  const cat = prediction ? categories[prediction.predicted_class] : undefined;
  const best = prediction?.top3[0];
  const stationPhase: StationPhase = stationView ? (phase as StationPhase) : "idle";

  // ------------------------------------------------------------------ station (analyzing / sorting / result)
  if (stationView) {
    const title = phase === "analyzing" ? "Looking a little closer." : phase === "sorting" && prediction ? `A perfect place for ${label(prediction.predicted_class).toLowerCase()}.`
      : phase === "sorted" ? "Sorted. One better decision." : "Let's take a second look.";
    const sub = phase === "analyzing" ? "Our model is reading the shape, texture and material." : phase === "sorting" && prediction ? `${label(prediction.predicted_class)} identified. Guiding your item to the right destination.`
      : phase === "sorted" ? "A small action, a material difference. Keep the cycle going." : "This one is a close call. A clearer image can make the difference.";
    const pill = phase === "analyzing" ? "MODEL IS ANALYZING" : phase === "uncertain" ? "LOW CONFIDENCE" : prediction ? `${label(prediction.predicted_class).toUpperCase()} · ${pct(prediction.confidence)} CONFIDENCE` : "";
    const steps = ["Image prepared", "Extracting features", `Comparing ${STATION_ORDER.length} categories`];
    return (
      <main className="mx-auto max-w-[1200px] px-5 pb-8 pt-6 md:px-8">
        <button onClick={remove} className="mb-6 flex items-center gap-1.5 text-xs text-muted hover:text-forest">← Back to scanner</button>
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div aria-live="polite">
            <h1 className="text-3xl font-semibold tracking-tight md:text-[40px]">{title}</h1>
            <p className="mt-2 text-sm text-muted">{sub}</p>
          </div>
          <span className={`rounded-full px-3 py-1.5 text-[10px] font-semibold tracking-wide ${phase === "uncertain" ? "bg-amber-soft text-amber-ink" : "bg-sage text-forest"}`}>{pill}</span>
        </div>

        <div className="mt-8 grid gap-5 lg:grid-cols-[1fr_340px]">
          <div className="h-[420px] md:h-[470px]">
            <BinStation order={STATION_ORDER} phase={stationPhase} target={prediction && !prediction.is_uncertain ? prediction.predicted_class : null}
              imageUrl={previewUrl} uncertainPair={prediction && prediction.top3.length > 1 ? [prediction.top3[0].class_name, prediction.top3[1].class_name] : null}
              onSorted={() => setPhase("sorted")} />
          </div>

          <aside className="card flex flex-col p-6" aria-live="polite">
            <div className="eyebrow">CLASSIFICATION</div>
            {phase === "analyzing" && (
              <>
                <h2 className="mt-3 text-2xl font-semibold leading-tight">Connecting the dots</h2>
                <p className="mt-2 text-xs text-muted">Comparing visual patterns</p>
                <ul className="mt-6 space-y-3.5 text-sm">
                  {steps.map((s, i) => (
                    <li key={s} className="flex items-center gap-3">
                      {i < step ? <Check className="h-4 w-4 text-forest" /> : i === step ? <Loader2 className="h-4 w-4 animate-spin text-forest" /> : <Circle className="h-4 w-4 text-mist" />}
                      <span className={i <= step ? "text-forest" : "text-muted"}>{s}</span>
                    </li>
                  ))}
                </ul>
                <div className="mt-auto pt-8"><Bar value={(step + 1) / 3} tone="dark" /><p className="mt-3 text-[11px] text-muted">Analyzing your image…</p><p className="mt-14 text-[11px] text-muted">Usually just a moment.</p></div>
              </>
            )}
            {(phase === "sorting" || phase === "sorted") && prediction && best && (
              <>
                <h2 className="mt-3 text-3xl font-semibold">{label(prediction.predicted_class)}</h2>
                <p className="mt-1 text-xs text-muted">{BINS[prediction.predicted_class]?.recyclable ? "Recyclable" : "General waste"}</p>
                <div className="mt-6 text-6xl font-semibold tracking-tight">{pct(prediction.confidence)}</div>
                <div className="eyebrow mt-1">MODEL CONFIDENCE</div>
                <div className="mt-3"><Bar value={prediction.confidence} /></div>
                <div className="mt-6 rounded-2xl bg-cream p-4">
                  <div className="text-sm font-semibold">{cat?.headline ?? "Check your local rules."}</div>
                  <p className="mt-1 text-xs text-muted">{cat?.guidance ?? "Recycling rules vary by location."}</p>
                </div>
                <div className="mt-auto pt-6">
                  {phase === "sorting" ? <div className="rounded-xl bg-sage px-4 py-3 text-xs font-semibold">Sorting into {label(prediction.predicted_class).toLowerCase()}…</div> : (
                    <div className="flex flex-col gap-2">
                      <button className="btn-primary" onClick={() => nav("/results")}>View full results <ArrowRight className="h-4 w-4" /></button>
                      <button className="btn-ghost !border-0" onClick={remove}>Classify another item</button>
                    </div>
                  )}
                </div>
              </>
            )}
            {phase === "uncertain" && prediction && (
              <>
                <h2 className="mt-3 text-2xl font-semibold leading-tight">Not quite sure yet</h2>
                <p className="mt-2 text-xs text-muted">Confidence is below the {Math.round(prediction.uncertainty_threshold * 100)}% threshold.</p>
                <div className="mt-6 text-xs text-muted">Top possibilities</div>
                <div className="mt-3 space-y-4">
                  {prediction.top3.slice(0, 2).map((t) => (
                    <div key={t.class_name}><div className="mb-1.5 flex justify-between text-sm"><span>{label(t.class_name)}</span><span className="font-medium">{pct(t.probability)}</span></div><Bar value={t.probability} tone="amber" /></div>
                  ))}
                </div>
                <p className="mt-6 text-xs text-muted">Try brighter light and a plain background. Avoid reflections.</p>
                <div className="mt-auto flex flex-col gap-2 pt-6">
                  <button className="btn-primary" onClick={remove}>Try another photo <ArrowUpRight className="h-4 w-4" /></button>
                  <button className="btn-soft" onClick={() => nav("/results")}>Review possibilities</button>
                </div>
              </>
            )}
          </aside>
        </div>

        <ol className="card mt-5 grid grid-cols-3 items-center gap-2 px-4 py-4 text-xs md:px-8 md:text-sm" aria-label="Progress">
          {["Upload a photo", "Identify the material", "Sort with confidence"].map((s, i) => {
            const done = i === 0 || (i === 1 && phase !== "analyzing") || (i === 2 && (phase === "sorted" || phase === "uncertain"));
            return <li key={s} className="flex items-center justify-center gap-2 text-center">{done ? <Check className="h-4 w-4 text-forest" /> : <span className="font-semibold text-muted">{i + 1}</span>}<span className={done ? "" : "text-muted"}>{s}</span></li>;
          })}
        </ol>
        <PageNote>Model predictions are guidance, not a guarantee. {disclaimer}</PageNote>
      </main>
    );
  }

  // ------------------------------------------------------------------ upload / ready
  return (
    <main className="mx-auto max-w-[1200px] px-5 pb-8 pt-8 md:px-8">
      <div className="eyebrow">THE SMARTER WAY TO RECYCLE</div>
      <div className="mt-3 flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-3xl font-semibold tracking-tight md:text-[40px]">One photo. The right bin.</h1>
          <p className="mt-2 text-sm text-muted">Give your waste a second chance. Start with a clear photo of a single item.</p>
        </div>
        <label className="card min-w-[200px] px-4 py-2.5">
          <span className="eyebrow block">MODEL</span>
          <span className="relative mt-0.5 flex items-center">
            <select className="w-full appearance-none bg-transparent pr-6 text-sm font-semibold outline-none" value={modelId ?? ""} onChange={(e) => setModelId(e.target.value)} disabled={!models.length} aria-label="Choose model">
              {models.length === 0 && <option value="">{modelsError ? "Unavailable" : "Loading…"}</option>}
              {models.map((m) => <option key={m.id} value={m.id}>{m.code} · {m.name}</option>)}
            </select>
            <ChevronDown className="pointer-events-none absolute right-0 h-3.5 w-3.5" />
          </span>
        </label>
      </div>
      {modelsError && <div className="mt-4"><Notice tone="warn">{modelsError} Classification needs the backend and trained models (see the README).</Notice></div>}

      <div className="mt-8 grid gap-5 lg:grid-cols-[1fr_380px]">
        <section className="card p-5" aria-label="Upload">
          {phase === "idle" ? <Dropzone onFile={accept} error={pickError} /> : (
            <div className="grid gap-6 md:grid-cols-[260px_1fr]">
              <img src={previewUrl ?? ""} alt="Your uploaded item" className="aspect-square w-full rounded-2xl object-cover" />
              <div className="flex flex-col justify-center">
                <div className="eyebrow">READY TO CLASSIFY</div>
                <h2 className="mt-2 text-3xl font-semibold">Let’s find its place.</h2>
                <p className="mt-4 text-sm font-medium">{file?.name}</p>
                <p className="text-xs text-muted">{file ? (file.size / 1e6).toFixed(1) : 0} MB · Image uploaded</p>
                <div className="mt-4 flex items-center gap-3">
                  <button className="btn-soft !py-2" onClick={() => replaceInput.current?.click()}>Replace</button>
                  <button className="flex items-center gap-1 text-xs text-muted hover:text-forest" onClick={remove}>Remove <X className="h-3 w-3" /></button>
                  <input ref={replaceInput} type="file" accept="image/jpeg,image/png,image/webp" className="sr-only" onChange={(e) => { if (e.target.files?.[0]) accept(e.target.files[0]); e.target.value = ""; }} />
                </div>
                {(error || pickError) && <div className="mt-4"><Notice tone="error">{error ?? pickError}</Notice></div>}
                <button className="btn-primary mt-6 self-start" onClick={classify} disabled={!modelId}>{phase === "error" ? "Try again" : "Classify item"} <ArrowUpRight className="h-4 w-4" /></button>
                {hasSample && <button className="mt-4 self-start text-xs font-medium hover:underline" onClick={loadSample} disabled={sampleBusy}>Try an uncertain example →</button>}
              </div>
            </div>
          )}
          {phase === "idle" && hasSample && <div className="mt-3 text-center"><button className="text-xs font-medium hover:underline" onClick={loadSample} disabled={sampleBusy}>{sampleBusy ? "Loading…" : "Try an example image →"}</button></div>}
        </section>

        <aside className="rounded-card bg-deep p-7 text-white">
          <h2 className="text-2xl font-semibold leading-tight">A little clarity<br />goes a long way.</h2>
          <ul className="mt-8 space-y-6">
            {[["One item at a time", "Keep other objects out of the frame."], ["Keep it in focus", "Good light helps us read the material."], ["Show the whole object", "Give edges and textures room to shine."]].map(([t, d], i) => (
              <li key={t} className="flex gap-4"><span className="mt-0.5 text-[11px] font-semibold text-lime">0{i + 1}</span><div><div className="text-sm font-semibold">{t}</div><div className="mt-0.5 text-xs text-white/60">{d}</div></div></li>
            ))}
          </ul>
        </aside>
      </div>

      <section className="mt-12">
        <h2 className="text-xl font-semibold">Six materials. One simple habit.</h2>
        <p className="mt-1 text-xs text-muted">Choose a category to explore what belongs.</p>
        <div className="mt-5 grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
          {CARD_ORDER.map((id) => (
            <button key={id} onClick={() => setSelected(id)} aria-pressed={selected === id}
              className={`card flex flex-col items-center gap-2 px-3 py-5 text-sm font-semibold transition-shadow ${selected === id ? "ring-2 ring-forest" : "hover:shadow-md"}`}>
              <MiniBin bin={BINS[id]} selected={selected === id} />{label(id)}
            </button>
          ))}
        </div>
        <div className="mt-4 rounded-2xl bg-sage p-5 text-sm" aria-live="polite">
          <span className="font-semibold">{label(selected)}.</span>{" "}
          <span className="text-ink">{categories[selected]?.summary ?? "Connect the backend to see guidance for this material."}</span>
          {categories[selected] && <div className="mt-1 text-xs text-muted">{categories[selected].guidance}</div>}
        </div>
      </section>
      <PageNote>{disclaimer}</PageNote>
    </main>
  );
}
