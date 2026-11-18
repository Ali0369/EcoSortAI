import { Link, useNavigate } from "react-router-dom";
import { ArrowRight, ArrowUpRight, Recycle } from "lucide-react";
import { Bar, EmptyState } from "../components/ui";
import { PageNote } from "../components/Footer";
import { BINS, label, pct } from "../lib/bins";
import { useScan } from "../state/ScanContext";

export default function Results() {
  const nav = useNavigate();
  const { prediction, previewUrl, file, categories, models, reset } = useScan();
  if (!prediction) {
    return (
      <main className="px-5 py-20"><EmptyState title="No result yet" text="Classify a photo first and your full results will appear here."
        action={<Link to="/scanner" className="btn-primary">Open the scanner <ArrowUpRight className="h-4 w-4" /></Link>} /></main>
    );
  }
  const unsure = prediction.is_uncertain;
  const id = prediction.predicted_class;
  const cat = categories[id];
  const model = models.find((m) => m.id === prediction.model_id);
  const again = () => { reset(); nav("/scanner"); };
  return (
    <main className="mx-auto max-w-[1200px] px-5 pb-8 pt-6 md:px-8">
      <Link to="/scanner" className="text-xs text-muted hover:text-forest">← Back to scanner</Link>
      <div className="mt-6 flex flex-wrap items-start justify-between gap-4">
        <div>
          <h1 className="text-3xl font-semibold tracking-tight md:text-[40px]">{unsure ? "A close call. Here’s what we found." : "A new life starts in the right bin."}</h1>
          <p className="mt-2 text-sm text-muted">{unsure ? "The model isn’t sure enough to pick one bin. Its top possibilities are below." : "Here’s what we found, and how to make your next move count."}</p>
        </div>
        <button className="btn-primary" onClick={again}>Classify another item <ArrowUpRight className="h-4 w-4" /></button>
      </div>

      <div className="mt-8 grid gap-5 lg:grid-cols-[1.45fr_1fr]">
        <section className="card grid gap-6 p-5 md:grid-cols-[250px_1fr]">
          {previewUrl && <img src={previewUrl} alt={file?.name ?? "Uploaded item"} className="aspect-square w-full rounded-2xl object-cover" />}
          <div className="flex flex-col justify-center">
            <div className="eyebrow">{unsure ? "LOW CONFIDENCE" : "✓ CLASSIFICATION COMPLETE"}</div>
            <h2 className="mt-2 text-4xl font-semibold">{unsure ? "Not sure yet" : label(id)}</h2>
            {!unsure && <span className="mt-3 inline-flex w-fit items-center gap-1.5 rounded-full bg-sage px-3 py-1 text-[11px] font-medium"><Recycle className="h-3 w-3" />{BINS[id]?.recyclable ? "Recyclable" : "General waste"}</span>}
            <div className="mt-5 text-5xl font-semibold tracking-tight">{pct(prediction.confidence)}</div>
            <div className="text-xs text-muted">Model confidence</div>
            <p className="mt-4 text-[11px] text-muted">{model ? `${model.code} · ${model.name}` : prediction.model_id} · {prediction.inference_ms.toFixed(0)} ms · uncertain below {Math.round(prediction.uncertainty_threshold * 100)}%</p>
          </div>
        </section>

        <section className="card p-6">
          <h2 className="text-lg font-semibold">The model’s top three</h2>
          <p className="mt-1 text-xs text-muted">A little transparency goes a long way.</p>
          <ol className="mt-6 space-y-5">
            {prediction.top3.map((t, i) => (
              <li key={t.class_name}><div className="mb-1.5 flex justify-between text-sm"><span><span className="mr-2 text-xs text-muted">0{i + 1}</span>{label(t.class_name)}</span><span className="font-semibold">{pct(t.probability)}</span></div><Bar value={t.probability} tone={i === 0 ? (unsure ? "amber" : "dark") : "light"} /></li>
            ))}
          </ol>
        </section>
      </div>

      <div className="mt-5 grid gap-5 lg:grid-cols-[1.45fr_1fr]">
        <section className="rounded-card bg-deep p-7 text-white">
          <h2 className="text-2xl font-semibold">{unsure ? "Get a clearer answer." : `Give this ${label(id).toLowerCase()} its best next chapter.`}</h2>
          {unsure ? (
            <ul className="mt-6 grid gap-5 sm:grid-cols-3">
              {[["Use brighter light", "Shadows hide texture."], ["Plain background", "Keep other objects out."], ["Avoid reflections", "Glare hides the material."]].map(([t, d], i) => (
                <li key={t}><div className="text-[11px] font-semibold text-lime">0{i + 1}</div><div className="mt-1 text-sm font-semibold">{t}</div><div className="mt-1 text-xs text-white/60">{d}</div></li>
              ))}
            </ul>
          ) : (
            <ul className="mt-6 grid gap-5 sm:grid-cols-3">
              {(cat?.steps ?? [{ title: "Check local rules", text: "Recycling rules vary by location." }]).map((s, i) => (
                <li key={s.title}><div className="text-[11px] font-semibold text-lime">0{i + 1}</div><div className="mt-1 text-sm font-semibold">{s.title}</div><div className="mt-1 text-xs text-white/60">{s.text}</div></li>
              ))}
            </ul>
          )}
          <p className="mt-8 text-[11px] text-white/60">Check your municipality’s guidance before placing it in the recycling bin.</p>
        </section>
        <section className="flex flex-col rounded-card bg-sage p-7">
          <h2 className="text-xl font-semibold">{unsure ? "Why only a close call?" : `Why ${label(id).toLowerCase()}?`}</h2>
          <p className="mt-3 text-sm text-ink">{cat?.summary ?? "See which parts of the image influenced the prediction."}</p>
          <Link to="/explain" className="btn-ghost mt-auto self-start">Explore the explanation <ArrowRight className="h-4 w-4" /></Link>
        </section>
      </div>
      <PageNote>AI can make mistakes. Classification is guidance only, not a guarantee of local recyclability.</PageNote>
    </main>
  );
}
