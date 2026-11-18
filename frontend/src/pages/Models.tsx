import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { Bar as RBar, BarChart, Cell, LabelList, ResponsiveContainer, XAxis, YAxis, Tooltip } from "recharts";
import { api, ApiError } from "../api/client";
import type { CompareResponse, MetricsPayload } from "../api/types";
import { EmptyState, Notice, Spinner } from "../components/ui";
import { PageNote } from "../components/Footer";
import { label, pct } from "../lib/bins";
import { useScan } from "../state/ScanContext";

const n1 = (x: number) => (x * 100).toFixed(1);

export default function Models() {
  const [data, setData] = useState<MetricsPayload | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [active, setActive] = useState<string>("");
  useEffect(() => {
    api.metrics().then((d) => { setData(d); setActive(d.default_model_id); }).catch((e: Error) => setErr(e.message));
  }, []);

  const order = useMemo(() => (data ? data.models.map((m) => m.id).filter((id) => data.metrics.models[id]) : []), [data]);
  if (err) return <main className="px-5 py-20"><EmptyState title="Metrics aren’t available yet" text={`${err} Run the training notebook, copy its artifacts into ml/artifacts, then start the backend.`} action={<Link to="/methodology" className="btn-ghost">Read the methodology</Link>} /></main>;
  if (!data) return <main className="px-5 py-20"><div className="flex justify-center"><Spinner label="Loading experiment results" /></div></main>;

  const M = data.metrics.models;
  const cur = M[active] ?? M[order[0]];
  const e2 = M["e2_efficientnetb0"], e3 = M["e3_hybrid_cbam"];
  const mc = data.metrics.statistics?.mcnemar_e2_vs_e3;
  const chart = order.map((id) => ({ id, name: M[id].code, full: M[id].name, acc: +(M[id].test_accuracy * 100).toFixed(1), proposed: data.models.find((m) => m.id === id)?.is_proposed }));
  const names = data.metrics.class_names;

  return (
    <main className="mx-auto max-w-[1200px] px-5 pb-8 pt-8 md:px-8">
      <div className="eyebrow">THE SCIENCE BEHIND THE SORT</div>
      <div className="mt-3 flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-3xl font-semibold tracking-tight md:text-[40px]">Better models. Measurable progress.</h1>
          <p className="mt-2 text-sm text-muted">Compare the trade-offs. Choose the intelligence behind your next sort.</p>
        </div>
        <label className="card min-w-[220px] px-4 py-2.5"><span className="eyebrow block">ACTIVE MODEL</span>
          <select className="mt-0.5 w-full bg-transparent text-sm font-semibold outline-none" value={active} onChange={(e) => setActive(e.target.value)} aria-label="Active model">
            {order.map((id) => <option key={id} value={id}>{M[id].code} · {M[id].name}</option>)}
          </select>
        </label>
      </div>

      <div className="mt-5 space-y-2">
        {data.metrics.smoke_test && <Notice tone="warn"><b>Smoke-test results.</b> These numbers come from a quick pipeline test with tiny data and are not valid experimental results. Re-run the full notebook.</Notice>}
        <Notice>Real results from your notebook run · {data.metrics.protocol ?? "held-out test split"} · macro-averaged metrics · generated {data.metrics.generated_utc?.slice(0, 10)}</Notice>
      </div>

      <div className="mt-6 grid grid-cols-2 gap-4 lg:grid-cols-4">
        {([["Accuracy", cur.test_accuracy, "Correct across all categories"], ["Precision", cur.macro_precision, "Fewer false-positive predictions"],
          ["Recall", cur.macro_recall, "More materials recognised"], ["F1-score", cur.macro_f1, "Balanced precision and recall"]] as const).map(([t, v, d], i) => (
          <div key={t} className={`rounded-card p-5 ${i === 0 ? "bg-deep text-white" : "card"}`}>
            <div className={`text-xs ${i === 0 ? "text-white/70" : "text-muted"}`}>{t}</div>
            <div className="mt-3 text-4xl font-semibold tracking-tight">{n1(v)}%</div>
            <div className={`mt-2 text-[11px] ${i === 0 ? "text-white/60" : "text-muted"}`}>{d}</div>
          </div>
        ))}
      </div>

      <div className="mt-5 grid gap-5 lg:grid-cols-[1.45fr_1fr]">
        <section className="card p-6">
          <h2 className="text-lg font-semibold">Accuracy by architecture</h2>
          <p className="text-xs text-muted">Held-out test set · Higher is better</p>
          <div className="mt-4 h-[280px]" role="img" aria-label="Bar chart of test accuracy for each model">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chart} layout="vertical" margin={{ left: 8, right: 44 }}>
                <XAxis type="number" domain={[0, 100]} tick={{ fontSize: 11 }} unit="%" />
                <YAxis type="category" dataKey="name" width={36} tick={{ fontSize: 12 }} />
                <Tooltip formatter={(v: unknown) => `${v}%`} labelFormatter={(_, p) => p?.[0]?.payload?.full ?? ""} />
                <RBar dataKey="acc" radius={[6, 6, 6, 6]} barSize={18}>
                  {chart.map((c) => <Cell key={c.id} fill={c.proposed ? "#173D33" : c.id === active ? "#7E9A86" : "#CFDCC8"} />)}
                  <LabelList dataKey="acc" position="right" formatter={(v: unknown) => `${v}%`} style={{ fontSize: 11, fill: "#354A3D" }} />
                </RBar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </section>

        <section className="rounded-card bg-deep p-7 text-white">
          <div className="text-[11px] font-semibold text-lime">DOES ATTENTION HELP?</div>
          <h2 className="mt-3 text-2xl font-semibold leading-snug">EfficientNetB0 with and without CBAM</h2>
          <p className="mt-3 text-xs leading-relaxed text-white/70">The proposed hybrid adds channel and spatial attention (CBAM) on top of EfficientNetB0 features. Everything else is identical, so the difference isolates the attention module.</p>
          {e2 && e3 ? (
            <div className="mt-6 border-t border-white/15 pt-5">
              <div className="flex items-baseline gap-3"><span className="text-4xl font-semibold text-lime">{(e3.test_accuracy - e2.test_accuracy >= 0 ? "+" : "") + ((e3.test_accuracy - e2.test_accuracy) * 100).toFixed(1)} pts</span><span className="text-xs text-white/60">test accuracy vs. EfficientNetB0</span></div>
              <p className="mt-3 text-xs text-white/75">{mc ? `McNemar exact test p = ${mc.p_value.toFixed(3)}: ${mc.p_value < 0.05 ? "a statistically detectable difference" : "no statistically detectable difference"} (${mc.a_right_b_wrong} vs ${mc.a_wrong_b_right} discordant test images).` : "Run the statistics cell in the notebook for a significance test."}</p>
            </div>
          ) : <p className="mt-6 border-t border-white/15 pt-5 text-xs text-white/70">Train both E2 and E3 to see this comparison.</p>}
        </section>
      </div>

      <section className="card mt-5 overflow-x-auto p-6">
        <h2 className="text-lg font-semibold">Performance, side by side</h2>
        <table className="mt-4 w-full min-w-[720px] text-left text-sm">
          <thead className="text-xs text-muted"><tr>{["Architecture", "Accuracy", "Precision", "Recall", "F1-score", "Train time", "CPU ms/img", "Size"].map((h) => <th key={h} className="pb-3 pr-4 font-medium">{h}</th>)}</tr></thead>
          <tbody>
            {order.map((id) => {
              const m = M[id], prop = data.models.find((x) => x.id === id)?.is_proposed;
              return (
                <tr key={id} className={`border-t border-line ${id === active ? "bg-sage" : ""}`}>
                  <td className="py-3 pr-4 font-medium">{m.code} · {m.name}{prop && <span className="ml-2 rounded bg-lime px-1.5 py-0.5 text-[10px]">proposed</span>}</td>
                  <td className="pr-4">{n1(m.test_accuracy)}%</td><td className="pr-4">{n1(m.macro_precision)}%</td><td className="pr-4">{n1(m.macro_recall)}%</td><td className="pr-4">{n1(m.macro_f1)}%</td>
                  <td className="pr-4">{(m.training_seconds / 60).toFixed(1)} min</td><td className="pr-4">{m.inference_ms_cpu != null ? m.inference_ms_cpu.toFixed(1) : "–"}</td><td>{m.model_size_mb.toFixed(1)} MB</td>
                </tr>
              );
            })}
          </tbody>
        </table>
        <p className="mt-3 text-[11px] text-muted">Metrics are macro-averaged on the held-out test split. Training time is for the logged run on the GPU that was used.</p>
      </section>

      <div className="mt-5 grid gap-5 lg:grid-cols-2">
        <section className="card p-6">
          <h2 className="text-lg font-semibold">Where the model gets it right</h2>
          <p className="text-xs text-muted">Confusion matrix · {cur.name} · row-normalised (%)</p>
          <div className="mt-4 overflow-x-auto">
            <table className="mx-auto text-center text-[11px]" aria-label="Confusion matrix">
              <thead><tr><th />{names.map((n) => <th key={n} className="px-1 pb-1 font-medium text-muted">{n.slice(0, 2).toUpperCase()}</th>)}</tr></thead>
              <tbody>
                {cur.confusion_matrix.map((row, i) => {
                  const total = row.reduce((a, b) => a + b, 0) || 1;
                  return (
                    <tr key={i}><th className="pr-2 text-right font-medium text-muted">{names[i].slice(0, 2).toUpperCase()}</th>
                      {row.map((v, j) => { const p = v / total; return <td key={j} className="p-0.5"><div className="flex h-9 w-11 items-center justify-center rounded-md" title={`actual ${names[i]} → predicted ${names[j]}: ${v}`} style={{ background: `rgba(23,61,51,${0.06 + p * 0.94})`, color: p > 0.45 ? "#fff" : "#173D33" }}>{Math.round(p * 100)}</div></td>; })}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <p className="mt-3 text-[11px] text-muted">{names.map((n) => `${n.slice(0, 2).toUpperCase()} ${label(n)}`).join(" · ")}</p>
        </section>

        <section className="card p-6">
          <h2 className="text-lg font-semibold">Per-category results</h2>
          <p className="text-xs text-muted">{cur.code} · {cur.name} · held-out test split</p>
          <table className="mt-4 w-full text-left text-sm">
            <thead className="text-xs text-muted"><tr><th className="pb-2 font-medium">Category</th><th className="font-medium">Precision</th><th className="font-medium">Recall</th><th className="font-medium">F1</th><th className="font-medium">Images</th></tr></thead>
            <tbody>
              {names.map((n) => { const r = cur.per_class?.[n]; return (
                <tr key={n} className="border-t border-line"><td className="py-2.5">{label(n)}</td>{r ? <><td>{n1(r.precision)}%</td><td>{n1(r.recall)}%</td><td>{n1(r["f1-score"])}%</td><td className="text-muted">{r.support}</td></> : <td colSpan={4} className="text-xs text-muted">not available</td>}</tr>
              ); })}
            </tbody>
          </table>
        </section>
      </div>

      <ComparePanel modelIds={order} nameOf={(id) => `${M[id].code} · ${M[id].name}`} />
      <PageNote>These figures come from one training run and one split. Small differences between models may be within noise.</PageNote>
    </main>
  );
}

function ComparePanel({ modelIds, nameOf }: { modelIds: string[]; nameOf: (id: string) => string }) {
  const { file, previewUrl } = useScan();
  const [chosen, setChosen] = useState<string[]>(modelIds);
  const [res, setRes] = useState<CompareResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const run = async () => {
    if (!file) return;
    setBusy(true); setErr(null);
    try { setRes(await api.compare(file, chosen)); } catch (e) { setErr(e instanceof ApiError ? e.message : "Comparison failed."); } finally { setBusy(false); }
  };
  useEffect(() => { setRes(null); }, [file]);
  return (
    <section className="card mt-5 p-6">
      <h2 className="text-lg font-semibold">Same image, different models</h2>
      <p className="text-xs text-muted">Run your uploaded photo through the models you pick and see where they agree.</p>
      {!file ? (
        <p className="mt-4 text-sm text-muted">Upload a photo in the <Link to="/scanner" className="font-medium text-forest underline">scanner</Link> first.</p>
      ) : (
        <>
          <div className="mt-4 flex flex-wrap items-center gap-4">
            {previewUrl && <img src={previewUrl} alt="Your image" className="h-16 w-16 rounded-xl object-cover" />}
            <div className="flex flex-wrap gap-2">
              {modelIds.map((id) => (
                <label key={id} className={`cursor-pointer rounded-full border px-3 py-1.5 text-xs ${chosen.includes(id) ? "border-forest bg-sage font-semibold" : "border-line text-muted"}`}>
                  <input type="checkbox" className="sr-only" checked={chosen.includes(id)} onChange={() => setChosen((c) => c.includes(id) ? c.filter((x) => x !== id) : [...c, id])} />{nameOf(id)}
                </label>
              ))}
            </div>
            <button className="btn-primary ml-auto" onClick={run} disabled={busy || chosen.length === 0}>{busy ? "Comparing…" : "Compare models"}</button>
          </div>
          {err && <div className="mt-4"><Notice tone="error">{err}</Notice></div>}
          {res && (
            <div className="mt-5 overflow-x-auto">
              <table className="w-full min-w-[520px] text-left text-sm">
                <thead className="text-xs text-muted"><tr><th className="pb-2 font-medium">Model</th><th className="font-medium">Prediction</th><th className="font-medium">Confidence</th><th className="font-medium">Status</th><th className="font-medium">Time</th></tr></thead>
                <tbody>{res.predictions.map((p) => (
                  <tr key={p.model_id} className="border-t border-line"><td className="py-2.5 font-medium">{nameOf(p.model_id)}</td><td>{label(p.predicted_class)}</td><td>{pct(p.confidence)}</td>
                    <td>{p.is_uncertain ? <span className="rounded bg-amber-soft px-2 py-0.5 text-xs text-amber-ink">uncertain</span> : <span className="rounded bg-sage px-2 py-0.5 text-xs">confident</span>}</td><td className="text-muted">{p.inference_ms.toFixed(0)} ms</td></tr>))}
                </tbody>
              </table>
              <p className="mt-3 text-xs text-muted">{res.all_agree ? "All selected models agree on this image." : "The models disagree on this image."}</p>
            </div>
          )}
        </>
      )}
    </section>
  );
}
