import { Link } from "react-router-dom";
import { ArrowRight, ArrowUpRight, Eye, GitCompare, ListChecks, Recycle } from "lucide-react";
import { IsoBin, MiniBin } from "../components/IsoBin";
import { BINS, CARD_ORDER } from "../lib/bins";

const HERO_ORDER = ["paper", "plastic", "glass", "cardboard", "metal", "trash"];

export default function Landing() {
  return (
    <main className="mx-auto max-w-[1200px] px-5 pb-8 md:px-8">
      {/* hero */}
      <section className="grid items-center gap-10 pb-10 pt-14 lg:grid-cols-[1fr_1.1fr]">
        <div>
          <span className="inline-flex items-center gap-2 rounded-full bg-sage px-3 py-1.5 text-[10px] font-medium tracking-wide"><span className="h-1.5 w-1.5 rounded-full bg-forest" />SMALL ACTIONS. REAL IMPACT.</span>
          <h1 className="mt-6 text-5xl font-semibold leading-[1.05] tracking-tight md:text-[64px]">A smarter sort.<br />A greener world.</h1>
          <p className="mt-6 max-w-md text-base text-muted">Turn everyday waste into a better decision. Identify it with AI. Understand it. Sort it right.</p>
          <div className="mt-8 flex flex-wrap gap-3">
            <Link to="/scanner" className="btn-primary">Start Sorting <ArrowUpRight className="h-4 w-4" /></Link>
            <Link to="/methodology" className="btn-ghost">Explore the technology <ArrowRight className="h-4 w-4" /></Link>
          </div>
          <p className="mt-5 text-[11px] text-muted">No sign-up needed · Six waste categories · Built for clarity</p>
        </div>

        <div className="relative mx-auto w-full max-w-[560px]" aria-hidden>
          <span className="absolute right-4 top-0 z-10 rounded-full bg-white px-3 py-1 text-[10px] font-medium shadow-card">✦ AI-POWERED CLASSIFICATION</span>
          <svg viewBox="0 0 400 120" className="absolute bottom-6 left-0 w-full" preserveAspectRatio="none"><polygon points="50,10 390,10 360,110 10,110" fill="#E6E3D8" /></svg>
          <div className="relative flex items-end justify-center gap-[1.5%] px-[6%] pb-12 pt-16">
            {HERO_ORDER.map((id, i) => <div key={id} className="w-[15%]" style={{ marginBottom: `${i * 1.4}%` }}><IsoBin bin={BINS[id]} /></div>)}
          </div>
          <div className="card absolute -bottom-3 right-0 flex items-center gap-3 p-3 pr-5">
            <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-sage"><Recycle className="h-5 w-5" /></div>
            <div><div className="text-sm font-semibold">Plastic bottle</div><div className="text-[11px] text-muted">Example · Recyclable</div><div className="mt-1 rounded-full bg-sage px-2 py-0.5 text-[10px] font-medium">Confidence shown on every result</div></div>
          </div>
        </div>
      </section>

      <ol className="card grid divide-y divide-line text-sm md:grid-cols-3 md:divide-x md:divide-y-0">
        {["Upload a photo", "Let AI do the thinking", "Give waste a second life"].map((s, i) => <li key={s} className="flex items-center gap-3 px-6 py-4"><span className="text-[11px] text-muted">0{i + 1}</span>{s}</li>)}
      </ol>

      {/* how it works */}
      <section className="mt-20">
        <div className="eyebrow">LESS GUESSWORK. BETTER HABITS.</div>
        <h2 className="mt-3 text-3xl font-semibold tracking-tight">Good for you. Better for the planet.</h2>
        <div className="mt-8 grid gap-4 md:grid-cols-3">
          {[["Snap it. Drop it.", "Upload a clear photo of one item. A bottle, a box, or that mystery container."], ["Get an intelligent answer.", "Our model identifies the material and shows you how confident it really is."], ["Make the right move.", "Watch your item find its bin, then follow simple, practical recycling guidance."]].map(([t, d], i) => (
            <div key={t} className="card flex min-h-[190px] flex-col p-6"><span className="text-[11px] text-muted">0{i + 1}</span><h3 className="mt-auto text-lg font-semibold">{t}</h3><p className="mt-2 text-xs text-muted">{d}</p></div>
          ))}
        </div>
      </section>

      {/* six destinations */}
      <section className="mt-20">
        <h2 className="text-3xl font-semibold tracking-tight">One scanner. Six destinations.</h2>
        <p className="mt-2 text-xs text-muted">From everyday packaging to everything in between.</p>
        <div className="mt-6 grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
          {CARD_ORDER.map((id) => <Link key={id} to="/scanner" className="card flex flex-col items-center gap-2 px-3 py-5 text-xs font-semibold transition-shadow hover:shadow-md"><MiniBin bin={BINS[id]} />{BINS[id].label}</Link>)}
        </div>
      </section>

      {/* explain band */}
      <section className="mt-20 grid gap-8 rounded-xl2 bg-deep p-8 text-white md:grid-cols-2 md:p-10">
        <div>
          <span className="inline-block rounded-full bg-white/10 px-3 py-1.5 text-[10px] font-semibold tracking-wide text-lime">DESIGNED TO EXPLAIN, NOT JUST PREDICT</span>
          <h2 className="mt-6 text-4xl font-semibold leading-tight">Intelligence you<br />can look into.</h2>
          <Link to="/models" className="btn-lime mt-8">Meet the models <ArrowUpRight className="h-4 w-4" /></Link>
        </div>
        <div className="grid gap-6 sm:grid-cols-2">
          {[[Eye, "See what AI sees", "Visual heatmaps reveal the features behind a prediction."], [GitCompare, "Compare the evidence", "Explore accuracy, trade-offs and class-level performance."], [ListChecks, "Know the next step", "Material-specific advice, with a reminder to check local rules."]].map(([Icon, t, d]) => {
            const I = Icon as typeof Eye;
            return <div key={t as string}><div className="flex items-center gap-2 text-sm font-semibold"><I className="h-4 w-4" />{t as string}</div><p className="mt-2 text-xs text-white/60">{d as string}</p></div>;
          })}
        </div>
      </section>

      <section className="mt-20 flex flex-wrap items-center justify-between gap-6 pb-4">
        <div><h2 className="text-3xl font-semibold tracking-tight">Your next small action starts here.</h2><p className="mt-2 text-xs text-muted">Make sorting second nature.</p></div>
        <Link to="/scanner" className="btn-primary">Start Sorting <ArrowUpRight className="h-4 w-4" /></Link>
      </section>
    </main>
  );
}
