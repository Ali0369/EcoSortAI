import type { ReactNode } from "react";
import { AlertTriangle, Info } from "lucide-react";
import { pct } from "../lib/bins";

export function Notice({ tone = "info", children }: { tone?: "info" | "warn" | "error"; children: ReactNode }) {
  const cls = tone === "error" ? "bg-red-50 text-red-800" : tone === "warn" ? "bg-amber-soft text-amber-ink" : "bg-sage text-ink";
  return (
    <div role={tone === "error" ? "alert" : "status"} className={`flex items-start gap-2 rounded-xl px-4 py-2.5 text-xs ${cls}`}>
      {tone === "info" ? <Info className="mt-0.5 h-3.5 w-3.5 shrink-0" aria-hidden /> : <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" aria-hidden />}
      <div>{children}</div>
    </div>
  );
}

export function Bar({ value, tone = "dark", height = 6 }: { value: number; tone?: "dark" | "light" | "amber"; height?: number }) {
  const color = tone === "dark" ? "#123C31" : tone === "amber" ? "#C9A15C" : "#CFDCC8";
  return (
    <div className="w-full overflow-hidden rounded-full bg-sage" style={{ height }} role="progressbar" aria-valuenow={Math.round(value * 100)} aria-valuemin={0} aria-valuemax={100}>
      <div className="h-full rounded-full transition-[width] duration-700" style={{ width: pct(Math.max(0, Math.min(1, value)), 1), background: color }} />
    </div>
  );
}

export function Eyebrow({ children }: { children: ReactNode }) {
  return <div className="eyebrow">{children}</div>;
}

export function Spinner({ label = "Loading" }: { label?: string }) {
  return <div role="status" className="flex items-center gap-2 text-sm text-muted"><span className="h-4 w-4 animate-spin rounded-full border-2 border-leaf border-t-forest" />{label}…</div>;
}

export function EmptyState({ title, text, action }: { title: string; text: string; action?: ReactNode }) {
  return (
    <div className="card mx-auto max-w-xl p-10 text-center">
      <h2 className="text-xl font-semibold">{title}</h2>
      <p className="mt-2 text-sm text-muted">{text}</p>
      {action && <div className="mt-6 flex justify-center">{action}</div>}
    </div>
  );
}
