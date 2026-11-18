import { Recycle } from "lucide-react";
import type { BinStyle } from "../lib/bins";

/** Isometric bin drawn in SVG with the colors sampled from the Figma design. */
export function IsoBin({ bin, active = false, dim = false }: { bin: BinStyle; active?: boolean; dim?: boolean }) {
  const bh = 110 * bin.height;
  const H = 46 + bh + 6;
  const iconTop = ((38 + bh / 2) / H) * 100;
  return (
    <div className="relative w-full transition-opacity" style={{ opacity: dim ? 0.55 : 1 }} aria-hidden>
      <svg viewBox={`0 0 80 ${H}`} className="block w-full" style={{ filter: active ? "drop-shadow(0 0 6px rgba(212,239,130,.9))" : "drop-shadow(0 6px 8px rgba(23,61,51,.12))" }}>
        <polygon points={`4,30 38,46 38,${46 + bh} 4,${30 + bh}`} fill={bin.front} />
        <polygon points={`38,46 72,30 72,${30 + bh} 38,${46 + bh}`} fill={bin.side} />
        <polygon points="4,30 38,14 72,30 38,46" fill={bin.rim} stroke={active ? "#D4EF82" : "none"} strokeWidth="2" strokeLinejoin="round" />
        <polygon points="15,30 38,19 61,30 38,41" fill="#122E25" opacity="0.92" />
      </svg>
      <Recycle className="absolute h-[18%] w-[18%] -translate-x-1/2 -translate-y-1/2" style={{ left: "26%", top: `${iconTop}%`, color: bin.id === "trash" || bin.id === "plastic" ? "rgba(255,255,255,.65)" : "rgba(23,61,51,.55)" }} />
    </div>
  );
}

/** Small flat bin used on category cards. */
export function MiniBin({ bin, selected = false }: { bin: BinStyle; selected?: boolean }) {
  return (
    <div className="mx-auto flex h-12 w-9 flex-col items-center" aria-hidden>
      <div className="h-1.5 w-8 rounded-sm" style={{ background: selected ? "#123C31" : bin.rim === "#D4EF82" ? "#173D33" : "#DCE7DF" }} />
      <div className="mt-px flex h-10 w-7 items-center justify-center rounded-b-md" style={{ background: bin.id === "plastic" ? "#D6E4BF" : "#E7EDE4" }}>
        <Recycle className="h-3.5 w-3.5 text-forest/70" />
      </div>
    </div>
  );
}
