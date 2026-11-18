import { useLayoutEffect, useRef, useState } from "react";
import { motion, useReducedMotion } from "framer-motion";
import { Check } from "lucide-react";
import { BINS, label } from "../lib/bins";
import { IsoBin } from "./IsoBin";

export type StationPhase = "idle" | "analyzing" | "sorting" | "sorted" | "uncertain";

interface Props {
  order: string[];
  phase: StationPhase;
  target?: string | null;                 // REAL predicted class (from the API)
  imageUrl?: string | null;
  uncertainPair?: [string, string] | null;
  statusLabel?: string;
  onSorted?: () => void;                  // called when the thumbnail has landed in the bin
}

interface Geom { w: number; h: number; targets: Record<string, { cx: number; top: number }> }

/**
 * The interactive sorting scene.  The thumbnail flies from the hover position into the bin whose id
 * equals `target` - the class returned by the backend.  With "reduce motion" enabled the flight is skipped.
 */
export function BinStation({ order, phase, target, imageUrl, uncertainPair, statusLabel, onSorted }: Props) {
  const reduced = useReducedMotion();
  const boxRef = useRef<HTMLDivElement>(null);
  const binRefs = useRef<Record<string, HTMLDivElement | null>>({});
  const [geom, setGeom] = useState<Geom>({ w: 0, h: 0, targets: {} });
  const doneFor = useRef<string | null>(null);

  useLayoutEffect(() => {
    const measure = () => {
      const box = boxRef.current;
      if (!box) return;
      const b = box.getBoundingClientRect();
      const targets: Geom["targets"] = {};
      for (const id of order) {
        const el = binRefs.current[id];
        if (el) { const r = el.getBoundingClientRect(); targets[id] = { cx: r.left - b.left + r.width / 2, top: r.top - b.top }; }
      }
      setGeom({ w: b.width, h: b.height, targets });
    };
    measure();
    const ro = new ResizeObserver(measure);
    if (boxRef.current) ro.observe(boxRef.current);
    return () => ro.disconnect();
  }, [order]);

  const size = Math.max(56, Math.min(92, geom.w * 0.14));
  const hover = { x: geom.w / 2 - size / 2, y: geom.h * 0.03, scale: 1, opacity: 1 };
  const t = target ? geom.targets[target] : undefined;
  const intoBin = t ? { x: t.cx - size / 2, y: t.top - size * 0.2, scale: 0.3, opacity: 0 } : hover;
  const sorting = phase === "sorting" && !!t;
  const showThumb = !!imageUrl && geom.w > 0 && phase !== "idle" && phase !== "sorted";
  const targetPos = t;

  const handleDone = () => {
    if (phase === "sorting" && target && doneFor.current !== target + imageUrl) {
      doneFor.current = target + imageUrl;
      onSorted?.();
    }
  };
  // No image to animate (or animation impossible): still complete the flow immediately.
  useLayoutEffect(() => {
    if (phase === "sorting" && (!imageUrl || !t)) handleDone();
  }, [phase, imageUrl, t]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div ref={boxRef} className="relative h-full min-h-[340px] w-full overflow-hidden rounded-card bg-station" role="img"
      aria-label={target && phase === "sorted" ? `The item was sorted into the ${label(target)} bin` : "Recycling station with six bins"}>
      <div className="absolute left-5 top-4 text-[10px] font-medium tracking-wider text-muted">YOUR RECYCLING STATION</div>

      {/* platform */}
      <svg className="absolute bottom-[14%] left-[3%] w-[94%]" viewBox="0 0 400 120" preserveAspectRatio="none" aria-hidden>
        <polygon points="60,10 380,10 340,110 20,110" fill="#E6E3D8" opacity="0.9" />
        <polygon points="20,110 340,110 340,118 20,118" fill="#D9D5C8" />
      </svg>

      {/* bins */}
      <div className="absolute bottom-[19%] left-[6%] right-[6%] flex items-end justify-center gap-[1.5%]">
        {order.map((id, i) => {
          const bin = BINS[id];
          if (!bin) return null;
          const isTarget = target === id && (phase === "sorting" || phase === "sorted");
          return (
            <motion.div key={id} ref={(el) => { binRefs.current[id] = el; }} className="w-[14%]"
              animate={phase === "sorted" && target === id && !reduced ? { y: [0, -10, 0] } : { y: 0 }} transition={{ duration: 0.5, ease: "easeOut" }}
              style={{ marginBottom: `${i * 0.6}%` }}>
              <IsoBin bin={bin} active={isTarget} dim={phase === "sorted" && !!target && target !== id} />
            </motion.div>
          );
        })}
      </div>

      {/* labels */}
      <div className="absolute bottom-[6%] left-[6%] right-[6%] flex justify-center gap-[1.5%] text-[11px] text-muted">
        {order.map((id) => (
          <div key={id} className="w-[14%] text-center">
            <span className={target === id && (phase === "sorting" || phase === "sorted") ? "font-semibold text-forest" : ""}>{label(id)}</span>
            {target === id && (phase === "sorting" || phase === "sorted") && <div className="mx-auto mt-1 h-0.5 w-10 rounded bg-lime" />}
          </div>
        ))}
      </div>

      {/* "+ PLASTIC" chip over the target bin */}
      {targetPos && (phase === "sorting" || phase === "sorted") && (
        <motion.div initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: reduced ? 0 : 0.2 }}
          className="absolute -translate-x-1/2 whitespace-nowrap rounded-md bg-forest px-2.5 py-1 text-[10px] font-semibold tracking-wide text-lime"
          style={{ left: targetPos.cx, top: Math.max(8, targetPos.top - 34) }}>
          + {label(target!).toUpperCase()}
        </motion.div>
      )}

      {/* thumbnail of the uploaded image */}
      {showThumb && (
        <motion.div key={imageUrl} className="absolute left-0 top-0 overflow-hidden rounded-2xl bg-white shadow-card ring-4 ring-white"
          style={{ width: size, height: size }} initial={hover} animate={sorting ? intoBin : hover}
          transition={sorting ? { duration: reduced ? 0.01 : 1.1, ease: [0.5, 0, 0.2, 1] } : { duration: 0.35 }}
          onAnimationComplete={handleDone}>
          <img src={imageUrl!} alt="" className="h-full w-full object-cover" draggable={false} />
          {phase === "analyzing" && !reduced && (
            <motion.div className="absolute left-0 right-0 h-0.5 bg-lime shadow-[0_0_8px_#D4EF82]" initial={{ top: "5%" }} animate={{ top: ["5%", "92%", "5%"] }} transition={{ duration: 1.8, repeat: Infinity, ease: "easeInOut" }} />
          )}
        </motion.div>
      )}

      {/* status pill under the thumbnail */}
      {geom.w > 0 && (phase === "analyzing" || phase === "uncertain") && (
        <div className="absolute left-1/2 -translate-x-1/2 whitespace-nowrap rounded-full px-3 py-1 text-[11px] font-medium shadow-card"
          style={{ top: geom.h * 0.03 + size + 8, background: phase === "uncertain" ? "#FBEBD3" : "#fff", color: phase === "uncertain" ? "#9A6A1F" : "#173D33" }}>
          {phase === "uncertain" ? (uncertainPair ? `${label(uncertainPair[0])} or ${label(uncertainPair[1])}?` : "Not quite sure yet") : statusLabel ?? "Reading material…"}
        </div>
      )}

      {/* success tile */}
      {phase === "sorted" && (
        <motion.div initial={reduced ? false : { scale: 0.6, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} transition={{ type: "spring", stiffness: 260, damping: 18 }}
          className="absolute left-1/2 top-[8%] flex -translate-x-1/2 flex-col items-center gap-2">
          <div className="flex h-16 w-24 items-center justify-center rounded-2xl bg-deep shadow-card"><Check className="h-8 w-8 text-lime" strokeWidth={3} /></div>
          <div className="rounded-full bg-white px-3 py-1 text-[11px] font-medium shadow-card">Right material. Right bin.</div>
        </motion.div>
      )}
    </div>
  );
}
