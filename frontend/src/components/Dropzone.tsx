import { useRef, useState } from "react";
import { ArrowUp, Upload } from "lucide-react";

export const MAX_MB = 10;
const TYPES = ["image/jpeg", "image/png", "image/webp"];

/** Client-side check (the backend validates again - never trust the browser alone). */
export function validateFile(f: File): string | null {
  if (!TYPES.includes(f.type)) return "Use a JPG, PNG or WebP image.";
  if (f.size > MAX_MB * 1024 * 1024) return `That image is larger than ${MAX_MB} MB.`;
  if (f.size === 0) return "That file is empty.";
  return null;
}

export function Dropzone({ onFile, error }: { onFile: (f: File) => void; error?: string | null }) {
  const input = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);
  const pick = (files: FileList | null) => { if (files && files[0]) onFile(files[0]); };
  return (
    <div
      onDragOver={(e) => { e.preventDefault(); setOver(true); }} onDragLeave={() => setOver(false)}
      onDrop={(e) => { e.preventDefault(); setOver(false); pick(e.dataTransfer.files); }}
      className={`flex h-full min-h-[300px] flex-col items-center justify-center rounded-2xl border-2 border-dashed px-6 py-10 text-center transition-colors ${over ? "border-forest bg-leaf/40" : "border-mist bg-station/60"}`}>
      <div className="mb-5 flex h-14 w-14 items-center justify-center rounded-2xl bg-sage"><ArrowUp className="h-6 w-6" aria-hidden /></div>
      <h2 className="text-xl font-semibold">Drop your waste photo here</h2>
      <p className="mt-1 text-xs text-muted">or choose an image from your device</p>
      <input ref={input} type="file" accept="image/jpeg,image/png,image/webp" className="sr-only" id="file-input" onChange={(e) => { pick(e.target.files); e.target.value = ""; }} />
      <button className="btn-primary mt-6" onClick={() => input.current?.click()}><Upload className="h-4 w-4" /> Upload image</button>
      <p className="mt-5 text-[11px] text-muted">JPG, PNG or WebP · Up to {MAX_MB} MB</p>
      <p className="mt-1 text-[10px] text-muted">Your photo is only used for this classification.</p>
      {error && <p role="alert" className="mt-4 rounded-lg bg-red-50 px-3 py-2 text-xs text-red-800">{error}</p>}
    </div>
  );
}
