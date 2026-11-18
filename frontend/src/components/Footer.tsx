import { Link } from "react-router-dom";
import { Recycle } from "lucide-react";

export function Footer() {
  return (
    <footer className="mx-auto mt-16 max-w-[1200px] border-t border-line px-5 py-8 md:px-8">
      <div className="flex flex-col gap-4 text-xs text-muted md:flex-row md:items-center md:justify-between">
        <div className="flex items-center gap-2 text-sm font-semibold text-forest"><Recycle className="h-4 w-4" aria-hidden /> EcoSort AI</div>
        <p>A little intelligence. A lighter footprint.</p>
        <div className="flex gap-5">
          <Link to="/methodology" className="hover:text-forest">Methodology</Link>
          <span>University AI semester project</span>
        </div>
      </div>
    </footer>
  );
}

export function PageNote({ children }: { children: React.ReactNode }) {
  return <p className="mx-auto mt-8 max-w-[1200px] px-5 text-xs text-muted md:px-8">{children}</p>;
}
