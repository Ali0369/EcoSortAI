import { useState } from "react";
import { Link, NavLink } from "react-router-dom";
import { ArrowUpRight, Menu, Recycle, X } from "lucide-react";

const NAV = [
  { to: "/scanner", label: "Scanner" },
  { to: "/models", label: "Model comparison" },
  { to: "/explain", label: "Explainable AI" },
  { to: "/methodology", label: "Methodology" },
];

export function Logo() {
  return (
    <Link to="/" className="flex items-center gap-2 text-forest" aria-label="EcoSort AI home">
      <Recycle className="h-5 w-5" aria-hidden />
      <span className="text-[17px] font-semibold tracking-tight">EcoSort</span>
      <span className="rounded-md bg-lime px-1.5 py-0.5 text-[10px] font-semibold leading-none">AI</span>
    </Link>
  );
}

export function Header() {
  const [open, setOpen] = useState(false);
  return (
    <header className="sticky top-0 z-40 border-b border-line bg-cream/90 backdrop-blur">
      <div className="mx-auto flex h-16 max-w-[1200px] items-center justify-between px-5 md:px-8">
        <Logo />
        <nav className="hidden items-center gap-1 md:flex" aria-label="Main">
          {NAV.map((n) => (
            <NavLink key={n.to} to={n.to}
              className={({ isActive }) => `rounded-lg px-3 py-1.5 text-[13px] transition-colors ${isActive ? "bg-sage font-medium text-forest" : "text-muted hover:text-forest"}`}>
              {n.label}
            </NavLink>
          ))}
        </nav>
        <Link to="/scanner" className="btn-primary hidden !py-2.5 md:inline-flex">Start Sorting <ArrowUpRight className="h-3.5 w-3.5" /></Link>
        <button className="rounded-lg p-2 md:hidden" onClick={() => setOpen((o) => !o)} aria-expanded={open} aria-label="Toggle menu">
          {open ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
        </button>
      </div>
      {open && (
        <nav className="border-t border-line bg-cream px-5 py-3 md:hidden" aria-label="Mobile">
          {NAV.map((n) => (
            <NavLink key={n.to} to={n.to} onClick={() => setOpen(false)}
              className={({ isActive }) => `block rounded-lg px-3 py-2.5 text-sm ${isActive ? "bg-sage font-medium" : "text-muted"}`}>
              {n.label}
            </NavLink>
          ))}
          <Link to="/scanner" onClick={() => setOpen(false)} className="btn-primary mt-2 w-full">Start Sorting</Link>
        </nav>
      )}
    </header>
  );
}
