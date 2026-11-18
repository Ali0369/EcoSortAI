/** Display configuration for the six bins (colors sampled from the Figma design). Order = design. */
export interface BinStyle { id: string; label: string; front: string; side: string; rim: string; height: number; recyclable: boolean }

export const BINS: Record<string, BinStyle> = {
  plastic:   { id: "plastic",   label: "Plastic",   front: "#204739", side: "#122E25", rim: "#D4EF82", height: 0.86, recyclable: true },
  paper:     { id: "paper",     label: "Paper",     front: "#A5B4A0", side: "#748B79", rim: "#DCE7DF", height: 0.86, recyclable: true },
  cardboard: { id: "cardboard", label: "Cardboard", front: "#D3C4A6", side: "#A99A7F", rim: "#ECE2D1", height: 1,    recyclable: true },
  glass:     { id: "glass",     label: "Glass",     front: "#B6C9CB", side: "#80989C", rim: "#DFE9ED", height: 1,    recyclable: true },
  metal:     { id: "metal",     label: "Metal",     front: "#D9DDD8", side: "#929F9B", rim: "#E5E8E5", height: 0.9,  recyclable: true },
  trash:     { id: "trash",     label: "Trash",     front: "#48534F", side: "#272F2D", rim: "#BCCDC2", height: 0.9,  recyclable: false },
};
/** Bin order in the sorting scene (Figma "station"). */
export const STATION_ORDER = ["paper", "plastic", "glass", "cardboard", "metal", "trash"];
/** Order of the category cards (Figma landing/scanner). */
export const CARD_ORDER = ["plastic", "paper", "cardboard", "glass", "metal", "trash"];

export const pct = (x: number, digits = 1) => `${(x * 100).toFixed(digits)}%`;
export const label = (id: string) => BINS[id]?.label ?? id.charAt(0).toUpperCase() + id.slice(1);
