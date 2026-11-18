import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { api } from "../api/client";
import type { CategoryInfo, ModelInfo, Prediction } from "../api/types";

interface ScanState {
  file: File | null;
  previewUrl: string | null;
  prediction: Prediction | null;
  models: ModelInfo[];
  modelsError: string | null;
  modelId: string | undefined;
  categories: Record<string, CategoryInfo>;
  disclaimer: string;
  setModelId: (id: string) => void;
  setFile: (f: File | null) => void;
  setPrediction: (p: Prediction | null) => void;
  reset: () => void;
}

const Ctx = createContext<ScanState | null>(null);

export function ScanProvider({ children }: { children: ReactNode }) {
  const [file, setFileState] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [modelsError, setModelsError] = useState<string | null>(null);
  const [modelId, setModelId] = useState<string | undefined>(undefined);
  const [categories, setCategories] = useState<Record<string, CategoryInfo>>({});
  const [disclaimer, setDisclaimer] = useState("Recycling rules vary by location. Always check your local collection guidelines.");

  useEffect(() => {
    api.models().then((m) => {
      setModels(m);
      setModelId((cur) => cur ?? m.find((x) => x.is_default)?.id ?? m[0]?.id);
    }).catch((e: Error) => setModelsError(e.message));
    api.categories().then((c) => {
      setCategories(Object.fromEntries(c.categories.map((x) => [x.id, x])));
      setDisclaimer(c.disclaimer);
    }).catch(() => { /* guidance falls back to generic text */ });
  }, []);

  const urlRef = useRef<string | null>(null);
  const setFile = useCallback((f: File | null) => {
    if (urlRef.current) URL.revokeObjectURL(urlRef.current);
    urlRef.current = f ? URL.createObjectURL(f) : null;
    setPreviewUrl(urlRef.current);
    setFileState(f);
    setPrediction(null);
  }, []);
  const reset = useCallback(() => setFile(null), [setFile]);

  const value = useMemo(() => ({ file, previewUrl, prediction, models, modelsError, modelId, categories, disclaimer, setModelId, setFile, setPrediction, reset }),
    [file, previewUrl, prediction, models, modelsError, modelId, categories, disclaimer, setFile, reset]);
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useScan() {
  const c = useContext(Ctx);
  if (!c) throw new Error("useScan must be used inside ScanProvider");
  return c;
}
