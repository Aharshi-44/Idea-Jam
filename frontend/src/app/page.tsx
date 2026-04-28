"use client";

import { useEffect, useMemo, useState } from "react";

type DetectResult = { detected: boolean; confidence: number };

function toObjectUrl(blob: Blob) {
  return URL.createObjectURL(blob);
}

export default function Home() {
  const [embedFile, setEmbedFile] = useState<File | null>(null);
  const [detectFile, setDetectFile] = useState<File | null>(null);
  const [embedPreviewUrl, setEmbedPreviewUrl] = useState<string | null>(null);
  const [generatedPreviewUrl, setGeneratedPreviewUrl] = useState<string | null>(null);
  const [detectResult, setDetectResult] = useState<DetectResult | null>(null);
  const [healthStatus, setHealthStatus] = useState("Checking backend health...");
  const [prompt, setPrompt] = useState("A futuristic city skyline at sunrise");
  const [pendingAction, setPendingAction] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const busy = useMemo(() => pendingAction !== null, [pendingAction]);

  useEffect(() => {
    let disposed = false;
    (async () => {
      try {
        const response = await fetch("/api/health");
        const payload = (await response.json()) as { status?: string; detail?: string };
        if (!disposed && response.ok && payload.status === "ok") {
          setHealthStatus("Backend connected");
          return;
        }
        if (!disposed) {
          setHealthStatus(payload.detail ?? "Backend unavailable");
        }
      } catch {
        if (!disposed) {
          setHealthStatus("Backend unavailable");
        }
      }
    })();
    return () => {
      disposed = true;
    };
  }, []);

  const embedSourcePreviewUrl = useMemo(() => {
    if (!embedFile) return null;
    return URL.createObjectURL(embedFile);
  }, [embedFile]);

  const detectSourcePreviewUrl = useMemo(() => {
    if (!detectFile) return null;
    return URL.createObjectURL(detectFile);
  }, [detectFile]);

  useEffect(() => {
    return () => {
      if (embedSourcePreviewUrl) {
        URL.revokeObjectURL(embedSourcePreviewUrl);
      }
    };
  }, [embedSourcePreviewUrl]);

  useEffect(() => {
    return () => {
      if (detectSourcePreviewUrl) {
        URL.revokeObjectURL(detectSourcePreviewUrl);
      }
    };
  }, [detectSourcePreviewUrl]);

  useEffect(() => {
    return () => {
      if (generatedPreviewUrl) {
        URL.revokeObjectURL(generatedPreviewUrl);
      }
    };
  }, [generatedPreviewUrl]);

  useEffect(() => {
    return () => {
      if (embedPreviewUrl) {
        URL.revokeObjectURL(embedPreviewUrl);
      }
    };
  }, [embedPreviewUrl]);

  async function handleDetect() {
    if (!detectFile) {
      setError("Please upload an image first.");
      return;
    }

    setPendingAction("Detecting watermark");
    setError(null);
    try {
      const formData = new FormData();
      formData.set("image", detectFile);
      const response = await fetch("/api/detect", { method: "POST", body: formData });
      const payload = (await response.json()) as DetectResult & { detail?: string };
      if (!response.ok) {
        throw new Error(payload.detail ?? "Detection failed.");
      }
      setDetectResult({
        detected: Boolean(payload.detected),
        confidence: Number(payload.confidence),
      });
    } catch (actionError) {
      setError(actionError instanceof Error ? actionError.message : "Unexpected error.");
    } finally {
      setPendingAction(null);
    }
  }

  async function handleGenerate() {
    if (!prompt.trim()) {
      setError("Prompt cannot be empty.");
      return;
    }

    setPendingAction("Generating watermarked image");
    setError(null);
    try {
      const response = await fetch("/api/generate", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ prompt }),
      });

      if (!response.ok) {
        const payload = (await response.json().catch(() => ({}))) as { detail?: string };
        throw new Error(payload.detail ?? "Generation failed.");
      }

      const blob = await response.blob();
      if (generatedPreviewUrl) URL.revokeObjectURL(generatedPreviewUrl);
      setGeneratedPreviewUrl(toObjectUrl(blob));
    } catch (actionError) {
      setError(actionError instanceof Error ? actionError.message : "Unexpected error.");
    } finally {
      setPendingAction(null);
    }
  }

  async function handleEmbed() {
    if (!embedFile) {
      setError("Please upload an image first.");
      return;
    }

    setPendingAction("Embedding watermark");
    setError(null);
    try {
      const formData = new FormData();
      formData.set("image", embedFile);

      const response = await fetch("/api/embed", { method: "POST", body: formData });
      if (!response.ok) {
        const payload = (await response.json().catch(() => ({}))) as { detail?: string };
        throw new Error(payload.detail ?? "Embedding failed.");
      }

      const blob = await response.blob();
      if (embedPreviewUrl) URL.revokeObjectURL(embedPreviewUrl);
      setEmbedPreviewUrl(toObjectUrl(blob));
    } catch (actionError) {
      setError(actionError instanceof Error ? actionError.message : "Unexpected error.");
    } finally {
      setPendingAction(null);
    }
  }

  return (
    <div className="mx-auto w-full max-w-7xl px-6 py-10">
      <main className="space-y-8">
        <section className="overflow-hidden rounded-[2rem] border border-white/60 bg-white/85 p-8 shadow-[0_20px_80px_rgba(15,23,42,0.12)] backdrop-blur">
          <div className="flex flex-col gap-8 lg:flex-row lg:items-end lg:justify-between">
            <div className="max-w-3xl">
              <div className="inline-flex items-center rounded-full border border-indigo-200 bg-indigo-50 px-3 py-1 text-xs font-semibold uppercase tracking-[0.22em] text-indigo-700">
                AI image generation with built-in watermarking
              </div>
              <h1 className="mt-4 text-4xl font-semibold tracking-tight text-zinc-950 sm:text-5xl">
                SEMANTIC AWARE WATERMARKING (SAW)
              </h1>
              <p className="mt-4 max-w-2xl text-base leading-7 text-zinc-600">One liner will be added here</p>
            </div>
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
              <StatusCard label="Generate" value="Watermarked by default" />
              <StatusCard label="Detect" value="Confidence-based scan" />
              <StatusCard label="Backend" value={healthStatus} />
            </div>
          </div>
        </section>

        <section className="overflow-hidden rounded-[2rem] border border-indigo-100 bg-gradient-to-br from-indigo-50 via-white to-cyan-50 p-6 shadow-[0_20px_60px_rgba(79,70,229,0.10)]">
          <div className="flex flex-col gap-5 lg:flex-row lg:items-end lg:justify-between">
            <div className="max-w-2xl">
              <SectionEyebrow>Primary Flow</SectionEyebrow>
              <h2 className="mt-2 text-2xl font-semibold text-zinc-950">Generate a Watermarked Image</h2>
              <p className="mt-2 text-sm leading-6 text-zinc-600">
                Enter a prompt and the backend will generate an image with the watermark embedded before it is
                returned.
              </p>
            </div>
            <div className="rounded-2xl border border-white/70 bg-white/80 px-4 py-3 text-sm text-zinc-600 shadow-sm">
              Invisible watermarking is applied automatically to generated outputs.
            </div>
          </div>
          <div className="mt-5 flex flex-col gap-3 md:flex-row">
            <input
              value={prompt}
              onChange={(event) => setPrompt(event.target.value)}
              placeholder="Enter a prompt to search"
              className="w-full rounded-2xl border border-white bg-white px-5 py-4 text-sm text-black shadow-sm outline-none ring-0 transition focus:border-indigo-300 focus:shadow-[0_0_0_4px_rgba(99,102,241,0.12)]"
            />
            <button
              type="button"
              disabled={busy}
              onClick={handleGenerate}
              className="rounded-2xl bg-zinc-950 px-6 py-4 text-sm font-medium text-white shadow-lg shadow-zinc-950/15 transition hover:bg-zinc-800 disabled:opacity-60"
            >
              Generate Watermarked Image
            </button>
          </div>
          {generatedPreviewUrl && (
            <div className="mt-6">
              <div className="mb-3 flex justify-end">
                <a
                  href={generatedPreviewUrl}
                  download="saw-generated-watermarked.png"
                  className="rounded-xl bg-emerald-600 px-4 py-2 text-sm font-medium text-white shadow-md shadow-emerald-600/20 transition hover:bg-emerald-500"
                >
                  Download Image
                </a>
              </div>
              <ImageCard
                title="Generated Watermarked Output"
                src={generatedPreviewUrl}
                alt="Generated watermarked image preview"
              />
            </div>
          )}
        </section>

        <section className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          <div className="rounded-[2rem] border border-white/70 bg-white/90 p-6 shadow-[0_16px_50px_rgba(15,23,42,0.08)] backdrop-blur">
            <SectionEyebrow>Manual Watermarking</SectionEyebrow>
            <h2 className="mt-2 text-2xl font-semibold text-zinc-950">Embed Watermark in Uploaded Image</h2>
            <p className="mt-2 text-sm leading-6 text-zinc-600">
              Upload any image and embed the watermark manually when you want to test your own input.
            </p>
            <label htmlFor="embed-image" className="mt-5 block text-sm font-medium text-zinc-700">
              Upload image
            </label>
            <input
              id="embed-image"
              type="file"
              accept="image/*"
              className="mt-2 block w-full rounded-2xl border border-dashed border-zinc-300 bg-zinc-50 px-3 py-3 text-sm text-black file:mr-3 file:rounded-xl file:border-0 file:bg-zinc-950 file:px-4 file:py-2.5 file:text-white"
              onChange={(event) => {
                setEmbedFile(event.target.files?.[0] ?? null);
                if (embedPreviewUrl) {
                  URL.revokeObjectURL(embedPreviewUrl);
                  setEmbedPreviewUrl(null);
                }
              }}
            />
            <button
              type="button"
              disabled={busy}
              onClick={handleEmbed}
              className="mt-4 rounded-2xl bg-indigo-600 px-5 py-3 text-sm font-medium text-white shadow-lg shadow-indigo-600/20 transition hover:bg-indigo-500 disabled:opacity-60"
            >
              Embed Watermark
            </button>
            <div className="mt-5 grid grid-cols-1 gap-4">
              {embedSourcePreviewUrl && (
                <ImageCard title="Uploaded Source" src={embedSourcePreviewUrl} alt="Uploaded source preview" />
              )}
              {embedPreviewUrl && (
                <ImageCard title="Embedded Output" src={embedPreviewUrl} alt="Embedded image preview" />
              )}
            </div>
          </div>

          <div className="rounded-[2rem] border border-white/70 bg-white/90 p-6 shadow-[0_16px_50px_rgba(15,23,42,0.08)] backdrop-blur">
            <SectionEyebrow>Verification</SectionEyebrow>
            <h2 className="mt-2 text-2xl font-semibold text-zinc-950">Detect Watermark</h2>
            <p className="mt-2 text-sm leading-6 text-zinc-600">
            Enter a prompt and the backend will generate an image with the watermark embedded before it is
              Upload an image to check whether the watermark is present and see the confidence score.
            </p>
            <label htmlFor="detect-image" className="mt-5 block text-sm font-medium text-zinc-700">
              Upload image
            </label>
            <input
              id="detect-image"
              type="file"
              accept="image/*"
              className="mt-2 block w-full rounded-2xl border border-dashed border-zinc-300 bg-zinc-50 px-3 py-3 text-sm text-black file:mr-3 file:rounded-xl file:border-0 file:bg-zinc-800 file:px-4 file:py-2.5 file:text-white"
              onChange={(event) => {
                setDetectFile(event.target.files?.[0] ?? null);
                setDetectResult(null);
              }}
            />
            <button
              type="button"
              disabled={busy}
              onClick={handleDetect}
              className="mt-4 rounded-2xl bg-cyan-600 px-5 py-3 text-sm font-medium text-white shadow-lg shadow-cyan-600/20 transition hover:bg-cyan-500 disabled:opacity-60"
            >
              Detect Watermark
            </button>
            {detectResult && (
              <div className="mt-4 rounded-2xl border border-emerald-100 bg-emerald-50/70 p-4 text-sm text-zinc-700">
                <p>
                  Result:{" "}
                  <span className="font-semibold">{detectResult.detected ? "Watermark detected" : "Not detected"}</span>
                </p>
                <p className="mt-1">Confidence: {detectResult.confidence.toFixed(4)}</p>
              </div>
            )}
            <div className="mt-5">
              {detectSourcePreviewUrl && (
                <ImageCard title="Detection Input" src={detectSourcePreviewUrl} alt="Detection input preview" />
              )}
            </div>
          </div>
        </section>

        {(pendingAction || error) && (
          <section className="rounded-[1.5rem] border border-white/70 bg-white/90 p-5 shadow-[0_12px_40px_rgba(15,23,42,0.08)]">
            {pendingAction && <p className="text-sm font-medium text-zinc-700">{pendingAction}...</p>}
            {error && <p className="mt-2 text-sm text-red-600">{error}</p>}
          </section>
        )}
      </main>
    </div>
  );
}

function SectionEyebrow({ children }: { children: React.ReactNode }) {
  return (
    <p className="text-xs font-semibold uppercase tracking-[0.22em] text-zinc-500">
      {children}
    </p>
  );
}

function StatusCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-2xl border border-white/70 bg-white/90 px-4 py-3 shadow-sm">
      <p className="text-xs font-semibold uppercase tracking-[0.18em] text-zinc-500">{label}</p>
      <p className="mt-2 text-sm font-medium text-zinc-900">{value}</p>
    </div>
  );
}

function ImageCard({ title, src, alt }: { title: string; src: string; alt: string }) {
  return (
    <div className="overflow-hidden rounded-[1.5rem] border border-white/70 bg-white shadow-[0_16px_40px_rgba(15,23,42,0.08)]">
      <div className="border-b border-zinc-100 bg-zinc-50/80 px-4 py-3 text-sm font-medium text-zinc-800">{title}</div>
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src={src} alt={alt} className="h-auto w-full bg-zinc-50 object-contain" />
    </div>
  );
}
