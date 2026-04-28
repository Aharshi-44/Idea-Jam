import { NextResponse } from "next/server";

import { buildBackendUrl, parseBackendError } from "@/lib/backend";

export async function POST(request: Request) {
  const contentType = request.headers.get("content-type") ?? "";
  const backendUrl = buildBackendUrl("/generate");

  try {
    if (contentType.includes("application/json")) {
      const body = (await request.json()) as { prompt?: string };
      const response = await fetch(backendUrl, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ prompt: body.prompt ?? "" }),
        cache: "no-store",
      });

      if (!response.ok) {
        const detail = await parseBackendError(response);
        return NextResponse.json({ detail }, { status: response.status });
      }

      const bytes = await response.arrayBuffer();
      return new NextResponse(bytes, {
        status: response.status,
        headers: { "content-type": response.headers.get("content-type") ?? "image/png" },
      });
    }

    const incomingFormData = await request.formData();
    const prompt = incomingFormData.get("prompt");
    if (typeof prompt !== "string") {
      return NextResponse.json({ detail: "Field 'prompt' is required." }, { status: 400 });
    }

    const formData = new FormData();
    formData.set("prompt", prompt);

    const response = await fetch(backendUrl, {
      method: "POST",
      body: formData,
      cache: "no-store",
    });

    if (!response.ok) {
      const detail = await parseBackendError(response);
      return NextResponse.json({ detail }, { status: response.status });
    }

    const bytes = await response.arrayBuffer();
    return new NextResponse(bytes, {
      status: response.status,
      headers: { "content-type": response.headers.get("content-type") ?? "image/png" },
    });
  } catch {
    return NextResponse.json(
      { detail: "Unable to reach backend /generate endpoint." },
      { status: 502 },
    );
  }
}
