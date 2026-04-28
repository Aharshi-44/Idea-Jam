import { NextResponse } from "next/server";

import { buildBackendUrl } from "@/lib/backend";

export async function GET() {
  try {
    const response = await fetch(buildBackendUrl("/health"), {
      method: "GET",
      cache: "no-store",
    });

    const text = await response.text();
    return new NextResponse(text, {
      status: response.status,
      headers: { "content-type": response.headers.get("content-type") ?? "application/json" },
    });
  } catch {
    return NextResponse.json(
      { status: "error", detail: "Unable to reach backend /health endpoint." },
      { status: 502 },
    );
  }
}
