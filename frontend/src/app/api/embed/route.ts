import { NextResponse } from "next/server";

import { buildBackendUrl } from "@/lib/backend";

export async function POST(request: Request) {
  const incomingFormData = await request.formData();
  const image = incomingFormData.get("image");

  if (!(image instanceof File)) {
    return NextResponse.json({ detail: "Field 'image' is required." }, { status: 400 });
  }

  const formData = new FormData();
  formData.set("image", image);

  try {
    const response = await fetch(buildBackendUrl("/embed"), {
      method: "POST",
      body: formData,
      cache: "no-store",
    });

    const bytes = await response.arrayBuffer();
    return new NextResponse(bytes, {
      status: response.status,
      headers: { "content-type": response.headers.get("content-type") ?? "image/png" },
    });
  } catch {
    return NextResponse.json(
      { detail: "Unable to reach backend /embed endpoint." },
      { status: 502 },
    );
  }
}
