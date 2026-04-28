import { NextResponse } from "next/server";

import { buildBackendUrl, parseBackendError } from "@/lib/backend";

export async function POST(request: Request) {
  const incomingFormData = await request.formData();
  const image = incomingFormData.get("image");

  if (!(image instanceof File)) {
    return NextResponse.json({ detail: "Field 'image' is required." }, { status: 400 });
  }

  const formData = new FormData();
  formData.set("image", image);

  try {
    const response = await fetch(buildBackendUrl("/detect"), {
      method: "POST",
      body: formData,
      cache: "no-store",
    });

    if (!response.ok) {
      const detail = await parseBackendError(response);
      return NextResponse.json({ detail }, { status: response.status });
    }

    const payload = await response.json();
    return NextResponse.json(payload);
  } catch {
    return NextResponse.json(
      { detail: "Unable to reach backend /detect endpoint." },
      { status: 502 },
    );
  }
}
