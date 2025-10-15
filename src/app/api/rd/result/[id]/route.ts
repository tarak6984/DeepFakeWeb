import { NextResponse } from "next/server";

export async function GET(_req: Request, context: { params: Promise<{ id: string }> }) {
  try {
    const { id } = await context.params || { id: '' };

    if (!id) {
      return NextResponse.json({ error: "Missing id" }, { status: 400 });
    }

    const localApiUrl = process.env.NEXT_PUBLIC_LOCAL_API_URL || 'http://localhost:8000';
    
    // Forward request to local deepfake detection API
    const upstream = await fetch(`${localApiUrl}/api/result/${id}`, {
      method: "GET",
      cache: "no-store",
    });

    const data = await upstream.json().catch(() => ({}));

    if (!upstream.ok) {
      return NextResponse.json(
        { error: "Local API error", status: upstream.status, response: data },
        { status: upstream.status }
      );
    }

    // Transform the response for frontend compatibility
    const transformedData = {
      ...data,
      // Ensure compatibility with existing frontend
      status: data.status || 'completed',
      api_type: 'local',
      // Map our format to expected frontend fields
      result: {
        prediction: data.prediction,
        confidence: data.confidence,
        fake_confidence: data.fake_confidence,
        real_confidence: data.real_confidence,
        media_type: data.media_type,
        models_used: data.models_used,
        device_used: data.device_used
      }
    };

    return NextResponse.json(transformedData);
  } catch (err: unknown) {
    return NextResponse.json({ 
      error: err instanceof Error ? err.message : "Unexpected error" 
    }, { status: 500 });
  }
}
