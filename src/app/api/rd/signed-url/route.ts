import { NextResponse } from "next/server";

export async function POST(req: Request) {
  try {
    const { fileName } = await req.json();

    if (!fileName || typeof fileName !== "string") {
      return NextResponse.json({ error: "Invalid fileName" }, { status: 400 });
    }

    // Local API doesn't need signed URLs - return direct upload endpoint
    const localApiUrl = process.env.NEXT_PUBLIC_LOCAL_API_URL || 'http://localhost:8000';
    
    return NextResponse.json({
      message: "Local API - upload directly to /api/upload",
      upload_url: `${localApiUrl}/api/upload`,
      fileName: fileName,
      // For compatibility with existing frontend code
      success: true,
      api_type: "local"
    });
    
  } catch (err: unknown) {
    return NextResponse.json({ 
      error: err instanceof Error ? err.message : "Unexpected error" 
    }, { status: 500 });
  }
}
