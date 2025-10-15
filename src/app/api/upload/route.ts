import { NextResponse } from "next/server";
import { getServerSession } from "next-auth/next";
import { authOptions } from "@/lib/auth";
import { checkServerUsage, consumeServerUsage } from "@/lib/server-usage-tracker";

export async function POST(req: Request) {
  try {
    // Check authentication status
    const session = await getServerSession(authOptions);
    const isAuthenticated = !!session?.user;
    
    // Check usage limits for anonymous users
    const usageCheck = checkServerUsage(req, isAuthenticated);
    
    if (!usageCheck.allowed) {
      return NextResponse.json(
        { 
          error: "Free scan limit reached",
          message: "You have used all your free scans. Sign up for unlimited access!",
          remaining: 0,
          resetTime: usageCheck.resetTime.toISOString(),
          requiresAuth: true
        },
        { status: 429 } // Too Many Requests
      );
    }
    
    const localApiUrl = process.env.NEXT_PUBLIC_LOCAL_API_URL || 'http://localhost:8000';
    
    // Get the form data from the request
    const formData = await req.formData();
    
    // Forward the request to the local deepfake API
    const response = await fetch(`${localApiUrl}/api/upload`, {
      method: 'POST',
      body: formData,
    });
    
    const data = await response.json();
    
    if (!response.ok) {
      return NextResponse.json(
        { error: "Local API error", status: response.status, response: data },
        { status: response.status }
      );
    }
    
    // Consume usage for anonymous users after successful processing
    if (!isAuthenticated) {
      consumeServerUsage(req, isAuthenticated);
    }
    
    // Transform response to ensure compatibility
    const transformedData = {
      ...data,
      success: true,
      api_type: 'local',
      // Map analysis_id to id for compatibility
      id: data.analysis_id,
      message: data.message || 'File uploaded successfully to local API',
      // Include usage info in response
      usage: {
        remaining: isAuthenticated ? -1 : usageCheck.remaining - 1,
        isAuthenticated,
        resetTime: usageCheck.resetTime.toISOString()
      }
    };
    
    return NextResponse.json(transformedData);
    
  } catch (err: unknown) {
    return NextResponse.json({ 
      error: err instanceof Error ? err.message : "Upload failed",
      success: false 
    }, { status: 500 });
  }
}