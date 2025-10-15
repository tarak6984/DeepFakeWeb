import { NextResponse } from "next/server";
import { getServerSession } from "next-auth/next";
import { authOptions } from "@/lib/auth";
import { checkServerUsage, getServerUsageStats } from "@/lib/server-usage-tracker";

export async function GET(req: Request) {
  try {
    // Check authentication status
    const session = await getServerSession(authOptions);
    const isAuthenticated = !!session?.user;
    
    // Get usage status
    const usageCheck = checkServerUsage(req, isAuthenticated);
    
    // Get server stats (for debugging/monitoring)
    const serverStats = getServerUsageStats();
    
    return NextResponse.json({
      success: true,
      usage: {
        allowed: usageCheck.allowed,
        remaining: usageCheck.remaining,
        used: isAuthenticated ? -1 : Math.max(0, 3 - usageCheck.remaining),
        limit: isAuthenticated ? -1 : 3,
        resetTime: usageCheck.resetTime.toISOString(),
        isAuthenticated: usageCheck.isAuthenticated,
      },
      user: isAuthenticated ? {
        id: session?.user?.id,
        name: session?.user?.name,
        email: session?.user?.email,
        plan: session?.user?.plan || 'FREE',
      } : null,
      // Server stats (only for authenticated users or in development)
      ...(isAuthenticated || process.env.NODE_ENV === 'development' ? {
        serverStats
      } : {}),
    });
    
  } catch (error) {
    console.error('Usage status error:', error);
    return NextResponse.json(
      { 
        error: "Failed to get usage status",
        success: false 
      }, 
      { status: 500 }
    );
  }
}