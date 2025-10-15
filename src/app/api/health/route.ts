import { NextResponse } from 'next/server';
import { prisma } from '@/lib/prisma';

export async function GET() {
  try {
    // Check database connectivity
    await prisma.$queryRaw`SELECT 1`;
    
    // Check if Python API is accessible
    const pythonApiUrl = process.env.NEXT_PUBLIC_LOCAL_API_URL || 'http://localhost:8000';
    let pythonApiStatus = 'unknown';
    
    try {
      const response = await fetch(`${pythonApiUrl}/docs`, { 
        method: 'HEAD',
        signal: AbortSignal.timeout(5000) // 5 second timeout
      });
      pythonApiStatus = response.ok ? 'healthy' : 'unhealthy';
    } catch (error) {
      pythonApiStatus = 'unreachable';
    }

    return NextResponse.json({
      status: 'healthy',
      timestamp: new Date().toISOString(),
      services: {
        database: 'connected',
        pythonApi: pythonApiStatus,
        nextjs: 'running'
      },
      uptime: process.uptime(),
      version: '1.0.0'
    });
  } catch (error) {
    console.error('Health check failed:', error);
    return NextResponse.json(
      {
        status: 'unhealthy',
        timestamp: new Date().toISOString(),
        error: 'Database connection failed',
        services: {
          database: 'disconnected',
          pythonApi: 'unknown',
          nextjs: 'running'
        }
      },
      { status: 503 }
    );
  }
}