import { NextRequest, NextResponse } from 'next/server';
import { getServerSession } from 'next-auth';
import { authOptions } from '@/lib/auth';
import { databaseService } from '@/lib/database-service';

export async function GET(request: NextRequest) {
  try {
    // Check authentication and admin role
    const session = await getServerSession(authOptions);
    if (!session?.user) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    // Check for admin role (temporarily disabled for testing)
    // if (session.user.role !== 'ADMIN' && session.user.role !== 'SUPER_ADMIN') {
    //   return NextResponse.json({ error: 'Forbidden: Admin access required' }, { status: 403 });
    // }

    // Get system statistics
    const systemStats = await databaseService.getSystemStats();
    
    // Get detailed analytics
    const analytics = await databaseService.getSystemAnalytics();
    
    // Get performance metrics
    const performance = {
      uptime: process.uptime(),
      memoryUsage: process.memoryUsage(),
      nodeVersion: process.version,
      platform: process.platform,
      arch: process.arch,
    };

    // Get recent activities
    const recentActivities = await databaseService.getRecentSystemActivities();

    // Calculate growth rates
    const growthRates = await databaseService.getGrowthRates();

    return NextResponse.json({
      systemStats,
      analytics,
      performance,
      recentActivities,
      growthRates,
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    console.error('Admin system stats API error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}

export async function POST(request: NextRequest) {
  try {
    // Check authentication and admin role
    const session = await getServerSession(authOptions);
    if (!session?.user) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    // if (session.user.role !== 'ADMIN' && session.user.role !== 'SUPER_ADMIN') {
    //   return NextResponse.json({ error: 'Forbidden: Admin access required' }, { status: 403 });
    // }

    const { action, data } = await request.json();

    if (!action) {
      return NextResponse.json({ error: 'Missing action' }, { status: 400 });
    }

    let result;
    switch (action) {
      case 'cleanupOldData':
        // Clean up old analyses and data
        result = await databaseService.cleanupOldData(data?.days || 30);
        break;
      
      case 'optimizeDatabase':
        // Optimize database performance
        result = await databaseService.optimizeDatabase();
        break;
      
      case 'generateReport':
        // Generate system report
        result = await databaseService.generateSystemReport(data?.type || 'full');
        break;
      
      case 'exportData':
        // Export system data
        result = await databaseService.exportSystemData(data?.format || 'json');
        break;

      default:
        return NextResponse.json({ error: 'Invalid action' }, { status: 400 });
    }

    return NextResponse.json({ success: true, data: result });
  } catch (error) {
    console.error('Admin system action error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}