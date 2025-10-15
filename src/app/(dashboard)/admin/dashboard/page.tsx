'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Separator } from '@/components/ui/separator';
import { useToast } from '@/hooks/use-toast';
import { 
  Users, 
  FileVideo, 
  Activity, 
  Server, 
  TrendingUp, 
  AlertCircle,
  RefreshCw,
  Download,
  Database,
  Trash2,
  Settings
} from 'lucide-react';
import { formatFileSize } from '@/lib/utils';

interface SystemStats {
  totalUsers: number;
  totalAnalyses: number;
  totalStorage: number;
  activeUsers: number;
  successRate: number;
  avgProcessingTime: number;
}

interface SystemAnalytics {
  dailyAnalyses: number;
  weeklyGrowth: number;
  monthlyGrowth: number;
  topRegions: Array<{ region: string; count: number }>;
  popularFeatures: Array<{ feature: string; usage: number }>;
}

interface PerformanceMetrics {
  uptime: number;
  memoryUsage: {
    rss: number;
    heapTotal: number;
    heapUsed: number;
    external: number;
  };
  nodeVersion: string;
  platform: string;
  arch: string;
}

interface RecentActivity {
  id: string;
  type: 'user_registration' | 'analysis_completed' | 'error' | 'system';
  message: string;
  timestamp: string;
  user?: string;
}

interface GrowthRates {
  userGrowth: number;
  analysisGrowth: number;
  storageGrowth: number;
}

interface SystemData {
  systemStats: SystemStats;
  analytics: SystemAnalytics;
  performance: PerformanceMetrics;
  recentActivities: RecentActivity[];
  growthRates: GrowthRates;
  timestamp: string;
}

export default function AdminDashboard() {
  const [systemData, setSystemData] = useState<SystemData | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const { toast } = useToast();

  const fetchSystemData = async () => {
    try {
      const response = await fetch('/api/admin/system');
      if (!response.ok) throw new Error('Failed to fetch system data');
      
      const data = await response.json();
      setSystemData(data);
    } catch (error) {
      console.error('Error fetching system data:', error);
      toast({
        title: 'Error',
        description: 'Failed to load system data',
        variant: 'destructive',
      });
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  const handleSystemAction = async (action: string, data?: any) => {
    try {
      setRefreshing(true);
      const response = await fetch('/api/admin/system', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action, data }),
      });

      if (!response.ok) throw new Error('Action failed');

      const result = await response.json();
      toast({
        title: 'Success',
        description: `${action} completed successfully`,
      });

      // Refresh data after action
      await fetchSystemData();
    } catch (error) {
      console.error('Error performing system action:', error);
      toast({
        title: 'Error',
        description: `Failed to perform ${action}`,
        variant: 'destructive',
      });
    } finally {
      setRefreshing(false);
    }
  };

  const formatUptime = (seconds: number) => {
    const days = Math.floor(seconds / 86400);
    const hours = Math.floor((seconds % 86400) / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    return `${days}d ${hours}h ${minutes}m`;
  };

  const formatMemoryUsage = (bytes: number) => {
    return (bytes / 1024 / 1024).toFixed(1) + ' MB';
  };

  useEffect(() => {
    fetchSystemData();
    
    // Auto-refresh every 30 seconds
    const interval = setInterval(() => {
      if (!refreshing) {
        fetchSystemData();
      }
    }, 30000);

    return () => clearInterval(interval);
  }, [refreshing]);

  if (loading) {
    return (
      <div className="flex justify-center items-center min-h-screen">
        <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-gray-900"></div>
      </div>
    );
  }

  if (!systemData) {
    return (
      <div className="flex justify-center items-center min-h-screen">
        <div className="text-center">
          <AlertCircle className="h-12 w-12 mx-auto mb-4 text-red-500" />
          <h2 className="text-xl font-semibold mb-2">Failed to load system data</h2>
          <Button onClick={() => fetchSystemData()}>Try Again</Button>
        </div>
      </div>
    );
  }

  const { systemStats, analytics, performance, recentActivities, growthRates } = systemData;

  return (
    <div className="container mx-auto px-4 py-8 space-y-8">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold">Admin Dashboard</h1>
          <p className="text-gray-600">System overview and management</p>
        </div>
        <div className="flex gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => fetchSystemData()}
            disabled={refreshing}
          >
            <RefreshCw className={`h-4 w-4 mr-2 ${refreshing ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => handleSystemAction('generateReport')}
            disabled={refreshing}
          >
            <Download className="h-4 w-4 mr-2" />
            Export Report
          </Button>
        </div>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Users</CardTitle>
            <Users className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{systemStats.totalUsers.toLocaleString()}</div>
            <div className="flex items-center text-xs text-muted-foreground">
              <TrendingUp className="h-3 w-3 mr-1" />
              +{growthRates.userGrowth}% from last month
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Analyses</CardTitle>
            <FileVideo className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{systemStats.totalAnalyses.toLocaleString()}</div>
            <div className="flex items-center text-xs text-muted-foreground">
              <TrendingUp className="h-3 w-3 mr-1" />
              +{growthRates.analysisGrowth}% from last month
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Storage Used</CardTitle>
            <Database className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{formatFileSize(systemStats.totalStorage)}</div>
            <div className="flex items-center text-xs text-muted-foreground">
              <TrendingUp className="h-3 w-3 mr-1" />
              +{growthRates.storageGrowth}% from last month
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Success Rate</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{systemStats.successRate}%</div>
            <Progress value={systemStats.successRate} className="mt-2" />
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* System Performance */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <Server className="h-5 w-5 mr-2" />
              System Performance
            </CardTitle>
            <CardDescription>Server metrics and health status</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-sm font-medium">Uptime</p>
                <p className="text-2xl font-bold">{formatUptime(performance.uptime)}</p>
              </div>
              <div>
                <p className="text-sm font-medium">Active Users</p>
                <p className="text-2xl font-bold">{systemStats.activeUsers}</p>
              </div>
            </div>
            
            <Separator />
            
            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-sm">Memory Usage</span>
                <span className="text-sm font-medium">
                  {formatMemoryUsage(performance.memoryUsage.heapUsed)} / {formatMemoryUsage(performance.memoryUsage.heapTotal)}
                </span>
              </div>
              <Progress 
                value={(performance.memoryUsage.heapUsed / performance.memoryUsage.heapTotal) * 100} 
                className="w-full"
              />
            </div>
            
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <p className="text-muted-foreground">Node Version</p>
                <p className="font-medium">{performance.nodeVersion}</p>
              </div>
              <div>
                <p className="text-muted-foreground">Platform</p>
                <p className="font-medium">{performance.platform} ({performance.arch})</p>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Analytics Overview */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <TrendingUp className="h-5 w-5 mr-2" />
              Analytics Overview
            </CardTitle>
            <CardDescription>Usage statistics and trends</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-sm font-medium">Daily Analyses</p>
                <p className="text-2xl font-bold">{analytics.dailyAnalyses}</p>
              </div>
              <div>
                <p className="text-sm font-medium">Avg Processing Time</p>
                <p className="text-2xl font-bold">{systemStats.avgProcessingTime}s</p>
              </div>
            </div>

            <Separator />

            <div>
              <p className="text-sm font-medium mb-2">Popular Features</p>
              <div className="space-y-2">
                {analytics.popularFeatures?.slice(0, 3).map((feature, index) => (
                  <div key={index} className="flex justify-between items-center">
                    <span className="text-sm">{feature.feature}</span>
                    <Badge variant="secondary">{feature.usage}%</Badge>
                  </div>
                ))}
              </div>
            </div>

            <div>
              <p className="text-sm font-medium mb-2">Top Regions</p>
              <div className="space-y-2">
                {analytics.topRegions?.slice(0, 3).map((region, index) => (
                  <div key={index} className="flex justify-between items-center">
                    <span className="text-sm">{region.region}</span>
                    <span className="text-sm font-medium">{region.count}</span>
                  </div>
                ))}
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Quick Actions */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center">
            <Settings className="h-5 w-5 mr-2" />
            Quick Actions
          </CardTitle>
          <CardDescription>System maintenance and management tools</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <Button
              variant="outline"
              onClick={() => handleSystemAction('cleanupOldData', { days: 30 })}
              disabled={refreshing}
              className="flex items-center justify-center"
            >
              <Trash2 className="h-4 w-4 mr-2" />
              Cleanup Old Data
            </Button>
            
            <Button
              variant="outline"
              onClick={() => handleSystemAction('optimizeDatabase')}
              disabled={refreshing}
              className="flex items-center justify-center"
            >
              <Database className="h-4 w-4 mr-2" />
              Optimize Database
            </Button>
            
            <Button
              variant="outline"
              onClick={() => handleSystemAction('exportData', { format: 'json' })}
              disabled={refreshing}
              className="flex items-center justify-center"
            >
              <Download className="h-4 w-4 mr-2" />
              Export System Data
            </Button>
            
            <Button
              variant="outline"
              onClick={() => handleSystemAction('generateReport', { type: 'performance' })}
              disabled={refreshing}
              className="flex items-center justify-center"
            >
              <FileVideo className="h-4 w-4 mr-2" />
              Generate Report
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Recent Activities */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center">
            <Activity className="h-5 w-5 mr-2" />
            Recent Activities
          </CardTitle>
          <CardDescription>Latest system events and user activities</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {recentActivities?.slice(0, 10).map((activity) => (
              <div key={activity.id} className="flex items-start space-x-3 p-3 rounded-lg bg-gray-50 dark:bg-gray-800">
                <div className="flex-shrink-0">
                  {activity.type === 'user_registration' && <Users className="h-4 w-4 text-green-500" />}
                  {activity.type === 'analysis_completed' && <FileVideo className="h-4 w-4 text-blue-500" />}
                  {activity.type === 'error' && <AlertCircle className="h-4 w-4 text-red-500" />}
                  {activity.type === 'system' && <Server className="h-4 w-4 text-purple-500" />}
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm text-gray-900 dark:text-gray-100">{activity.message}</p>
                  <div className="flex items-center mt-1 space-x-2">
                    <p className="text-xs text-gray-500">
                      {new Date(activity.timestamp).toLocaleString()}
                    </p>
                    {activity.user && (
                      <Badge variant="outline" className="text-xs">
                        {activity.user}
                      </Badge>
                    )}
                  </div>
                </div>
              </div>
            ))}
            
            {(!recentActivities || recentActivities.length === 0) && (
              <p className="text-center text-gray-500 py-8">No recent activities</p>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}