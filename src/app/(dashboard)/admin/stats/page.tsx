'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Separator } from '@/components/ui/separator';
import { useToast } from '@/hooks/use-toast';
import { 
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  LineChart,
  Line,
  Area,
  AreaChart
} from 'recharts';
import { 
  TrendingUp,
  TrendingDown,
  Users,
  FileVideo,
  HardDrive,
  Clock,
  CheckCircle,
  XCircle,
  AlertCircle,
  RefreshCw,
  Download,
  Calendar
} from 'lucide-react';
import { formatFileSize } from '@/lib/utils';

interface StatsData {
  systemStats: {
    totalUsers: number;
    totalAnalyses: number;
    totalStorage: number;
    activeUsers: number;
    successRate: number;
    avgProcessingTime: number;
  };
  analytics: {
    dailyAnalyses: number;
    weeklyGrowth: number;
    monthlyGrowth: number;
    topRegions: Array<{ region: string; count: number }>;
    popularFeatures: Array<{ feature: string; usage: number }>;
  };
  growthRates: {
    userGrowth: number;
    analysisGrowth: number;
    storageGrowth: number;
  };
  performance: {
    uptime: number;
    memoryUsage: {
      heapUsed: number;
      heapTotal: number;
    };
  };
}

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8'];

export default function AdminStatsPage() {
  const [statsData, setStatsData] = useState<StatsData | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [timeRange, setTimeRange] = useState('7d');
  const { toast } = useToast();

  const fetchStats = async () => {
    try {
      const response = await fetch('/api/admin/system');
      if (!response.ok) throw new Error('Failed to fetch stats');
      
      const data = await response.json();
      setStatsData({
        systemStats: data.systemStats,
        analytics: data.analytics,
        growthRates: data.growthRates,
        performance: data.performance
      });
    } catch (error) {
      console.error('Error fetching stats:', error);
      toast({
        title: 'Error',
        description: 'Failed to load statistics',
        variant: 'destructive',
      });
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  const formatUptime = (seconds: number) => {
    const days = Math.floor(seconds / 86400);
    const hours = Math.floor((seconds % 86400) / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    return `${days}d ${hours}h ${minutes}m`;
  };

  const generateMockChartData = () => {
    // Generate mock data for the last 7 days
    const data = [];
    const now = new Date();
    
    for (let i = 6; i >= 0; i--) {
      const date = new Date(now);
      date.setDate(date.getDate() - i);
      
      data.push({
        date: date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
        analyses: Math.floor(Math.random() * 50) + 20,
        users: Math.floor(Math.random() * 15) + 5,
        storage: Math.floor(Math.random() * 1000) + 500
      });
    }
    return data;
  };

  const chartData = generateMockChartData();

  useEffect(() => {
    fetchStats();
    
    // Auto-refresh every 60 seconds
    const interval = setInterval(() => {
      if (!refreshing) {
        setRefreshing(true);
        fetchStats();
      }
    }, 60000);

    return () => clearInterval(interval);
  }, [refreshing]);

  if (loading) {
    return (
      <div className="flex justify-center items-center min-h-screen">
        <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-gray-900"></div>
      </div>
    );
  }

  if (!statsData) {
    return (
      <div className="flex justify-center items-center min-h-screen">
        <div className="text-center">
          <AlertCircle className="h-12 w-12 mx-auto mb-4 text-red-500" />
          <h2 className="text-xl font-semibold mb-2">Failed to load statistics</h2>
          <Button onClick={() => fetchStats()}>Try Again</Button>
        </div>
      </div>
    );
  }

  const { systemStats, analytics, growthRates, performance } = statsData;

  return (
    <div className="container mx-auto px-4 py-8 space-y-8">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold">System Statistics</h1>
          <p className="text-gray-600">Detailed analytics and performance metrics</p>
        </div>
        <div className="flex gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => {
              setRefreshing(true);
              fetchStats();
            }}
            disabled={refreshing}
          >
            <RefreshCw className={`h-4 w-4 mr-2 ${refreshing ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
          <Button
            variant="outline"
            size="sm"
          >
            <Download className="h-4 w-4 mr-2" />
            Export Report
          </Button>
        </div>
      </div>

      {/* Key Metrics Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Users</CardTitle>
            <Users className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{systemStats.totalUsers.toLocaleString()}</div>
            <div className="flex items-center text-xs">
              {growthRates.userGrowth >= 0 ? (
                <TrendingUp className="h-3 w-3 mr-1 text-green-500" />
              ) : (
                <TrendingDown className="h-3 w-3 mr-1 text-red-500" />
              )}
              <span className={growthRates.userGrowth >= 0 ? 'text-green-500' : 'text-red-500'}>
                {Math.abs(growthRates.userGrowth)}%
              </span>
              <span className="text-muted-foreground ml-1">vs last month</span>
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
            <div className="flex items-center text-xs">
              {growthRates.analysisGrowth >= 0 ? (
                <TrendingUp className="h-3 w-3 mr-1 text-green-500" />
              ) : (
                <TrendingDown className="h-3 w-3 mr-1 text-red-500" />
              )}
              <span className={growthRates.analysisGrowth >= 0 ? 'text-green-500' : 'text-red-500'}>
                {Math.abs(growthRates.analysisGrowth)}%
              </span>
              <span className="text-muted-foreground ml-1">vs last month</span>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Storage Used</CardTitle>
            <HardDrive className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{formatFileSize(systemStats.totalStorage)}</div>
            <div className="flex items-center text-xs">
              {growthRates.storageGrowth >= 0 ? (
                <TrendingUp className="h-3 w-3 mr-1 text-green-500" />
              ) : (
                <TrendingDown className="h-3 w-3 mr-1 text-red-500" />
              )}
              <span className={growthRates.storageGrowth >= 0 ? 'text-green-500' : 'text-red-500'}>
                {Math.abs(growthRates.storageGrowth)}%
              </span>
              <span className="text-muted-foreground ml-1">vs last month</span>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Success Rate</CardTitle>
            <CheckCircle className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{systemStats.successRate}%</div>
            <Progress value={systemStats.successRate} className="mt-2" />
            <div className="text-xs text-muted-foreground mt-1">
              Avg processing: {systemStats.avgProcessingTime}s
            </div>
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Usage Trends */}
        <Card>
          <CardHeader>
            <CardTitle>Usage Trends (Last 7 Days)</CardTitle>
            <CardDescription>Daily analyses and user activity</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" />
                <YAxis />
                <Tooltip />
                <Area
                  type="monotone"
                  dataKey="analyses"
                  stackId="1"
                  stroke="#8884d8"
                  fill="#8884d8"
                  fillOpacity={0.6}
                  name="Analyses"
                />
                <Area
                  type="monotone"
                  dataKey="users"
                  stackId="1"
                  stroke="#82ca9d"
                  fill="#82ca9d"
                  fillOpacity={0.6}
                  name="Active Users"
                />
              </AreaChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Regional Distribution */}
        <Card>
          <CardHeader>
            <CardTitle>Regional Distribution</CardTitle>
            <CardDescription>Usage by geographic region</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={analytics.topRegions}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ region, count, percent }: any) => `${region}: ${((percent || 0) * 100).toFixed(0)}%`}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="count"
                >
                  {analytics.topRegions.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Feature Usage */}
        <Card>
          <CardHeader>
            <CardTitle>Popular Features</CardTitle>
            <CardDescription>Most used analysis types</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {analytics.popularFeatures.map((feature, index) => (
                <div key={index} className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span>{feature.feature}</span>
                    <span>{feature.usage}%</span>
                  </div>
                  <Progress value={feature.usage} className="h-2" />
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* System Health */}
        <Card>
          <CardHeader>
            <CardTitle>System Health</CardTitle>
            <CardDescription>Performance indicators</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex justify-between items-center">
              <span className="text-sm">Uptime</span>
              <Badge variant="outline">{formatUptime(performance.uptime)}</Badge>
            </div>
            
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>Memory Usage</span>
                <span>
                  {Math.round(performance.memoryUsage.heapUsed / 1024 / 1024)}MB / 
                  {Math.round(performance.memoryUsage.heapTotal / 1024 / 1024)}MB
                </span>
              </div>
              <Progress 
                value={(performance.memoryUsage.heapUsed / performance.memoryUsage.heapTotal) * 100} 
                className="h-2"
              />
            </div>

            <div className="flex justify-between items-center">
              <span className="text-sm">Active Users</span>
              <Badge>{systemStats.activeUsers}</Badge>
            </div>

            <div className="flex justify-between items-center">
              <span className="text-sm">Daily Analyses</span>
              <Badge variant="secondary">{analytics.dailyAnalyses}</Badge>
            </div>
          </CardContent>
        </Card>

        {/* Growth Metrics */}
        <Card>
          <CardHeader>
            <CardTitle>Growth Metrics</CardTitle>
            <CardDescription>Month-over-month changes</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex justify-between items-center">
              <span className="text-sm">User Growth</span>
              <div className="flex items-center">
                {growthRates.userGrowth >= 0 ? (
                  <TrendingUp className="h-4 w-4 mr-1 text-green-500" />
                ) : (
                  <TrendingDown className="h-4 w-4 mr-1 text-red-500" />
                )}
                <span className={growthRates.userGrowth >= 0 ? 'text-green-600' : 'text-red-600'}>
                  {growthRates.userGrowth >= 0 ? '+' : ''}{growthRates.userGrowth}%
                </span>
              </div>
            </div>

            <div className="flex justify-between items-center">
              <span className="text-sm">Analysis Growth</span>
              <div className="flex items-center">
                {growthRates.analysisGrowth >= 0 ? (
                  <TrendingUp className="h-4 w-4 mr-1 text-green-500" />
                ) : (
                  <TrendingDown className="h-4 w-4 mr-1 text-red-500" />
                )}
                <span className={growthRates.analysisGrowth >= 0 ? 'text-green-600' : 'text-red-600'}>
                  {growthRates.analysisGrowth >= 0 ? '+' : ''}{growthRates.analysisGrowth}%
                </span>
              </div>
            </div>

            <div className="flex justify-between items-center">
              <span className="text-sm">Storage Growth</span>
              <div className="flex items-center">
                {growthRates.storageGrowth >= 0 ? (
                  <TrendingUp className="h-4 w-4 mr-1 text-green-500" />
                ) : (
                  <TrendingDown className="h-4 w-4 mr-1 text-red-500" />
                )}
                <span className={growthRates.storageGrowth >= 0 ? 'text-green-600' : 'text-red-600'}>
                  {growthRates.storageGrowth >= 0 ? '+' : ''}{growthRates.storageGrowth}%
                </span>
              </div>
            </div>

            <Separator />

            <div className="text-center">
              <div className="text-sm text-muted-foreground mb-2">Weekly Growth</div>
              <div className="flex items-center justify-center">
                {analytics.weeklyGrowth >= 0 ? (
                  <TrendingUp className="h-5 w-5 mr-2 text-green-500" />
                ) : (
                  <TrendingDown className="h-5 w-5 mr-2 text-red-500" />
                )}
                <span className={`text-lg font-semibold ${analytics.weeklyGrowth >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                  {analytics.weeklyGrowth >= 0 ? '+' : ''}{analytics.weeklyGrowth}%
                </span>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Storage Analysis Chart */}
      <Card>
        <CardHeader>
          <CardTitle>Storage Usage Over Time</CardTitle>
          <CardDescription>Daily storage consumption trends</CardDescription>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" />
              <YAxis />
              <Tooltip formatter={(value) => [`${value} MB`, 'Storage Used']} />
              <Line
                type="monotone"
                dataKey="storage"
                stroke="#8884d8"
                strokeWidth={2}
                dot={{ r: 4 }}
                activeDot={{ r: 6 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>
    </div>
  );
}