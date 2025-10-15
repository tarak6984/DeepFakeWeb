'use client';

import { useState, useEffect, useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { 
  BarChart, 
  Bar, 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell
} from 'recharts';
import {
  Zap,
  Clock,
  TrendingUp,
  AlertTriangle,
  CheckCircle,
  FileImage,
  Volume2,
  Calendar,
  Target,
} from 'lucide-react';
import { motion } from 'framer-motion';
import { usageTracker, UsageStats } from '@/lib/usage-tracker';

export function UsageDashboard() {
  const [stats, setStats] = useState<UsageStats | null>(null);
  const [isClient, setIsClient] = useState(false);

  useEffect(() => {
    setIsClient(true);
    if (typeof window !== 'undefined') {
      setStats(usageTracker.getUsageStats());
    }
  }, []);

  const trackerData = useMemo(() => {
    if (!isClient || !stats) return null;
    
    return {
      remainingScans: usageTracker.getRemainingScans(),
      usagePercentage: usageTracker.getUsagePercentage(),
      weeklyUsage: usageTracker.getWeeklyUsage(),
      fileTypeUsage: usageTracker.getMonthlyUsageByFileType(),
      confidenceDistribution: usageTracker.getConfidenceDistribution(),
      predictionStats: usageTracker.getPredictionStats(),
    };
  }, [isClient, stats]);

  if (!isClient || !stats || !trackerData) {
    return <div>Loading usage statistics...</div>;
  }

  const { remainingScans, usagePercentage, weeklyUsage, fileTypeUsage, confidenceDistribution, predictionStats } = trackerData;

  const getUsageColor = (percentage: number) => {
    if (percentage >= 90) return 'text-red-600 dark:text-red-400';
    if (percentage >= 70) return 'text-yellow-600 dark:text-yellow-400';
    return 'text-green-600 dark:text-green-400';
  };

  const getUsageBadgeVariant = (percentage: number) => {
    if (percentage >= 90) return 'destructive';
    if (percentage >= 70) return 'secondary';
    return 'default';
  };

  const COLORS = ['#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#06b6d4'];

  return (
    <div className="space-y-6">
      {/* Usage Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium">Monthly Usage</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex items-center justify-between mb-2">
                <span className={`text-2xl font-bold ${getUsageColor(usagePercentage)}`}>
                  {stats.monthlyScans}
                </span>
                <Badge variant={getUsageBadgeVariant(usagePercentage)}>
                  {usagePercentage.toFixed(1)}%
                </Badge>
              </div>
              <Progress value={usagePercentage} className="h-2" />
              <p className="text-xs text-muted-foreground mt-1">
                {remainingScans} scans remaining
              </p>
            </CardContent>
          </Card>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.1 }}
        >
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium">Free Tier Limit</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex items-center gap-2 mb-2">
                <Target className="h-4 w-4 text-primary" />
                <span className="text-2xl font-bold">{stats.freeTierLimit}</span>
              </div>
              <p className="text-xs text-muted-foreground">
                Audio & Image files per month
              </p>
              <div className="mt-2">
                <Badge variant="outline" className="text-xs">
                  Reality Defender Free
                </Badge>
              </div>
            </CardContent>
          </Card>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
        >
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium">Total Scans</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex items-center gap-2 mb-2">
                <CheckCircle className="h-4 w-4 text-green-500" />
                <span className="text-2xl font-bold">{stats.totalScans}</span>
              </div>
              <p className="text-xs text-muted-foreground">
                All-time analyses completed
              </p>
            </CardContent>
          </Card>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.3 }}
        >
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium">Next Reset</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex items-center gap-2 mb-2">
                <Calendar className="h-4 w-4 text-primary" />
                <span className="text-lg font-bold">
                  {new Date(new Date().getFullYear(), new Date().getMonth() + 1, 1)
                    .toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}
                </span>
              </div>
              <p className="text-xs text-muted-foreground">
                Monthly quota resets
              </p>
            </CardContent>
          </Card>
        </motion.div>
      </div>

      {/* Usage Alerts */}
      {usagePercentage >= 80 && (
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.3 }}
        >
          <Card className="border-yellow-200 bg-yellow-50 dark:border-yellow-800 dark:bg-yellow-950">
            <CardContent className="pt-6">
              <div className="flex items-center gap-3">
                <AlertTriangle className="h-5 w-5 text-yellow-600" />
                <div>
                  <h3 className="font-medium text-yellow-800 dark:text-yellow-200">
                    Usage Alert: {usagePercentage.toFixed(1)}% of monthly quota used
                  </h3>
                  <p className="text-sm text-yellow-700 dark:text-yellow-300">
                    You have {remainingScans} scans remaining this month. Consider upgrading for unlimited access.
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </motion.div>
      )}

      {/* Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Weekly Usage Chart */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <TrendingUp className="h-4 w-4" />
              Weekly Usage Trend
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={weeklyUsage}>
                  <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
                  <XAxis dataKey="day" tick={{ fontSize: 12 }} />
                  <YAxis tick={{ fontSize: 12 }} />
                  <Tooltip 
                    content={({ active, payload, label }) => {
                      if (active && payload && payload.length) {
                        return (
                          <div className="bg-background border rounded-lg shadow-lg p-3">
                            <p className="font-medium">Date: {label}</p>
                            <p className="text-sm">Scans: {payload[0].value}</p>
                          </div>
                        );
                      }
                      return null;
                    }}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="count" 
                    stroke="#8884d8" 
                    strokeWidth={2}
                    dot={{ fill: '#8884d8', strokeWidth: 2, r: 4 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* File Type Distribution */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <FileImage className="h-4 w-4" />
              File Type Distribution
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-64">
              {Object.keys(fileTypeUsage).length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={Object.entries(fileTypeUsage).map(([type, count]) => ({
                        name: type.charAt(0).toUpperCase() + type.slice(1),
                        value: count,
                        percentage: ((count / stats.monthlyScans) * 100).toFixed(1)
                      }))}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      label={({ name, percentage }) => `${name} (${percentage}%)`}
                      outerRadius={80}
                      fill="#8884d8"
                      dataKey="value"
                    >
                      {Object.entries(fileTypeUsage).map((_, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              ) : (
                <div className="flex items-center justify-center h-full text-muted-foreground">
                  <div className="text-center">
                    <FileImage className="h-8 w-8 mx-auto mb-2 opacity-50" />
                    <p className="text-sm">No usage data yet</p>
                  </div>
                </div>
              )}
            </div>
          </CardContent>
        </Card>

        {/* Confidence Distribution */}
        <Card>
          <CardHeader>
            <CardTitle>Confidence Score Distribution</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={confidenceDistribution}>
                  <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
                  <XAxis dataKey="range" tick={{ fontSize: 12 }} />
                  <YAxis tick={{ fontSize: 12 }} />
                  <Tooltip />
                  <Bar 
                    dataKey="count" 
                    fill="#8884d8"
                    radius={[4, 4, 0, 0]}
                  />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* Prediction Statistics */}
        <Card>
          <CardHeader>
            <CardTitle>Detection Results Summary</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {Object.entries(predictionStats).map(([prediction, count], index) => (
                <motion.div
                  key={prediction}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.1 }}
                  className="flex items-center justify-between p-3 rounded-lg border"
                >
                  <div className="flex items-center gap-2">
                    <div 
                      className="w-3 h-3 rounded-full"
                      style={{ 
                        backgroundColor: prediction === 'authentic' ? '#10b981' : 
                                        prediction === 'manipulated' ? '#ef4444' : '#f59e0b'
                      }}
                    />
                    <span className="font-medium text-sm capitalize">
                      {prediction}
                    </span>
                  </div>
                  <div className="text-right">
                    <div className="font-bold">{count}</div>
                    <div className="text-xs text-muted-foreground">
                      {((count / stats.totalScans) * 100).toFixed(1)}%
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
            {stats.totalScans === 0 && (
              <div className="text-center py-8 text-muted-foreground">
                <CheckCircle className="h-8 w-8 mx-auto mb-2 opacity-50" />
                <p className="text-sm">Upload files to see detection statistics</p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Optimization Tips */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Zap className="h-4 w-4" />
            Free Tier Optimization Tips
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            <div className="p-4 rounded-lg bg-muted/30">
              <h4 className="font-medium text-sm mb-2">📊 Batch Analysis</h4>
              <p className="text-xs text-muted-foreground">
                Upload multiple files at once to make the most of your 50 monthly scans.
              </p>
            </div>
            <div className="p-4 rounded-lg bg-muted/30">
              <h4 className="font-medium text-sm mb-2">🎯 Focus on Suspicious Content</h4>
              <p className="text-xs text-muted-foreground">
                Save scans by manually reviewing files before uploading obviously authentic content.
              </p>
            </div>
            <div className="p-4 rounded-lg bg-muted/30">
              <h4 className="font-medium text-sm mb-2">📅 Track Your Usage</h4>
              <p className="text-xs text-muted-foreground">
                Monitor your monthly progress to pace your usage throughout the month.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}