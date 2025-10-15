'use client'

import { useState, useEffect } from 'react'
import { useSession } from 'next-auth/react'
import { useRouter } from 'next/navigation'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { Separator } from '@/components/ui/separator'
import { 
  BarChart3, 
  Shield, 
  Clock, 
  Upload, 
  Users, 
  TrendingUp,
  FileText,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Image as ImageIcon,
  Video,
  Volume2,
  Settings,
  LogOut,
  User,
  Plus
} from 'lucide-react'
import { motion } from 'framer-motion'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, LineChart, Line } from 'recharts'
import { format } from 'date-fns'
import Link from 'next/link'

interface UserStats {
  total: number
  authentic: number
  manipulated: number
  inconclusive: number
  byMediaType: {
    image: number
    audio: number
    video: number
  }
  avgProcessingTime: number
  avgConfidence: number
  recentActivity: any[]
}

interface UsageData {
  monthlyScans: number
  totalScans: number
  monthlyLimit: number
  remaining: number
}

export default function DashboardPage() {
  const { data: session, status } = useSession()
  const router = useRouter()
  const [stats, setStats] = useState<UserStats | null>(null)
  const [usage, setUsage] = useState<UsageData | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    if (status === 'loading') return
    if (status === 'unauthenticated') {
      router.push('/auth/signin')
      return
    }

    fetchDashboardData()
  }, [status, router])

  const fetchDashboardData = async () => {
    try {
      const [statsRes, usageRes] = await Promise.all([
        fetch('/api/user/stats'),
        fetch('/api/user/usage')
      ])

      if (statsRes.ok) {
        const statsData = await statsRes.json()
        setStats(statsData)
      }

      if (usageRes.ok) {
        const usageData = await usageRes.json()
        setUsage(usageData)
      }
    } catch (error) {
      console.error('Error fetching dashboard data:', error)
    } finally {
      setLoading(false)
    }
  }

  if (status === 'loading' || loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto mb-4"></div>
          <p className="text-muted-foreground">Loading dashboard...</p>
        </div>
      </div>
    )
  }

  if (!session) return null

  // Chart data
  const chartData = stats ? [
    { name: 'Authentic', value: stats.authentic, color: '#10b981' },
    { name: 'Manipulated', value: stats.manipulated, color: '#ef4444' },
    { name: 'Inconclusive', value: stats.inconclusive, color: '#f59e0b' }
  ] : []

  const mediaTypeData = stats ? [
    { name: 'Images', count: stats.byMediaType.image, icon: '🖼️' },
    { name: 'Audio', count: stats.byMediaType.audio, icon: '🎵' },
    { name: 'Video', count: stats.byMediaType.video, icon: '🎥' }
  ] : []

  const usagePercentage = usage ? (usage.monthlyScans / usage.monthlyLimit) * 100 : 0

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-muted/10">
      {/* Header */}
      <div className="border-b bg-card/50 backdrop-blur-sm">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="w-10 h-10 bg-primary/10 rounded-lg flex items-center justify-center">
                <Shield className="w-6 h-6 text-primary" />
              </div>
              <div>
                <h1 className="text-2xl font-bold">ITL Deepfake Detective</h1>
                <p className="text-muted-foreground">Welcome back, {session.user.name || session.user.email}</p>
              </div>
            </div>
            
            <div className="flex items-center gap-3">
              <Badge variant="outline" className="capitalize">
                {session.user.plan?.toLowerCase() || 'Free'} Plan
              </Badge>
              <Button variant="outline" size="sm" asChild>
                <Link href="/profile">
                  <User className="w-4 h-4 mr-2" />
                  Profile
                </Link>
              </Button>
              <Button variant="outline" size="sm" asChild>
                <Link href="/upload">
                  <Plus className="w-4 h-4 mr-2" />
                  New Analysis
                </Link>
              </Button>
            </div>
          </div>
        </div>
      </div>

      <div className="container mx-auto px-4 py-8">
        <div className="space-y-8">
          {/* Quick Stats */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }}
            >
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Total Analyses</CardTitle>
                  <BarChart3 className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{stats?.total || 0}</div>
                  <p className="text-xs text-muted-foreground">
                    Lifetime analyses
                  </p>
                </CardContent>
              </Card>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
            >
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Monthly Usage</CardTitle>
                  <TrendingUp className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{usage?.monthlyScans || 0}</div>
                  <Progress value={usagePercentage} className="mt-2" />
                  <p className="text-xs text-muted-foreground mt-2">
                    {usage?.remaining || 0} remaining this month
                  </p>
                </CardContent>
              </Card>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
            >
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Avg Processing</CardTitle>
                  <Clock className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">
                    {stats?.avgProcessingTime ? `${(stats.avgProcessingTime / 1000).toFixed(1)}s` : '0s'}
                  </div>
                  <p className="text-xs text-muted-foreground">
                    Per analysis
                  </p>
                </CardContent>
              </Card>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 }}
            >
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Avg Confidence</CardTitle>
                  <Shield className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">
                    {stats?.avgConfidence ? `${Math.round(stats.avgConfidence * 100)}%` : '0%'}
                  </div>
                  <p className="text-xs text-muted-foreground">
                    Detection accuracy
                  </p>
                </CardContent>
              </Card>
            </motion.div>
          </div>

          {/* Charts Section */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* Results Breakdown */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5 }}
            >
              <Card>
                <CardHeader>
                  <CardTitle>Analysis Results</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <Pie
                          data={chartData}
                          cx="50%"
                          cy="50%"
                          outerRadius={80}
                          fill="#8884d8"
                          dataKey="value"
                        >
                          {chartData.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={entry.color} />
                          ))}
                        </Pie>
                        <Tooltip />
                      </PieChart>
                    </ResponsiveContainer>
                  </div>
                  <div className="flex justify-center space-x-4 mt-4">
                    {chartData.map((item) => (
                      <div key={item.name} className="flex items-center">
                        <div 
                          className="w-3 h-3 rounded-full mr-2" 
                          style={{ backgroundColor: item.color }}
                        />
                        <span className="text-sm">{item.name}: {item.value}</span>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </motion.div>

            {/* Media Types */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.6 }}
            >
              <Card>
                <CardHeader>
                  <CardTitle>Media Types Analyzed</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {mediaTypeData.map((media) => (
                      <div key={media.name} className="flex items-center justify-between">
                        <div className="flex items-center">
                          <span className="text-xl mr-3">{media.icon}</span>
                          <span className="font-medium">{media.name}</span>
                        </div>
                        <Badge variant="secondary">{media.count}</Badge>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          </div>

          {/* Recent Activity */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.7 }}
          >
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center justify-between">
                  Recent Activity
                  <Button variant="outline" size="sm" asChild>
                    <Link href="/history">View All</Link>
                  </Button>
                </CardTitle>
              </CardHeader>
              <CardContent>
                {stats?.recentActivity && stats.recentActivity.length > 0 ? (
                  <div className="space-y-4">
                    {stats.recentActivity.slice(0, 5).map((activity, index) => {
                      const Icon = activity.prediction === 'AUTHENTIC' ? CheckCircle : 
                                  activity.prediction === 'MANIPULATED' ? XCircle : AlertTriangle
                      const color = activity.prediction === 'AUTHENTIC' ? 'text-green-600' : 
                                   activity.prediction === 'MANIPULATED' ? 'text-red-600' : 'text-yellow-600'
                      
                      return (
                        <div key={index} className="flex items-center space-x-4">
                          <Icon className={`h-5 w-5 ${color}`} />
                          <div className="flex-1">
                            <p className="text-sm font-medium">
                              {activity.mediaType.toLowerCase()} analysis completed
                            </p>
                            <p className="text-xs text-muted-foreground">
                              {format(new Date(activity.date), 'MMM d, yyyy h:mm a')} • 
                              {Math.round(activity.confidence * 100)}% confidence
                            </p>
                          </div>
                          <Badge 
                            variant={activity.prediction === 'AUTHENTIC' ? 'secondary' : 
                                    activity.prediction === 'MANIPULATED' ? 'destructive' : 'outline'}
                          >
                            {activity.prediction.toLowerCase()}
                          </Badge>
                        </div>
                      )
                    })}
                  </div>
                ) : (
                  <div className="text-center py-8">
                    <Upload className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                    <p className="text-muted-foreground">No analyses yet</p>
                    <Button className="mt-4" asChild>
                      <Link href="/upload">
                        <Plus className="w-4 h-4 mr-2" />
                        Start Your First Analysis
                      </Link>
                    </Button>
                  </div>
                )}
              </CardContent>
            </Card>
          </motion.div>

          {/* Quick Actions */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <Card className="cursor-pointer hover:shadow-lg transition-shadow">
              <Link href="/upload">
                <CardContent className="flex items-center space-x-4 p-6">
                  <div className="w-12 h-12 bg-primary/10 rounded-lg flex items-center justify-center">
                    <Upload className="w-6 h-6 text-primary" />
                  </div>
                  <div>
                    <h3 className="font-semibold">New Analysis</h3>
                    <p className="text-sm text-muted-foreground">Upload and analyze media files</p>
                  </div>
                </CardContent>
              </Link>
            </Card>

            <Card className="cursor-pointer hover:shadow-lg transition-shadow">
              <Link href="/history">
                <CardContent className="flex items-center space-x-4 p-6">
                  <div className="w-12 h-12 bg-blue-100 dark:bg-blue-900 rounded-lg flex items-center justify-center">
                    <FileText className="w-6 h-6 text-blue-600" />
                  </div>
                  <div>
                    <h3 className="font-semibold">View History</h3>
                    <p className="text-sm text-muted-foreground">Browse past analyses</p>
                  </div>
                </CardContent>
              </Link>
            </Card>

            <Card className="cursor-pointer hover:shadow-lg transition-shadow">
              <Link href="/settings">
                <CardContent className="flex items-center space-x-4 p-6">
                  <div className="w-12 h-12 bg-green-100 dark:bg-green-900 rounded-lg flex items-center justify-center">
                    <Settings className="w-6 h-6 text-green-600" />
                  </div>
                  <div>
                    <h3 className="font-semibold">Settings</h3>
                    <p className="text-sm text-muted-foreground">Manage preferences</p>
                  </div>
                </CardContent>
              </Link>
            </Card>
          </div>
        </div>
      </div>
    </div>
  )
}