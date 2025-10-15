'use client'

import { useState, useEffect } from 'react'
import { useSession, signOut } from 'next-auth/react'
import { useRouter } from 'next/navigation'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Separator } from '@/components/ui/separator'
import { Switch } from '@/components/ui/switch'
import { Badge } from '@/components/ui/badge'
import { 
  User, 
  Mail, 
  Calendar, 
  Settings, 
  Shield, 
  Bell, 
  Lock,
  Download,
  Trash2,
  Eye,
  EyeOff,
  Save,
  ArrowLeft,
  Crown
} from 'lucide-react'
import { motion } from 'framer-motion'
import { toast } from 'sonner'
import Link from 'next/link'
import { format } from 'date-fns'

interface UserProfile {
  id: string
  name?: string
  email: string
  image?: string
  role: string
  plan: string
  createdAt: string
  preferences?: {
    theme: string
    notifications: boolean
    dataRetention: number
    shareAnalytics: boolean
  }
  usage?: {
    totalScans: number
    monthlyScans: number
    monthlyLimit: number
  }
}

export default function ProfilePage() {
  const { data: session, status, update } = useSession()
  const router = useRouter()
  const [profile, setProfile] = useState<UserProfile | null>(null)
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    notifications: true,
    shareAnalytics: false,
    dataRetention: 90,
    theme: 'system'
  })

  useEffect(() => {
    if (status === 'loading') return
    if (status === 'unauthenticated') {
      router.push('/auth/signin')
      return
    }

    fetchProfile()
  }, [status, router])

  const fetchProfile = async () => {
    try {
      const response = await fetch('/api/user/profile')
      if (response.ok) {
        const profileData = await response.json()
        setProfile(profileData)
        setFormData({
          name: profileData.name || '',
          email: profileData.email || '',
          notifications: profileData.preferences?.notifications ?? true,
          shareAnalytics: profileData.preferences?.shareAnalytics ?? false,
          dataRetention: profileData.preferences?.dataRetention ?? 90,
          theme: profileData.preferences?.theme ?? 'system'
        })
      }
    } catch (error) {
      console.error('Error fetching profile:', error)
      toast.error('Failed to load profile')
    } finally {
      setLoading(false)
    }
  }

  const handleSave = async () => {
    setSaving(true)
    try {
      const response = await fetch('/api/user/profile', {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: formData.name,
          preferences: {
            notifications: formData.notifications,
            shareAnalytics: formData.shareAnalytics,
            dataRetention: formData.dataRetention,
            theme: formData.theme
          }
        })
      })

      if (response.ok) {
        await update({ name: formData.name })
        toast.success('Profile updated successfully')
        await fetchProfile() // Refresh profile data
      } else {
        throw new Error('Failed to update profile')
      }
    } catch (error) {
      console.error('Error updating profile:', error)
      toast.error('Failed to update profile')
    } finally {
      setSaving(false)
    }
  }

  const handleDeleteAccount = async () => {
    if (!confirm('Are you sure you want to delete your account? This action cannot be undone.')) {
      return
    }

    try {
      const response = await fetch('/api/user/delete', {
        method: 'DELETE'
      })

      if (response.ok) {
        toast.success('Account deleted successfully')
        await signOut({ callbackUrl: '/' })
      } else {
        throw new Error('Failed to delete account')
      }
    } catch (error) {
      console.error('Error deleting account:', error)
      toast.error('Failed to delete account')
    }
  }

  const handleExportData = async () => {
    try {
      const response = await fetch('/api/user/export')
      if (response.ok) {
        const blob = await response.blob()
        const url = URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        a.download = `deepfake-detective-data-${new Date().toISOString().split('T')[0]}.json`
        a.click()
        URL.revokeObjectURL(url)
        toast.success('Data exported successfully')
      } else {
        throw new Error('Failed to export data')
      }
    } catch (error) {
      console.error('Error exporting data:', error)
      toast.error('Failed to export data')
    }
  }

  if (status === 'loading' || loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto mb-4"></div>
          <p className="text-muted-foreground">Loading profile...</p>
        </div>
      </div>
    )
  }

  if (!session || !profile) return null

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-muted/10">
      {/* Header */}
      <div className="border-b bg-card/50 backdrop-blur-sm">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center gap-4">
            <Button variant="outline" size="sm" asChild>
              <Link href="/dashboard">
                <ArrowLeft className="w-4 h-4 mr-2" />
                Back to Dashboard
              </Link>
            </Button>
            <div>
              <h1 className="text-2xl font-bold">Profile Settings</h1>
              <p className="text-muted-foreground">Manage your account and preferences</p>
            </div>
          </div>
        </div>
      </div>

      <div className="container mx-auto px-4 py-8">
        <div className="max-w-4xl mx-auto space-y-8">
          {/* Profile Overview */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
          >
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <User className="w-5 h-5" />
                  Account Information
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="flex items-center space-x-4">
                  <div className="w-20 h-20 bg-primary/10 rounded-full flex items-center justify-center">
                    {profile.image ? (
                      <img 
                        src={profile.image} 
                        alt="Profile" 
                        className="w-20 h-20 rounded-full object-cover"
                      />
                    ) : (
                      <User className="w-10 h-10 text-primary" />
                    )}
                  </div>
                  <div className="flex-1">
                    <div className="flex items-center gap-2">
                      <h3 className="text-xl font-semibold">{profile.name || 'User'}</h3>
                      <Badge variant="outline" className="capitalize">
                        {profile.role.toLowerCase()}
                      </Badge>
                      {profile.plan !== 'FREE' && (
                        <Badge className="capitalize">
                          <Crown className="w-3 h-3 mr-1" />
                          {profile.plan.toLowerCase()}
                        </Badge>
                      )}
                    </div>
                    <p className="text-muted-foreground">{profile.email}</p>
                    <p className="text-sm text-muted-foreground">
                      Member since {format(new Date(profile.createdAt), 'MMMM yyyy')}
                    </p>
                  </div>
                </div>

                {/* Usage Stats */}
                {profile.usage && (
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-6 pt-6 border-t">
                    <div className="text-center">
                      <div className="text-2xl font-bold text-primary">{profile.usage.totalScans}</div>
                      <div className="text-sm text-muted-foreground">Total Analyses</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-blue-600">{profile.usage.monthlyScans}</div>
                      <div className="text-sm text-muted-foreground">This Month</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-green-600">
                        {profile.usage.monthlyLimit === -1 ? '∞' : profile.usage.monthlyLimit - profile.usage.monthlyScans}
                      </div>
                      <div className="text-sm text-muted-foreground">Remaining</div>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          </motion.div>

          {/* Personal Information */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
          >
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Settings className="w-5 h-5" />
                  Personal Information
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="name">Full Name</Label>
                    <Input
                      id="name"
                      value={formData.name}
                      onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                      placeholder="Enter your full name"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="email">Email Address</Label>
                    <Input
                      id="email"
                      value={formData.email}
                      disabled
                      className="bg-muted"
                    />
                    <p className="text-xs text-muted-foreground">
                      Email cannot be changed. Contact support if needed.
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </motion.div>

          {/* Preferences */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
          >
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Bell className="w-5 h-5" />
                  Preferences
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label>Email Notifications</Label>
                    <p className="text-sm text-muted-foreground">
                      Receive notifications about analysis results and account updates
                    </p>
                  </div>
                  <Switch
                    checked={formData.notifications}
                    onCheckedChange={(checked) => 
                      setFormData({ ...formData, notifications: checked })
                    }
                  />
                </div>

                <Separator />

                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label>Share Analytics</Label>
                    <p className="text-sm text-muted-foreground">
                      Help improve our service by sharing anonymous usage data
                    </p>
                  </div>
                  <Switch
                    checked={formData.shareAnalytics}
                    onCheckedChange={(checked) => 
                      setFormData({ ...formData, shareAnalytics: checked })
                    }
                  />
                </div>

                <Separator />

                <div className="space-y-2">
                  <Label htmlFor="retention">Data Retention Period</Label>
                  <select
                    id="retention"
                    className="w-full px-3 py-2 border border-input bg-background rounded-md"
                    value={formData.dataRetention}
                    onChange={(e) => 
                      setFormData({ ...formData, dataRetention: parseInt(e.target.value) })
                    }
                  >
                    <option value={30}>30 days</option>
                    <option value={90}>90 days</option>
                    <option value={180}>6 months</option>
                    <option value={365}>1 year</option>
                  </select>
                  <p className="text-sm text-muted-foreground">
                    How long to keep your analysis history
                  </p>
                </div>
              </CardContent>
            </Card>
          </motion.div>

          {/* Data Management */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
          >
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Shield className="w-5 h-5" />
                  Data Management
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <Button
                    variant="outline"
                    onClick={handleExportData}
                    className="flex items-center gap-2"
                  >
                    <Download className="w-4 h-4" />
                    Export My Data
                  </Button>
                  
                  <Button
                    variant="outline"
                    asChild
                  >
                    <Link href="/history">
                      <Eye className="w-4 h-4 mr-2" />
                      View Analysis History
                    </Link>
                  </Button>
                </div>

                <div className="p-4 bg-muted/30 rounded-lg">
                  <h4 className="font-medium text-sm mb-2">Data Privacy</h4>
                  <p className="text-xs text-muted-foreground">
                    Your data is processed locally and stored securely. You can export or delete 
                    your data at any time. We never share your personal information with third parties.
                  </p>
                </div>
              </CardContent>
            </Card>
          </motion.div>

          {/* Save Button */}
          <div className="flex justify-between items-center">
            <Button
              variant="destructive"
              onClick={handleDeleteAccount}
              className="flex items-center gap-2"
            >
              <Trash2 className="w-4 h-4" />
              Delete Account
            </Button>

            <Button
              onClick={handleSave}
              disabled={saving}
              className="flex items-center gap-2"
            >
              {saving ? (
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white" />
              ) : (
                <Save className="w-4 h-4" />
              )}
              {saving ? 'Saving...' : 'Save Changes'}
            </Button>
          </div>
        </div>
      </div>
    </div>
  )
}