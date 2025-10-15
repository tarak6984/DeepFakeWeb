'use client';

import { useState, useEffect } from 'react';
import { useSession } from 'next-auth/react';
import { useRouter } from 'next/navigation';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { Separator } from '@/components/ui/separator';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Progress } from '@/components/ui/progress';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  Settings,
  User,
  Bell,
  Shield,
  Palette,
  Download,
  Trash2,
  Save,
  AlertTriangle,
  CheckCircle,
  Moon,
  Sun,
  Monitor,
  ArrowLeft,
  Mail,
  Database,
} from 'lucide-react';
import { motion } from 'framer-motion';
import Link from 'next/link';

export default function SettingsPage() {
  const { data: session, status } = useSession();
  const router = useRouter();
  
  // States for different settings
  const [theme, setTheme] = useState<'light' | 'dark' | 'system'>('system');
  const [notifications, setNotifications] = useState({
    email: true,
    browser: true,
    analysis: true,
    security: true,
  });
  const [preferences, setPreferences] = useState({
    autoDownload: false,
    maxHistoryItems: 50,
    defaultConfidenceThreshold: 0.7,
    compactView: false,
    showAdvancedFeatures: true,
  });
  const [isSaving, setIsSaving] = useState(false);
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
    loadSettings();
  }, []);

  const loadSettings = () => {
    // Load theme
    const savedTheme = localStorage.getItem('deepfake_theme') as 'light' | 'dark' | 'system';
    if (savedTheme) {
      setTheme(savedTheme);
    }

    // Load preferences
    const savedPrefs = localStorage.getItem('deepfake_preferences');
    if (savedPrefs) {
      try {
        const parsedPrefs = JSON.parse(savedPrefs);
        setPreferences(prev => ({ ...prev, ...parsedPrefs }));
      } catch (error) {
        console.error('Failed to load preferences:', error);
      }
    }

    // Load notification settings
    const savedNotifications = localStorage.getItem('deepfake_notifications');
    if (savedNotifications) {
      try {
        const parsedNotifications = JSON.parse(savedNotifications);
        setNotifications(prev => ({ ...prev, ...parsedNotifications }));
      } catch (error) {
        console.error('Failed to load notification settings:', error);
      }
    }
  };

  const handleThemeChange = (newTheme: 'light' | 'dark' | 'system') => {
    setTheme(newTheme);
    localStorage.setItem('deepfake_theme', newTheme);

    // Apply theme immediately
    const root = document.documentElement;
    if (newTheme === 'system') {
      const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
      root.classList.toggle('dark', mediaQuery.matches);
    } else {
      root.classList.toggle('dark', newTheme === 'dark');
    }
  };

  const handleSaveSettings = async () => {
    setIsSaving(true);
    setMessage(null);

    try {
      // Save to localStorage
      localStorage.setItem('deepfake_preferences', JSON.stringify(preferences));
      localStorage.setItem('deepfake_notifications', JSON.stringify(notifications));

      // If user is logged in, also save to server
      if (session) {
        const response = await fetch('/api/user/profile', {
          method: 'PATCH',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            preferences,
            notifications,
            theme,
          }),
        });

        if (!response.ok) {
          throw new Error('Failed to save settings to server');
        }
      }

      setMessage({ type: 'success', text: 'Settings saved successfully!' });
    } catch (error) {
      console.error('Failed to save settings:', error);
      setMessage({ type: 'error', text: 'Failed to save settings. Please try again.' });
    } finally {
      setIsSaving(false);
      // Clear message after 3 seconds
      setTimeout(() => setMessage(null), 3000);
    }
  };

  const handleClearData = () => {
    if (window.confirm('Are you sure you want to clear all local data? This cannot be undone.')) {
      localStorage.removeItem('deepfake_analysis_history');
      localStorage.removeItem('deepfake_preferences');
      localStorage.removeItem('deepfake_notifications');
      setMessage({ type: 'success', text: 'Local data cleared successfully!' });
      setTimeout(() => setMessage(null), 3000);
    }
  };

  const handleExportData = () => {
    const data = {
      theme,
      preferences,
      notifications,
      history: JSON.parse(localStorage.getItem('deepfake_analysis_history') || '[]'),
      exportDate: new Date().toISOString(),
    };

    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `deepfake-detective-settings-${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);

    setMessage({ type: 'success', text: 'Settings exported successfully!' });
    setTimeout(() => setMessage(null), 3000);
  };

  if (status === 'loading' || !mounted) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-muted/20">
      <div className="container mx-auto p-6 max-w-4xl">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div className="flex items-center space-x-4">
            <Button variant="ghost" size="sm" asChild>
              <Link href="/dashboard">
                <ArrowLeft className="w-4 h-4 mr-2" />
                Back
              </Link>
            </Button>
            <div>
              <h1 className="text-3xl font-bold flex items-center">
                <Settings className="w-8 h-8 mr-3 text-primary" />
                Settings
              </h1>
              <p className="text-muted-foreground">
                Manage your preferences and account settings
              </p>
            </div>
          </div>
          <Button onClick={handleSaveSettings} disabled={isSaving}>
            {isSaving ? (
              <>
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2" />
                Saving...
              </>
            ) : (
              <>
                <Save className="w-4 h-4 mr-2" />
                Save Settings
              </>
            )}
          </Button>
        </div>

        {/* Messages */}
        {message && (
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            className="mb-6"
          >
            <Alert variant={message.type === 'error' ? 'destructive' : 'default'}>
              {message.type === 'error' ? (
                <AlertTriangle className="h-4 w-4" />
              ) : (
                <CheckCircle className="h-4 w-4" />
              )}
              <AlertDescription>{message.text}</AlertDescription>
            </Alert>
          </motion.div>
        )}

        <div className="grid gap-6 md:grid-cols-2">
          {/* Theme Settings */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
          >
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center">
                  <Palette className="w-5 h-5 mr-2" />
                  Appearance
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label>Theme</Label>
                  <Select value={theme} onValueChange={(value: 'light' | 'dark' | 'system') => handleThemeChange(value)}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="light">
                        <div className="flex items-center">
                          <Sun className="w-4 h-4 mr-2" />
                          Light
                        </div>
                      </SelectItem>
                      <SelectItem value="dark">
                        <div className="flex items-center">
                          <Moon className="w-4 h-4 mr-2" />
                          Dark
                        </div>
                      </SelectItem>
                      <SelectItem value="system">
                        <div className="flex items-center">
                          <Monitor className="w-4 h-4 mr-2" />
                          System
                        </div>
                      </SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label>Compact View</Label>
                    <p className="text-sm text-muted-foreground">
                      Use a more compact layout for tables and lists
                    </p>
                  </div>
                  <Switch
                    checked={preferences.compactView}
                    onCheckedChange={(checked) =>
                      setPreferences(prev => ({ ...prev, compactView: checked }))
                    }
                  />
                </div>

                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label>Advanced Features</Label>
                    <p className="text-sm text-muted-foreground">
                      Show advanced analysis options and detailed reports
                    </p>
                  </div>
                  <Switch
                    checked={preferences.showAdvancedFeatures}
                    onCheckedChange={(checked) =>
                      setPreferences(prev => ({ ...prev, showAdvancedFeatures: checked }))
                    }
                  />
                </div>
              </CardContent>
            </Card>
          </motion.div>

          {/* Notification Settings */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
          >
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center">
                  <Bell className="w-5 h-5 mr-2" />
                  Notifications
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label>Email Notifications</Label>
                    <p className="text-sm text-muted-foreground">
                      Receive email updates about your account
                    </p>
                  </div>
                  <Switch
                    checked={notifications.email}
                    onCheckedChange={(checked) =>
                      setNotifications(prev => ({ ...prev, email: checked }))
                    }
                  />
                </div>

                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label>Browser Notifications</Label>
                    <p className="text-sm text-muted-foreground">
                      Show notifications in your browser
                    </p>
                  </div>
                  <Switch
                    checked={notifications.browser}
                    onCheckedChange={(checked) =>
                      setNotifications(prev => ({ ...prev, browser: checked }))
                    }
                  />
                </div>

                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label>Analysis Completed</Label>
                    <p className="text-sm text-muted-foreground">
                      Notify when analysis is finished
                    </p>
                  </div>
                  <Switch
                    checked={notifications.analysis}
                    onCheckedChange={(checked) =>
                      setNotifications(prev => ({ ...prev, analysis: checked }))
                    }
                  />
                </div>

                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label>Security Alerts</Label>
                    <p className="text-sm text-muted-foreground">
                      Important security and account notifications
                    </p>
                  </div>
                  <Switch
                    checked={notifications.security}
                    onCheckedChange={(checked) =>
                      setNotifications(prev => ({ ...prev, security: checked }))
                    }
                  />
                </div>
              </CardContent>
            </Card>
          </motion.div>

          {/* Analysis Preferences */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
          >
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center">
                  <Shield className="w-5 h-5 mr-2" />
                  Analysis Preferences
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label>Default Confidence Threshold</Label>
                  <div className="space-y-2">
                    <Progress value={preferences.defaultConfidenceThreshold * 100} />
                    <Input
                      type="range"
                      min="0.1"
                      max="1"
                      step="0.1"
                      value={preferences.defaultConfidenceThreshold}
                      onChange={(e) =>
                        setPreferences(prev => ({
                          ...prev,
                          defaultConfidenceThreshold: parseFloat(e.target.value)
                        }))
                      }
                      className="w-full"
                    />
                    <p className="text-sm text-muted-foreground">
                      Current: {Math.round(preferences.defaultConfidenceThreshold * 100)}%
                    </p>
                  </div>
                </div>

                <div className="space-y-2">
                  <Label>Max History Items</Label>
                  <Input
                    type="number"
                    min="10"
                    max="500"
                    value={preferences.maxHistoryItems}
                    onChange={(e) =>
                      setPreferences(prev => ({
                        ...prev,
                        maxHistoryItems: parseInt(e.target.value) || 50
                      }))
                    }
                  />
                  <p className="text-sm text-muted-foreground">
                    Number of analyses to keep in history
                  </p>
                </div>

                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label>Auto-download Reports</Label>
                    <p className="text-sm text-muted-foreground">
                      Automatically download analysis reports
                    </p>
                  </div>
                  <Switch
                    checked={preferences.autoDownload}
                    onCheckedChange={(checked) =>
                      setPreferences(prev => ({ ...prev, autoDownload: checked }))
                    }
                  />
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
                <CardTitle className="flex items-center">
                  <Database className="w-5 h-5 mr-2" />
                  Data Management
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label>Export Data</Label>
                  <p className="text-sm text-muted-foreground mb-2">
                    Download your settings and analysis history
                  </p>
                  <Button variant="outline" onClick={handleExportData} className="w-full">
                    <Download className="w-4 h-4 mr-2" />
                    Export Settings & History
                  </Button>
                </div>

                <Separator />

                <div className="space-y-2">
                  <Label className="text-destructive">Clear Local Data</Label>
                  <p className="text-sm text-muted-foreground mb-2">
                    Remove all locally stored data including history and preferences
                  </p>
                  <Button 
                    variant="destructive" 
                    onClick={handleClearData}
                    className="w-full"
                  >
                    <Trash2 className="w-4 h-4 mr-2" />
                    Clear All Data
                  </Button>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        </div>

        {/* Account Section (if logged in) */}
        {session && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5 }}
            className="mt-6"
          >
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center">
                  <User className="w-5 h-5 mr-2" />
                  Account Information
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid md:grid-cols-2 gap-4">
                  <div>
                    <Label>Name</Label>
                    <p className="text-sm font-medium">{session.user?.name || 'Not provided'}</p>
                  </div>
                  <div>
                    <Label>Email</Label>
                    <p className="text-sm font-medium">{session.user?.email}</p>
                  </div>
                </div>
                <div className="flex justify-end">
                  <Button variant="outline" asChild>
                    <Link href="/profile">
                      <User className="w-4 h-4 mr-2" />
                      Edit Profile
                    </Link>
                  </Button>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        )}

        <div className="mt-8 text-center">
          <p className="text-sm text-muted-foreground">
            Settings are saved locally and {session ? 'synced to your account' : 'will be synced when you sign in'}
          </p>
        </div>
      </div>
    </div>
  );
}