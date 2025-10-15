'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useSession, signOut, signIn } from 'next-auth/react';
import { Button } from '@/components/ui/button';
import { Switch } from '@/components/ui/switch';
  import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
  DropdownMenuSeparator,
} from '@/components/ui/dropdown-menu';
import {
  Menu,
  X,
  Shield,
  Sun,
  Moon,
  Monitor,
  Settings,
  History,
  User,
  LogOut,
  Home,
  UserCircle,
  Plus,
  ChevronDown,
  BarChart3,
  Users,
  ShieldCheck
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

interface NavbarProps {
  onMenuToggle?: () => void;
  showMobileMenu?: boolean;
}

export function Navbar({ onMenuToggle, showMobileMenu }: NavbarProps) {
  const [theme, setTheme] = useState<'light' | 'dark' | 'system'>('system');
  const [mounted, setMounted] = useState(false);
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const { data: session, status } = useSession();
  const pathname = usePathname();

  useEffect(() => {
    setMounted(true);
    // Load theme from localStorage
    const savedTheme = localStorage.getItem('deepfake_theme') as 'light' | 'dark' | 'system';
    if (savedTheme) {
      setTheme(savedTheme);
    }
  }, []);

  useEffect(() => {
    if (!mounted) return;

    const root = document.documentElement;
    
    if (theme === 'system') {
      const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
      root.classList.toggle('dark', mediaQuery.matches);
      
      const handler = (e: MediaQueryListEvent) => {
        root.classList.toggle('dark', e.matches);
      };
      
      mediaQuery.addEventListener('change', handler);
      return () => mediaQuery.removeEventListener('change', handler);
    } else {
      root.classList.toggle('dark', theme === 'dark');
    }
  }, [theme, mounted]);

  const handleThemeChange = (newTheme: 'light' | 'dark' | 'system') => {
    setTheme(newTheme);
    localStorage.setItem('deepfake_theme', newTheme);
  };

  const isAdmin = session?.user?.role === 'ADMIN' || session?.user?.role === 'SUPER_ADMIN';

  const navigationItems = session ? [
    { href: '/dashboard', label: 'Dashboard', active: pathname === '/dashboard' },
    { href: '/upload', label: 'Analyze', active: pathname === '/upload' },
    { href: '/history', label: 'History', active: pathname === '/history' },
    ...(isAdmin ? [
      { href: '/admin/dashboard', label: 'Admin', active: pathname.startsWith('/admin') }
    ] : [])
  ] : [
    { href: '/', label: 'Home', active: pathname === '/' },
    { href: '/upload', label: 'Analyze', active: pathname === '/upload' },
    { href: '/results', label: 'Results', active: pathname === '/results' },
  ];

  const ThemeIcon = theme === 'light' ? Sun : theme === 'dark' ? Moon : Monitor;

  if (!mounted) {
    return (
      <nav className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="container flex h-16 items-center justify-between px-4">
          <div className="flex items-center space-x-2">
            <div className="h-6 w-6" /> {/* Placeholder to avoid hydration mismatch */}
            <span className="font-semibold">ITL Deepfake Detective</span>
          </div>
          <div className="w-8 h-8" /> {/* Placeholder for theme toggle */}
        </div>
      </nav>
    );
  }

  return (
    <nav className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container flex h-16 items-center justify-between px-4">
        {/* Logo and Brand */}
        <div className="flex items-center space-x-2">
          <Link href="/" className="flex items-center space-x-2 hover:opacity-80 transition-opacity">
            <Shield className="h-6 w-6 text-primary" />
            <span className="font-semibold text-foreground">ITL Deepfake Detective</span>
          </Link>
        </div>

        {/* Desktop Navigation */}
        <div className="hidden md:flex items-center space-x-6">
          {navigationItems.map((item) => (
            <Link
              key={item.href}
              href={item.href}
              className={`text-sm font-medium transition-colors hover:text-primary ${
                item.active 
                  ? 'text-primary border-b-2 border-primary pb-1' 
                  : 'text-muted-foreground'
              }`}
            >
              {item.label}
            </Link>
          ))}
        </div>

        {/* Actions */}
        <div className="flex items-center space-x-2">
          {/* Authentication Section */}
          {session ? (
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="ghost" size="sm" className="flex items-center space-x-2">
                  <User className="h-4 w-4" />
                  <span className="hidden sm:inline text-sm">{session.user?.name || session.user?.email}</span>
                  <ChevronDown className="h-3 w-3" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="w-48">
                <DropdownMenuItem asChild>
                  <Link href="/profile" className="flex items-center w-full">
                    <Settings className="mr-2 h-4 w-4" />
                    Profile & Settings
                  </Link>
                </DropdownMenuItem>
                <DropdownMenuItem asChild>
                  <Link href="/dashboard" className="flex items-center w-full">
                    <Home className="mr-2 h-4 w-4" />
                    Dashboard
                  </Link>
                </DropdownMenuItem>
                {isAdmin && (
                  <>
                    <DropdownMenuSeparator />
                    <DropdownMenuItem asChild>
                      <Link href="/admin/dashboard" className="flex items-center w-full">
                        <ShieldCheck className="mr-2 h-4 w-4" />
                        Admin Dashboard
                      </Link>
                    </DropdownMenuItem>
                    <DropdownMenuItem asChild>
                      <Link href="/admin/users" className="flex items-center w-full">
                        <Users className="mr-2 h-4 w-4" />
                        User Management
                      </Link>
                    </DropdownMenuItem>
                    <DropdownMenuItem asChild>
                      <Link href="/admin/stats" className="flex items-center w-full">
                        <BarChart3 className="mr-2 h-4 w-4" />
                        System Stats
                      </Link>
                    </DropdownMenuItem>
                  </>
                )}
                <DropdownMenuSeparator />
                <DropdownMenuItem 
                  onClick={() => signOut()}
                  className="text-red-600 dark:text-red-400"
                >
                  <LogOut className="mr-2 h-4 w-4" />
                  Sign Out
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          ) : (
            <div className="hidden sm:flex items-center space-x-2">
              <Button 
                variant="ghost" 
                size="sm"
                onClick={() => signIn()}
              >
                Sign In
              </Button>
              <Button 
                size="sm"
                onClick={() => signIn()}
              >
                Get Started
              </Button>
            </div>
          )}

          {/* Theme Toggle */}
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" size="sm" className="w-9 px-0">
                <ThemeIcon className="h-4 w-4" />
                <span className="sr-only">Toggle theme</span>
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuItem onClick={() => handleThemeChange('light')}>
                <Sun className="mr-2 h-4 w-4" />
                Light
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => handleThemeChange('dark')}>
                <Moon className="mr-2 h-4 w-4" />
                Dark
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => handleThemeChange('system')}>
                <Monitor className="mr-2 h-4 w-4" />
                System
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>

          {/* Mobile Menu Toggle */}
          <Button
            variant="ghost"
            size="sm"
            className="md:hidden w-9 px-0"
            onClick={() => {
              setIsMenuOpen(!isMenuOpen);
              onMenuToggle?.();
            }}
          >
            {(showMobileMenu ?? isMenuOpen) ? (
              <X className="h-4 w-4" />
            ) : (
              <Menu className="h-4 w-4" />
            )}
            <span className="sr-only">Toggle menu</span>
          </Button>
        </div>
      </div>

      {/* Mobile Navigation */}
      <AnimatePresence>
        {(showMobileMenu ?? isMenuOpen) && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="md:hidden border-t overflow-hidden"
          >
            <div className="container px-4 py-4 space-y-3">
              {navigationItems.map((item) => (
                <Link
                  key={item.href}
                  href={item.href}
                  className={`block px-3 py-2 text-sm font-medium rounded-md transition-colors ${
                    item.active
                      ? 'bg-primary text-primary-foreground'
                      : 'text-muted-foreground hover:text-foreground hover:bg-accent'
                  }`}
                >
                  {item.label}
                </Link>
              ))}
              
              {/* Mobile Authentication */}
              {session ? (
                <>
                  <div className="pt-3 border-t space-y-1">
                    <Link
                      href="/profile"
                      className="flex items-center px-3 py-2 text-sm font-medium rounded-md text-muted-foreground hover:text-foreground hover:bg-accent"
                    >
                      <Settings className="mr-2 h-4 w-4" />
                      Profile & Settings
                    </Link>
                    {isAdmin && (
                      <>
                        <div className="px-3 py-2 text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                          Admin
                        </div>
                        <Link
                          href="/admin/dashboard"
                          className="flex items-center px-3 py-2 text-sm font-medium rounded-md text-muted-foreground hover:text-foreground hover:bg-accent"
                        >
                          <ShieldCheck className="mr-2 h-4 w-4" />
                          Admin Dashboard
                        </Link>
                        <Link
                          href="/admin/users"
                          className="flex items-center px-3 py-2 text-sm font-medium rounded-md text-muted-foreground hover:text-foreground hover:bg-accent"
                        >
                          <Users className="mr-2 h-4 w-4" />
                          User Management
                        </Link>
                        <Link
                          href="/admin/stats"
                          className="flex items-center px-3 py-2 text-sm font-medium rounded-md text-muted-foreground hover:text-foreground hover:bg-accent"
                        >
                          <BarChart3 className="mr-2 h-4 w-4" />
                          System Stats
                        </Link>
                      </>
                    )}
                    <button
                      onClick={() => signOut()}
                      className="w-full flex items-center px-3 py-2 text-sm font-medium rounded-md text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/20"
                    >
                      <LogOut className="mr-2 h-4 w-4" />
                      Sign Out
                    </button>
                  </div>
                </>
              ) : (
                <div className="pt-3 border-t space-y-2">
                  <Button
                    variant="ghost"
                    className="w-full justify-start"
                    onClick={() => signIn()}
                  >
                    <User className="mr-2 h-4 w-4" />
                    Sign In
                  </Button>
                  <Button
                    className="w-full justify-start"
                    onClick={() => signIn()}
                  >
                    <Plus className="mr-2 h-4 w-4" />
                    Get Started
                  </Button>
                </div>
              )}
              
              <div className="pt-3 border-t">
                <div className="flex items-center justify-between px-3 py-2 text-sm">
                  <span className="text-muted-foreground">Theme</span>
                  <div className="flex items-center space-x-2">
                    <Button
                      variant={theme === 'light' ? 'default' : 'outline'}
                      size="sm"
                      onClick={() => handleThemeChange('light')}
                      className="w-8 h-8 p-0"
                    >
                      <Sun className="h-3 w-3" />
                    </Button>
                    <Button
                      variant={theme === 'dark' ? 'default' : 'outline'}
                      size="sm"
                      onClick={() => handleThemeChange('dark')}
                      className="w-8 h-8 p-0"
                    >
                      <Moon className="h-3 w-3" />
                    </Button>
                    <Button
                      variant={theme === 'system' ? 'default' : 'outline'}
                      size="sm"
                      onClick={() => handleThemeChange('system')}
                      className="w-8 h-8 p-0"
                    >
                      <Monitor className="h-3 w-3" />
                    </Button>
                  </div>
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </nav>
  );
}
