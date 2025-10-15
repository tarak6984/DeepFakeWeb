'use client';

import { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { 
  Clock, 
  Zap, 
  AlertTriangle, 
  CheckCircle, 
  Star, 
  Sparkles,
  X,
  UserPlus
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { useUsageManager } from '@/lib/unified-usage-manager';
import { signIn } from 'next-auth/react';
import { toast } from 'sonner';

interface UsageLimitBannerProps {
  className?: string;
  showWhenAuthenticated?: boolean;
  showOnlyWhenLimitReached?: boolean;
}

export function UsageLimitBanner({ 
  className, 
  showWhenAuthenticated = false,
  showOnlyWhenLimitReached = false 
}: UsageLimitBannerProps) {
  const [isVisible, setIsVisible] = useState(true);
  const [mounted, setMounted] = useState(false);
  
  const {
    isAuthenticated,
    stats,
    remaining,
    used,
    message,
    timeUntilReset,
    isLimitReached,
    shouldShowUpgrade,
  } = useUsageManager();

  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) return null;

  // Don't show for authenticated users unless specified
  if (isAuthenticated && !showWhenAuthenticated) {
    return null;
  }

  // Only show when limit is reached if specified
  if (showOnlyWhenLimitReached && !isLimitReached) {
    return null;
  }

  // Don't show if user dismissed it
  if (!isVisible) {
    return null;
  }

  const handleSignUp = () => {
    signIn(undefined, { callbackUrl: '/dashboard' });
  };

  const handleDismiss = () => {
    setIsVisible(false);
    toast.info('Banner hidden for this session');
  };

  const getBannerVariant = () => {
    if (isLimitReached) return 'destructive';
    if (remaining === 1) return 'warning';
    return 'default';
  };

  const getBannerIcon = () => {
    if (isLimitReached) return AlertTriangle;
    if (remaining === 1) return Clock;
    return CheckCircle;
  };

  const getBannerColor = () => {
    if (isLimitReached) return 'border-red-200 bg-red-50 dark:bg-red-950/20';
    if (remaining === 1) return 'border-yellow-200 bg-yellow-50 dark:bg-yellow-950/20';
    return 'border-blue-300 bg-blue-100 dark:bg-blue-950/30';
  };

  const Icon = getBannerIcon();

  return (
    <AnimatePresence>
      {isVisible && (
        <motion.div
          initial={{ opacity: 0, y: -20, scale: 0.95 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          exit={{ opacity: 0, y: -20, scale: 0.95 }}
          transition={{ duration: 0.3 }}
          className={className}
        >
          <Card className={`${getBannerColor()} border-2 shadow-xl`}>
            <CardContent className="p-8">
              <div className="flex items-start justify-between gap-4">
                {/* Dismiss button */}
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={handleDismiss}
                  className="absolute top-2 right-2 h-8 w-8 p-0 opacity-60 hover:opacity-100"
                >
                  <X className="h-4 w-4" />
                </Button>

                {/* Icon */}
                <div className="flex-shrink-0">
                  <div className={`w-16 h-16 rounded-full flex items-center justify-center ${
                    isLimitReached ? 'bg-red-100 dark:bg-red-900/30' : 
                    remaining === 1 ? 'bg-yellow-100 dark:bg-yellow-900/30' : 
                    'bg-blue-200 dark:bg-blue-900/40'
                  }`}>
                    <Icon className={`w-8 h-8 ${
                      isLimitReached ? 'text-red-600 dark:text-red-400' :
                      remaining === 1 ? 'text-yellow-600 dark:text-yellow-400' :
                      'text-blue-700 dark:text-blue-300'
                    }`} />
                  </div>
                </div>

                {/* Content */}
                <div className="flex-1 space-y-4">
                  <div>
                    <h3 className="font-bold text-xl mb-2">
                      {isLimitReached ? 'Free Scans Used Up!' : 
                       remaining === 1 ? 'Last Free Scan!' : 
                       'Free Trial Active'}
                    </h3>
                    <p className="text-lg text-muted-foreground">
                      {message}
                    </p>
                  </div>

                  {/* Progress bar for non-authenticated users */}
                  {!isAuthenticated && (
                    <div className="space-y-3">
                      <div className="flex justify-between text-base font-medium">
                        <span>Used: {used}/{stats.limit}</span>
                        <span>Remaining: {remaining}</span>
                      </div>
                      <Progress 
                        value={stats.percentage} 
                        className={`h-3 ${
                          isLimitReached ? '[&>[data-state="complete"]]:bg-red-500' :
                          remaining === 1 ? '[&>[data-state="complete"]]:bg-yellow-500' :
                          '[&>[data-state="complete"]]:bg-blue-600'
                        }`}
                      />
                      {!isLimitReached && (
                        <p className="text-sm text-muted-foreground flex items-center gap-1">
                          <Clock className="w-4 h-4" />
                          Resets in: {timeUntilReset}
                        </p>
                      )}
                    </div>
                  )}

                  {/* Benefits list */}
                  {(isLimitReached || shouldShowUpgrade) && (
                    <div className="space-y-3">
                      <p className="text-base font-semibold">Get unlimited access with a free account:</p>
                      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-sm text-muted-foreground">
                        <div className="flex items-center gap-2">
                          <Zap className="w-4 h-4 text-green-600" />
                          Unlimited scans
                        </div>
                        <div className="flex items-center gap-2">
                          <Star className="w-4 h-4 text-green-600" />
                          Analysis history
                        </div>
                        <div className="flex items-center gap-2">
                          <Sparkles className="w-4 h-4 text-green-600" />
                          Advanced reports
                        </div>
                        <div className="flex items-center gap-2">
                          <CheckCircle className="w-4 h-4 text-green-600" />
                          Custom preferences
                        </div>
                      </div>
                    </div>
                  )}
                </div>

                {/* Action buttons */}
                <div className="flex-shrink-0 space-y-3">
                  {(isLimitReached || shouldShowUpgrade) && (
                    <Button 
                      onClick={handleSignUp}
                      size="lg"
                      className="bg-primary hover:bg-primary/90 text-primary-foreground font-semibold px-6 py-3 text-base"
                    >
                      <UserPlus className="w-5 h-5 mr-2" />
                      Sign Up Free
                    </Button>
                  )}
                  
                  {remaining > 0 && !isAuthenticated && (
                    <Badge variant="secondary" className="text-sm font-semibold py-1 px-3">
                      {remaining} Free {remaining === 1 ? 'Scan' : 'Scans'} Left
                    </Badge>
                  )}
                </div>
              </div>
            </CardContent>
          </Card>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

// Compact version for smaller spaces
export function UsageLimitCompact() {
  const { isAuthenticated, remaining, isLimitReached, shouldShowUpgrade } = useUsageManager();
  
  if (isAuthenticated) return null;

  const handleSignUp = () => {
    signIn(undefined, { callbackUrl: '/dashboard' });
  };

  if (isLimitReached) {
    return (
      <Alert className="border-red-200 bg-red-50 dark:bg-red-950/20">
        <AlertTriangle className="h-4 w-4 text-red-600 dark:text-red-400" />
        <AlertDescription className="flex items-center justify-between w-full">
          <span className="text-sm">Free scans used up!</span>
          <Button 
            size="sm" 
            variant="destructive"
            onClick={handleSignUp}
            className="h-7 text-xs"
          >
            Sign Up Free
          </Button>
        </AlertDescription>
      </Alert>
    );
  }

  if (shouldShowUpgrade) {
    return (
      <Alert className="border-yellow-200 bg-yellow-50 dark:bg-yellow-950/20">
        <Clock className="h-4 w-4 text-yellow-600 dark:text-yellow-400" />
        <AlertDescription className="flex items-center justify-between w-full">
          <span className="text-sm">{remaining} free scan left</span>
          <Button 
            size="sm" 
            variant="outline"
            onClick={handleSignUp}
            className="h-7 text-xs border-yellow-300 hover:bg-yellow-100"
          >
            Get Unlimited
          </Button>
        </AlertDescription>
      </Alert>
    );
  }

  return (
    <div className="flex items-center gap-2 text-sm text-muted-foreground">
      <CheckCircle className="w-4 h-4 text-green-500" />
      <span>{remaining} free scans remaining</span>
    </div>
  );
}