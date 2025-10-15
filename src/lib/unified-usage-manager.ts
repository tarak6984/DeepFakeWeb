// Unified Usage Manager - Handles both anonymous and authenticated users
import { useSession } from 'next-auth/react';
import { anonymousUsageTracker } from './anonymous-usage-tracker';
import { usageTracker } from './usage-tracker';

export interface UsageStats {
  isAuthenticated: boolean;
  canPerformScan: boolean;
  remaining: number;
  used: number;
  limit: number;
  percentage: number;
  resetTime?: string;
  plan?: string;
}

export class UnifiedUsageManager {
  constructor() {}

  private isClient(): boolean {
    return typeof window !== 'undefined';
  }

  // Get usage statistics for current user (authenticated or anonymous)
  public getUsageStats(session?: any): UsageStats {
    if (session?.user) {
      // Authenticated user - unlimited scans
      const authStats = usageTracker.getUsageStats();
      return {
        isAuthenticated: true,
        canPerformScan: true,
        remaining: -1, // Unlimited
        used: authStats.totalScans,
        limit: -1, // Unlimited
        percentage: 0,
        plan: session.user.plan || 'FREE',
      };
    } else {
      // Anonymous user - limited scans
      const anonStats = anonymousUsageTracker.getUsageStats();
      return {
        isAuthenticated: false,
        canPerformScan: anonStats.remaining > 0,
        remaining: anonStats.remaining,
        used: anonStats.scansUsed,
        limit: anonStats.maxScans,
        percentage: anonStats.percentage,
        resetTime: anonStats.resetTime,
      };
    }
  }

  // Check if user can perform a scan
  public canPerformScan(session?: any): boolean {
    if (session?.user) {
      return true; // Authenticated users have unlimited scans
    } else {
      return anonymousUsageTracker.canPerformScan();
    }
  }

  // Consume a scan (for anonymous users only, authenticated users don't consume)
  public consumeScan(session?: any): boolean {
    if (session?.user) {
      // For authenticated users, we track in the database but don't limit
      // This will be handled by the API routes and database service
      return true;
    } else {
      // For anonymous users, consume from localStorage limit
      return anonymousUsageTracker.consumeScan();
    }
  }

  // Get remaining scans
  public getRemainingScans(session?: any): number {
    if (session?.user) {
      return -1; // Unlimited
    } else {
      return anonymousUsageTracker.getRemainingScans();
    }
  }

  // Get used scans count
  public getUsedScans(session?: any): number {
    if (session?.user) {
      return usageTracker.getUsageStats().totalScans;
    } else {
      return anonymousUsageTracker.getUsedScans();
    }
  }

  // Clear anonymous usage (when user signs up)
  public clearAnonymousUsage(): void {
    anonymousUsageTracker.clearUsage();
  }

  // Get time until reset for anonymous users
  public getTimeUntilReset(session?: any): string {
    if (session?.user) {
      return 'Never'; // Authenticated users don't have reset limits
    } else {
      return anonymousUsageTracker.getTimeUntilReset();
    }
  }

  // Get usage message for UI
  public getUsageMessage(session?: any): string {
    if (session?.user) {
      return 'Unlimited scans available';
    } else {
      const remaining = this.getRemainingScans();
      if (remaining === 0) {
        return 'Free scan limit reached. Sign up for unlimited access!';
      } else if (remaining === 1) {
        return `${remaining} free scan remaining. Sign up for unlimited access!`;
      } else {
        return `${remaining} free scans remaining`;
      }
    }
  }

  // Get upgrade message for anonymous users who hit limit
  public getUpgradeMessage(): string {
    return 'Get unlimited scans by creating a free account! Enjoy advanced features, analysis history, and detailed reports.';
  }
}

// React hook for easy usage in components
export function useUsageManager() {
  const { data: session } = useSession();
  const manager = new UnifiedUsageManager();

  const stats = manager.getUsageStats(session);
  const canScan = manager.canPerformScan(session);
  const remaining = manager.getRemainingScans(session);
  const used = manager.getUsedScans(session);
  const message = manager.getUsageMessage(session);
  const timeUntilReset = manager.getTimeUntilReset(session);

  const consumeScan = () => manager.consumeScan(session);
  const clearAnonymousUsage = () => manager.clearAnonymousUsage();

  return {
    // User info
    isAuthenticated: !!session?.user,
    user: session?.user,
    
    // Usage stats
    stats,
    canScan,
    remaining,
    used,
    message,
    timeUntilReset,
    
    // Actions
    consumeScan,
    clearAnonymousUsage,
    
    // Utility
    isLimitReached: !canScan && !session?.user,
    shouldShowUpgrade: !session?.user && remaining <= 1,
    upgradeMessage: manager.getUpgradeMessage(),
  };
}

// Export singleton instance
export const unifiedUsageManager = new UnifiedUsageManager();