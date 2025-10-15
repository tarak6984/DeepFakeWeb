// Anonymous Usage Tracker - Manages scan limits for guest users
import { toast } from 'sonner';

const STORAGE_KEYS = {
  ANONYMOUS_USAGE: 'deepfake_anonymous_usage',
  LAST_RESET: 'deepfake_last_reset',
} as const;

interface AnonymousUsage {
  scansUsed: number;
  maxScans: number;
  lastReset: string;
  sessionId: string;
}

export class AnonymousUsageTracker {
  private readonly MAX_FREE_SCANS = 3;
  private readonly RESET_INTERVAL = 24 * 60 * 60 * 1000; // 24 hours in milliseconds

  constructor() {
    this.initializeIfNeeded();
  }

  private isClient(): boolean {
    return typeof window !== 'undefined';
  }

  private generateSessionId(): string {
    return `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  private initializeIfNeeded(): void {
    if (!this.isClient()) return;

    const stored = this.getStoredUsage();
    if (!stored || this.shouldReset(stored)) {
      this.resetUsage();
    }
  }

  private getStoredUsage(): AnonymousUsage | null {
    if (!this.isClient()) return null;

    try {
      const stored = localStorage.getItem(STORAGE_KEYS.ANONYMOUS_USAGE);
      return stored ? JSON.parse(stored) : null;
    } catch (error) {
      console.error('Failed to read anonymous usage:', error);
      return null;
    }
  }

  private setStoredUsage(usage: AnonymousUsage): void {
    if (!this.isClient()) return;

    try {
      localStorage.setItem(STORAGE_KEYS.ANONYMOUS_USAGE, JSON.stringify(usage));
    } catch (error) {
      console.error('Failed to store anonymous usage:', error);
    }
  }

  private shouldReset(usage: AnonymousUsage): boolean {
    const now = Date.now();
    const lastReset = new Date(usage.lastReset).getTime();
    return now - lastReset > this.RESET_INTERVAL;
  }

  private resetUsage(): void {
    const newUsage: AnonymousUsage = {
      scansUsed: 0,
      maxScans: this.MAX_FREE_SCANS,
      lastReset: new Date().toISOString(),
      sessionId: this.generateSessionId(),
    };
    this.setStoredUsage(newUsage);
  }

  // Public methods
  public getRemainingScans(): number {
    const usage = this.getStoredUsage();
    if (!usage) return this.MAX_FREE_SCANS;
    
    return Math.max(0, usage.maxScans - usage.scansUsed);
  }

  public getUsedScans(): number {
    const usage = this.getStoredUsage();
    return usage?.scansUsed || 0;
  }

  public getMaxScans(): number {
    return this.MAX_FREE_SCANS;
  }

  public hasScansRemaining(): boolean {
    return this.getRemainingScans() > 0;
  }

  public canPerformScan(): boolean {
    return this.hasScansRemaining();
  }

  public consumeScan(): boolean {
    if (!this.canPerformScan()) {
      return false;
    }

    const usage = this.getStoredUsage();
    if (!usage) return false;

    const updatedUsage: AnonymousUsage = {
      ...usage,
      scansUsed: usage.scansUsed + 1,
    };

    this.setStoredUsage(updatedUsage);

    // Show helpful messages
    const remaining = this.MAX_FREE_SCANS - updatedUsage.scansUsed;
    if (remaining === 0) {
      toast.error('Free scan limit reached!', {
        description: 'Sign up for unlimited scans and advanced features.',
        duration: 5000,
      });
    } else if (remaining === 1) {
      toast.warning(`${remaining} free scan remaining`, {
        description: 'Sign up to get unlimited scans!',
        duration: 4000,
      });
    } else if (remaining > 0) {
      toast.success('Analysis completed!', {
        description: `${remaining} free scans remaining.`,
        duration: 3000,
      });
    }

    return true;
  }

  public getUsageStats() {
    const usage = this.getStoredUsage();
    if (!usage) {
      return {
        scansUsed: 0,
        maxScans: this.MAX_FREE_SCANS,
        remaining: this.MAX_FREE_SCANS,
        percentage: 0,
        resetTime: new Date().toISOString(),
        sessionId: '',
      };
    }

    const remaining = Math.max(0, usage.maxScans - usage.scansUsed);
    const percentage = (usage.scansUsed / usage.maxScans) * 100;

    return {
      scansUsed: usage.scansUsed,
      maxScans: usage.maxScans,
      remaining,
      percentage: Math.round(percentage),
      resetTime: usage.lastReset,
      sessionId: usage.sessionId,
    };
  }

  public getTimeUntilReset(): string {
    const usage = this.getStoredUsage();
    if (!usage) return '24 hours';

    const resetTime = new Date(usage.lastReset).getTime() + this.RESET_INTERVAL;
    const now = Date.now();
    const timeLeft = resetTime - now;

    if (timeLeft <= 0) return 'Available now';

    const hours = Math.floor(timeLeft / (60 * 60 * 1000));
    const minutes = Math.floor((timeLeft % (60 * 60 * 1000)) / (60 * 1000));

    if (hours > 0) {
      return `${hours}h ${minutes}m`;
    }
    return `${minutes}m`;
  }

  public clearUsage(): void {
    if (!this.isClient()) return;
    
    try {
      localStorage.removeItem(STORAGE_KEYS.ANONYMOUS_USAGE);
      localStorage.removeItem(STORAGE_KEYS.LAST_RESET);
    } catch (error) {
      console.error('Failed to clear anonymous usage:', error);
    }
  }

  // For debugging
  public getDebugInfo() {
    const usage = this.getStoredUsage();
    return {
      stored: usage,
      remaining: this.getRemainingScans(),
      canScan: this.canPerformScan(),
      timeUntilReset: this.getTimeUntilReset(),
      shouldReset: usage ? this.shouldReset(usage) : false,
    };
  }
}

// Export singleton instance
export const anonymousUsageTracker = new AnonymousUsageTracker();