interface UsageStats {
  totalScans: number;
  monthlyScans: number;
  currentMonth: string;
  lastReset: string;
  scanHistory: Array<{
    date: string;
    fileType: string;
    fileName: string;
    confidence: number;
    prediction: string;
  }>;
  freeTierLimit: number;
}

class UsageTracker {
  private storageKey = 'rd_usage_stats';
  private freeTierLimit = 50; // Reality Defender free tier limit

  getUsageStats(): UsageStats {
    // SSR-safe localStorage access
    if (typeof window === 'undefined') {
      return this.createInitialStats();
    }
    
    const stored = localStorage.getItem(this.storageKey);
    if (!stored) {
      return this.createInitialStats();
    }

    const stats: UsageStats = JSON.parse(stored);
    
    // Check if we need to reset monthly count
    const currentMonth = new Date().toISOString().slice(0, 7); // YYYY-MM
    if (stats.currentMonth !== currentMonth) {
      stats.monthlyScans = 0;
      stats.currentMonth = currentMonth;
      stats.lastReset = new Date().toISOString();
      this.saveUsageStats(stats);
    }

    return stats;
  }

  private createInitialStats(): UsageStats {
    const currentMonth = new Date().toISOString().slice(0, 7);
    const stats: UsageStats = {
      totalScans: 0,
      monthlyScans: 0,
      currentMonth,
      lastReset: new Date().toISOString(),
      scanHistory: [],
      freeTierLimit: this.freeTierLimit,
    };
    
    // Only save if we're on the client
    if (typeof window !== 'undefined') {
      this.saveUsageStats(stats);
    }
    return stats;
  }

  incrementUsage(fileType: string, fileName: string, confidence: number, prediction: string): UsageStats {
    const stats = this.getUsageStats();
    
    stats.totalScans += 1;
    stats.monthlyScans += 1;
    
    stats.scanHistory.unshift({
      date: new Date().toISOString(),
      fileType,
      fileName,
      confidence,
      prediction,
    });
    
    // Keep only last 100 entries to prevent storage bloat
    if (stats.scanHistory.length > 100) {
      stats.scanHistory = stats.scanHistory.slice(0, 100);
    }
    
    this.saveUsageStats(stats);
    return stats;
  }

  private saveUsageStats(stats: UsageStats): void {
    localStorage.setItem(this.storageKey, JSON.stringify(stats));
  }

  getRemainingScans(): number {
    const stats = this.getUsageStats();
    return Math.max(0, this.freeTierLimit - stats.monthlyScans);
  }

  getUsagePercentage(): number {
    const stats = this.getUsageStats();
    return Math.min(100, (stats.monthlyScans / this.freeTierLimit) * 100);
  }

  canMakeScan(): boolean {
    return this.getRemainingScans() > 0;
  }

  getMonthlyUsageByFileType(): Record<string, number> {
    const stats = this.getUsageStats();
    const currentMonth = new Date().toISOString().slice(0, 7);
    
    const monthlyHistory = stats.scanHistory.filter(
      scan => scan.date.slice(0, 7) === currentMonth
    );
    
    const usage: Record<string, number> = {};
    monthlyHistory.forEach(scan => {
      const type = scan.fileType.split('/')[0]; // Get 'image', 'audio', etc.
      usage[type] = (usage[type] || 0) + 1;
    });
    
    return usage;
  }

  getConfidenceDistribution(): { range: string; count: number }[] {
    const stats = this.getUsageStats();
    const ranges = [
      { min: 0, max: 0.2, label: '0-20%' },
      { min: 0.2, max: 0.4, label: '20-40%' },
      { min: 0.4, max: 0.6, label: '40-60%' },
      { min: 0.6, max: 0.8, label: '60-80%' },
      { min: 0.8, max: 1, label: '80-100%' },
    ];
    
    return ranges.map(range => ({
      range: range.label,
      count: stats.scanHistory.filter(
        scan => scan.confidence >= range.min && scan.confidence < range.max
      ).length,
    }));
  }

  getPredictionStats(): Record<string, number> {
    const stats = this.getUsageStats();
    const predictions: Record<string, number> = {};
    
    stats.scanHistory.forEach(scan => {
      predictions[scan.prediction] = (predictions[scan.prediction] || 0) + 1;
    });
    
    return predictions;
  }

  getWeeklyUsage(): Array<{ day: string; count: number }> {
    const stats = this.getUsageStats();
    const last7Days = Array.from({ length: 7 }, (_, i) => {
      const date = new Date();
      date.setDate(date.getDate() - i);
      return date.toISOString().slice(0, 10);
    }).reverse();
    
    return last7Days.map(day => ({
      day: day.slice(5), // MM-DD format
      count: stats.scanHistory.filter(scan => scan.date.slice(0, 10) === day).length,
    }));
  }
}

export const usageTracker = new UsageTracker();
export type { UsageStats };