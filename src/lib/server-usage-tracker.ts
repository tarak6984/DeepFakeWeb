// Server-side Anonymous Usage Tracker
// This validates usage on the server side using session/IP tracking

export interface ServerUsageCheck {
  allowed: boolean;
  remaining: number;
  resetTime: Date;
  isAuthenticated: boolean;
}

const USAGE_CACHE = new Map<string, { count: number; resetTime: Date }>();
const MAX_FREE_SCANS = 3;
const RESET_INTERVAL = 24 * 60 * 60 * 1000; // 24 hours

function getClientIdentifier(request: Request): string {
  // Try to get IP from various headers (for production with proxies)
  const forwarded = request.headers.get('x-forwarded-for');
  const realIp = request.headers.get('x-real-ip');
  const remoteAddr = request.headers.get('x-remote-addr');
  
  // In development, headers might not be available
  const ip = forwarded?.split(',')[0] || realIp || remoteAddr || 'unknown';
  
  // Include User-Agent for additional uniqueness (but keep it simple)
  const userAgent = request.headers.get('user-agent') || 'unknown';
  const shortUA = userAgent.split(' ')[0]; // Just get the first part
  
  return `${ip}_${shortUA}`;
}

function cleanupExpiredEntries(): void {
  const now = new Date();
  for (const [key, data] of USAGE_CACHE.entries()) {
    if (now > data.resetTime) {
      USAGE_CACHE.delete(key);
    }
  }
}

export function checkServerUsage(request: Request, isAuthenticated: boolean = false): ServerUsageCheck {
  // Authenticated users have unlimited access
  if (isAuthenticated) {
    return {
      allowed: true,
      remaining: -1, // Unlimited
      resetTime: new Date(Date.now() + RESET_INTERVAL),
      isAuthenticated: true,
    };
  }

  // Clean up expired entries periodically
  cleanupExpiredEntries();

  const clientId = getClientIdentifier(request);
  const now = new Date();
  const resetTime = new Date(now.getTime() + RESET_INTERVAL);

  // Get or create usage data for this client
  let usageData = USAGE_CACHE.get(clientId);
  
  if (!usageData || now > usageData.resetTime) {
    // Create new or reset expired usage data
    usageData = {
      count: 0,
      resetTime,
    };
    USAGE_CACHE.set(clientId, usageData);
  }

  const remaining = Math.max(0, MAX_FREE_SCANS - usageData.count);

  return {
    allowed: remaining > 0,
    remaining,
    resetTime: usageData.resetTime,
    isAuthenticated: false,
  };
}

export function consumeServerUsage(request: Request, isAuthenticated: boolean = false): boolean {
  // Authenticated users don't consume usage
  if (isAuthenticated) {
    return true;
  }

  const clientId = getClientIdentifier(request);
  const usageData = USAGE_CACHE.get(clientId);
  
  if (!usageData) {
    return false; // Should have been created by checkServerUsage
  }

  if (usageData.count >= MAX_FREE_SCANS) {
    return false; // Already at limit
  }

  // Increment usage count
  usageData.count += 1;
  USAGE_CACHE.set(clientId, usageData);

  return true;
}

// Get current usage stats for debugging/logging
export function getServerUsageStats(): {
  totalClients: number;
  activeClients: number;
  totalUsage: number;
} {
  cleanupExpiredEntries();
  
  let totalUsage = 0;
  for (const data of USAGE_CACHE.values()) {
    totalUsage += data.count;
  }

  return {
    totalClients: USAGE_CACHE.size,
    activeClients: USAGE_CACHE.size, // After cleanup, all are active
    totalUsage,
  };
}