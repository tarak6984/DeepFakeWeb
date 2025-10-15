/**
 * Confidence thresholds configuration for the deepfake detection system.
 * These thresholds have been updated to provide more conservative and accurate detection.
 * 
 * Updated: October 2025 - Increased thresholds to 95% system for production use
 */

export const CONFIDENCE_THRESHOLDS = {
  // Main classification thresholds (must match backend config.yaml)
  LOW_RISK_MAX: 0.9,     // 0-90%: Low risk - likely authentic
  MEDIUM_RISK_MAX: 0.95, // 90-95%: Medium risk - uncertain
  HIGH_RISK_MIN: 0.95,   // 95-100%: High risk - likely fake

  // Legacy thresholds (deprecated - kept for reference)
  LEGACY_LOW_RISK_MAX: 0.3,
  LEGACY_MEDIUM_RISK_MAX: 0.7,
} as const;

export const RISK_LEVEL_RANGES = {
  LOW_RISK: '0-90%',
  MEDIUM_RISK: '90-95%', 
  HIGH_RISK: '95-100%'
} as const;

export const RISK_LEVEL_DESCRIPTIONS = {
  LOW_RISK: 'Content appears authentic with low likelihood of manipulation',
  MEDIUM_RISK: 'Analysis inconclusive - further verification may be needed', 
  HIGH_RISK: 'High confidence detection of potential manipulation or deepfake content'
} as const;

export const CONFIDENCE_COLORS = {
  LOW_RISK: '#10b981',    // green
  MEDIUM_RISK: '#f59e0b', // yellow  
  HIGH_RISK: '#ef4444'    // red
} as const;

/**
 * Helper function to get risk level based on confidence score
 */
export function getRiskLevel(confidence: number): 'LOW_RISK' | 'MEDIUM_RISK' | 'HIGH_RISK' {
  if (confidence < CONFIDENCE_THRESHOLDS.LOW_RISK_MAX) return 'LOW_RISK';
  if (confidence < CONFIDENCE_THRESHOLDS.MEDIUM_RISK_MAX) return 'MEDIUM_RISK';
  return 'HIGH_RISK';
}

/**
 * Helper function to get risk level label
 */
export function getRiskLevelLabel(confidence: number): string {
  const level = getRiskLevel(confidence);
  return level.replace('_', ' ').toLowerCase().replace(/\b\w/g, l => l.toUpperCase());
}

/**
 * Helper function to get color for confidence score
 */
export function getConfidenceColor(confidence: number): string {
  const level = getRiskLevel(confidence);
  switch (level) {
    case 'LOW_RISK': return CONFIDENCE_COLORS.LOW_RISK;
    case 'MEDIUM_RISK': return CONFIDENCE_COLORS.MEDIUM_RISK;
    case 'HIGH_RISK': return CONFIDENCE_COLORS.HIGH_RISK;
  }
}

/**
 * Helper function to get CSS text color class for confidence score
 */
export function getConfidenceTextColorClass(confidence: number): string {
  if (confidence < CONFIDENCE_THRESHOLDS.LOW_RISK_MAX) return 'text-green-600 dark:text-green-400';
  if (confidence < CONFIDENCE_THRESHOLDS.MEDIUM_RISK_MAX) return 'text-yellow-600 dark:text-yellow-400';
  return 'text-red-600 dark:text-red-400';
}