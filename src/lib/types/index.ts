export interface FileUploadState {
  file: File | null;
  preview: string | null;
  progress: number;
  status: 'idle' | 'uploading' | 'processing' | 'completed' | 'error';
  error?: string;
}

export interface UploadProgress {
  percentage: number;
  stage: 'uploading' | 'preprocessing' | 'analyzing_audio' | 'analyzing_video' | 'analyzing_image' | 'model_processing' | 'ensemble_analysis' | 'generating_results' | 'completing';
  message: string;
  detailedMessage?: string;
  subStep?: string;
  stagesCompleted?: string[];
  
  // Time tracking
  timeElapsed?: number;
  estimatedTimeRemaining?: number;
  startTime?: number;
  
  // Processing counts
  processedFrames?: number;
  totalFrames?: number;
  processedSegments?: number;
  totalSegments?: number;
  processedSamples?: number;
  totalSamples?: number;
  
  // Model-specific details
  analysisDetails?: {
    activeModels: string[];
    completedModels: number;
    totalModels: number;
    modelStatuses: Record<string, string>;
    currentPhase: string;
    currentModelName?: string;
    modelProgress?: number;
    modelConfidence?: number;
  };
  
  // Stage-specific progress
  stageProgress?: {
    upload?: { completed: boolean; progress: number };
    preprocessing?: { completed: boolean; progress: number; subSteps?: string[] };
    analysis?: { completed: boolean; progress: number; subSteps?: string[] };
    results?: { completed: boolean; progress: number; subSteps?: string[] };
  };
  
  // Real-time confidence updates
  intermediateResults?: {
    currentConfidence?: number;
    confidenceTrend?: 'increasing' | 'decreasing' | 'stable';
    detectedAnomalies?: string[];
    processingQuality?: 'excellent' | 'good' | 'fair' | 'poor';
  };
  
  // Animation and display hints
  showProgressBar?: boolean;
  progressBarVariant?: 'default' | 'success' | 'warning' | 'danger';
  animationSpeed?: 'slow' | 'normal' | 'fast';
}

export interface ChartDataPoint {
  name: string;
  value: number;
  color?: string;
}

export interface TimelineDataPoint {
  timestamp: number;
  frame?: number;
  confidence: number;
  anomalies?: string[];
}

export interface AudioSegment {
  start: number;
  end: number;
  confidence: number;
  anomalies?: string[];
}

export interface WaveformData {
  data: number[];
  duration: number;
  sampleRate: number;
}

export interface ConfidenceLevel {
  value: number;
  label: 'High' | 'Medium' | 'Low';
  color: string;
  textColor: string;
}

export interface DetectionCategory {
  name: string;
  percentage: number;
  color: string;
  description?: string;
}

export interface MediaMetadata {
  filename: string;
  fileSize: number;
  fileType: string;
  duration?: number;
  resolution?: string;
  codec?: string;
  bitrate?: number;
  createdAt?: string;
}

export interface APIStatus {
  status: 'online' | 'offline' | 'demo_mode' | 'error';
  version?: string;
  latency?: number;
  lastCheck?: string;
}

export interface ExportOptions {
  format: 'json' | 'pdf' | 'csv';
  includeMetadata: boolean;
  includeCharts: boolean;
  includeTimeline: boolean;
}

export interface NavigationItem {
  href: string;
  label: string;
  icon: string;
  active?: boolean;
}

export interface AlertConfig {
  type: 'success' | 'error' | 'warning' | 'info';
  title: string;
  message: string;
  duration?: number;
  dismissible?: boolean;
}

// UI Component Props Types
export interface BaseComponentProps {
  className?: string;
  children?: React.ReactNode;
}

export interface LoadingSpinnerProps extends BaseComponentProps {
  size?: 'sm' | 'md' | 'lg';
  color?: string;
}

export interface EmptyStateProps extends BaseComponentProps {
  icon?: string;
  title: string;
  description?: string;
  action?: {
    label: string;
    onClick: () => void;
  };
}

export interface StatCardProps extends BaseComponentProps {
  title: string;
  value: string | number;
  subtitle?: string;
  icon?: React.ReactNode;
  trend?: {
    value: number;
    isPositive: boolean;
  };
}

// Animation and Motion Types
export interface MotionVariants {
  initial: object;
  animate: object;
  exit?: object;
}

export interface PageTransition {
  duration: number;
  ease: string;
}

// Form and Input Types
export interface FieldError {
  field: string;
  message: string;
}

export interface FormValidationResult {
  isValid: boolean;
  errors: FieldError[];
}

// Utility Types
export type Maybe<T> = T | null | undefined;
export type Optional<T, K extends keyof T> = Omit<T, K> & Partial<Pick<T, K>>;
export type RequiredFields<T, K extends keyof T> = T & Required<Pick<T, K>>;

// Event Handler Types
export type FileUploadHandler = (file: File) => void | Promise<void>;
export type AnalysisCompleteHandler = (result: unknown) => void;
export type ThemeChangeHandler = (theme: 'light' | 'dark' | 'system') => void;

// API Response Types
export interface APIResponse<T = unknown> {
  success: boolean;
  data?: T;
  error?: {
    code: string;
    message: string;
    details?: unknown;
  };
  meta?: {
    timestamp: string;
    requestId: string;
    version: string;
  };
}

// Explainability Types
export interface ExplanationReason {
  id: string;
  category: 'visual' | 'audio' | 'metadata' | 'model' | 'temporal' | 'technical';
  type: 'evidence' | 'anomaly' | 'pattern' | 'inconsistency' | 'artifact';
  severity: 'high' | 'medium' | 'low';
  confidence: number;
  title: string;
  description: string;
  technicalDetails?: string;
  affectedRegions?: Array<{
    startTime?: number;
    endTime?: number;
    frame?: number;
    coordinates?: { x: number; y: number; width: number; height: number; };
  }>;
  supportingEvidence?: string[];
  modelSources?: string[];
}

export interface ExplanationEvidence {
  id: string;
  type: 'visual_artifact' | 'audio_distortion' | 'compression_anomaly' | 'temporal_inconsistency' | 'statistical_anomaly';
  location: {
    frame?: number;
    timestamp?: number;
    coordinates?: { x: number; y: number; width: number; height: number; };
    frequency?: number;
  };
  severity: number; // 0-1 scale
  description: string;
  visualData?: {
    heatmapData?: number[][];
    highlightRegions?: Array<{ x: number; y: number; width: number; height: number; }>;
    annotationImage?: string;
  };
}

export interface ModelInsight {
  modelName: string;
  modelType: 'deepfake_detector' | 'face_analysis' | 'audio_analysis' | 'compression_analysis' | 'ensemble';
  confidence: number;
  prediction: 'authentic' | 'manipulated' | 'inconclusive';
  keyFindings: string[];
  technicalScore: number;
  processingTime: number;
  reasoning: string;
  supportingMetrics?: Record<string, number>;
}

export interface ExplanationSummary {
  primaryReason: string;
  secondaryReasons: string[];
  overallConfidence: number;
  authenticityIndicators: Array<{
    factor: string;
    weight: number;
    contribution: 'positive' | 'negative' | 'neutral';
    explanation: string;
  }>;
  riskFactors: Array<{
    factor: string;
    severity: 'critical' | 'high' | 'medium' | 'low';
    likelihood: number;
    impact: string;
  }>;
  recommendedActions?: string[];
}

export interface DetailedExplanation {
  id: string;
  analysisId: string;
  summary: ExplanationSummary;
  reasons: ExplanationReason[];
  evidence: ExplanationEvidence[];
  modelInsights: ModelInsight[];
  temporalAnalysis?: {
    frameByFrameReasons: Array<{
      frame: number;
      timestamp: number;
      primaryConcerns: string[];
      confidenceChange?: number;
    }>;
    overallTrends: string[];
  };
  metadataAnalysis?: {
    fileProperties: Array<{
      property: string;
      expectedValue?: string | number;
      actualValue: string | number;
      assessment: 'normal' | 'suspicious' | 'anomalous';
      explanation: string;
    }>;
    processingHistory?: string[];
  };
  generatedAt: string;
  processingVersion: string;
}

export interface ExplanationConfig {
  includeModelInsights: boolean;
  includeTechnicalDetails: boolean;
  includeVisualEvidence: boolean;
  simplifyForGeneralAudience: boolean;
  maxReasonsToShow: number;
  evidenceVisualization: boolean;
}

// Feature Flags
export interface FeatureFlags {
  enableBetaFeatures: boolean;
  enableAdvancedAnalytics: boolean;
  enableExperimentalCharts: boolean;
  enableOfflineMode: boolean;
  enableExplanations: boolean;
  enableAdvancedExplanations: boolean;
}

export const DEFAULT_FEATURE_FLAGS: FeatureFlags = {
  enableBetaFeatures: false,
  enableAdvancedAnalytics: true,
  enableExperimentalCharts: false,
  enableOfflineMode: false,
  enableExplanations: true,
  enableAdvancedExplanations: true,
};
