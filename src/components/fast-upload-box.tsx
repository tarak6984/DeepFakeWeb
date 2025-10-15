'use client';

import { useState, useCallback, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import {
  Upload,
  Image as ImageIcon,
  Video,
  Volume2,
  CheckCircle,
  AlertTriangle,
  X,
  Zap,
  Clock,
  Cpu,
  Activity,
  Eye,
  Layers,
  TrendingUp,
  TrendingDown,
  Minus,
  Brain,
} from 'lucide-react';
import type { FileUploadState, UploadProgress } from '@/lib/types';

interface FastUploadBoxProps {
  onFileSelect: (file: File) => Promise<void>;
  onFileRemove: () => void;
  uploadState: FileUploadState;
  uploadProgress?: UploadProgress;
  className?: string;
}

const getFileIcon = (type: string) => {
  if (type.startsWith('image/')) return ImageIcon;
  if (type.startsWith('video/')) return Video;
  if (type.startsWith('audio/')) return Volume2;
  return Upload;
};

const getStatusColor = (status: string) => {
  switch (status) {
    case 'completed': return 'text-green-600 dark:text-green-400';
    case 'error': return 'text-red-600 dark:text-red-400';
    case 'uploading': return 'text-blue-600 dark:text-blue-400';
    case 'processing': return 'text-yellow-600 dark:text-yellow-400';
    default: return 'text-muted-foreground';
  }
};

const getStatusIcon = (status: string) => {
  switch (status) {
    case 'completed': return CheckCircle;
    case 'error': return AlertTriangle;
    default: return Zap;
  }
};

const formatTime = (seconds: number): string => {
  if (seconds < 60) return `${Math.floor(seconds)}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${Math.floor(seconds % 60)}s`;
  return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`;
};

const getStageIcon = (stage: string) => {
  switch (stage) {
    case 'uploading': return Upload;
    case 'preprocessing': return Layers;
    case 'analyzing_audio': return Volume2;
    case 'analyzing_video': return Video;
    case 'analyzing_image': return ImageIcon;
    case 'model_processing': return Brain;
    case 'ensemble_analysis': return Cpu;
    case 'generating_results': return Activity;
    case 'completing': return CheckCircle;
    default: return Zap;
  }
};

const getConfidenceTrendIcon = (trend?: 'increasing' | 'decreasing' | 'stable') => {
  switch (trend) {
    case 'increasing': return TrendingUp;
    case 'decreasing': return TrendingDown;
    case 'stable': return Minus;
    default: return Activity;
  }
};

export function FastUploadBox({
  onFileSelect,
  onFileRemove,
  uploadState,
  uploadProgress,
  className = '',
}: FastUploadBoxProps) {
  const [dragOver, setDragOver] = useState(false);
  const [realTimeElapsed, setRealTimeElapsed] = useState(0);

  // Real-time timer effect
  useEffect(() => {
    let interval: NodeJS.Timeout;
    
    if (uploadState.status === 'processing' && uploadProgress?.startTime) {
      interval = setInterval(() => {
        const elapsed = Math.floor((Date.now() - uploadProgress.startTime!) / 1000);
        setRealTimeElapsed(elapsed);
      }, 1000);
    } else {
      setRealTimeElapsed(0);
    }
    
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [uploadState.status, uploadProgress?.startTime]);

  const handleFileChange = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      onFileSelect(file);
    }
  }, [onFileSelect]);

  const handleDrop = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    setDragOver(false);
    const file = event.dataTransfer.files?.[0];
    if (file) {
      onFileSelect(file);
    }
  }, [onFileSelect]);

  const handleDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    setDragOver(true);
  }, []);

  const handleDragLeave = useCallback(() => {
    setDragOver(false);
  }, []);

  const FileIcon = uploadState.file ? getFileIcon(uploadState.file.type) : Upload;
  const StatusIcon = getStatusIcon(uploadState.status);

  return (
    <Card className={`w-full ${className}`}>
      <CardContent className="p-6">
        {!uploadState.file ? (
          // Upload Zone
          <div
            className={`
              border-2 border-dashed rounded-lg p-8 text-center transition-colors cursor-pointer
              ${dragOver 
                ? 'border-primary bg-primary/5' 
                : 'border-muted-foreground/25 hover:border-primary/50 hover:bg-muted/20'
              }
            `}
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onClick={() => document.getElementById('file-input')?.click()}
          >
            <Upload className="w-12 h-12 mx-auto mb-4 text-muted-foreground" />
            <h3 className="text-lg font-semibold mb-2">Upload Media File</h3>
            <p className="text-muted-foreground mb-4">
              Drop your file here or click to browse
            </p>
            <Badge variant="secondary" className="mb-2">
              <Zap className="w-3 h-3 mr-1" />
              Unlimited Local Processing
            </Badge>
            <p className="text-xs text-muted-foreground">
              Supports: Images, Audio, Video • Max 100MB
            </p>
            <input
              id="file-input"
              type="file"
              className="hidden"
              accept="image/*,video/*,audio/*"
              onChange={handleFileChange}
            />
          </div>
        ) : (
          // File Processing Zone
          <div className="space-y-4">
            {/* File Info */}
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <FileIcon className="w-8 h-8 text-primary" />
                <div>
                  <p className="font-medium truncate max-w-xs">{uploadState.file.name}</p>
                  <p className="text-sm text-muted-foreground">
                    {uploadState.file.size < 1024 * 1024 
                      ? `${(uploadState.file.size / 1024).toFixed(1)} KB`
                      : `${(uploadState.file.size / 1024 / 1024).toFixed(1)} MB`
                    }
                  </p>
                </div>
              </div>
              <Button
                variant="ghost"
                size="sm"
                onClick={onFileRemove}
                className="text-muted-foreground hover:text-foreground"
              >
                <X className="w-4 h-4" />
              </Button>
            </div>

            {/* Enhanced Status & Progress */}
            <div className="space-y-4">
              {/* Main Status Header */}
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  {uploadProgress && (() => {
                    const StageIcon = getStageIcon(uploadProgress.stage);
                    return (
                      <div className={`p-2 rounded-full bg-gradient-to-r ${
                        uploadState.status === 'processing' 
                          ? 'from-blue-500/20 to-purple-500/20 animate-pulse' 
                          : 'from-green-500/20 to-emerald-500/20'
                      }`}>
                        <StageIcon className={`w-5 h-5 ${getStatusColor(uploadState.status)}`} />
                      </div>
                    );
                  })()}
                  <div>
                    <span className={`text-lg font-semibold ${getStatusColor(uploadState.status)}`}>
                      {uploadProgress?.message || `Status: ${uploadState.status}`}
                    </span>
                    {uploadProgress?.detailedMessage && (
                      <p className="text-sm text-muted-foreground mt-1">
                        {uploadProgress.detailedMessage}
                      </p>
                    )}
                  </div>
                </div>
                
                {uploadProgress?.percentage && (
                  <div className="text-right">
                    <span className="text-2xl font-bold text-primary">
                      {uploadProgress.percentage}%
                    </span>
                    <p className="text-xs text-muted-foreground">Complete</p>
                  </div>
                )}
              </div>

              {/* Sub-step indicator */}
              {uploadProgress?.subStep && uploadState.status !== 'completed' && (
                <div className="flex items-center gap-2 text-sm">
                  <Activity className={`w-4 h-4 text-blue-500 ${
                    uploadState.status === 'processing' ? 'animate-spin' : ''
                  }`} />
                  <span className="font-medium text-blue-700 dark:text-blue-300">
                    {uploadProgress.subStep}
                  </span>
                </div>
              )}

              {/* Enhanced Progress Bar */}
              {uploadState.progress > 0 && (
                <div className="space-y-2">
                  <Progress 
                    value={uploadState.status === 'completed' ? 100 : uploadState.progress} 
                    className={`w-full h-3 ${
                      uploadState.status === 'completed' ? 'bg-green-200' :
                      uploadProgress?.progressBarVariant === 'success' ? 'bg-green-200' :
                      uploadProgress?.progressBarVariant === 'warning' ? 'bg-yellow-200' :
                      uploadProgress?.progressBarVariant === 'danger' ? 'bg-red-200' :
                      'bg-blue-200'
                    }`} 
                  />
                  
                  {/* Stage-specific progress indicators */}
                  {(uploadProgress?.stageProgress || uploadState.status === 'completed') && (
                    <div className="flex justify-between text-xs">
                      {uploadState.status === 'completed' ? (
                        // Show all stages as completed when analysis is done
                        ['upload', 'preprocessing', 'analysis', 'results'].map((stage) => (
                          <div key={stage} className="flex flex-col items-center gap-1">
                            <div className="w-2 h-2 rounded-full bg-green-500" />
                            <span className="capitalize text-green-600">
                              {stage}
                            </span>
                          </div>
                        ))
                      ) : (
                        // Show normal progress indicators during processing
                        uploadProgress?.stageProgress && Object.entries(uploadProgress.stageProgress).map(([stage, data]) => (
                          <div key={stage} className="flex flex-col items-center gap-1">
                            <div className={`w-2 h-2 rounded-full ${
                              data.completed ? 'bg-green-500' : 
                              data.progress > 0 ? 'bg-blue-500 animate-pulse' : 'bg-gray-300'
                            }`} />
                            <span className={`capitalize ${
                              data.completed ? 'text-green-600' : 'text-muted-foreground'
                            }`}>
                              {stage}
                            </span>
                          </div>
                        ))
                      )}
                    </div>
                  )}
                </div>
              )}

              {/* Detailed Processing Information */}
              {uploadProgress && uploadState.status === 'processing' && (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {/* Timing Information */}
                  <div className="space-y-2">
                    <div className="flex items-center gap-2 text-sm font-medium">
                      <Clock className="w-4 h-4 text-blue-500" />
                      <span>Timing</span>
                    </div>
                    <div className="space-y-1 text-xs">
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Elapsed:</span>
                        <span className="font-mono font-medium">
                          {formatTime(realTimeElapsed || uploadProgress.timeElapsed || 0)}
                        </span>
                      </div>
                      {uploadProgress.estimatedTimeRemaining && (
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Remaining:</span>
                          <span className="font-mono font-medium text-blue-600">
                            {formatTime(uploadProgress.estimatedTimeRemaining)}
                          </span>
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Processing Counts */}
                  {(uploadProgress.totalFrames || uploadProgress.totalSegments || uploadProgress.totalSamples) && (
                    <div className="space-y-2">
                      <div className="flex items-center gap-2 text-sm font-medium">
                        <Eye className="w-4 h-4 text-purple-500" />
                        <span>Processing</span>
                      </div>
                      <div className="space-y-1 text-xs">
                        {uploadProgress.totalFrames && (
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">Frames:</span>
                            <span className="font-mono font-medium">
                              {uploadProgress.processedFrames || 0}/{uploadProgress.totalFrames}
                            </span>
                          </div>
                        )}
                        {uploadProgress.totalSegments && (
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">Segments:</span>
                            <span className="font-mono font-medium">
                              {uploadProgress.processedSegments || 0}/{uploadProgress.totalSegments}
                            </span>
                          </div>
                        )}
                        {uploadProgress.totalSamples && (
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">Samples:</span>
                            <span className="font-mono font-medium">
                              {uploadProgress.processedSamples || 0}/{uploadProgress.totalSamples}
                            </span>
                          </div>
                        )}
                      </div>
                    </div>
                  )}

                  {/* Model Information */}
                  {uploadProgress.analysisDetails && (
                    <div className="space-y-2">
                      <div className="flex items-center gap-2 text-sm font-medium">
                        <Brain className="w-4 h-4 text-green-500" />
                        <span>Models</span>
                      </div>
                      <div className="space-y-1 text-xs">
                        {uploadProgress.analysisDetails.currentModelName && (
                          <div>
                            <span className="text-muted-foreground">Current:</span>
                            <p className="font-medium text-green-600 truncate">
                              {uploadProgress.analysisDetails.currentModelName}
                            </p>
                          </div>
                        )}
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Progress:</span>
                          <span className="font-mono font-medium">
                            {uploadProgress.analysisDetails.completedModels}/{uploadProgress.analysisDetails.totalModels}
                          </span>
                        </div>
                        {uploadProgress.analysisDetails.modelProgress && (
                          <div className="mt-1">
                            <Progress 
                              value={uploadProgress.analysisDetails.modelProgress} 
                              className="h-1" 
                            />
                          </div>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* Real-time Confidence Updates */}
              {uploadProgress?.intermediateResults && (
                <div className="p-3 bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-lg">
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <Activity className="w-4 h-4 text-blue-500" />
                      <span className="text-sm font-medium">Live Analysis</span>
                    </div>
                    {uploadProgress.intermediateResults.currentConfidence && (() => {
                      const TrendIcon = getConfidenceTrendIcon(uploadProgress.intermediateResults.confidenceTrend);
                      return (
                        <div className="flex items-center gap-1">
                          <TrendIcon className={`w-4 h-4 ${
                            uploadProgress.intermediateResults.confidenceTrend === 'increasing' ? 'text-green-500' :
                            uploadProgress.intermediateResults.confidenceTrend === 'decreasing' ? 'text-red-500' :
                            'text-gray-500'
                          }`} />
                          <span className="text-sm font-bold text-blue-600">
                            {Math.round(uploadProgress.intermediateResults.currentConfidence * 100)}%
                          </span>
                        </div>
                      );
                    })()}
                  </div>
                  
                  {uploadProgress.intermediateResults.processingQuality && (
                    <div className="flex items-center gap-2 text-xs">
                      <span className="text-muted-foreground">Quality:</span>
                      <Badge variant="outline" className={`text-xs ${
                        uploadProgress.intermediateResults.processingQuality === 'excellent' ? 'border-green-500 text-green-600' :
                        uploadProgress.intermediateResults.processingQuality === 'good' ? 'border-blue-500 text-blue-600' :
                        uploadProgress.intermediateResults.processingQuality === 'fair' ? 'border-yellow-500 text-yellow-600' :
                        'border-red-500 text-red-600'
                      }`}>
                        {uploadProgress.intermediateResults.processingQuality}
                      </Badge>
                    </div>
                  )}
                  
                  {uploadProgress.intermediateResults.detectedAnomalies && uploadProgress.intermediateResults.detectedAnomalies.length > 0 && (
                    <div className="mt-2">
                      <span className="text-xs text-muted-foreground">Detected:</span>
                      <div className="flex flex-wrap gap-1 mt-1">
                        {uploadProgress.intermediateResults.detectedAnomalies.slice(0, 3).map((anomaly, index) => (
                          <Badge key={index} variant="secondary" className="text-xs">
                            {anomaly}
                          </Badge>
                        ))}
                        {uploadProgress.intermediateResults.detectedAnomalies.length > 3 && (
                          <Badge variant="secondary" className="text-xs">
                            +{uploadProgress.intermediateResults.detectedAnomalies.length - 3} more
                          </Badge>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* Completion Status */}
              {uploadState.status === 'completed' && (
                <div className="p-4 bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 rounded-lg">
                  <div className="flex items-center gap-3">
                    <div className="p-2 bg-green-500 rounded-full">
                      <CheckCircle className="w-5 h-5 text-white" />
                    </div>
                    <div>
                      <h3 className="text-lg font-semibold text-green-700 dark:text-green-300">
                        Analysis Complete!
                      </h3>
                      <p className="text-sm text-green-600 dark:text-green-400">
                        Processing finished in {formatTime(realTimeElapsed || uploadProgress?.timeElapsed || 0)}
                      </p>
                    </div>
                  </div>
                </div>
              )}

              {/* Error Display */}
              {uploadState.error && (
                <div className="p-4 bg-gradient-to-r from-red-50 to-rose-50 dark:from-red-900/20 dark:to-rose-900/20 rounded-lg">
                  <div className="flex items-start gap-3">
                    <AlertTriangle className="w-5 h-5 text-red-500 mt-0.5" />
                    <div>
                      <h4 className="font-medium text-red-700 dark:text-red-300 mb-1">
                        Processing Error
                      </h4>
                      <p className="text-sm text-red-600 dark:text-red-400">
                        {uploadState.error}
                      </p>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}