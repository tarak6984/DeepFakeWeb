'use client';

import { useState, useCallback, useRef } from 'react';
import { useDropzone } from 'react-dropzone';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import {
  Upload,
  File,
  FileImage,
  FileVideo,
  FileAudio,
  X,
  AlertCircle,
  CheckCircle2,
  Loader2,
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { formatFileSize, getFileIcon } from '@/lib/storage';
import { usageTracker } from '@/lib/usage-tracker';
import type { FileUploadState, UploadProgress } from '@/lib/types';

interface UploadBoxProps {
  onFileSelect: (file: File) => void;
  onFileRemove: () => void;
  uploadState: FileUploadState;
  uploadProgress?: UploadProgress;
  className?: string;
  accept?: Record<string, string[]>;
  maxSize?: number;
  disabled?: boolean;
}

// Local API supports all media types - unlimited usage!
const DEFAULT_ACCEPT = {
  'image/*': ['.png', '.jpg', '.jpeg', '.gif', '.webp'],
  'audio/*': ['.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a'],
  'video/*': ['.mp4', '.webm', '.mov', '.avi', '.mkv'],
};

// Local API file size limit: 100MB for all types
const DEFAULT_MAX_SIZE = 100 * 1024 * 1024; // 100MB

export function UploadBox({
  onFileSelect,
  onFileRemove,
  uploadState,
  uploadProgress,
  className,
  accept = DEFAULT_ACCEPT,
  maxSize = DEFAULT_MAX_SIZE,
  disabled = false,
}: UploadBoxProps) {
  const [preview, setPreview] = useState<string | null>(null);

  const onDrop = useCallback(
    (acceptedFiles: File[], rejectedFiles: any[]) => {
      if (disabled || uploadState.status === 'processing') return;

      // Local API has no limits - unlimited processing!

      if (rejectedFiles.length > 0) {
        const error = rejectedFiles[0].errors[0];
        console.error('File rejection:', error);
        return;
      }

      if (acceptedFiles.length > 0) {
        const file = acceptedFiles[0];
        
        // Validate file size for local API (100MB for all types)
        if (file.size > maxSize) {
          alert(`File must be smaller than ${Math.round(maxSize / (1024 * 1024))}MB`);
          return;
        }
        
        onFileSelect(file);
        
        // Create preview for images and videos
        if (file.type.startsWith('image/')) {
          const reader = new FileReader();
          reader.onload = () => setPreview(reader.result as string);
          reader.readAsDataURL(file);
        } else if (file.type.startsWith('video/')) {
          const url = URL.createObjectURL(file);
          setPreview(url);
        } else {
          setPreview(null);
        }
      }
    },
    [onFileSelect, disabled, uploadState.status]
  );

  const { getRootProps, getInputProps, isDragActive, isDragReject } = useDropzone({
    onDrop,
    accept,
    maxSize,
    multiple: false,
    disabled: disabled || uploadState.status === 'processing',
    onDropRejected: (rejectedFiles) => {
      const firstError = rejectedFiles[0]?.errors[0];
      console.error('Upload rejected:', firstError?.message);
    },
  });

  const handleRemoveFile = () => {
    onFileRemove();
    setPreview(null);
    if (preview && preview.startsWith('blob:')) {
      URL.revokeObjectURL(preview);
    }
  };

  const getStatusIcon = () => {
    switch (uploadState.status) {
      case 'uploading':
      case 'processing':
        return <Loader2 className="h-6 w-6 animate-spin text-primary" />;
      case 'completed':
        return <CheckCircle2 className="h-6 w-6 text-green-600" />;
      case 'error':
        return <AlertCircle className="h-6 w-6 text-red-600" />;
      default:
        return <Upload className="h-6 w-6 text-muted-foreground" />;
    }
  };

  const getStatusText = () => {
    if (uploadProgress) {
      return uploadProgress.message;
    }
    
    switch (uploadState.status) {
      case 'uploading':
        return 'Uploading file to secure servers...';
      case 'processing':
        return 'AI models analyzing for manipulation patterns...';
      case 'completed':
        return 'Analysis completed! View detailed results.';
      case 'error':
        return uploadState.error || 'Upload failed';
      default:
        return 'Drag & drop your file here, or click to browse';
    }
  };

  const getDetailedProgress = () => {
    if (!uploadProgress) return null;
    
    const stages = [
      { key: 'uploading', label: 'File Upload', icon: Upload },
      { key: 'preprocessing', label: 'Media Processing', icon: Loader2 },
      { key: 'model_processing', label: 'AI Analysis', icon: CheckCircle2 },
      { key: 'generating_results', label: 'Results Ready', icon: CheckCircle2 }
    ];
    
    return (
      <div className="space-y-3">
        <div className="flex items-center justify-between text-sm">
          <span className="font-medium">{uploadProgress.message}</span>
          <span className="text-muted-foreground">{uploadProgress.percentage}%</span>
        </div>
        <Progress value={uploadProgress.percentage} className="h-2" />
        
        {/* Analysis Stages */}
        <div className="grid grid-cols-2 gap-2 mt-4">
          {stages.map((stage, index) => {
            const isActive = uploadProgress.stage === stage.key;
            const isCompleted = uploadProgress.stagesCompleted?.includes(stage.key);
            const Icon = stage.icon;
            
            return (
              <div
                key={stage.key}
                className={`flex items-center gap-2 p-2 rounded-lg text-xs transition-colors ${
                  isActive 
                    ? 'bg-primary/10 text-primary border border-primary/20' 
                    : isCompleted 
                    ? 'bg-green-50 text-green-700 dark:bg-green-900/20 dark:text-green-400'
                    : 'bg-muted/50 text-muted-foreground'
                }`}
              >
                <Icon className={`h-3 w-3 ${
                  isActive ? 'animate-spin' : ''
                }`} />
                <span className="font-medium">{stage.label}</span>
                {isCompleted && <CheckCircle2 className="h-3 w-3 ml-auto" />}
              </div>
            );
          })}
        </div>
        
        {/* Real-time Hints */}
        <div className="mt-3 p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
          <div className="flex items-start gap-2">
            <div className="w-2 h-2 bg-blue-500 rounded-full mt-2 animate-pulse" />
            <div className="text-xs text-blue-700 dark:text-blue-300">
              <div className="font-medium mb-1">What's happening now:</div>
              <div>{getAnalysisHint(uploadProgress)}</div>
            </div>
          </div>
        </div>
      </div>
    );
  };
  
  const getAnalysisHint = (progress: UploadProgress) => {
    switch (progress.stage) {
      case 'uploading':
        return 'Securely uploading your file to Reality Defender servers for analysis.';
      case 'preprocessing':
        return 'Extracting frames, audio tracks, and metadata for comprehensive analysis.';
      case 'model_processing':
      case 'analyzing_audio':
      case 'analyzing_video':
      case 'analyzing_image':
      case 'ensemble_analysis':
        if (progress.analysisDetails) {
          const { activeModels, completedModels, totalModels } = progress.analysisDetails;
          return `AI models analyzing: ${completedModels}/${totalModels} completed. Active: ${activeModels?.join(', ') || 'Multiple models'}.`;
        }
        return 'Multiple AI models examining your media for manipulation patterns and deepfake signatures.';
      case 'generating_results':
      case 'completing':
        return 'Compiling results from all AI models and generating detailed analysis report.';
      default:
        return 'Processing your media file...';
    }
  };

  const getAcceptedFormats = () => {
    const formats = Object.values(accept).flat();
    const unique = [...new Set(formats)];
    return unique.slice(0, 6).join(', ') + (unique.length > 6 ? '...' : '');
  };

  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          {getStatusIcon()}
          Upload Media File
        </CardTitle>
        <CardDescription>
          {uploadState.file ? uploadState.file.name : 'Select a file to analyze for deepfake detection'}
        </CardDescription>
      </CardHeader>
      <CardContent>
        <AnimatePresence mode="wait">
          {uploadState.file ? (
            <motion.div
              key="file-selected"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
              transition={{ duration: 0.2 }}
            >
              <div className="space-y-4">
                {/* File Preview */}
                <div className="flex items-center gap-4 p-4 bg-muted/50 rounded-lg">
                  <div className="w-16 h-16 rounded-lg bg-background flex items-center justify-center flex-shrink-0 overflow-hidden">
                    {preview ? (
                      uploadState.file?.type.startsWith('image/') ? (
                        <img
                          src={preview}
                          alt="Preview"
                          className="w-full h-full object-cover rounded-lg"
                        />
                      ) : uploadState.file?.type.startsWith('video/') ? (
                        <video
                          src={preview}
                          className="w-full h-full object-cover rounded-lg"
                          muted
                        />
                      ) : null
                    ) : (
                      <span className="text-2xl">
                        {getFileIcon(uploadState.file?.type || '')}
                      </span>
                    )}
                  </div>
                  
                  <div className="flex-1 min-w-0">
                    <h4 className="font-medium truncate" title={uploadState.file.name}>
                      {uploadState.file.name}
                    </h4>
                    <p className="text-sm text-muted-foreground">
                      {formatFileSize(uploadState.file.size)}
                    </p>
                    <div className="flex items-center gap-2 mt-1">
                      <Badge variant="outline" className="text-xs">
                        {uploadState.file.type.split('/')[0]}
                      </Badge>
                      {uploadState.status === 'completed' && (
                        <Badge variant="secondary" className="text-xs text-green-600">
                          <CheckCircle2 className="w-3 h-3 mr-1" />
                          Analyzed
                        </Badge>
                      )}
                    </div>
                  </div>

                  {uploadState.status !== 'processing' && uploadState.status !== 'uploading' && (
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={handleRemoveFile}
                      className="flex-shrink-0"
                    >
                      <X className="h-4 w-4" />
                    </Button>
                  )}
                </div>

                {/* Progress Bar */}
                {(uploadState.status === 'uploading' || uploadState.status === 'processing') && (
                  <div className="space-y-2">
                    {uploadProgress ? (
                      getDetailedProgress()
                    ) : (
                      <>
                        <Progress value={uploadState.progress} className="w-full" />
                        <div className="flex justify-between text-sm">
                          <span className="text-muted-foreground">{getStatusText()}</span>
                          <span className="text-muted-foreground">{uploadState.progress}%</span>
                        </div>
                      </>
                    )}
                  </div>
                )}

                {/* Error Message */}
                {uploadState.status === 'error' && uploadState.error && (
                  <div className="flex items-center gap-2 p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
                    <AlertCircle className="h-4 w-4 text-red-600 dark:text-red-400 flex-shrink-0" />
                    <p className="text-sm text-red-700 dark:text-red-300">{uploadState.error}</p>
                  </div>
                )}

                {/* Success Message */}
                {uploadState.status === 'completed' && (
                  <div className="flex items-center gap-2 p-3 bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg">
                    <CheckCircle2 className="h-4 w-4 text-green-600 dark:text-green-400 flex-shrink-0" />
                    <p className="text-sm text-green-700 dark:text-green-300">
                      File analyzed successfully! Check the results below.
                    </p>
                  </div>
                )}
              </div>
            </motion.div>
          ) : (
            <motion.div
              key="upload-area"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
              transition={{ duration: 0.2 }}
            >
              <div
                {...getRootProps()}
                className={`
                  upload-zone
                  relative min-h-[200px] p-8 rounded-lg transition-all duration-200 cursor-pointer
                  flex flex-col items-center justify-center text-center space-y-4
                  ${isDragActive ? 'active border-primary bg-primary/5' : ''}
                  ${isDragReject ? 'border-red-500 bg-red-50 dark:bg-red-900/20' : ''}
                  ${disabled ? 'opacity-50 cursor-not-allowed' : 'hover:border-primary/50'}
                `}
              >
                <input {...getInputProps()} />
                
                <motion.div
                  animate={{
                    y: isDragActive ? -4 : 0,
                    scale: isDragActive ? 1.05 : 1,
                  }}
                  transition={{ duration: 0.2 }}
                  className="flex flex-col items-center space-y-3"
                >
                  <div className={`
                    w-16 h-16 rounded-full flex items-center justify-center
                    ${isDragActive ? 'bg-primary text-primary-foreground' : 'bg-muted'}
                    ${isDragReject ? 'bg-red-100 text-red-600 dark:bg-red-900/40' : ''}
                  `}>
                    {isDragReject ? (
                      <AlertCircle className="h-8 w-8" />
                    ) : (
                      <Upload className={`h-8 w-8 ${isDragActive ? 'animate-bounce-subtle' : ''}`} />
                    )}
                  </div>
                  
                  <div>
                    <h3 className="font-medium mb-1">
                      {isDragReject
                        ? 'Invalid file type or size'
                        : isDragActive
                        ? 'Drop your file here'
                        : 'Drag & drop your file here'
                      }
                    </h3>
                    <p className="text-sm text-muted-foreground">
                      {isDragReject
                        ? 'Please select a valid media file'
                        : 'or click to browse your files'
                      }
                    </p>
                  </div>
                  
                  {!isDragReject && (
                    <Button variant="outline" size="sm" disabled={disabled}>
                      Choose File
                    </Button>
                  )}
                </motion.div>

                {/* Scan line animation when processing */}
                {uploadState.status === 'processing' && (
                  <div className="absolute inset-0 overflow-hidden rounded-lg pointer-events-none">
                    <div className="animate-scan absolute inset-y-0 w-1 bg-gradient-to-b from-transparent via-primary to-transparent opacity-50" />
                  </div>
                )}
              </div>
              
              {/* File Type Info */}
              <div className="mt-4 text-center">
                <p className="text-sm text-muted-foreground mb-2">Supported formats:</p>
                <p className="text-xs text-muted-foreground">
                  {getAcceptedFormats()}
                </p>
                <p className="text-xs text-muted-foreground mt-1">
                  Max size: {Math.round(maxSize / (1024 * 1024))}MB
                </p>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </CardContent>
    </Card>
  );
}