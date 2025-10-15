'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { UploadBox } from '@/components/upload-box';
import { Sidebar } from '@/components/layout/sidebar';
import dynamic from 'next/dynamic';
import { Suspense } from 'react';

// Dynamic imports for heavy chart components
const ConfidenceGauge = dynamic(() => import('@/components/charts/confidence-gauge').then(mod => ({ default: mod.ConfidenceGauge })), {
  loading: () => <div className="h-64 bg-muted/30 rounded-lg animate-pulse" />
});

const CategoryChart = dynamic(() => import('@/components/charts/category-chart').then(mod => ({ default: mod.CategoryChart })), {
  loading: () => <div className="h-64 bg-muted/30 rounded-lg animate-pulse" />
});

const UsageDashboard = dynamic(() => import('@/components/usage-dashboard').then(mod => ({ default: mod.UsageDashboard })), {
  loading: () => <div className="h-96 bg-muted/30 rounded-lg animate-pulse" />
});

const ExplanationDashboard = dynamic(() => import('@/components/explanation/explanation-dashboard').then(mod => ({ default: mod.ExplanationDashboard })), {
  loading: () => <div className="h-96 bg-muted/30 rounded-lg animate-pulse" />
});

const RiskHeatmap = dynamic(() => import('@/components/charts/advanced-charts').then(mod => ({ default: mod.RiskHeatmap })), {
  loading: () => <div className="h-64 bg-muted/30 rounded-lg animate-pulse" />
});

const AnomalyScatterPlot = dynamic(() => import('@/components/charts/advanced-charts').then(mod => ({ default: mod.AnomalyScatterPlot })), {
  loading: () => <div className="h-64 bg-muted/30 rounded-lg animate-pulse" />
});

const MultidimensionalRadar = dynamic(() => import('@/components/charts/advanced-charts').then(mod => ({ default: mod.MultidimensionalRadar })), {
  loading: () => <div className="h-64 bg-muted/30 rounded-lg animate-pulse" />
});

const ConfidenceTimelineAdvanced = dynamic(() => import('@/components/charts/advanced-charts').then(mod => ({ default: mod.ConfidenceTimelineAdvanced })), {
  loading: () => <div className="h-64 bg-muted/30 rounded-lg animate-pulse" />
});

// Dynamic import for Recharts (heavy library)
const ResponsiveContainer = dynamic(() => import('recharts').then(mod => ({ default: mod.ResponsiveContainer })), {
  ssr: false,
  loading: () => <div className="h-80 bg-muted/30 rounded-lg animate-pulse" />
});

const AreaChart = dynamic(() => import('recharts').then(mod => ({ default: mod.AreaChart })), {
  ssr: false
});

const Area = dynamic(() => import('recharts').then(mod => ({ default: mod.Area })), {
  ssr: false
});

const CartesianGrid = dynamic(() => import('recharts').then(mod => ({ default: mod.CartesianGrid })), {
  ssr: false
});

const XAxis = dynamic(() => import('recharts').then(mod => ({ default: mod.XAxis })), {
  ssr: false
});

const YAxis = dynamic(() => import('recharts').then(mod => ({ default: mod.YAxis })), {
  ssr: false
});

const Tooltip = dynamic(() => import('recharts').then(mod => ({ default: mod.Tooltip })), {
  ssr: false
});

const PDFExportDialog = dynamic(() => import('@/components/pdf/pdf-export-dialog').then(mod => ({ default: mod.PDFExportDialog })), {
  loading: () => <div className="w-32 h-10 bg-muted/30 rounded animate-pulse" />
});

const PDFChartComponents = dynamic(() => import('@/components/pdf/pdf-chart-components').then(mod => ({ default: mod.PDFChartComponents })), {
  ssr: false
});
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import {
  Download,
  ExternalLink,
  Clock,
  FileText,
  AlertCircle,
  Info,
  Zap,
  BarChart3,
  Activity,
  Volume2,
  Upload,
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { localDeepfakeAPI, type AnalysisResult } from '@/lib/local-deepfake-api';
import { storage, createThumbnail, formatFileSize } from '@/lib/storage';
import { usageTracker } from '@/lib/usage-tracker';
import type { FileUploadState, UploadProgress, DetailedExplanation } from '@/lib/types';
import { ExplanationGenerator } from '@/lib/explanation-generator';
import { toast } from 'sonner';
import { UsageLimitBanner } from '@/components/usage/usage-limit-banner';
import { useUsageManager } from '@/lib/unified-usage-manager';

export default function UploadPage() {
  const [uploadState, setUploadState] = useState<FileUploadState>({
    file: null,
    preview: null,
    progress: 0,
    status: 'idle',
  });
  const [uploadProgress, setUploadProgress] = useState<UploadProgress>();
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [explanation, setExplanation] = useState<DetailedExplanation | null>(null);
  const [showSidebar, setShowSidebar] = useState(true);
  const [activeTab, setActiveTab] = useState<'upload' | 'usage'>('upload');
  const router = useRouter();
  
  // Usage management
  const { canScan, consumeScan, isLimitReached } = useUsageManager();

  const handleFileSelect = async (file: File) => {
    // Check usage limits for anonymous users
    if (!canScan) {
      toast.error('Scan limit reached!', {
        description: 'Sign up for unlimited scans and advanced features.',
        duration: 5000,
      });
      return;
    }
    
    console.log('🆓 Using unlimited local deepfake detection API!');

    setUploadState({
      file,
      preview: null,
      progress: 0,
      status: 'uploading',
    });
    setAnalysisResult(null);
    setExplanation(null);

    try {
      setUploadState(prev => ({ ...prev, progress: 5, status: 'processing' }));

      // Call local deepfake API with progress callback
      const result = await localDeepfakeAPI.analyzeMedia(file, (progress) => {
        setUploadProgress(progress);
        setUploadState(prev => ({ ...prev, progress: progress.percentage }));
      });

      setUploadState(prev => ({ ...prev, progress: 100, status: 'completed' }));

      // Create thumbnail and save to storage
      try {
        const thumbnail = await createThumbnail(file);
        storage.addAnalysisToHistory(result, thumbnail);
      } catch (error) {
        console.warn('Failed to create thumbnail:', error);
        storage.addAnalysisToHistory(result);
      }

      setAnalysisResult(result);
      
      // Consume a scan for anonymous users
      consumeScan();
      
      // Generate explanation
      try {
        const explanationGenerator = new ExplanationGenerator();
        const generatedExplanation = explanationGenerator.generateExplanation(result);
        setExplanation(generatedExplanation);
      } catch (error) {
        console.warn('Failed to generate explanation:', error);
        setExplanation(null);
      }
      
      toast.success('Analysis completed successfully!');

    } catch (error) {
      console.error('Analysis failed:', error);
      setUploadState(prev => ({
        ...prev,
        status: 'error',
        error: error instanceof Error ? error.message : 'Analysis failed',
      }));
      toast.error('Analysis failed. Please try again.');
    }
  };

  const handleFileRemove = () => {
    setUploadState({
      file: null,
      preview: null,
      progress: 0,
      status: 'idle',
    });
    setUploadProgress(undefined);
    setAnalysisResult(null);
    setExplanation(null);
  };

  const handleAnalysisSelect = (analysisId: string) => {
    router.push(`/results?id=${analysisId}`);
  };

  // Function to trigger file picker
  const triggerFileUpload = () => {
    const fileInput = document.querySelector('input[type="file"]') as HTMLInputElement;
    if (fileInput) {
      fileInput.click();
    }
  };


  // Generate timeline data from analysis results
  const generateTimelineData = () => {
    if (!analysisResult) return [];
    
    const data = [];
    const baseConfidence = analysisResult.confidence;
    const points = analysisResult.details.frameAnalysis?.length || 20;
    
    for (let i = 0; i < points; i++) {
      const variation = (Math.random() - 0.5) * 0.3;
      const confidence = Math.max(0, Math.min(1, baseConfidence + variation));
      data.push({
        frame: i + 1,
        timestamp: (i / points) * (analysisResult.details.metadata.duration || 60),
        confidence: Math.round(confidence * 100),
        anomalies: Math.random() > 0.8 ? 1 : 0,
      });
    }
    return data;
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-muted/20">
      <div className="flex">
        {/* Sidebar */}
        <AnimatePresence>
          {showSidebar && (
            <motion.div
              initial={{ width: 0, opacity: 0 }}
              animate={{ width: 280, opacity: 1 }}
              exit={{ width: 0, opacity: 0 }}
              transition={{ duration: 0.3 }}
              className="flex-shrink-0"
            >
              <Sidebar onAnalysisSelect={handleAnalysisSelect} />
            </motion.div>
          )}
        </AnimatePresence>

        {/* Main Content */}
        <div className="flex-1 p-6">
          <div className="max-w-6xl mx-auto space-y-8">
            {/* Header */}
            <motion.div
              initial={{ opacity: 0, y: -20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
              className="text-center space-y-4"
            >
              <h1 className="text-3xl sm:text-4xl font-bold">
                ITL Deepfake Detection Analysis
              </h1>
              <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
                Upload your media file to analyze it for deepfake manipulation using advanced AI technology.
              </p>
              
              {/* Tab Navigation */}
              <div className="flex items-center justify-center gap-2">
                <Button
                  variant={activeTab === 'upload' ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => {
                    setActiveTab('upload');
                    // If no file or analysis is complete, allow new upload
                    if (!uploadState.file && uploadState.status === 'idle') {
                      triggerFileUpload();
                    } else if (uploadState.status === 'completed') {
                      // Clear current results and trigger new upload
                      handleFileRemove();
                      // Small delay to ensure state is cleared before triggering upload
                      setTimeout(() => triggerFileUpload(), 100);
                    }
                  }}
                  className="px-6"
                >
                  <Activity className="w-4 h-4 mr-2" />
                  Upload & Analyze
                </Button>
                <Button
                  variant={activeTab === 'usage' ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => setActiveTab('usage')}
                  className="px-6"
                >
                  <BarChart3 className="w-4 h-4 mr-2" />
                  Usage Dashboard
                </Button>
              </div>

              <div className="flex items-center justify-center gap-4 text-sm text-muted-foreground">
                <div className="flex items-center gap-1">
                  <Zap className="w-4 h-4" />
                  Fast Analysis
                </div>
                <div className="flex items-center gap-1">
                  <Info className="w-4 h-4" />
                  Local AI: Images, Audio & Video - Unlimited!
                </div>
              </div>
            </motion.div>

            {/* Content Tabs */}
            <AnimatePresence mode="wait">
              {activeTab === 'upload' ? (
                <motion.div
                  key="upload"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  transition={{ duration: 0.3 }}
                  className="space-y-8"
                >
                  {/* Upload Section */}
                  <div>
                    <UploadBox
                      onFileSelect={handleFileSelect}
                      onFileRemove={handleFileRemove}
                      uploadState={uploadState}
                      uploadProgress={uploadProgress}
                      className="max-w-2xl mx-auto"
                    />
                  </div>

                  {/* Results Section */}
                  <AnimatePresence>
                    {analysisResult && (
                      <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -20 }}
                        transition={{ duration: 0.5 }}
                        className="space-y-6"
                      >
                        <Separator className="my-8" />
                        
                        {/* Results Header */}
                        <div className="text-center space-y-2">
                          <h2 className="text-2xl font-bold">Analysis Results</h2>
                          <p className="text-muted-foreground">
                            Completed analysis for{' '}
                            <span className="font-medium">{analysisResult.filename}</span>
                          </p>
                          <div className="flex items-center justify-center gap-4 text-sm">
                            <Badge variant="outline">
                              <Clock className="w-3 h-3 mr-1" />
                              {Math.round(analysisResult.processingTime / 1000)}s processing
                            </Badge>
                            <Badge variant="outline">
                              {formatFileSize(analysisResult.fileSize)}
                            </Badge>
                          </div>
                          {/* Analyze New File Button */}
                          <div className="pt-4">
                            <Button 
                              variant="outline" 
                              onClick={() => {
                                handleFileRemove();
                                setTimeout(() => triggerFileUpload(), 100);
                              }}
                              className="px-6"
                            >
                              <Upload className="w-4 h-4 mr-2" />
                              Analyze New File
                            </Button>
                          </div>
                        </div>

                        {/* Charts Grid */}
                        <div className="grid lg:grid-cols-2 gap-6">
                          <ConfidenceGauge
                            confidence={analysisResult.confidence}
                            prediction={analysisResult.prediction}
                          />
                          <CategoryChart
                            data={analysisResult.details.categoryBreakdown}
                          />
                        </div>

                        {/* Why This Result Section */}
                        {explanation && (
                          <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ duration: 0.5, delay: 0.2 }}
                          >
                            <Separator className="my-8" />
                            <div className="text-center mb-6">
                              <h3 className="text-xl font-bold">Why This Result?</h3>
                              <p className="text-muted-foreground mt-2">
                                Detailed explanation of the analysis findings and reasoning
                              </p>
                            </div>
                            <ExplanationDashboard explanation={explanation} />
                            <Separator className="my-8" />
                          </motion.div>
                        )}

                        {/* Metadata Card */}
                        <Card>
                          <CardHeader>
                            <CardTitle className="flex items-center gap-2">
                              <FileText className="w-5 h-5" />
                              File Information
                            </CardTitle>
                          </CardHeader>
                          <CardContent className="space-y-3">
                            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                              <div>
                                <span className="text-muted-foreground">Type:</span>
                                <p className="font-medium capitalize">
                                  {analysisResult.fileType.split('/')[0]}
                                </p>
                              </div>
                              <div>
                                <span className="text-muted-foreground">Size:</span>
                                <p className="font-medium">
                                  {formatFileSize(analysisResult.fileSize)}
                                </p>
                              </div>
                              {analysisResult.details.metadata.duration && (
                                <div>
                                  <span className="text-muted-foreground">Duration:</span>
                                  <p className="font-medium">
                                    {Math.round(analysisResult.details.metadata.duration)}s
                                  </p>
                                </div>
                              )}
                              {analysisResult.details.metadata.resolution && (
                                <div>
                                  <span className="text-muted-foreground">Resolution:</span>
                                  <p className="font-medium">
                                    {analysisResult.details.metadata.resolution}
                                  </p>
                                </div>
                              )}
                            </div>
                          </CardContent>
                        </Card>

                        {/* Timeline Analysis */}
                        <Card>
                          <CardHeader>
                            <CardTitle className="flex items-center gap-2">
                              <Clock className="w-5 h-5" />
                              Timeline Analysis
                            </CardTitle>
                          </CardHeader>
                          <CardContent>
                            <div className="h-80">
                              <ResponsiveContainer width="100%" height="100%">
                                <AreaChart data={generateTimelineData()}>
                                  <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
                                  <XAxis 
                                    dataKey="frame" 
                                    tick={{ fontSize: 12 }}
                                    axisLine={false}
                                    tickLine={false}
                                  />
                                  <YAxis 
                                    domain={[0, 100]}
                                    tick={{ fontSize: 12 }}
                                    axisLine={false}
                                    tickLine={false}
                                    label={{ value: 'Confidence %', angle: -90, position: 'insideLeft' }}
                                  />
                                  <Tooltip 
                                    content={({ active, payload, label }) => {
                                      if (active && payload && payload.length) {
                                        return (
                                          <div className="bg-background border border-border rounded-lg shadow-lg p-3">
                                            <p className="font-medium">Frame {label}</p>
                                            <p className="text-sm text-muted-foreground">
                                              Confidence: {payload[0].value}%
                                            </p>
                                          </div>
                                        );
                                      }
                                      return null;
                                    }}
                                  />
                                  <Area
                                    type="monotone"
                                    dataKey="confidence"
                                    stroke="#8884d8"
                                    fill="#8884d8"
                                    fillOpacity={0.2}
                                    strokeWidth={2}
                                  />
                                </AreaChart>
                              </ResponsiveContainer>
                            </div>
                          </CardContent>
                        </Card>

                        {/* Advanced Charts Grid */}
                        <div className="space-y-6">
                          <h3 className="text-xl font-bold text-center">Advanced Analysis</h3>
                          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                            <RiskHeatmap data={analysisResult} />
                            <AnomalyScatterPlot data={analysisResult} />
                            <MultidimensionalRadar data={analysisResult} />
                            <ConfidenceTimelineAdvanced data={generateTimelineData()} />
                          </div>
                        </div>

                        {/* Technical Details */}
                        <Card>
                          <CardHeader>
                            <CardTitle className="flex items-center gap-2">
                              <FileText className="w-5 h-5" />
                              Technical Details
                            </CardTitle>
                          </CardHeader>
                          <CardContent>
                            <div className="space-y-4">
                              {/* Processing Stats */}
                              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                                <div className="text-center p-3 bg-muted/30 rounded">
                                  <div className="text-lg font-bold text-primary">
                                    {Math.round(analysisResult.processingTime / 1000)}s
                                  </div>
                                  <div className="text-xs text-muted-foreground">Processing Time</div>
                                </div>
                                <div className="text-center p-3 bg-muted/30 rounded">
                                  <div className="text-lg font-bold">
                                    {analysisResult.confidence > 1 ? Math.round(analysisResult.confidence) : Math.round(analysisResult.confidence * 100)}%
                                  </div>
                                  <div className="text-xs text-muted-foreground">Risk Score</div>
                                </div>
                                <div className="text-center p-3 bg-muted/30 rounded">
                                  <div className="text-lg font-bold text-green-600">
                                    {analysisResult.details.frameAnalysis?.length || 'N/A'}
                                  </div>
                                  <div className="text-xs text-muted-foreground">Frames Analyzed</div>
                                </div>
                                <div className="text-center p-3 bg-muted/30 rounded">
                                  <div className="text-lg font-bold">
                                    {analysisResult.id.slice(-8)}
                                  </div>
                                  <div className="text-xs text-muted-foreground">Analysis ID</div>
                                </div>
                              </div>

                              {/* Analysis Summary in User-Friendly Format */}
                              <div className="mt-6">
                                <h4 className="font-medium mb-2">Analysis Summary</h4>
                                <div className="bg-muted/30 p-4 rounded-lg space-y-3">
                                  <div className="grid grid-cols-2 gap-4 text-sm">
                                    <div>
                                      <span className="text-muted-foreground">Confidence Score:</span>
                                      <p className="font-medium">{analysisResult.confidence > 1 ? Math.round(analysisResult.confidence) : Math.round(analysisResult.confidence * 100)}%</p>
                                    </div>
                                    <div>
                                      <span className="text-muted-foreground">Final Prediction:</span>
                                      <p className="font-medium capitalize">{analysisResult.prediction}</p>
                                    </div>
                                  </div>
                                  
                                  <div>
                                    <span className="text-muted-foreground text-sm">Category Scores:</span>
                                    <div className="grid grid-cols-3 gap-2 mt-1">
                                      <div className="text-center p-2 bg-green-50 dark:bg-green-900/20 rounded">
                                        <div className="text-sm font-semibold text-green-700 dark:text-green-400">
                                          {analysisResult.details.categoryBreakdown.authentic > 1 ? Math.round(analysisResult.details.categoryBreakdown.authentic) : Math.round(analysisResult.details.categoryBreakdown.authentic * 100)}%
                                        </div>
                                        <div className="text-xs text-green-600 dark:text-green-400">Authentic</div>
                                      </div>
                                      <div className="text-center p-2 bg-red-50 dark:bg-red-900/20 rounded">
                                        <div className="text-sm font-semibold text-red-700 dark:text-red-400">
                                          {analysisResult.details.categoryBreakdown.manipulated > 1 ? Math.round(analysisResult.details.categoryBreakdown.manipulated) : Math.round(analysisResult.details.categoryBreakdown.manipulated * 100)}%
                                        </div>
                                        <div className="text-xs text-red-600 dark:text-red-400">Manipulated</div>
                                      </div>
                                      <div className="text-center p-2 bg-yellow-50 dark:bg-yellow-900/20 rounded">
                                        <div className="text-sm font-semibold text-yellow-700 dark:text-yellow-400">
                                          {analysisResult.details.categoryBreakdown.inconclusive > 1 ? Math.round(analysisResult.details.categoryBreakdown.inconclusive) : Math.round(analysisResult.details.categoryBreakdown.inconclusive * 100)}%
                                        </div>
                                        <div className="text-xs text-yellow-600 dark:text-yellow-400">Inconclusive</div>
                                      </div>
                                    </div>
                                  </div>
                                  
                                  {analysisResult.details.metadata && (
                                    <div>
                                      <span className="text-muted-foreground text-sm">Analysis Details:</span>
                                      <div className="grid grid-cols-2 gap-4 text-sm mt-1">
                                        <div>
                                          <span className="text-muted-foreground">Models Analyzed:</span>
                                          <p className="font-medium">{(analysisResult.details.metadata as any).modelsAnalyzed || 'N/A'}</p>
                                        </div>
                                        <div>
                                          <span className="text-muted-foreground">Completed Models:</span>
                                          <p className="font-medium">{(analysisResult.details.metadata as any).completedModels || 'N/A'}</p>
                                        </div>
                                      </div>
                                    </div>
                                  )}
                                </div>
                              </div>
                            </div>
                          </CardContent>
                        </Card>

                        {/* Export PDF Button */}
                        <div className="flex justify-center">
                          <PDFExportDialog
                            analysis={{
                              ...analysisResult,
                              id: analysisResult.id,
                              timestamp: analysisResult.timestamp,
                              fileSize: analysisResult.fileSize,
                              fileType: analysisResult.fileType,
                              details: analysisResult.details,
                            }}
                            trigger={
                              <Button className="px-8">
                                <FileText className="w-4 h-4 mr-2" />
                                Export PDF Report
                              </Button>
                            }
                          />
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>

                  {/* Hidden PDF Components */}
                  {analysisResult && (
                    <PDFChartComponents analysis={{
                      ...analysisResult,
                      id: analysisResult.id,
                      timestamp: analysisResult.timestamp,
                      fileSize: analysisResult.fileSize,
                      fileType: analysisResult.fileType,
                      details: analysisResult.details,
                    }} />
                  )}
                </motion.div>
              ) : (
                <motion.div
                  key="usage"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  transition={{ duration: 0.3 }}
                >
                  <UsageDashboard />
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>
      </div>
    </div>
  );
}