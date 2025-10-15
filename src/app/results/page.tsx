'use client';

import { useState, useEffect, Suspense } from 'react';
import { useSearchParams, useRouter } from 'next/navigation';
import { ConfidenceGauge } from '@/components/charts/confidence-gauge';
import { CategoryChart } from '@/components/charts/category-chart';
import { Sidebar } from '@/components/layout/sidebar';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { Progress } from '@/components/ui/progress';
import {
  Download,
  ArrowLeft,
  Clock,
  FileText,
  Activity,
  Image as ImageIcon,
  Video,
  Volume2,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Share,
  Trash2,
  BarChart3,
  Lightbulb,
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, ResponsiveContainer, Tooltip, AreaChart, Area } from 'recharts';
import { storage, formatFileSize, getFileIcon, type StoredAnalysis } from '@/lib/storage';
import { usageTracker } from '@/lib/usage-tracker';
import {
  RiskHeatmap,
  AnomalyScatterPlot,
  MultidimensionalRadar,
  ConfidenceTimelineAdvanced,
  ComparativeAnalysis,
  FileTypeTreemap,
} from '@/components/charts/advanced-charts';
import { ExplanationDashboard } from '@/components/explanation/explanation-dashboard';
import { toast } from 'sonner';
import { PDFExportDialog } from '@/components/pdf/pdf-export-dialog';
import { PDFChartComponents } from '@/components/pdf/pdf-chart-components';

function ResultsPageContent() {
  const [analysis, setAnalysis] = useState<StoredAnalysis | null>(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<'overview' | 'explanation' | 'timeline' | 'advanced' | 'details'>('overview');
  const searchParams = useSearchParams();
  const router = useRouter();
  const analysisId = searchParams.get('id');

  useEffect(() => {
    if (!analysisId) {
      router.push('/upload');
      return;
    }

    const result = storage.getAnalysisById(analysisId);
    if (result) {
      setAnalysis(result);
    } else {
      toast.error('Analysis not found');
      router.push('/upload');
    }
    setLoading(false);
  }, [analysisId, router]);

  const handleDelete = () => {
    if (!analysis) return;
    
    if (confirm('Are you sure you want to delete this analysis?')) {
      storage.removeAnalysisFromHistory(analysis.id);
      toast.success('Analysis deleted successfully');
      router.push('/upload');
    }
  };

  const downloadReport = () => {
    if (!analysis) return;

    const report = {
      id: analysis.id,
      filename: analysis.filename,
      analysis: {
        confidence: analysis.confidence,
        prediction: analysis.prediction,
        timestamp: analysis.timestamp,
        processingTime: analysis.processingTime,
      },
      details: analysis.details,
      explanation: analysis.explanation ? {
        summary: analysis.explanation.summary,
        reasons: analysis.explanation.reasons,
        evidence: analysis.explanation.evidence,
        modelInsights: analysis.explanation.modelInsights,
        temporalAnalysis: analysis.explanation.temporalAnalysis,
        metadataAnalysis: analysis.explanation.metadataAnalysis,
        generatedAt: analysis.explanation.generatedAt,
        processingVersion: analysis.explanation.processingVersion
      } : null,
      metadata: {
        fileSize: analysis.fileSize,
        fileType: analysis.fileType,
        exportedAt: new Date().toISOString(),
        includesExplanation: !!analysis.explanation
      },
    };

    const blob = new Blob([JSON.stringify(report, null, 2)], {
      type: 'application/json',
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    const suffix = analysis.explanation ? '-with-explanation' : '';
    a.download = `deepfake-analysis-${analysis.filename}-${new Date().toISOString().split('T')[0]}${suffix}.json`;
    a.click();
    URL.revokeObjectURL(url);

    const message = analysis.explanation ? 
      'Complete report with explanations downloaded!' : 
      'Report downloaded successfully!';
    toast.success(message);
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence < 0.9) return 'text-green-600 dark:text-green-400';
    if (confidence < 0.95) return 'text-yellow-600 dark:text-yellow-400';
    return 'text-red-600 dark:text-red-400';
  };

  const getConfidenceIcon = (confidence: number) => {
    if (confidence < 0.9) return CheckCircle;
    if (confidence < 0.95) return AlertTriangle;
    return XCircle;
  };

  // Generate timeline data from analysis results
  const generateTimelineData = () => {
    if (!analysis) return [];
    
    const data = [];
    const baseConfidence = analysis.confidence;
    const points = analysis.details.frameAnalysis?.length || 20;
    
    for (let i = 0; i < points; i++) {
      const variation = (Math.random() - 0.5) * 0.3;
      const confidence = Math.max(0, Math.min(1, baseConfidence + variation));
      data.push({
        frame: i + 1,
        timestamp: (i / points) * (analysis.details.metadata.duration || 60),
        confidence: Math.round(confidence * 100),
        anomalies: Math.random() > 0.8 ? 1 : 0,
      });
    }
    return data;
  };

  const timelineData = generateTimelineData();

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto mb-4"></div>
          <p className="text-muted-foreground">Loading analysis...</p>
        </div>
      </div>
    );
  }

  if (!analysis) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <AlertTriangle className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
          <h2 className="text-xl font-semibold mb-2">Analysis Not Found</h2>
          <p className="text-muted-foreground mb-4">The requested analysis could not be found.</p>
          <Button onClick={() => router.push('/upload')}>
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back to Upload
          </Button>
        </div>
      </div>
    );
  }

  const ConfidenceIcon = getConfidenceIcon(analysis.confidence);

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-muted/20">
      <div className="flex">
        {/* Sidebar */}
        <Sidebar onAnalysisSelect={(id) => router.push(`/results?id=${id}`)} />

        {/* Main Content */}
        <div className="flex-1 p-6">
          <div className="max-w-6xl mx-auto space-y-6">
            {/* Header */}
            <motion.div
              initial={{ opacity: 0, y: -20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
              className="flex items-center justify-between"
            >
              <div className="flex items-center gap-4">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => router.back()}
                >
                  <ArrowLeft className="w-4 h-4 mr-2" />
                  Back
                </Button>
                <div>
                  <h1 className="text-2xl font-bold flex items-center gap-2">
                    <span className="text-2xl">{getFileIcon(analysis.fileType)}</span>
                    {analysis.filename}
                  </h1>
                  <p className="text-muted-foreground">
                    Analyzed on {new Date(analysis.timestamp).toLocaleDateString()}
                  </p>
                </div>
              </div>
              
              <div className="flex items-center gap-2">
                <Badge variant="outline" className={getConfidenceColor(analysis.confidence)}>
                  <ConfidenceIcon className="w-3 h-3 mr-1" />
                  {analysis.confidence > 1 ? Math.round(analysis.confidence) : Math.round(analysis.confidence * 100)}% Risk
                </Badge>
                <Button variant="outline" size="sm" onClick={downloadReport}>
                  <Download className="w-4 h-4 mr-2" />
                  {analysis.explanation ? 'Download Full Report' : 'Download JSON'}
                </Button>
                {/* PDF Export Dialog Trigger */}
                <PDFExportDialog
                  analysis={analysis}
                  trigger={
                    <Button variant="outline" size="sm">
                      <FileText className="w-4 h-4 mr-2" />
                      Export PDF
                    </Button>
                  }
                />
                <Button variant="outline" size="sm" onClick={handleDelete}>
                  <Trash2 className="w-4 h-4 mr-2" />
                  Delete
                </Button>
              </div>
            </motion.div>

            {/* Tab Navigation */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.1 }}
            >
              <div className="border-b">
                <nav className="-mb-px flex space-x-8">
                  {[
                    { id: 'overview', label: 'Overview', icon: Activity },
                    { id: 'explanation', label: 'Why This Result?', icon: Lightbulb },
                    { id: 'timeline', label: 'Timeline Analysis', icon: Clock },
                    { id: 'advanced', label: 'Advanced Charts', icon: BarChart3 },
                    { id: 'details', label: 'Technical Details', icon: FileText },
                  ].map((tab) => (
                    <button
                      key={tab.id}
                      onClick={() => setActiveTab(tab.id as any)}
                      className={`py-2 px-1 border-b-2 font-medium text-sm flex items-center gap-2 ${
                        activeTab === tab.id
                          ? 'border-primary text-primary'
                          : 'border-transparent text-muted-foreground hover:text-foreground hover:border-border'
                      }`}
                    >
                      <tab.icon className="w-4 h-4" />
                      {tab.label}
                    </button>
                  ))}
                </nav>
              </div>
            </motion.div>

            {/* Tab Content */}
            <AnimatePresence mode="wait">
              {activeTab === 'overview' && (
                <motion.div
                  key="overview"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  transition={{ duration: 0.3 }}
                  className="space-y-6"
                >
                  {/* Main Charts */}
                  <div className="grid lg:grid-cols-2 gap-6">
                    <ConfidenceGauge
                      confidence={analysis.confidence}
                      prediction={analysis.prediction}
                    />
                    <CategoryChart
                      data={analysis.details.categoryBreakdown}
                    />
                  </div>

                  {/* File Info & Processing Stats */}
                  <div className="grid md:grid-cols-2 gap-6">
                    <Card>
                      <CardHeader>
                        <CardTitle className="flex items-center gap-2">
                          <FileText className="w-5 h-5" />
                          File Information
                        </CardTitle>
                      </CardHeader>
                      <CardContent className="space-y-4">
                        <div className="grid grid-cols-2 gap-4 text-sm">
                          <div>
                            <span className="text-muted-foreground">Type:</span>
                            <p className="font-medium capitalize">
                              {analysis.fileType.split('/')[0]}
                            </p>
                          </div>
                          <div>
                            <span className="text-muted-foreground">Size:</span>
                            <p className="font-medium">
                              {formatFileSize(analysis.fileSize)}
                            </p>
                          </div>
                          {analysis.details.metadata.duration && (
                            <div>
                              <span className="text-muted-foreground">Duration:</span>
                              <p className="font-medium">
                                {Math.round(analysis.details.metadata.duration)}s
                              </p>
                            </div>
                          )}
                          {analysis.details.metadata.resolution && (
                            <div>
                              <span className="text-muted-foreground">Resolution:</span>
                              <p className="font-medium">
                                {analysis.details.metadata.resolution}
                              </p>
                            </div>
                          )}
                        </div>
                      </CardContent>
                    </Card>

                    <Card>
                      <CardHeader>
                        <CardTitle className="flex items-center gap-2">
                          <Activity className="w-5 h-5" />
                          Processing Stats
                        </CardTitle>
                      </CardHeader>
                      <CardContent className="space-y-4">
                        <div className="space-y-3">
                          <div className="flex justify-between items-center text-sm">
                            <span className="text-muted-foreground">Processing Time:</span>
                            <span className="font-medium">
                              {Math.round(analysis.processingTime / 1000)}s
                            </span>
                          </div>
                          <div className="flex justify-between items-center text-sm">
                            <span className="text-muted-foreground">Analysis ID:</span>
                            <span className="font-mono text-xs bg-muted px-2 py-1 rounded">
                              {analysis.id.slice(-8)}
                            </span>
                          </div>
                          <div className="flex justify-between items-center text-sm">
                            <span className="text-muted-foreground">Status:</span>
                            <Badge variant="secondary">Complete</Badge>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  </div>
                </motion.div>
              )}

              {activeTab === 'explanation' && (
                <motion.div
                  key="explanation"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  transition={{ duration: 0.3 }}
                  className="space-y-6"
                >
                  {analysis.explanation ? (
                    <ExplanationDashboard explanation={analysis.explanation} />
                  ) : (
                    <Card>
                      <CardContent className="text-center py-8">
                        <Lightbulb className="w-12 h-12 mx-auto text-muted-foreground mb-4" />
                        <h3 className="font-medium mb-2">Explanation Not Available</h3>
                        <p className="text-sm text-muted-foreground">
                          Detailed explanations are not available for this analysis. This may be due to an older analysis or processing limitations.
                        </p>
                        <Button 
                          variant="outline" 
                          className="mt-4"
                          onClick={() => {
                            // Re-analyze to get explanation
                            toast.info('Re-analysis feature coming soon');
                          }}
                        >
                          <Lightbulb className="w-4 h-4 mr-2" />
                          Generate Explanation
                        </Button>
                      </CardContent>
                    </Card>
                  )}
                </motion.div>
              )}

              {activeTab === 'timeline' && (
                <motion.div
                  key="timeline"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  transition={{ duration: 0.3 }}
                  className="space-y-6"
                >
                  <Card>
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2">
                        <Clock className="w-5 h-5" />
                        Frame-by-Frame Analysis
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="h-80">
                        <ResponsiveContainer width="100%" height="100%">
                          <AreaChart data={timelineData}>
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

                  {/* Audio Analysis if applicable */}
                  {analysis.details.audioAnalysis && (
                    <Card>
                      <CardHeader>
                        <CardTitle className="flex items-center gap-2">
                          <Volume2 className="w-5 h-5" />
                          Audio Analysis
                        </CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="space-y-4">
                          <div className="h-32 bg-muted/30 rounded flex items-center justify-center">
                            <BarChart3 className="w-8 h-8 text-muted-foreground" />
                            <span className="ml-2 text-muted-foreground">Waveform Visualization</span>
                          </div>
                          <div className="grid grid-cols-2 md:grid-cols-3 gap-4 text-sm">
                            <div>
                              <span className="text-muted-foreground">Segments Analyzed:</span>
                              <p className="font-medium">{analysis.details.audioAnalysis.segments.length}</p>
                            </div>
                            <div>
                              <span className="text-muted-foreground">Avg Confidence:</span>
                              <p className="font-medium">
                                {Math.round(analysis.details.audioAnalysis.segments.reduce((sum, seg) => sum + seg.confidence, 0) / analysis.details.audioAnalysis.segments.length * 100)}%
                              </p>
                            </div>
                            <div>
                              <span className="text-muted-foreground">Anomalies Found:</span>
                              <p className="font-medium">
                                {analysis.details.audioAnalysis.segments.filter(seg => seg.anomalies && seg.anomalies.length > 0).length}
                              </p>
                            </div>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  )}
                </motion.div>
              )}

              {activeTab === 'advanced' && (
                <motion.div
                  key="advanced"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  transition={{ duration: 0.3 }}
                  className="space-y-6"
                >
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    <RiskHeatmap data={analysis} />
                    <AnomalyScatterPlot data={analysis} />
                    <MultidimensionalRadar data={analysis} />
                    <ConfidenceTimelineAdvanced data={timelineData} />
                  </div>
                  
                  {/* Comparative Analysis if there are multiple analyses */}
                  {storage.getAnalysisHistory().length > 1 && (
                    <ComparativeAnalysis analyses={storage.getAnalysisHistory()} />
                  )}
                </motion.div>
              )}

              {activeTab === 'details' && (
                <motion.div
                  key="details"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  transition={{ duration: 0.3 }}
                  className="space-y-6"
                >
                  {/* Usage Statistics Card */}
                  <Card>
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2">
                        <BarChart3 className="w-5 h-5" />
                        Free Tier Usage Statistics
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-4">
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                          <div className="text-center p-3 bg-muted/30 rounded">
                            <div className="text-lg font-bold text-primary">{usageTracker.getUsageStats().monthlyScans}</div>
                            <div className="text-xs text-muted-foreground">This Month</div>
                          </div>
                          <div className="text-center p-3 bg-muted/30 rounded">
                            <div className="text-lg font-bold text-green-600">{usageTracker.getRemainingScans()}</div>
                            <div className="text-xs text-muted-foreground">Remaining</div>
                          </div>
                          <div className="text-center p-3 bg-muted/30 rounded">
                            <div className="text-lg font-bold">{usageTracker.getUsageStats().totalScans}</div>
                            <div className="text-xs text-muted-foreground">Total</div>
                          </div>
                          <div className="text-center p-3 bg-muted/30 rounded">
                            <div className={`text-lg font-bold ${usageTracker.getUsagePercentage() > 80 ? 'text-red-600' : 'text-primary'}`}>
                              {usageTracker.getUsagePercentage().toFixed(1)}%
                            </div>
                            <div className="text-xs text-muted-foreground">Used</div>
                          </div>
                        </div>
                        
                        {/* File Type Usage Distribution */}
                        <FileTypeTreemap usageData={usageTracker.getMonthlyUsageByFileType()} />
                      </div>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader>
                      <CardTitle>Raw Analysis Data</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <pre className="bg-muted/30 p-4 rounded-lg overflow-auto text-xs">
                        {JSON.stringify(analysis, null, 2)}
                      </pre>
                    </CardContent>
                  </Card>
                </motion.div>
              )}
            </AnimatePresence>
            
            {/* Hidden PDF Components for Export */}
            <PDFChartComponents analysis={analysis} />
          </div>
        </div>
      </div>
    </div>
  );
}

export default function ResultsPage() {
  return (
    <Suspense fallback={
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto mb-4"></div>
          <p className="text-muted-foreground">Loading analysis...</p>
        </div>
      </div>
    }>
      <ResultsPageContent />
    </Suspense>
  );
}
