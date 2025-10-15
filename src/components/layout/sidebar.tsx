'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import {
  History,
  FileImage,
  FileVideo,
  FileAudio,
  File,
  Trash2,
  Download,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Clock,
  ChevronRight,
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { storage, type StoredAnalysis } from '@/lib/storage';
import { formatFileSize, getFileIcon } from '@/lib/storage';

interface SidebarProps {
  className?: string;
  onAnalysisSelect?: (analysisId: string) => void;
}

export function Sidebar({ className, onAnalysisSelect }: SidebarProps) {
  const [analyses, setAnalyses] = useState<StoredAnalysis[]>([]);
  const [isCollapsed, setIsCollapsed] = useState(false);

  useEffect(() => {
    loadAnalyses();
    
    // Listen for storage changes
    const handleStorageChange = () => {
      loadAnalyses();
    };
    
    window.addEventListener('storage', handleStorageChange);
    return () => window.removeEventListener('storage', handleStorageChange);
  }, []);

  const loadAnalyses = () => {
    const history = storage.getAnalysisHistory();
    setAnalyses(history);
  };

  const handleDeleteAnalysis = (analysisId: string) => {
    storage.removeAnalysisFromHistory(analysisId);
    loadAnalyses();
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence < 0.3) return 'text-green-600 dark:text-green-400';
    if (confidence < 0.7) return 'text-yellow-600 dark:text-yellow-400';
    return 'text-red-600 dark:text-red-400';
  };

  const getConfidenceIcon = (confidence: number) => {
    if (confidence < 0.3) return CheckCircle;
    if (confidence < 0.7) return AlertTriangle;
    return XCircle;
  };

  const formatTimestamp = (timestamp: string) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diff = now.getTime() - date.getTime();
    const minutes = Math.floor(diff / (1000 * 60));
    const hours = Math.floor(diff / (1000 * 60 * 60));
    const days = Math.floor(diff / (1000 * 60 * 60 * 24));

    if (minutes < 1) return 'Just now';
    if (minutes < 60) return `${minutes}m ago`;
    if (hours < 24) return `${hours}h ago`;
    if (days < 7) return `${days}d ago`;
    return date.toLocaleDateString();
  };

  if (isCollapsed) {
    return (
      <motion.div
        initial={{ width: 280 }}
        animate={{ width: 60 }}
        transition={{ duration: 0.3 }}
        className={`bg-card border-r h-screen sticky top-16 overflow-hidden ${className}`}
      >
        <div className="p-3">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setIsCollapsed(false)}
            className="w-full p-2"
          >
            <History className="h-5 w-5" />
          </Button>
        </div>
      </motion.div>
    );
  }

  return (
    <motion.div
      initial={{ width: 60 }}
      animate={{ width: 280 }}
      transition={{ duration: 0.3 }}
      className={`bg-card border-r h-screen sticky top-16 overflow-hidden ${className}`}
    >
      <div className="flex flex-col h-full">
        {/* Header */}
        <div className="p-4 border-b">
          <div className="flex items-center justify-between">
            <h2 className="font-semibold text-sm flex items-center gap-2">
              <History className="h-4 w-4" />
              Recent Results
            </h2>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setIsCollapsed(true)}
              className="h-6 w-6 p-0"
            >
              <ChevronRight className="h-3 w-3" />
            </Button>
          </div>
          <p className="text-xs text-muted-foreground mt-1">
            {analyses.length} {analyses.length === 1 ? 'analysis' : 'analyses'}
          </p>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-2">
          <AnimatePresence>
            {analyses.length === 0 ? (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="text-center py-8 px-4"
              >
                <History className="h-8 w-8 text-muted-foreground mx-auto mb-3" />
                <p className="text-sm text-muted-foreground mb-2">No recent analyses</p>
                <p className="text-xs text-muted-foreground">
                  Upload and analyze files to see results here
                </p>
              </motion.div>
            ) : (
              <div className="space-y-2">
                {analyses.map((analysis, index) => {
                  const ConfidenceIcon = getConfidenceIcon(analysis.confidence);
                  
                  return (
                    <motion.div
                      key={analysis.id}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: index * 0.1 }}
                    >
                      <Card
                        className="cursor-pointer hover:bg-accent/50 transition-colors group"
                        onClick={() => onAnalysisSelect?.(analysis.id)}
                      >
                        <CardContent className="p-3">
                          <div className="flex items-start gap-2">
                            {/* File Preview/Icon */}
                            <div className="w-10 h-10 rounded bg-muted flex items-center justify-center flex-shrink-0">
                              {analysis.thumbnailBlob ? (
                                <img
                                  src={analysis.thumbnailBlob}
                                  alt={analysis.filename}
                                  className="w-full h-full object-cover rounded"
                                />
                              ) : (
                                <span className="text-lg">{getFileIcon(analysis.fileType)}</span>
                              )}
                            </div>

                            {/* Content */}
                            <div className="flex-1 min-w-0">
                              <div className="flex items-start justify-between gap-1">
                                <h4 className="text-xs font-medium truncate" title={analysis.filename}>
                                  {analysis.filename}
                                </h4>
                                <Badge
                                  variant="secondary"
                                  className={`text-xs px-1 py-0 ${getConfidenceColor(analysis.confidence)}`}
                                >
                                  {analysis.confidence > 1 ? Math.round(analysis.confidence) : Math.round(analysis.confidence * 100)}%
                                </Badge>
                              </div>
                              
                              <div className="flex items-center gap-1 mt-1">
                                <ConfidenceIcon className={`h-3 w-3 ${getConfidenceColor(analysis.confidence)}`} />
                                <span className="text-xs text-muted-foreground capitalize">
                                  {analysis.prediction}
                                </span>
                              </div>

                              <div className="flex items-center justify-between mt-2">
                                <span className="text-xs text-muted-foreground">
                                  {formatFileSize(analysis.fileSize)}
                                </span>
                                <span className="text-xs text-muted-foreground flex items-center gap-1">
                                  <Clock className="h-2.5 w-2.5" />
                                  {formatTimestamp(analysis.timestamp)}
                                </span>
                              </div>
                            </div>
                          </div>

                          {/* Actions */}
                          <div className="flex items-center justify-end gap-1 mt-2 opacity-0 group-hover:opacity-100 transition-opacity">
                            <Link href={`/results?id=${analysis.id}`}>
                              <Button
                                variant="ghost"
                                size="sm"
                                className="h-6 w-6 p-0"
                                onClick={(e) => e.stopPropagation()}
                              >
                                <ChevronRight className="h-3 w-3" />
                              </Button>
                            </Link>
                            <Button
                              variant="ghost"
                              size="sm"
                              className="h-6 w-6 p-0 hover:text-destructive"
                              onClick={(e) => {
                                e.stopPropagation();
                                handleDeleteAnalysis(analysis.id);
                              }}
                            >
                              <Trash2 className="h-3 w-3" />
                            </Button>
                          </div>
                        </CardContent>
                      </Card>
                    </motion.div>
                  );
                })}
              </div>
            )}
          </AnimatePresence>
        </div>

        {/* Footer */}
        {analyses.length > 0 && (
          <div className="p-3 border-t">
            <div className="flex gap-2">
              <Button
                variant="outline"
                size="sm"
                className="flex-1 text-xs h-7"
                onClick={() => {
                  const data = storage.exportData();
                  const blob = new Blob([data], { type: 'application/json' });
                  const url = URL.createObjectURL(blob);
                  const a = document.createElement('a');
                  a.href = url;
                  a.download = `deepfake-analysis-${new Date().toISOString().split('T')[0]}.json`;
                  a.click();
                  URL.revokeObjectURL(url);
                }}
              >
                <Download className="h-3 w-3 mr-1" />
                Export
              </Button>
              <Button
                variant="outline"
                size="sm"
                className="flex-1 text-xs h-7 hover:text-destructive"
                onClick={() => {
                  if (confirm('Clear all analysis history?')) {
                    storage.clearAnalysisHistory();
                    loadAnalyses();
                  }
                }}
              >
                <Trash2 className="h-3 w-3 mr-1" />
                Clear
              </Button>
            </div>
          </div>
        )}
      </div>
    </motion.div>
  );
}