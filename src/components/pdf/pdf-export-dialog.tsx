'use client';

import { useState } from 'react';
import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';
import { Switch } from '@/components/ui/switch';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { 
  FileText, 
  Download, 
  Settings, 
  BarChart3, 
  Clock, 
  Database,
  Loader2,
  CheckCircle 
} from 'lucide-react';
import { toast } from 'sonner';
import { pdfGenerator, PDFGenerationOptions } from '@/lib/pdf-generator';
import { StoredAnalysis } from '@/lib/storage';

interface PDFExportDialogProps {
  analysis: StoredAnalysis;
  trigger?: React.ReactNode;
}

export function PDFExportDialog({ analysis, trigger }: PDFExportDialogProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [progress, setProgress] = useState(0);
  const [options, setOptions] = useState<PDFGenerationOptions>({
    includeCharts: true,
    includeTimeline: true,
    includeAdvancedCharts: false,
    includeRawData: false,
  });

  const handleExport = async () => {
    setIsGenerating(true);
    setProgress(0);

    try {
      // Simulate progress updates
      const progressInterval = setInterval(() => {
        setProgress(prev => {
          if (prev >= 90) {
            clearInterval(progressInterval);
            return 90;
          }
          return prev + 10;
        });
      }, 200);

      // Generate PDF
      await pdfGenerator.downloadReport(analysis, options);
      
      clearInterval(progressInterval);
      setProgress(100);
      
      toast.success('PDF report generated successfully!', {
        description: 'The report has been downloaded to your device.',
      });

      setTimeout(() => {
        setIsOpen(false);
        setProgress(0);
      }, 1000);

    } catch (error) {
      toast.error('Failed to generate PDF report', {
        description: 'Please try again or contact support if the issue persists.',
      });
      console.error('PDF export error:', error);
    } finally {
      setIsGenerating(false);
    }
  };

  const getEstimatedSize = () => {
    let size = 0.5; // Base size in MB
    if (options.includeCharts) size += 1.0;
    if (options.includeTimeline) size += 0.5;
    if (options.includeAdvancedCharts) size += 1.5;
    if (options.includeRawData) size += 0.3;
    return size.toFixed(1);
  };

  const getEstimatedPages = () => {
    let pages = 3; // Base pages (header, summary, details, disclaimer)
    if (options.includeCharts) pages += 1;
    if (options.includeTimeline) pages += 1;
    if (options.includeAdvancedCharts) pages += 2;
    if (options.includeRawData) pages += 1;
    return pages;
  };

  return (
    <Dialog open={isOpen} onOpenChange={setIsOpen}>
      <DialogTrigger asChild>
        {trigger || (
          <Button variant="outline" size="sm">
            <FileText className="w-4 h-4 mr-2" />
            Export PDF
          </Button>
        )}
      </DialogTrigger>
      
      <DialogContent className="sm:max-w-[500px]">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <FileText className="w-5 h-5" />
            Export PDF Report
          </DialogTitle>
          <DialogDescription>
            Customize your PDF report options. The report will include a comprehensive analysis
            of {analysis.filename}.
          </DialogDescription>
        </DialogHeader>

        {isGenerating ? (
          <div className="space-y-6 py-6">
            <div className="text-center space-y-4">
              {progress < 100 ? (
                <Loader2 className="w-12 h-12 animate-spin mx-auto text-primary" />
              ) : (
                <CheckCircle className="w-12 h-12 mx-auto text-green-600" />
              )}
              
              <div className="space-y-2">
                <p className="text-sm font-medium">
                  {progress < 100 ? 'Generating PDF report...' : 'Report generated successfully!'}
                </p>
                <Progress value={progress} className="w-full" />
                <p className="text-xs text-muted-foreground">
                  {progress}% complete
                </p>
              </div>
            </div>
          </div>
        ) : (
          <div className="space-y-6">
            {/* Report Preview Info */}
            <div className="bg-muted/30 p-4 rounded-lg space-y-2">
              <div className="flex items-center justify-between text-sm">
                <span className="text-muted-foreground">Estimated size:</span>
                <Badge variant="secondary">{getEstimatedSize()} MB</Badge>
              </div>
              <div className="flex items-center justify-between text-sm">
                <span className="text-muted-foreground">Estimated pages:</span>
                <Badge variant="secondary">{getEstimatedPages()} pages</Badge>
              </div>
              <div className="flex items-center justify-between text-sm">
                <span className="text-muted-foreground">Format:</span>
                <Badge variant="secondary">PDF (A4)</Badge>
              </div>
            </div>

            {/* Export Options */}
            <div className="space-y-4">
              <h4 className="font-medium text-sm">Report Content</h4>
              
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    <BarChart3 className="w-4 h-4 text-muted-foreground" />
                    <div>
                      <p className="text-sm font-medium">Include Charts</p>
                      <p className="text-xs text-muted-foreground">
                        Confidence gauge and category breakdown
                      </p>
                    </div>
                  </div>
                  <Switch
                    checked={options.includeCharts}
                    onCheckedChange={(checked) =>
                      setOptions(prev => ({ ...prev, includeCharts: checked }))
                    }
                  />
                </div>

                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    <Clock className="w-4 h-4 text-muted-foreground" />
                    <div>
                      <p className="text-sm font-medium">Timeline Analysis</p>
                      <p className="text-xs text-muted-foreground">
                        Frame-by-frame confidence timeline
                      </p>
                    </div>
                  </div>
                  <Switch
                    checked={options.includeTimeline}
                    onCheckedChange={(checked) =>
                      setOptions(prev => ({ ...prev, includeTimeline: checked }))
                    }
                  />
                </div>

                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    <Settings className="w-4 h-4 text-muted-foreground" />
                    <div>
                      <p className="text-sm font-medium">Advanced Charts</p>
                      <p className="text-xs text-muted-foreground">
                        Heatmaps, scatter plots, and radar charts
                      </p>
                    </div>
                  </div>
                  <Switch
                    checked={options.includeAdvancedCharts}
                    onCheckedChange={(checked) =>
                      setOptions(prev => ({ ...prev, includeAdvancedCharts: checked }))
                    }
                  />
                </div>

                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    <Database className="w-4 h-4 text-muted-foreground" />
                    <div>
                      <p className="text-sm font-medium">Raw Data</p>
                      <p className="text-xs text-muted-foreground">
                        Complete JSON analysis data
                      </p>
                    </div>
                  </div>
                  <Switch
                    checked={options.includeRawData}
                    onCheckedChange={(checked) =>
                      setOptions(prev => ({ ...prev, includeRawData: checked }))
                    }
                  />
                </div>
              </div>
            </div>

            {/* Report Content Summary */}
            <div className="bg-blue-50 dark:bg-blue-950/20 p-3 rounded-lg">
              <p className="text-xs text-blue-700 dark:text-blue-300">
                💡 <strong>Note:</strong> The PDF report will always include an executive summary, 
                technical details, and disclaimer regardless of the options selected above.
              </p>
            </div>
          </div>
        )}

        <DialogFooter>
          {!isGenerating && (
            <>
              <Button variant="outline" onClick={() => setIsOpen(false)}>
                Cancel
              </Button>
              <Button onClick={handleExport} className="min-w-[120px]">
                <Download className="w-4 h-4 mr-2" />
                Generate PDF
              </Button>
            </>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}