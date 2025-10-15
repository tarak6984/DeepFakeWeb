'use client';

import { Alert, AlertDescription } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { Info, Shield } from 'lucide-react';

export function ConfidenceThresholdNotice() {
  return (
    <Alert className="border-blue-200 bg-blue-50 dark:bg-blue-950/20 dark:border-blue-800">
      <Shield className="h-4 w-4 text-blue-600 dark:text-blue-400" />
      <AlertDescription className="text-blue-800 dark:text-blue-200">
        <div className="space-y-2">
          <div className="font-medium">Enhanced Confidence System (95% Threshold)</div>
          <div className="text-sm space-y-1">
            <p>We've upgraded our detection system for higher accuracy:</p>
            <div className="flex flex-wrap gap-2 mt-2">
              <Badge variant="secondary" className="text-green-700 bg-green-100 dark:bg-green-900/30 dark:text-green-300">
                0-90%: Low Risk
              </Badge>
              <Badge variant="secondary" className="text-yellow-700 bg-yellow-100 dark:bg-yellow-900/30 dark:text-yellow-300">
                90-95%: Medium Risk  
              </Badge>
              <Badge variant="secondary" className="text-red-700 bg-red-100 dark:bg-red-900/30 dark:text-red-300">
                95-100%: High Risk
              </Badge>
            </div>
            <p className="text-xs mt-2 text-muted-foreground">
              Only content with 95%+ confidence is flagged as deepfake, ensuring fewer false positives.
            </p>
          </div>
        </div>
      </AlertDescription>
    </Alert>
  );
}