'use client';

import { PieChart, Pie, Cell, ResponsiveContainer } from 'recharts';
import { motion } from 'framer-motion';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { AlertTriangle, CheckCircle, XCircle } from 'lucide-react';

interface ConfidenceGaugeProps {
  confidence: number;
  prediction: 'authentic' | 'manipulated' | 'inconclusive';
  className?: string;
}

export function ConfidenceGauge({
  confidence,
  prediction,
  className,
}: ConfidenceGaugeProps) {
  // Handle both 0-1 range (0.99) and 0-100 range (99) safely
  const percentage = confidence > 1 ? Math.round(confidence) : Math.round(confidence * 100);
  
  // Calculate gauge data
  const gaugeData = [
    { name: 'confidence', value: percentage },
    { name: 'remaining', value: 100 - percentage },
  ];

  // Get color based on confidence level (Reality Defender style - 70% threshold)
  const getConfidenceColor = () => {
    if (confidence < 0.3) return '#10b981'; // green - Low Risk (0-30%)
    if (confidence < 0.7) return '#f59e0b'; // yellow - Medium Risk (30-70%)
    return '#ef4444'; // red - High Risk (70-100%)
  };

  // Get icon based on prediction
  const getPredictionIcon = () => {
    switch (prediction) {
      case 'authentic':
        return <CheckCircle className="h-5 w-5 text-green-600 dark:text-green-400" />;
      case 'manipulated':
        return <XCircle className="h-5 w-5 text-red-600 dark:text-red-400" />;
      case 'inconclusive':
        return <AlertTriangle className="h-5 w-5 text-yellow-600 dark:text-yellow-400" />;
    }
  };

  const getConfidenceLabel = () => {
    if (confidence < 0.3) return 'Low Risk';
    if (confidence < 0.7) return 'Medium Risk';
    return 'High Risk';
  };

  const getBadgeVariant = () => {
    switch (prediction) {
      case 'authentic':
        return 'secondary';
      case 'manipulated':
        return 'destructive';
      case 'inconclusive':
        return 'outline';
    }
  };

  return (
    <Card className={className}>
      <CardHeader className="pb-4">
        <CardTitle className="flex items-center justify-between">
          <span>Confidence Score</span>
          <Badge variant={getBadgeVariant()} className="capitalize">
            {getPredictionIcon()}
            <span className="ml-1">{prediction}</span>
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="relative">
          {/* Gauge Chart */}
          <div className="relative mx-auto w-48 h-48">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={gaugeData}
                  cx="50%"
                  cy="50%"
                  startAngle={180}
                  endAngle={0}
                  innerRadius={60}
                  outerRadius={80}
                  paddingAngle={0}
                  dataKey="value"
                >
                  <Cell fill={getConfidenceColor()} />
                  <Cell fill="transparent" />
                </Pie>
              </PieChart>
            </ResponsiveContainer>

            {/* Center Content */}
            <div className="absolute inset-0 flex flex-col items-center justify-center">
              <motion.div
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ delay: 0.5, type: "spring", stiffness: 200 }}
                className="text-center"
              >
                <div
                  className="text-3xl font-bold mb-1"
                  style={{ color: getConfidenceColor() }}
                >
                  {percentage}%
                </div>
                <div className="text-sm text-muted-foreground font-medium">
                  {getConfidenceLabel()}
                </div>
              </motion.div>
            </div>

            {/* Animated Background Ring */}
            <div className="absolute inset-0">
              <svg className="w-full h-full -rotate-90" viewBox="0 0 192 192">
                <circle
                  cx="96"
                  cy="96"
                  r="70"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="4"
                  className="text-muted/20"
                />
                <motion.circle
                  cx="96"
                  cy="96"
                  r="70"
                  fill="none"
                  stroke={getConfidenceColor()}
                  strokeWidth="4"
                  strokeLinecap="round"
                  strokeDasharray={`${2 * Math.PI * 70}`}
                  initial={{ strokeDashoffset: 2 * Math.PI * 70 }}
                  animate={{ strokeDashoffset: 2 * Math.PI * 70 * (1 - percentage / 100) }}
                  transition={{ duration: 1.5, ease: "easeInOut" }}
                  style={{
                    filter: 'drop-shadow(0 0 6px currentColor)',
                    opacity: 0.8,
                  }}
                />
              </svg>
            </div>
          </div>

          {/* Risk Level Indicators */}
          <div className="mt-6 space-y-3">
            <div className="flex items-center justify-between text-sm">
              <span className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full bg-green-500"></div>
                Low Risk
              </span>
              <span className="text-muted-foreground">0-30%</span>
            </div>
            <div className="flex items-center justify-between text-sm">
              <span className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
                Medium Risk
              </span>
              <span className="text-muted-foreground">30-70%</span>
            </div>
            <div className="flex items-center justify-between text-sm">
              <span className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full bg-red-500"></div>
                High Risk
              </span>
              <span className="text-muted-foreground">70-100%</span>
            </div>
          </div>

          {/* Additional Info */}
          <div className="mt-6 p-4 bg-muted/50 rounded-lg">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium">Detection Result</span>
              {getPredictionIcon()}
            </div>
            <p className="text-xs text-muted-foreground leading-relaxed">
              {prediction === 'authentic' && 
                'This file appears to be authentic with low likelihood of manipulation. AI analysis found consistent patterns typical of genuine content.'}
              {prediction === 'manipulated' && 
                'This file shows signs of potential manipulation or deepfake generation. Multiple AI models detected anomalous patterns consistent with synthetic content.'}
              {prediction === 'inconclusive' && 
                'The analysis could not determine with high confidence whether this file is authentic or manipulated. Mixed signals from different detection methods require further investigation.'}
            </p>
            
            <div className="mt-3 pt-2 border-t border-muted">
              <div className="flex items-center gap-2 text-xs text-muted-foreground">
                <span className="font-medium">💡 Tip:</span>
                <span>Check the "Why This Result?" tab for detailed explanations and AI reasoning.</span>
              </div>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}