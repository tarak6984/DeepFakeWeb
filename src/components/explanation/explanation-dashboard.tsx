'use client';

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Separator } from '@/components/ui/separator';
import { Progress } from '@/components/ui/progress';
import {
  Brain,
  Eye,
  Volume2,
  FileText,
  Clock,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Info,
  ChevronRight,
  ChevronDown,
  Lightbulb,
  Shield,
  TrendingUp,
  TrendingDown,
  Minus,
  ZoomIn
} from 'lucide-react';
import type { DetailedExplanation, ExplanationReason } from '@/lib/types';

interface ExplanationDashboardProps {
  explanation: DetailedExplanation;
  className?: string;
}

export function ExplanationDashboard({ explanation, className }: ExplanationDashboardProps) {
  const [activeTab, setActiveTab] = useState<'summary' | 'reasons' | 'insights' | 'evidence'>('summary');
  const [expandedReasons, setExpandedReasons] = useState<Set<string>>(new Set());

  const toggleReasonExpansion = (reasonId: string) => {
    const newExpanded = new Set(expandedReasons);
    if (newExpanded.has(reasonId)) {
      newExpanded.delete(reasonId);
    } else {
      newExpanded.add(reasonId);
    }
    setExpandedReasons(newExpanded);
  };

  const getSeverityColor = (severity: 'high' | 'medium' | 'low') => {
    switch (severity) {
      case 'high': return 'text-red-600 dark:text-red-400 bg-red-100 dark:bg-red-900/20';
      case 'medium': return 'text-yellow-600 dark:text-yellow-400 bg-yellow-100 dark:bg-yellow-900/20';
      case 'low': return 'text-green-600 dark:text-green-400 bg-green-100 dark:bg-green-900/20';
    }
  };

  const getCategoryIcon = (category: ExplanationReason['category']) => {
    switch (category) {
      case 'visual': return Eye;
      case 'audio': return Volume2;
      case 'model': return Brain;
      case 'temporal': return Clock;
      case 'metadata': return FileText;
      case 'technical': return Info;
      default: return Info;
    }
  };

  const getContributionIcon = (contribution: 'positive' | 'negative' | 'neutral') => {
    switch (contribution) {
      case 'positive': return TrendingUp;
      case 'negative': return TrendingDown;
      case 'neutral': return Minus;
    }
  };

  return (
    <div className={className}>
      {/* Tab Navigation */}
      <div className="mb-6">
        <div className="border-b">
          <nav className="-mb-px flex space-x-8">
            {[
              { id: 'summary', label: 'Summary', icon: Lightbulb },
              { id: 'reasons', label: 'Detailed Reasons', icon: Brain },
              { id: 'insights', label: 'AI Insights', icon: Eye },
              { id: 'evidence', label: 'Evidence', icon: Shield },
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                className={`py-2 px-1 border-b-2 font-medium text-sm flex items-center gap-2 transition-colors ${
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
      </div>

      {/* Tab Content */}
      <AnimatePresence mode="wait">
        {activeTab === 'summary' && (
          <motion.div
            key="summary"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.3 }}
            className="space-y-6"
          >
            {/* Primary Reason */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Lightbulb className="w-5 h-5" />
                  Why This Classification?
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-lg leading-relaxed mb-4">
                  {explanation.summary.primaryReason}
                </p>
                {explanation.summary.secondaryReasons.length > 0 && (
                  <div>
                    <h4 className="font-medium mb-2 text-muted-foreground">Additional factors:</h4>
                    <ul className="space-y-1">
                      {explanation.summary.secondaryReasons.map((reason, index) => (
                        <li key={index} className="flex items-center gap-2 text-sm">
                          <ChevronRight className="w-3 h-3 text-muted-foreground" />
                          {reason}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Authenticity Indicators */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Shield className="w-5 h-5" />
                  Authenticity Indicators
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {explanation.summary.authenticityIndicators.map((indicator, index) => {
                    const ContributionIcon = getContributionIcon(indicator.contribution);
                    return (
                      <div key={index} className="flex items-center justify-between p-3 bg-muted/30 rounded-lg">
                        <div className="flex items-center gap-3">
                          <ContributionIcon className={`w-4 h-4 ${
                            indicator.contribution === 'positive' ? 'text-green-600' :
                            indicator.contribution === 'negative' ? 'text-red-600' :
                            'text-muted-foreground'
                          }`} />
                          <div>
                            <div className="font-medium">{indicator.factor}</div>
                            <div className="text-sm text-muted-foreground">{indicator.explanation}</div>
                          </div>
                        </div>
                        <div className="text-right">
                          <div className="text-sm font-medium">
                            Weight: {Math.round(indicator.weight * 100)}%
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </CardContent>
            </Card>

            {/* Risk Factors */}
            {explanation.summary.riskFactors.length > 0 && (
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <AlertTriangle className="w-5 h-5" />
                    Risk Assessment
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {explanation.summary.riskFactors.map((risk, index) => (
                      <div key={index} className="border-l-4 border-yellow-500 pl-4 py-2">
                        <div className="flex items-center gap-2 mb-1">
                          <Badge variant="outline" className={
                            risk.severity === 'critical' ? 'border-red-500 text-red-700' :
                            risk.severity === 'high' ? 'border-orange-500 text-orange-700' :
                            risk.severity === 'medium' ? 'border-yellow-500 text-yellow-700' :
                            'border-blue-500 text-blue-700'
                          }>
                            {risk.severity.toUpperCase()}
                          </Badge>
                          <span className="font-medium">{risk.factor}</span>
                          <span className="text-sm text-muted-foreground">
                            ({Math.round(risk.likelihood * 100)}% likelihood)
                          </span>
                        </div>
                        <p className="text-sm text-muted-foreground">{risk.impact}</p>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Recommendations */}
            {explanation.summary.recommendedActions && explanation.summary.recommendedActions.length > 0 && (
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Info className="w-5 h-5" />
                    Recommendations
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <ul className="space-y-2">
                    {explanation.summary.recommendedActions.map((action, index) => (
                      <li key={index} className="flex items-start gap-2">
                        <CheckCircle className="w-4 h-4 mt-0.5 text-green-600 flex-shrink-0" />
                        <span className="text-sm">{action}</span>
                      </li>
                    ))}
                  </ul>
                </CardContent>
              </Card>
            )}
          </motion.div>
        )}

        {activeTab === 'reasons' && (
          <motion.div
            key="reasons"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.3 }}
            className="space-y-4"
          >
            {explanation.reasons.map((reason, index) => {
              const CategoryIcon = getCategoryIcon(reason.category);
              const isExpanded = expandedReasons.has(reason.id);

              return (
                <Card key={reason.id}>
                  <CardContent className="p-0">
                    <button
                      onClick={() => toggleReasonExpansion(reason.id)}
                      className="w-full p-4 text-left hover:bg-muted/50 transition-colors"
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                          <CategoryIcon className="w-5 h-5 text-muted-foreground" />
                          <div>
                            <h3 className="font-medium">{reason.title}</h3>
                            <div className="flex items-center gap-2 mt-1">
                              <Badge className={getSeverityColor(reason.severity)}>
                                {reason.severity}
                              </Badge>
                              <span className="text-sm text-muted-foreground capitalize">
                                {reason.category} • {reason.type}
                              </span>
                            </div>
                          </div>
                        </div>
                        <div className="flex items-center gap-2">
                          <div className="text-right">
                            <div className="text-sm font-medium">
                              {Math.round(reason.confidence * 100)}% confidence
                            </div>
                          </div>
                          {isExpanded ? (
                            <ChevronDown className="w-4 h-4" />
                          ) : (
                            <ChevronRight className="w-4 h-4" />
                          )}
                        </div>
                      </div>
                    </button>

                    <AnimatePresence>
                      {isExpanded && (
                        <motion.div
                          initial={{ height: 0, opacity: 0 }}
                          animate={{ height: 'auto', opacity: 1 }}
                          exit={{ height: 0, opacity: 0 }}
                          transition={{ duration: 0.2 }}
                          className="overflow-hidden"
                        >
                          <div className="px-4 pb-4 border-t">
                            <div className="pt-4 space-y-3">
                              <div>
                                <h4 className="font-medium mb-1">Description</h4>
                                <p className="text-sm text-muted-foreground leading-relaxed">
                                  {reason.description}
                                </p>
                              </div>

                              {reason.technicalDetails && (
                                <div>
                                  <h4 className="font-medium mb-1">Technical Details</h4>
                                  <p className="text-xs font-mono bg-muted/50 p-2 rounded">
                                    {reason.technicalDetails}
                                  </p>
                                </div>
                              )}

                              {reason.modelSources && reason.modelSources.length > 0 && (
                                <div>
                                  <h4 className="font-medium mb-1">Source Models</h4>
                                  <div className="flex flex-wrap gap-1">
                                    {reason.modelSources.map((model, idx) => (
                                      <Badge key={idx} variant="secondary" className="text-xs">
                                        {model}
                                      </Badge>
                                    ))}
                                  </div>
                                </div>
                              )}

                              {reason.supportingEvidence && reason.supportingEvidence.length > 0 && (
                                <div>
                                  <h4 className="font-medium mb-1">Supporting Evidence</h4>
                                  <ul className="text-sm text-muted-foreground space-y-1">
                                    {reason.supportingEvidence.map((evidence, idx) => (
                                      <li key={idx} className="flex items-start gap-2">
                                        <ZoomIn className="w-3 h-3 mt-0.5 flex-shrink-0" />
                                        {evidence}
                                      </li>
                                    ))}
                                  </ul>
                                </div>
                              )}
                            </div>
                          </div>
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </CardContent>
                </Card>
              );
            })}
          </motion.div>
        )}

        {activeTab === 'insights' && (
          <motion.div
            key="insights"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.3 }}
            className="space-y-6"
          >
            {explanation.modelInsights.map((insight, index) => (
              <Card key={index}>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Brain className="w-5 h-5" />
                    {insight.modelName}
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex items-center justify-between">
                    <Badge variant="outline" className="capitalize">
                      {insight.modelType.replace('_', ' ')}
                    </Badge>
                    <div className="text-right">
                      <div className="text-sm text-muted-foreground">Confidence</div>
                      <div className="font-bold">{Math.round(insight.confidence * 100)}%</div>
                    </div>
                  </div>

                  <div>
                    <h4 className="font-medium mb-2">AI Reasoning</h4>
                    <p className="text-sm text-muted-foreground leading-relaxed">
                      {insight.reasoning}
                    </p>
                  </div>

                  <div>
                    <h4 className="font-medium mb-2">Key Findings</h4>
                    <ul className="space-y-1">
                      {insight.keyFindings.map((finding, idx) => (
                        <li key={idx} className="flex items-start gap-2 text-sm">
                          <CheckCircle className="w-3 h-3 mt-0.5 text-green-600 flex-shrink-0" />
                          {finding}
                        </li>
                      ))}
                    </ul>
                  </div>

                  <div className="grid grid-cols-2 gap-4 pt-2 border-t">
                    <div className="text-center">
                      <div className="text-2xl font-bold text-primary">
                        {insight.technicalScore.toFixed(3)}
                      </div>
                      <div className="text-xs text-muted-foreground">Technical Score</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-muted-foreground">
                        {Math.round(insight.processingTime / 1000)}s
                      </div>
                      <div className="text-xs text-muted-foreground">Processing Time</div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </motion.div>
        )}

        {activeTab === 'evidence' && (
          <motion.div
            key="evidence"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.3 }}
            className="space-y-6"
          >
            {explanation.evidence.length > 0 ? (
              <div className="space-y-4">
                {explanation.evidence.map((evidence) => (
                  <Card key={evidence.id}>
                    <CardContent className="p-4">
                      <div className="flex items-start justify-between mb-3">
                        <div>
                          <Badge variant="outline" className="mb-2 capitalize">
                            {evidence.type.replace('_', ' ')}
                          </Badge>
                          <p className="text-sm">{evidence.description}</p>
                        </div>
                        <div className="text-right">
                          <div className="text-sm text-muted-foreground">Severity</div>
                          <Progress 
                            value={evidence.severity * 100} 
                            className="w-20 mt-1"
                          />
                        </div>
                      </div>

                      {evidence.location && (
                        <div className="flex gap-4 text-xs text-muted-foreground">
                          {evidence.location.frame && (
                            <span>Frame: {evidence.location.frame}</span>
                          )}
                          {evidence.location.timestamp && (
                            <span>Time: {evidence.location.timestamp.toFixed(2)}s</span>
                          )}
                          {evidence.location.frequency && (
                            <span>Frequency: {evidence.location.frequency}Hz</span>
                          )}
                        </div>
                      )}
                    </CardContent>
                  </Card>
                ))}
              </div>
            ) : (
              <Card>
                <CardContent className="text-center py-8">
                  <Shield className="w-12 h-12 mx-auto text-muted-foreground mb-4" />
                  <h3 className="font-medium mb-2">No Specific Evidence Found</h3>
                  <p className="text-sm text-muted-foreground">
                    The analysis did not identify specific anomalies or artifacts that require detailed examination.
                  </p>
                </CardContent>
              </Card>
            )}

            {/* Metadata Analysis */}
            {explanation.metadataAnalysis && (
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <FileText className="w-5 h-5" />
                    File Properties Analysis
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    {explanation.metadataAnalysis.fileProperties.map((prop, index) => (
                      <div key={index} className="flex items-center justify-between p-2 bg-muted/30 rounded">
                        <div>
                          <span className="font-medium">{prop.property}:</span>
                          <span className="ml-2">{prop.actualValue}</span>
                        </div>
                        <Badge variant={prop.assessment === 'normal' ? 'secondary' : 'destructive'}>
                          {prop.assessment}
                        </Badge>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}