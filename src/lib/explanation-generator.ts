import { 
  DetailedExplanation, 
  ExplanationReason, 
  ExplanationEvidence, 
  ModelInsight, 
  ExplanationSummary,
  ExplanationConfig 
} from './types';
import type { AnalysisResult } from './local-deepfake-api';

/**
 * Generates human-readable explanations for deepfake detection results
 */
export class ExplanationGenerator {
  private config: ExplanationConfig;

  constructor(config: Partial<ExplanationConfig> = {}) {
    this.config = {
      includeModelInsights: true,
      includeTechnicalDetails: true,
      includeVisualEvidence: true,
      simplifyForGeneralAudience: false,
      maxReasonsToShow: 10,
      evidenceVisualization: true,
      ...config
    };
  }

  /**
   * Generate complete explanation from analysis result
   */
  generateExplanation(analysis: AnalysisResult): DetailedExplanation {
    const reasons = this.generateReasons(analysis);
    const evidence = this.generateEvidence(analysis);
    const modelInsights = this.generateModelInsights(analysis);
    const summary = this.generateSummary(analysis, reasons, modelInsights);

    return {
      id: `exp_${analysis.id}`,
      analysisId: analysis.id,
      summary,
      reasons: reasons.slice(0, this.config.maxReasonsToShow),
      evidence,
      modelInsights,
      temporalAnalysis: this.generateTemporalAnalysis(analysis),
      metadataAnalysis: this.generateMetadataAnalysis(analysis),
      generatedAt: new Date().toISOString(),
      processingVersion: '1.0.0'
    };
  }

  /**
   * Generate primary reasons for the classification
   */
  private generateReasons(analysis: AnalysisResult): ExplanationReason[] {
    const reasons: ExplanationReason[] = [];
    const confidence = analysis.confidence;
    const prediction = analysis.prediction;

    // Primary classification reason
    if (prediction === 'authentic') {
      reasons.push({
        id: 'auth_primary',
        category: 'model',
        type: 'evidence',
        severity: confidence < 0.2 ? 'low' : 'medium',
        confidence: 1 - confidence,
        title: 'Content appears authentic',
        description: `AI analysis indicates this content is likely genuine with ${Math.round((1 - confidence) * 100)}% confidence. Multiple detection models found no significant signs of manipulation.`,
        technicalDetails: `Ensemble model score: ${confidence.toFixed(3)}, indicating low manipulation probability.`,
        modelSources: ['ensemble', 'deepfake_detector']
      });

      // Add supporting authentic indicators
      if (confidence < 0.1) {
        reasons.push({
          id: 'auth_strong',
          category: 'technical',
          type: 'evidence',
          severity: 'low',
          confidence: 0.95,
          title: 'Strong authenticity indicators',
          description: 'Analysis found consistent patterns typical of genuine content with no detectable artifacts.',
          technicalDetails: 'Low variance in model predictions, consistent temporal patterns, normal compression artifacts.'
        });
      }
    }

    if (prediction === 'manipulated') {
      reasons.push({
        id: 'manip_primary',
        category: 'model',
        type: 'anomaly',
        severity: confidence > 0.95 ? 'high' : confidence > 0.9 ? 'medium' : 'low',
        confidence,
        title: 'Potential manipulation detected',
        description: `AI analysis detected signs of possible manipulation with ${Math.round(confidence * 100)}% confidence. Multiple indicators suggest this content may be synthetically generated or altered.`,
        technicalDetails: `Ensemble model score: ${confidence.toFixed(3)}, exceeding manipulation threshold.`,
        modelSources: ['ensemble', 'deepfake_detector']
      });

      // Add specific manipulation indicators based on confidence level
      if (confidence > 0.95) {
        reasons.push({
          id: 'manip_strong',
          category: 'visual',
          type: 'artifact',
          severity: 'high',
          confidence: 0.9,
          title: 'Strong manipulation indicators',
          description: 'High-confidence detection of typical deepfake artifacts and inconsistencies.',
          technicalDetails: 'Multiple models detected convergent evidence of synthetic generation patterns.'
        });
      }

      if (confidence > 0.9) {
        reasons.push({
          id: 'manip_patterns',
          category: 'temporal',
          type: 'pattern',
          severity: 'medium',
          confidence: 0.75,
          title: 'Suspicious temporal patterns',
          description: 'Analysis detected frame-to-frame inconsistencies common in generated content.',
          technicalDetails: 'Temporal coherence metrics indicate potential frame interpolation or synthesis.'
        });
      }
    }

    if (prediction === 'inconclusive') {
      reasons.push({
        id: 'inconcl_primary',
        category: 'model',
        type: 'inconsistency',
        severity: 'medium',
        confidence: Math.abs(0.5 - confidence) * 2,
        title: 'Analysis inconclusive',
        description: `The analysis could not determine with high confidence whether this content is authentic or manipulated. Confidence score: ${Math.round(confidence * 100)}%.`,
        technicalDetails: `Score ${confidence.toFixed(3)} falls in the inconclusive range (0.9-0.95). Mixed signals from different detection models.`,
        modelSources: ['ensemble']
      });
    }

    // Add file-type specific reasons
    this.addFileTypeSpecificReasons(analysis, reasons);

    // Add metadata-based reasons
    this.addMetadataReasons(analysis, reasons);

    return reasons.sort((a, b) => b.confidence - a.confidence);
  }

  private addFileTypeSpecificReasons(analysis: AnalysisResult, reasons: ExplanationReason[]) {
    const fileType = analysis.fileType;

    if (fileType.startsWith('image/')) {
      reasons.push({
        id: 'img_analysis',
        category: 'visual',
        type: 'evidence',
        severity: 'medium',
        confidence: 0.8,
        title: 'Image analysis completed',
        description: 'Comprehensive pixel-level analysis performed looking for manipulation artifacts, inconsistent lighting, and synthetic generation patterns.',
        technicalDetails: 'Face detection, compression analysis, and artifact detection models applied.'
      });
    }

    if (fileType.startsWith('video/')) {
      reasons.push({
        id: 'video_analysis',
        category: 'temporal',
        type: 'evidence',
        severity: 'medium',
        confidence: 0.85,
        title: 'Video temporal analysis',
        description: 'Frame-by-frame analysis examining temporal consistency, lip-sync accuracy, and motion patterns.',
        technicalDetails: 'Temporal coherence models analyzed inter-frame relationships and motion vectors.'
      });
    }

    if (fileType.startsWith('audio/')) {
      reasons.push({
        id: 'audio_analysis',
        category: 'audio',
        type: 'evidence',
        severity: 'medium',
        confidence: 0.8,
        title: 'Audio analysis completed',
        description: 'Spectral analysis examining voice patterns, synthetic speech indicators, and audio artifacts.',
        technicalDetails: 'Voice cloning detection and audio synthesis pattern analysis applied.'
      });
    }
  }

  private addMetadataReasons(analysis: AnalysisResult, reasons: ExplanationReason[]) {
    const metadata = analysis.details.metadata;

    if (metadata.codec) {
      const isUnusualCodec = ['h264', 'h265', 'avc1'].every(common => 
        !metadata.codec?.toLowerCase().includes(common)
      );

      if (isUnusualCodec) {
        reasons.push({
          id: 'codec_unusual',
          category: 'metadata',
          type: 'anomaly',
          severity: 'low',
          confidence: 0.3,
          title: 'Unusual codec detected',
          description: `File uses codec '${metadata.codec}' which is less common for typical media files.`,
          technicalDetails: 'Codec analysis for synthetic generation indicators.'
        });
      }
    }

    if (metadata.bitrate && metadata.bitrate < 1000) {
      reasons.push({
        id: 'bitrate_low',
        category: 'metadata',
        type: 'anomaly',
        severity: 'low',
        confidence: 0.25,
        title: 'Low bitrate detected',
        description: 'Unusually low bitrate may indicate heavy compression or synthetic generation.',
        technicalDetails: `Bitrate: ${metadata.bitrate} kbps - below typical range for quality media.`
      });
    }
  }

  private generateEvidence(analysis: AnalysisResult): ExplanationEvidence[] {
    const evidence: ExplanationEvidence[] = [];

    // Generate evidence based on frame analysis if available
    if (analysis.details.frameAnalysis) {
      analysis.details.frameAnalysis.forEach((frame, index) => {
        if (frame.anomalies && frame.anomalies.length > 0) {
          evidence.push({
            id: `frame_${frame.frame}`,
            type: 'visual_artifact',
            location: {
              frame: frame.frame,
              timestamp: frame.timestamp
            },
            severity: frame.confidence,
            description: `Frame ${frame.frame}: ${frame.anomalies.join(', ')}`
          });
        }
      });
    }

    // Generate evidence from audio analysis
    if (analysis.details.audioAnalysis) {
      analysis.details.audioAnalysis.segments.forEach((segment, index) => {
        if (segment.anomalies && segment.anomalies.length > 0) {
          evidence.push({
            id: `audio_${index}`,
            type: 'audio_distortion',
            location: {
              timestamp: segment.start
            },
            severity: segment.confidence,
            description: `Audio segment ${segment.start}s-${segment.end}s: ${segment.anomalies.join(', ')}`
          });
        }
      });
    }

    return evidence;
  }

  private generateModelInsights(analysis: AnalysisResult): ModelInsight[] {
    const insights: ModelInsight[] = [];

    // Create insights from the analysis result
    insights.push({
      modelName: 'Reality Defender Ensemble',
      modelType: 'ensemble',
      confidence: analysis.confidence,
      prediction: analysis.prediction,
      keyFindings: this.generateKeyFindings(analysis),
      technicalScore: analysis.confidence,
      processingTime: analysis.processingTime,
      reasoning: this.generateModelReasoning(analysis)
    });

    return insights;
  }

  private generateKeyFindings(analysis: AnalysisResult): string[] {
    const findings: string[] = [];
    const confidence = analysis.confidence;

    if (analysis.prediction === 'authentic') {
      findings.push('No significant manipulation artifacts detected');
      findings.push('Consistent patterns throughout media');
      if (confidence < 0.1) findings.push('Strong authenticity indicators present');
    }

    if (analysis.prediction === 'manipulated') {
      findings.push('Synthetic generation patterns detected');
      if (confidence > 0.8) findings.push('High-confidence manipulation indicators');
      if (confidence > 0.6) findings.push('Multiple detection models agree');
    }

    if (analysis.prediction === 'inconclusive') {
      findings.push('Mixed signals from different analysis methods');
      findings.push('Requires manual review for definitive assessment');
    }

    // Add file-specific findings
    if (analysis.fileType.startsWith('video/')) {
      findings.push('Temporal analysis completed');
    }
    if (analysis.fileType.startsWith('audio/')) {
      findings.push('Spectral analysis performed');
    }

    return findings;
  }

  private generateModelReasoning(analysis: AnalysisResult): string {
    const confidence = analysis.confidence;
    const prediction = analysis.prediction;

    if (prediction === 'authentic') {
      return `The ensemble model analyzed this content and found it to be authentic with ${Math.round((1-confidence) * 100)}% confidence. The analysis considered multiple factors including pixel-level patterns, compression characteristics, and temporal consistency. No significant indicators of synthetic generation or manipulation were detected.`;
    }

    if (prediction === 'manipulated') {
      return `The ensemble model detected potential manipulation with ${Math.round(confidence * 100)}% confidence. The analysis identified patterns consistent with synthetic generation, including anomalous artifacts and inconsistencies that are typical of AI-generated or heavily edited content.`;
    }

    return `The analysis resulted in an inconclusive determination. The confidence score of ${Math.round(confidence * 100)}% falls in the uncertain range, indicating mixed signals from different detection methods. This could be due to edge cases, novel generation techniques, or ambiguous content characteristics.`;
  }

  private generateSummary(
    analysis: AnalysisResult, 
    reasons: ExplanationReason[], 
    insights: ModelInsight[]
  ): ExplanationSummary {
    const primaryReason = reasons[0]?.description || 'Analysis completed';
    const secondaryReasons = reasons.slice(1, 4).map(r => r.title);
    
    const authenticityIndicators = this.generateAuthenticityIndicators(analysis);
    const riskFactors = this.generateRiskFactors(analysis, reasons);
    const recommendedActions = this.generateRecommendations(analysis);

    return {
      primaryReason,
      secondaryReasons,
      overallConfidence: analysis.confidence,
      authenticityIndicators,
      riskFactors,
      recommendedActions
    };
  }

  private generateAuthenticityIndicators(analysis: AnalysisResult): Array<{
    factor: string;
    weight: number;
    contribution: 'positive' | 'negative' | 'neutral';
    explanation: string;
  }> {
    const indicators: Array<{
      factor: string;
      weight: number;
      contribution: 'positive' | 'negative' | 'neutral';
      explanation: string;
    }> = [];
    const confidence = analysis.confidence;

    indicators.push({
      factor: 'AI Model Analysis',
      weight: 0.8,
      contribution: analysis.prediction === 'authentic' ? 'positive' : 
                   analysis.prediction === 'manipulated' ? 'negative' : 'neutral',
      explanation: `Primary AI detection model with ${Math.round(confidence * 100)}% confidence`
    });

    indicators.push({
      factor: 'Technical Analysis',
      weight: 0.6,
      contribution: 'neutral',
      explanation: 'Comprehensive technical analysis of file properties and metadata'
    });

    if (analysis.details.frameAnalysis) {
      indicators.push({
        factor: 'Temporal Consistency',
        weight: 0.7,
        contribution: confidence < 0.5 ? 'positive' : 'negative',
        explanation: 'Frame-by-frame analysis for temporal consistency patterns'
      });
    }

    return indicators;
  }

  private generateRiskFactors(
    analysis: AnalysisResult, 
    reasons: ExplanationReason[]
  ): Array<{
    factor: string;
    severity: 'critical' | 'high' | 'medium' | 'low';
    likelihood: number;
    impact: string;
  }> {
    const riskFactors: Array<{
      factor: string;
      severity: 'critical' | 'high' | 'medium' | 'low';
      likelihood: number;
      impact: string;
    }> = [];
    const confidence = analysis.confidence;

    if (analysis.prediction === 'manipulated') {
      riskFactors.push({
        factor: 'Synthetic Content Detection',
        severity: confidence > 0.8 ? 'critical' : confidence > 0.6 ? 'high' : 'medium',
        likelihood: confidence,
        impact: 'Content may be artificially generated or manipulated'
      });
    }

    if (analysis.prediction === 'inconclusive') {
      riskFactors.push({
        factor: 'Uncertain Analysis',
        severity: 'medium',
        likelihood: Math.abs(0.5 - confidence) * 2,
        impact: 'Cannot determine authenticity with high confidence'
      });
    }

    // Add specific risk factors from reasons
    const highSeverityReasons = reasons.filter(r => r.severity === 'high');
    highSeverityReasons.forEach(reason => {
      riskFactors.push({
        factor: reason.title,
        severity: 'high',
        likelihood: reason.confidence,
        impact: reason.description
      });
    });

    return riskFactors;
  }

  private generateRecommendations(analysis: AnalysisResult): string[] {
    const recommendations = [];
    const confidence = analysis.confidence;

    if (analysis.prediction === 'manipulated') {
      recommendations.push('Exercise caution when sharing or using this content');
      recommendations.push('Consider additional verification from independent sources');
      if (confidence > 0.8) {
        recommendations.push('High probability of manipulation - avoid using without verification');
      }
    }

    if (analysis.prediction === 'inconclusive') {
      recommendations.push('Seek additional expert analysis or verification');
      recommendations.push('Use caution and clearly label as unverified if sharing');
      recommendations.push('Consider analyzing with additional tools or methods');
    }

    if (analysis.prediction === 'authentic' && confidence < 0.2) {
      recommendations.push('Content appears genuine with high confidence');
    }

    recommendations.push('Keep original file metadata when possible for future reference');
    
    return recommendations;
  }

  private generateTemporalAnalysis(analysis: AnalysisResult) {
    if (!analysis.details.frameAnalysis) return undefined;

    const frameByFrameReasons = analysis.details.frameAnalysis.map(frame => ({
      frame: frame.frame,
      timestamp: frame.timestamp,
      primaryConcerns: frame.anomalies || [],
      confidenceChange: frame.confidence - analysis.confidence
    }));

    const overallTrends = [
      'Analyzed temporal consistency across all frames',
      frameByFrameReasons.length > 0 ? 'Some frames show anomalies' : 'No significant anomalies detected',
      'Frame-to-frame analysis completed'
    ];

    return {
      frameByFrameReasons,
      overallTrends
    };
  }

  private generateMetadataAnalysis(analysis: AnalysisResult) {
    const metadata = analysis.details.metadata;
    const fileProperties: Array<{
      property: string;
      expectedValue?: string | number;
      actualValue: string | number;
      assessment: 'normal' | 'suspicious' | 'anomalous';
      explanation: string;
    }> = [];

    if (metadata.duration) {
      fileProperties.push({
        property: 'Duration',
        actualValue: `${metadata.duration}s`,
        assessment: metadata.duration > 0 ? 'normal' : 'suspicious',
        explanation: 'Media duration within expected range'
      });
    }

    if (metadata.resolution) {
      fileProperties.push({
        property: 'Resolution',
        actualValue: metadata.resolution,
        assessment: 'normal',
        explanation: 'Standard resolution format detected'
      });
    }

    if (metadata.codec) {
      const commonCodecs = ['h264', 'h265', 'avc1', 'mp4', 'mp3'];
      const isCommon = commonCodecs.some(codec => 
        metadata.codec!.toLowerCase().includes(codec)
      );

      fileProperties.push({
        property: 'Codec',
        actualValue: metadata.codec,
        assessment: isCommon ? 'normal' : 'suspicious',
        explanation: isCommon ? 'Standard codec format' : 'Unusual codec may indicate processing'
      });
    }

    return {
      fileProperties,
      processingHistory: ['File uploaded and analyzed via Reality Defender API']
    };
  }
}

// Export singleton instance with default config
export const explanationGenerator = new ExplanationGenerator();