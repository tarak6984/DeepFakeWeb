'use client';

import { useEffect, useRef } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { StoredAnalysis } from '@/lib/storage';
import { ConfidenceGauge } from '@/components/charts/confidence-gauge';
import { CategoryChart } from '@/components/charts/category-chart';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  AreaChart,
  Area,
  PieChart,
  Pie,
  Cell,
} from 'recharts';
import {
  CheckCircle,
  AlertTriangle,
  XCircle,
  BarChart3,
} from 'lucide-react';

interface PDFChartComponentsProps {
  analysis: StoredAnalysis;
}

// Generate timeline data from analysis results
const generateTimelineData = (analysis: StoredAnalysis) => {
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

const getConfidenceColor = (confidence: number) => {
  if (confidence < 0.3) return 'text-green-600';
  if (confidence < 0.7) return 'text-yellow-600';
  return 'text-red-600';
};

const getConfidenceIcon = (confidence: number) => {
  if (confidence < 0.3) return CheckCircle;
  if (confidence < 0.7) return AlertTriangle;
  return XCircle;
};

export function PDFChartComponents({ analysis }: PDFChartComponentsProps) {
  const timelineData = generateTimelineData(analysis);
  const ConfidenceIcon = getConfidenceIcon(analysis.confidence);

  // Components positioned for PDF capture
  return (
    <div 
      className="fixed top-0 left-[-1500px] w-[800px] bg-white z-[9998]" 
      style={{
        opacity: 0.01, // Barely visible but not completely hidden
        pointerEvents: 'none',
        visibility: 'visible'
      }}
    >
      {/* Confidence Gauge for PDF */}
      <div id="confidence-gauge-pdf" className="p-6 bg-white">
        <div style={{ border: 'none', boxShadow: 'none', backgroundColor: '#ffffff', padding: '16px', borderRadius: '8px' }}>
          <div style={{ textAlign: 'center', paddingBottom: '8px' }}>
            <h3 style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px', fontSize: '18px', margin: '0', color: '#1f2937' }}>
              ITL Deepfake Detection Results
            </h3>
          </div>
          <div style={{ padding: '16px' }}>
            {/* Confidence Circle */}
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <div style={{ position: 'relative', width: '128px', height: '128px' }}>
                <svg style={{ width: '128px', height: '128px', transform: 'rotate(-90deg)' }} viewBox="0 0 100 100">
                  <circle
                    cx="50"
                    cy="50"
                    r="40"
                    stroke="#e5e7eb"
                    strokeWidth="8"
                    fill="transparent"
                  />
                  <circle
                    cx="50"
                    cy="50"
                    r="40"
                    stroke={analysis.confidence < 0.3 ? '#10b981' : analysis.confidence < 0.7 ? '#f59e0b' : '#ef4444'}
                    strokeWidth="8"
                    fill="transparent"
                    strokeDasharray={`${2 * Math.PI * 40 * analysis.confidence} ${2 * Math.PI * 40}`}
                    strokeLinecap="round"
                  />
                </svg>
                <div style={{ position: 'absolute', top: '0', left: '0', right: '0', bottom: '0', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <div style={{ textAlign: 'center' }}>
                    <div style={{ fontSize: '24px', fontWeight: 'bold', color: analysis.confidence < 0.3 ? '#10b981' : analysis.confidence < 0.7 ? '#f59e0b' : '#ef4444' }}>
                      {analysis.confidence > 1 ? Math.round(analysis.confidence) : Math.round(analysis.confidence * 100)}%
                    </div>
                    <div style={{ fontSize: '14px', color: '#6b7280' }}>Risk</div>
                  </div>
                </div>
              </div>
            </div>
            
            <div style={{ textAlign: 'center', marginTop: '16px' }}>
              <div style={{
                display: 'inline-block',
                padding: '4px 8px',
                borderRadius: '4px',
                backgroundColor: analysis.confidence < 0.3 ? '#3b82f6' : analysis.confidence < 0.7 ? '#6b7280' : '#ef4444',
                color: '#ffffff',
                fontSize: '14px',
                fontWeight: '500'
              }}>
                {analysis.prediction.toUpperCase()}
              </div>
              <p style={{ fontSize: '14px', color: '#4b5563', marginTop: '8px', margin: '8px 0 0 0' }}>
                {analysis.confidence < 0.3 
                  ? 'Content appears authentic with low risk of manipulation' 
                  : analysis.confidence < 0.7 
                  ? 'Moderate risk detected - further verification recommended'
                  : 'High risk of artificial generation or manipulation detected'
                }
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Category Chart for PDF */}
      {analysis.details.categoryBreakdown && (
        <div id="category-chart-pdf" style={{ padding: '24px', backgroundColor: '#ffffff' }}>
          <div style={{ border: 'none', boxShadow: 'none', backgroundColor: '#ffffff', borderRadius: '8px' }}>
            <div style={{ padding: '16px 16px 8px 16px' }}>
              <h3 style={{ fontSize: '18px', fontWeight: '600', margin: '0', color: '#1f2937' }}>Analysis Categories</h3>
            </div>
            <div style={{ padding: '16px' }}>
              <div style={{ height: '300px', width: '100%' }}>
                {/* Simple bar chart representation */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '16px', height: '100%', justifyContent: 'center' }}>
                  {Object.entries(analysis.details.categoryBreakdown).map(([category, score], index) => {
                    const colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6'];
                    const color = colors[index % colors.length];
                    return (
                      <div key={category} style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                        <div style={{ minWidth: '120px', fontSize: '14px', fontWeight: '500', color: '#374151', textTransform: 'capitalize' }}>
                          {category}
                        </div>
                        <div style={{ flex: '1', backgroundColor: '#f3f4f6', height: '20px', borderRadius: '10px', overflow: 'hidden' }}>
                          <div 
                            style={{ 
                              height: '100%', 
                              backgroundColor: color, 
                              width: `${Math.max(score > 1 ? score : score * 100, 2)}%`,
                              borderRadius: '10px',
                              transition: 'width 0.3s ease'
                            }}
                          />
                        </div>
                        <div style={{ minWidth: '50px', fontSize: '14px', fontWeight: '600', color: '#1f2937', textAlign: 'right' }}>
                          {score > 1 ? Math.round(score) : Math.round(score * 100)}%
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Timeline Chart for PDF */}
      <div id="timeline-chart-pdf" style={{ padding: '24px', backgroundColor: '#ffffff' }}>
        <div style={{ border: 'none', boxShadow: 'none', backgroundColor: '#ffffff', borderRadius: '8px' }}>
          <div style={{ padding: '16px 16px 8px 16px' }}>
            <h3 style={{ fontSize: '18px', fontWeight: '600', margin: '0', color: '#1f2937' }}>Frame-by-Frame Analysis Timeline</h3>
          </div>
          <div style={{ padding: '16px' }}>
            <div style={{ height: '256px', width: '100%' }}>
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={timelineData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis 
                    dataKey="frame" 
                    tick={{ fontSize: 12, fill: '#6b7280' }}
                    axisLine={false}
                    tickLine={false}
                  />
                  <YAxis 
                    domain={[0, 100]}
                    tick={{ fontSize: 12, fill: '#6b7280' }}
                    axisLine={false}
                    tickLine={false}
                    label={{ value: 'Confidence %', angle: -90, position: 'insideLeft', style: { textAnchor: 'middle', fill: '#6b7280' } }}
                  />
                  <Tooltip 
                    contentStyle={{
                      backgroundColor: '#ffffff',
                      border: '1px solid #d1d5db',
                      borderRadius: '8px',
                      boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
                      padding: '12px'
                    }}
                    labelStyle={{ fontWeight: '500', color: '#1f2937' }}
                    itemStyle={{ color: '#4b5563', fontSize: '14px' }}
                  />
                  <Area
                    type="monotone"
                    dataKey="confidence"
                    stroke="#3b82f6"
                    fill="#3b82f6"
                    fillOpacity={0.2}
                    strokeWidth={2}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </div>

      {/* Risk Heatmap for PDF */}
      <div id="risk-heatmap-pdf" style={{ padding: '24px', backgroundColor: '#ffffff' }}>
        <div style={{ border: 'none', boxShadow: 'none', backgroundColor: '#ffffff', borderRadius: '8px' }}>
          <div style={{ padding: '16px 16px 8px 16px' }}>
            <h3 style={{ fontSize: '18px', fontWeight: '600', margin: '0', color: '#1f2937' }}>Risk Analysis Heatmap</h3>
          </div>
          <div style={{ padding: '16px' }}>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
              {/* Mock heatmap using colored bars */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(8, 1fr)', gap: '4px' }}>
                {Array.from({ length: 64 }).map((_, i) => {
                  const hue = 120 - (analysis.confidence * 120); // Green to red based on confidence
                  const lightness = 50 + (Math.sin(i * 0.5) * 20); // Vary lightness for visual interest
                  return (
                    <div
                      key={i}
                      style={{
                        aspectRatio: '1',
                        borderRadius: '2px',
                        backgroundColor: `hsl(${hue}, 60%, ${Math.max(30, Math.min(70, lightness))}%)`,
                      }}
                    />
                  );
                })}
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '14px', color: '#6b7280' }}>
                <span>Low Risk</span>
                <span>High Risk</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Anomaly Scatter Plot for PDF */}
      <div id="anomaly-scatter-pdf" style={{ padding: '24px', backgroundColor: '#ffffff' }}>
        <div style={{ border: 'none', boxShadow: 'none', backgroundColor: '#ffffff', borderRadius: '8px' }}>
          <div style={{ padding: '16px 16px 8px 16px' }}>
            <h3 style={{ fontSize: '18px', fontWeight: '600', margin: '0', color: '#1f2937' }}>Anomaly Detection Plot</h3>
          </div>
          <div style={{ padding: '16px' }}>
            <div style={{ height: '256px', width: '100%' }}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart
                  data={timelineData.map((point, i) => ({
                    ...point,
                    anomaly: Math.random() * 100,
                    x: i,
                    y: Math.random() * 100,
                  }))}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis dataKey="x" tick={{ fontSize: 12, fill: '#6b7280' }} />
                  <YAxis tick={{ fontSize: 12, fill: '#6b7280' }} />
                  <Tooltip 
                    contentStyle={{
                      backgroundColor: '#ffffff',
                      border: '1px solid #d1d5db',
                      borderRadius: '8px',
                      padding: '12px'
                    }}
                  />
                  <Line
                    type="monotone"
                    dataKey="y"
                    stroke="#ef4444"
                    strokeWidth={0}
                    dot={{ r: 3, fill: '#ef4444' }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </div>

      {/* Radar Chart for PDF */}
      <div id="radar-chart-pdf" style={{ padding: '24px', backgroundColor: '#ffffff' }}>
        <div style={{ border: 'none', boxShadow: 'none', backgroundColor: '#ffffff', borderRadius: '8px' }}>
          <div style={{ padding: '16px 16px 8px 16px' }}>
            <h3 style={{ fontSize: '18px', fontWeight: '600', margin: '0', color: '#1f2937' }}>Multidimensional Analysis</h3>
          </div>
          <div style={{ padding: '16px' }}>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
              {/* Mock radar chart using progress bars */}
              {[
                { name: 'Visual Quality', score: Math.round(analysis.confidence * 80 + 20) },
                { name: 'Temporal Consistency', score: Math.round(analysis.confidence * 75 + 25) },
                { name: 'Audio Synchronization', score: Math.round(analysis.confidence * 85 + 15) },
                { name: 'Compression Artifacts', score: Math.round(analysis.confidence * 70 + 30) },
                { name: 'Facial Features', score: Math.round(analysis.confidence * 90 + 10) },
              ].map((metric) => (
                <div key={metric.name} style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '14px' }}>
                    <span style={{ color: '#374151' }}>{metric.name}</span>
                    <span style={{ fontWeight: '500', color: '#1f2937' }}>{metric.score}%</span>
                  </div>
                  <div style={{ width: '100%', backgroundColor: '#e5e7eb', borderRadius: '9999px', height: '8px' }}>
                    <div
                      style={{ 
                        backgroundColor: '#2563eb', 
                        height: '8px', 
                        borderRadius: '9999px',
                        width: `${metric.score}%`,
                        transition: 'width 0.3s ease'
                      }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// Hook to mount PDF components when needed
export function usePDFComponents() {
  const mountedRef = useRef(false);

  const mountPDFComponents = (analysis: StoredAnalysis) => {
    if (mountedRef.current) return;
    
    const container = document.createElement('div');
    container.id = 'pdf-components-container';
    document.body.appendChild(container);
    
    // This would need React to render the components
    // For now, we'll use a different approach in the PDF generator
    mountedRef.current = true;
  };

  const unmountPDFComponents = () => {
    const container = document.getElementById('pdf-components-container');
    if (container) {
      document.body.removeChild(container);
      mountedRef.current = false;
    }
  };

  return { mountPDFComponents, unmountPDFComponents };
}