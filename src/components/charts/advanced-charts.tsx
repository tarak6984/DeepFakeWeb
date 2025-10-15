'use client';

import React from 'react';
import { 
  RadarChart, 
  Radar, 
  PolarGrid, 
  PolarAngleAxis, 
  PolarRadiusAxis,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ComposedChart,
  Line,
  Area,
  Bar,
  Legend,
  LineChart,
  Treemap,
  Cell
} from 'recharts';
import { motion } from 'framer-motion';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';

// Risk Heatmap Component
export function RiskHeatmap({ data }: { data: any }) {
  const heatmapData = [
    { name: 'Facial Analysis', authentic: 85, manipulated: 15, risk: 'low' },
    { name: 'Voice Patterns', authentic: 70, manipulated: 30, risk: 'medium' },
    { name: 'Temporal Consistency', authentic: 90, manipulated: 10, risk: 'low' },
    { name: 'Metadata Analysis', authentic: 95, manipulated: 5, risk: 'low' },
    { name: 'Compression Artifacts', authentic: 80, manipulated: 20, risk: 'low' },
  ];

  const getRiskColor = (risk: string) => {
    switch (risk) {
      case 'low': return '#10b981';
      case 'medium': return '#f59e0b';
      case 'high': return '#ef4444';
      default: return '#6b7280';
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          Risk Analysis Heatmap
          <Badge variant="secondary">Multi-Layer Detection</Badge>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {heatmapData.map((item, index) => (
            <motion.div
              key={item.name}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.1 }}
              className="flex items-center justify-between p-3 rounded-lg border"
            >
              <div className="flex-1">
                <div className="font-medium text-sm">{item.name}</div>
                <div className="flex items-center gap-2 mt-1">
                  <div className="flex-1 bg-muted rounded-full h-2 overflow-hidden">
                    <div 
                      className="h-full transition-all duration-1000"
                      style={{ 
                        width: `${item.authentic}%`,
                        backgroundColor: getRiskColor(item.risk)
                      }}
                    />
                  </div>
                  <div className="text-xs text-muted-foreground min-w-[3rem]">
                    {item.authentic}%
                  </div>
                </div>
              </div>
              <Badge 
                variant={item.risk === 'low' ? 'default' : item.risk === 'medium' ? 'secondary' : 'destructive'}
                className="ml-2"
              >
                {item.risk.toUpperCase()}
              </Badge>
            </motion.div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

// Anomaly Detection Scatter Plot
export function AnomalyScatterPlot({ data }: { data: any }) {
  const scatterData = Array.from({ length: 50 }, (_, i) => ({
    x: Math.random() * 100,
    y: Math.random() * 100,
    confidence: Math.random(),
    anomaly: Math.random() > 0.7,
  }));

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-background border rounded-lg shadow-lg p-3">
          <p className="font-medium">Frame Analysis</p>
          <p className="text-sm">Position: ({data.x.toFixed(1)}, {data.y.toFixed(1)})</p>
          <p className="text-sm">Confidence: {(data.confidence * 100).toFixed(1)}%</p>
          <p className="text-sm">
            Status: <span className={data.anomaly ? 'text-red-500' : 'text-green-500'}>
              {data.anomaly ? 'Anomaly Detected' : 'Normal'}
            </span>
          </p>
        </div>
      );
    }
    return null;
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Anomaly Detection Analysis</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart data={scatterData}>
              <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
              <XAxis 
                type="number" 
                dataKey="x" 
                domain={[0, 100]}
                tick={{ fontSize: 12 }}
              />
              <YAxis 
                type="number" 
                dataKey="y" 
                domain={[0, 100]}
                tick={{ fontSize: 12 }}
              />
              <Tooltip content={<CustomTooltip />} />
              <Scatter 
                dataKey="confidence" 
                fill="#8884d8"
                shape={(props: any) => {
                  const { cx, cy, payload } = props;
                  return (
                    <circle 
                      cx={cx} 
                      cy={cy} 
                      r={payload.anomaly ? 6 : 3}
                      fill={payload.anomaly ? '#ef4444' : '#10b981'}
                      opacity={0.7}
                    />
                  );
                }}
              />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
        <div className="flex items-center gap-4 mt-2 text-xs text-muted-foreground">
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 rounded-full bg-green-500" />
            Normal Frames
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 rounded-full bg-red-500" />
            Anomalies Detected
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

// Radar Chart for Multi-Dimensional Analysis
export function MultidimensionalRadar({ data }: { data: any }) {
  const radarData = [
    {
      metric: 'Facial Consistency',
      score: 85,
      fullMark: 100,
    },
    {
      metric: 'Temporal Flow',
      score: 78,
      fullMark: 100,
    },
    {
      metric: 'Lighting Analysis',
      score: 92,
      fullMark: 100,
    },
    {
      metric: 'Compression Patterns',
      score: 88,
      fullMark: 100,
    },
    {
      metric: 'Metadata Integrity',
      score: 95,
      fullMark: 100,
    },
    {
      metric: 'Pixel Artifacts',
      score: 76,
      fullMark: 100,
    },
  ];

  return (
    <Card>
      <CardHeader>
        <CardTitle>Multi-Dimensional Analysis</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <RadarChart data={radarData}>
              <PolarGrid />
              <PolarAngleAxis 
                dataKey="metric" 
                tick={{ fontSize: 10 }}
              />
              <PolarRadiusAxis 
                angle={90} 
                domain={[0, 100]}
                tick={{ fontSize: 10 }}
              />
              <Radar
                name="Analysis Score"
                dataKey="score"
                stroke="#8884d8"
                fill="#8884d8"
                fillOpacity={0.3}
                strokeWidth={2}
              />
            </RadarChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
}

// Confidence Timeline with Anomaly Markers
export function ConfidenceTimelineAdvanced({ data }: { data: any }) {
  const timelineData = Array.from({ length: 30 }, (_, i) => ({
    frame: i + 1,
    confidence: 65 + Math.sin(i / 5) * 15 + (Math.random() - 0.5) * 10,
    anomaly: Math.random() > 0.8 ? 100 : null,
    threshold: 95,
  }));

  return (
    <Card>
      <CardHeader>
        <CardTitle>Confidence Timeline Analysis</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={timelineData}>
              <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
              <XAxis 
                dataKey="frame" 
                tick={{ fontSize: 12 }}
              />
              <YAxis 
                domain={[0, 100]}
                tick={{ fontSize: 12 }}
              />
              <Tooltip 
                content={({ active, payload, label }) => {
                  if (active && payload && payload.length) {
                    return (
                      <div className="bg-background border rounded-lg shadow-lg p-3">
                        <p className="font-medium">Frame {label}</p>
                        <p className="text-sm">
                          Confidence: {payload[0]?.value?.toFixed(1)}%
                        </p>
                        {payload[1]?.value && (
                          <p className="text-sm text-red-500">⚠️ Anomaly Detected</p>
                        )}
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
                fillOpacity={0.3}
                strokeWidth={2}
              />
              <Line
                type="monotone"
                dataKey="threshold"
                stroke="#f59e0b"
                strokeDasharray="5 5"
                dot={false}
              />
              <Scatter
                dataKey="anomaly"
                fill="#ef4444"
                shape="diamond"
              />
              <Legend />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
}

// Comparative Analysis Chart
export function ComparativeAnalysis({ analyses }: { analyses: any[] }) {
  const comparisonData = analyses.slice(0, 5).map((analysis, index) => ({
    name: `Analysis ${index + 1}`,
    confidence: analysis.confidence * 100,
    authentic: (1 - analysis.confidence) * 100,
    processing_time: analysis.processingTime / 1000,
  }));

  return (
    <Card>
      <CardHeader>
        <CardTitle>Comparative Analysis</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={comparisonData}>
              <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar dataKey="confidence" fill="#ef4444" name="Manipulation Risk %" />
              <Bar dataKey="authentic" fill="#10b981" name="Authenticity %" />
              <Line 
                type="monotone" 
                dataKey="processing_time" 
                stroke="#f59e0b" 
                name="Processing Time (s)"
                yAxisId="right"
              />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
}

// File Type Usage Treemap
export function FileTypeTreemap({ usageData }: { usageData: Record<string, number> }) {
  const treemapData = Object.entries(usageData).map(([type, count]) => ({
    name: type.charAt(0).toUpperCase() + type.slice(1),
    value: count,
    percentage: (count / Object.values(usageData).reduce((a, b) => a + b, 0) * 100).toFixed(1),
  }));

  const COLORS = ['#8884d8', '#82ca9d', '#ffc658', '#ff7300', '#8dd1e1'];

  return (
    <Card>
      <CardHeader>
        <CardTitle>File Type Analysis Distribution</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <Treemap
              data={treemapData}
              dataKey="value"
              stroke="#fff"
              content={(props: any) => {
                const { x, y, width, height, index, name, value, percentage } = props;
                return (
                  <g>
                    <rect
                      x={x}
                      y={y}
                      width={width}
                      height={height}
                      fill={COLORS[index % COLORS.length]}
                      opacity={0.8}
                    />
                    {width > 60 && height > 30 && (
                      <>
                        <text
                          x={x + width / 2}
                          y={y + height / 2 - 6}
                          textAnchor="middle"
                          fill="#fff"
                          fontSize="12"
                          fontWeight="bold"
                        >
                          {name}
                        </text>
                        <text
                          x={x + width / 2}
                          y={y + height / 2 + 6}
                          textAnchor="middle"
                          fill="#fff"
                          fontSize="10"
                        >
                          {value} ({percentage}%)
                        </text>
                      </>
                    )}
                  </g>
                );
              }}
            />
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
}
