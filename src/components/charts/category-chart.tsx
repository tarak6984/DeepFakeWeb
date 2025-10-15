'use client';

import { BarChart, Bar, XAxis, YAxis, CartesianGrid, ResponsiveContainer, PieChart, Pie, Cell, Tooltip } from 'recharts';
import { motion } from 'framer-motion';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
// Toggle group removed - using buttons instead
import { BarChart3, PieChart as PieChartIcon } from 'lucide-react';
import { useState } from 'react';

interface CategoryData {
  authentic: number;
  manipulated: number;
  inconclusive: number;
}

interface CategoryChartProps {
  data: CategoryData;
  className?: string;
}

const COLORS = {
  authentic: '#10b981',
  manipulated: '#ef4444',
  inconclusive: '#f59e0b',
};

export function CategoryChart({ data, className }: CategoryChartProps) {
  const [viewType, setViewType] = useState<'pie' | 'bar'>('pie');

  // Transform data for charts - handle both 0-1 and 0-100 ranges safely
  const normalizeValue = (value: number) => value > 1 ? Math.round(value) : Math.round(value * 100);
  
  const chartData = [
    {
      name: 'Authentic',
      value: normalizeValue(data.authentic),
      percentage: `${normalizeValue(data.authentic)}%`,
      color: COLORS.authentic,
    },
    {
      name: 'Manipulated',
      value: normalizeValue(data.manipulated),
      percentage: `${normalizeValue(data.manipulated)}%`,
      color: COLORS.manipulated,
    },
    {
      name: 'Inconclusive',
      value: normalizeValue(data.inconclusive),
      percentage: `${normalizeValue(data.inconclusive)}%`,
      color: COLORS.inconclusive,
    },
  ];

  const CustomTooltip = ({ active, payload }: { active?: boolean; payload?: Array<{ name: string; value: number; color: string }> }) => {
    if (active && payload && payload.length) {
      const data = payload[0];
      return (
        <div className="bg-background border border-border rounded-lg shadow-lg p-3">
          <p className="font-medium">{data.name}</p>
          <p className="text-sm text-muted-foreground">
            <span style={{ color: data.color }}>●</span> {data.value}%
          </p>
        </div>
      );
    }
    return null;
  };

  const renderCustomizedLabel = (entry: any) => {
    return `${entry.percent?.toFixed(0) || 0}%`;
  };

  return (
    <Card className={className}>
      <CardHeader className="pb-4">
        <div className="flex items-center justify-between">
          <CardTitle>Category Breakdown</CardTitle>
          <div className="flex items-center gap-2">
            <Button
              variant={viewType === 'pie' ? 'default' : 'outline'}
              size="sm"
              onClick={() => setViewType('pie')}
              className="h-8 w-8 p-0"
            >
              <PieChartIcon className="h-4 w-4" />
            </Button>
            <Button
              variant={viewType === 'bar' ? 'default' : 'outline'}
              size="sm"
              onClick={() => setViewType('bar')}
              className="h-8 w-8 p-0"
            >
              <BarChart3 className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <motion.div
          key={viewType}
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.3 }}
        >
          {viewType === 'pie' ? (
            <div className="space-y-6">
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={chartData}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      label={renderCustomizedLabel}
                      outerRadius={80}
                      fill="#8884d8"
                      dataKey="value"
                      animationBegin={0}
                      animationDuration={1000}
                    >
                      {chartData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip content={<CustomTooltip />} />
                  </PieChart>
                </ResponsiveContainer>
              </div>

              {/* Legend */}
              <div className="grid grid-cols-1 gap-3">
                {chartData.map((item, index) => (
                  <motion.div
                    key={item.name}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.1 + 0.5 }}
                    className="flex items-center justify-between p-3 bg-muted/30 rounded-lg"
                  >
                    <div className="flex items-center gap-3">
                      <div
                        className="w-4 h-4 rounded-full"
                        style={{ backgroundColor: item.color }}
                      />
                      <span className="font-medium text-sm">{item.name}</span>
                    </div>
                    <Badge variant="secondary" className="font-semibold">
                      {item.percentage}
                    </Badge>
                  </motion.div>
                ))}
              </div>
            </div>
          ) : (
            <div className="space-y-6">
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
                    <XAxis 
                      dataKey="name" 
                      tick={{ fontSize: 12 }}
                      axisLine={false}
                      tickLine={false}
                    />
                    <YAxis 
                      tick={{ fontSize: 12 }}
                      axisLine={false}
                      tickLine={false}
                      domain={[0, 100]}
                    />
                    <Tooltip content={<CustomTooltip />} />
                    <Bar 
                      dataKey="value" 
                      fill="#8884d8"
                      radius={[4, 4, 0, 0]}
                      animationBegin={0}
                      animationDuration={1000}
                    >
                      {chartData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>

              {/* Statistics */}
              <div className="grid grid-cols-3 gap-4">
                {chartData.map((item, index) => (
                  <motion.div
                    key={item.name}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: index * 0.1 + 0.5 }}
                    className="text-center p-4 border rounded-lg"
                  >
                    <div
                      className="text-2xl font-bold mb-1"
                      style={{ color: item.color }}
                    >
                      {item.percentage}
                    </div>
                    <div className="text-sm text-muted-foreground font-medium">
                      {item.name}
                    </div>
                  </motion.div>
                ))}
              </div>
            </div>
          )}
        </motion.div>

        {/* Summary */}
        <div className="mt-6 p-4 bg-muted/50 rounded-lg">
          <h4 className="font-medium mb-2">Analysis Summary</h4>
          <p className="text-sm text-muted-foreground">
            {data.authentic > data.manipulated && data.authentic > data.inconclusive && 
              'The analysis indicates this file is likely authentic with low signs of manipulation.'}
            {data.manipulated > data.authentic && data.manipulated > data.inconclusive && 
              'The analysis suggests this file may be manipulated or artificially generated.'}
            {data.inconclusive >= data.authentic && data.inconclusive >= data.manipulated && 
              'The analysis results are inconclusive, requiring further investigation.'}
          </p>
        </div>
      </CardContent>
    </Card>
  );
}