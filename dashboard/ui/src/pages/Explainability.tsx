import React from 'react';
import { useCyberCastAPI } from '../hooks/useCyberCastAPI';
import { Card, CardHeader, CardTitle, CardContent } from '../components/ui/card';
import { FileSearch, ArrowUpRight, ArrowDownRight, Clock } from 'lucide-react';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid, Cell } from 'recharts';

export default function Explainability() {
  const { forecasts, loading, error } = useCyberCastAPI();

  if (loading) return <div className="p-8 text-center text-muted-foreground">Loading explainability data...</div>;
  if (error) return <div className="p-8 text-center text-destructive">Error loading data: {error}</div>;
  if (!forecasts || forecasts.length === 0) return <div className="p-8 text-center text-muted-foreground">No data available.</div>;

  const latestForecast = forecasts[forecasts.length - 1];
  const { top_features, temporal_evidence } = latestForecast;

  // Formatting chart data for Recharts
  const featureChartData = top_features.map((f: any) => ({
    name: f.feature,
    importance: f.importance,
    direction: f.direction
  })).sort((a: any, b: any) => Math.abs(b.importance) - Math.abs(a.importance));

  const temporalChartData = temporal_evidence.map((t: any, idx: number) => ({
    name: `t${t.window_offset}`,
    importance: t.importance,
    time: t.window_offset
  }));

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-card border border-border p-3 shadow-lg rounded-md text-sm">
          <p className="font-semibold">{payload[0].payload.name}</p>
          <p className="text-muted-foreground">Importance: {payload[0].value.toFixed(5)}</p>
          {payload[0].payload.direction && (
            <p className={payload[0].payload.direction === 'increases_risk' ? 'text-destructive' : 'text-emerald-500'}>
              {payload[0].payload.direction === 'increases_risk' ? 'Increases Risk' : 'Decreases Risk'}
            </p>
          )}
        </div>
      );
    }
    return null;
  };

  return (
    <div className="flex-1 space-y-4 p-8 pt-6">
      <div className="flex items-center justify-between space-y-2 mb-8">
        <div>
          <h2 className="text-3xl font-bold tracking-tight">Explainability & Attribution</h2>
          <p className="text-muted-foreground mt-1 flex items-center">
            <FileSearch className="w-4 h-4 mr-1 text-emerald-500" />
            Local Perturbation Analysis
          </p>
        </div>
      </div>

      <div className="rounded-md bg-secondary/50 p-4 border border-border mb-6">
        <p className="text-sm text-muted-foreground leading-relaxed">
          <strong className="text-foreground">Scientific Disclaimer:</strong> This module uses perturbation-based attribution to calculate model sensitivity to specific features and temporal windows. <strong>This analysis does not establish definitive causal relationships</strong>, but highlights which inputs most heavily influence the current probability score.
        </p>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Top Feature Contributors</CardTitle>
          </CardHeader>
          <CardContent>
            {featureChartData.length > 0 ? (
              <div className="h-[400px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={featureChartData} layout="vertical" margin={{ top: 5, right: 30, left: 100, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#333" horizontal={true} vertical={false} />
                    <XAxis type="number" stroke="#888888" fontSize={12} tickFormatter={(val) => val.toFixed(3)} />
                    <YAxis type="category" dataKey="name" stroke="#888888" fontSize={11} width={90} tickLine={false} axisLine={false} />
                    <Tooltip content={<CustomTooltip />} />
                    <Bar dataKey="importance" radius={[0, 4, 4, 0]}>
                      {featureChartData.map((entry: any, index: number) => (
                        <Cell key={`cell-${index}`} fill={entry.direction === 'increases_risk' ? '#dc2626' : '#10b981'} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            ) : (
              <p className="text-muted-foreground">Insufficient features extracted for attribution.</p>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Temporal Window Importance</CardTitle>
          </CardHeader>
          <CardContent>
            {temporalChartData.length > 0 ? (
              <div className="h-[400px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={temporalChartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#333" vertical={false} />
                    <XAxis dataKey="name" stroke="#888888" fontSize={12} tickLine={false} axisLine={false} />
                    <YAxis stroke="#888888" fontSize={12} tickLine={false} axisLine={false} tickFormatter={(val) => val.toFixed(3)} />
                    <Tooltip content={<CustomTooltip />} />
                    <Bar dataKey="importance" radius={[4, 4, 0, 0]}>
                       {temporalChartData.map((entry: any, index: number) => (
                        <Cell key={`cell-${index}`} fill={entry.importance > 0 ? '#c084fc' : '#aa3bff'} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            ) : (
              <p className="text-muted-foreground">Insufficient temporal extraction for attribution.</p>
            )}
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-4 mt-4 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle className="text-destructive flex items-center">
              <ArrowUpRight className="mr-2 h-5 w-5" /> Positive Contributors (Increases Risk)
            </CardTitle>
          </CardHeader>
          <CardContent>
             <ul className="space-y-2">
                {featureChartData.filter((f: any) => f.direction === 'increases_risk').map((f: any, i: number) => (
                  <li key={i} className="flex justify-between items-center text-sm border-b border-border pb-2 last:border-0">
                    <span className="font-mono">{f.name}</span>
                    <span className="text-muted-foreground">+{f.importance.toFixed(5)}</span>
                  </li>
                ))}
                {featureChartData.filter((f: any) => f.direction === 'increases_risk').length === 0 && (
                  <li className="text-sm text-muted-foreground italic">None observed</li>
                )}
             </ul>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-emerald-500 flex items-center">
              <ArrowDownRight className="mr-2 h-5 w-5" /> Negative Contributors (Decreases Risk)
            </CardTitle>
          </CardHeader>
          <CardContent>
             <ul className="space-y-2">
                {featureChartData.filter((f: any) => f.direction === 'decreases_risk').map((f: any, i: number) => (
                  <li key={i} className="flex justify-between items-center text-sm border-b border-border pb-2 last:border-0">
                    <span className="font-mono">{f.name}</span>
                    <span className="text-muted-foreground">{f.importance.toFixed(5)}</span>
                  </li>
                ))}
                {featureChartData.filter((f: any) => f.direction === 'decreases_risk').length === 0 && (
                  <li className="text-sm text-muted-foreground italic">None observed</li>
                )}
             </ul>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
