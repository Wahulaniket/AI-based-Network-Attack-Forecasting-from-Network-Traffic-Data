import React, { useState } from 'react';
import { useCyberCastAPI } from '../hooks/useCyberCastAPI';
import { Card, CardHeader, CardTitle, CardContent } from '../components/ui/card';
import { Badge } from '../components/ui/badge';
import { Clock, ShieldAlert, CheckCircle2 } from 'lucide-react';
import { cn } from '../lib/utils';

export default function AttackTimeline() {
  const { forecasts, modelInfo, loading, error } = useCyberCastAPI();
  const [selectedPoint, setSelectedPoint] = useState<number | null>(null);

  if (loading) return <div className="p-8 text-center text-muted-foreground">Loading timeline...</div>;
  if (error) return <div className="p-8 text-center text-destructive">Error loading timeline: {error}</div>;
  if (!forecasts || forecasts.length === 0) return <div className="p-8 text-center text-muted-foreground">No timeline data available.</div>;

  const threshold = modelInfo?.threshold || 0.98;
  
  // Reversing so newest is at the top
  const timelineData = [...forecasts].reverse();

  return (
    <div className="flex-1 space-y-4 p-8 pt-6 max-w-7xl mx-auto w-full">
      <div className="flex items-center justify-between space-y-2 mb-8">
        <div>
          <h2 className="text-3xl font-bold tracking-tight">Attack Timeline</h2>
          <p className="text-muted-foreground mt-1 flex items-center">
            <Clock className="w-4 h-4 mr-1 text-emerald-500" />
            Chronological Sequence of Events
          </p>
        </div>
      </div>

      <div className="grid md:grid-cols-3 gap-8">
        <div className="md:col-span-1 border-r border-border pr-4 h-[70vh] overflow-y-auto">
          <div className="space-y-4">
            {timelineData.map((item, index) => {
              const isHighRisk = item.attack_probability >= threshold;
              const isSelected = selectedPoint === index;

              return (
                <div 
                  key={index} 
                  className={cn(
                    "p-4 rounded-lg border cursor-pointer transition-colors relative group",
                    isSelected ? "bg-secondary border-primary" : "bg-card border-border hover:bg-secondary/50",
                    isHighRisk && !isSelected ? "border-destructive/30" : ""
                  )}
                  onClick={() => setSelectedPoint(index)}
                >
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-xs font-mono text-muted-foreground">{item.timestamp.split('T').join(' ')}</span>
                    {isHighRisk ? <ShieldAlert className="w-4 h-4 text-destructive" /> : <CheckCircle2 className="w-4 h-4 text-emerald-500" />}
                  </div>
                  <div className="font-semibold text-sm">
                    Prob: {(item.attack_probability * 100).toFixed(2)}%
                  </div>
                  <div className="text-xs text-muted-foreground mt-1 truncate">
                    {item.current_stage !== "UNKNOWN / INSUFFICIENT EVIDENCE" ? item.current_stage : 'No heuristic stage'}
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        <div className="md:col-span-2">
          {selectedPoint !== null ? (
            <Card className="sticky top-6">
              <CardHeader>
                <CardTitle className="flex justify-between items-center">
                  <span>Timeline Inspection</span>
                  <span className="font-mono text-sm text-muted-foreground font-normal">
                    {timelineData[selectedPoint].timestamp.replace('T', ' ')}
                  </span>
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                {timelineData[selectedPoint].attack_probability >= threshold ? (
                   <div className="p-4 bg-destructive/10 border border-destructive/20 rounded-md text-destructive">
                      <div className="font-bold text-lg mb-1">ELEVATED THREAT DETECTED</div>
                      <div className="text-sm">Model probability {(timelineData[selectedPoint].attack_probability * 100).toFixed(2)}% exceeds threshold {(threshold * 100).toFixed(2)}%</div>
                   </div>
                ) : (
                   <div className="p-4 bg-emerald-500/10 border border-emerald-500/20 rounded-md text-emerald-500">
                      <div className="font-bold text-lg mb-1">INSUFFICIENT MODEL-LEVEL ATTACK RISK</div>
                      <div className="text-sm">Model probability {(timelineData[selectedPoint].attack_probability * 100).toFixed(2)}% is below threshold {(threshold * 100).toFixed(2)}%</div>
                   </div>
                )}

                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-1 border border-border p-4 rounded-md">
                    <p className="text-xs text-muted-foreground uppercase font-semibold">Potential Traffic Pattern</p>
                    <p className="font-medium text-lg">{timelineData[selectedPoint].current_stage}</p>
                    {timelineData[selectedPoint].current_stage !== "UNKNOWN / INSUFFICIENT EVIDENCE" && (
                      <Badge variant="outline" className="mt-1">
                        Confidence: {timelineData[selectedPoint].stage_confidence.toUpperCase()}
                      </Badge>
                    )}
                  </div>
                  <div className="space-y-1 border border-border p-4 rounded-md">
                    <p className="text-xs text-muted-foreground uppercase font-semibold">Heuristic Forecast</p>
                    <p className="font-medium text-lg">{timelineData[selectedPoint].forecasted_next_stage || 'None'}</p>
                    {timelineData[selectedPoint].forecasted_next_stage && (
                      <Badge variant="outline" className="mt-1">
                        Confidence: {timelineData[selectedPoint].forecast_confidence.toUpperCase()}
                      </Badge>
                    )}
                  </div>
                </div>

                <div>
                  <h4 className="text-sm font-semibold mb-3 border-b border-border pb-2">Heuristic Stage Evidence</h4>
                  {timelineData[selectedPoint].stage_evidence && timelineData[selectedPoint].stage_evidence.length > 0 ? (
                    <ul className="space-y-3">
                      {timelineData[selectedPoint].stage_evidence.map((ev: any, idx: number) => (
                        <li key={idx} className="bg-secondary/30 p-3 rounded border border-border">
                          <div className="font-mono text-sm text-primary mb-1">{ev.feature}</div>
                          <div className="text-sm text-muted-foreground">{ev.reason}</div>
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <p className="text-sm text-muted-foreground italic">No heuristic signature detected in this window.</p>
                  )}
                </div>

              </CardContent>
            </Card>
          ) : (
            <div className="h-full flex items-center justify-center border-2 border-dashed border-border rounded-lg text-muted-foreground p-8 text-center">
              <div>
                <Clock className="w-12 h-12 mx-auto mb-4 opacity-20" />
                <p>Select a point on the timeline to inspect detailed records.</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
