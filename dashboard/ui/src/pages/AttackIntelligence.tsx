import React from 'react';
import { useCyberCastAPI } from '../hooks/useCyberCastAPI';
import { Card, CardHeader, CardTitle, CardContent } from '../components/ui/card';
import { Badge } from '../components/ui/badge';
import { ShieldCheck, ShieldAlert, Database, Radar } from 'lucide-react';

export default function AttackIntelligence() {
  const { forecasts, modelInfo, loading, error } = useCyberCastAPI();

  if (loading) return <div className="p-8 text-center text-muted-foreground">Loading ATT&CK Intelligence...</div>;
  if (error) return <div className="p-8 text-center text-destructive">Error loading data: {error}</div>;
  if (!forecasts || forecasts.length === 0) return <div className="p-8 text-center text-muted-foreground">No data available.</div>;

  const latestForecast = forecasts[forecasts.length - 1];
  const threshold = modelInfo?.threshold || 0.98;
  const isHighRisk = latestForecast.attack_probability >= threshold;

  return (
    <div className="flex-1 space-y-4 p-8 pt-6">
      <div className="flex items-center justify-between space-y-2 mb-8">
        <div>
          <h2 className="text-3xl font-bold tracking-tight">ATT&CK Intelligence & Forecast</h2>
          <p className="text-muted-foreground mt-1 flex items-center">
            <Radar className="w-4 h-4 mr-1 text-emerald-500" />
            Evidence-Based Stage Mapping
          </p>
        </div>
      </div>

      <div className="rounded-md bg-secondary/50 p-4 border border-border mb-6">
        <p className="text-sm text-muted-foreground leading-relaxed">
          <strong className="text-foreground">Scientific Honesty Disclaimer:</strong> The LSTM model strictly predicts generalized attack probability (Model-Level Attack Risk). The heuristic stage mapper relies on traffic signatures to identify "Potential Traffic Patterns". It is <strong>NOT</strong> an empirically validated transition model or a direct ATT&CK classifier.
        </p>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <Card className={isHighRisk ? 'border-destructive/50' : 'border-emerald-500/50'}>
          <CardHeader>
            <CardTitle>Model-Level Attack Risk</CardTitle>
          </CardHeader>
          <CardContent>
            {isHighRisk ? (
              <div>
                <div className="text-2xl font-bold text-destructive">ELEVATED ATTACK RISK DETECTED</div>
                <p className="text-sm text-muted-foreground mt-2">
                  The deep learning model indicates traffic patterns have crossed the critical threshold ({(threshold*100).toFixed(2)}%).
                </p>
              </div>
            ) : (
              <div>
                <div className="text-2xl font-bold text-emerald-500">INSUFFICIENT MODEL-LEVEL ATTACK RISK</div>
                <p className="text-sm text-muted-foreground mt-2">
                  The deep learning model indicates traffic patterns remain below the critical threshold ({(threshold*100).toFixed(2)}%).
                </p>
              </div>
            )}
            <div className="mt-4 pt-4 border-t border-border">
               <span className="text-sm font-medium">Recorded Probability: </span>
               <span className={`font-mono ml-2 ${isHighRisk ? 'text-destructive' : 'text-emerald-500'}`}>
                 {(latestForecast.attack_probability * 100).toFixed(4)}%
               </span>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Potential Traffic Pattern (ATT&CK Evidence)</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div>
                <div className="text-xl font-semibold">{latestForecast.current_stage}</div>
                {!isHighRisk && latestForecast.current_stage !== "UNKNOWN / INSUFFICIENT EVIDENCE" && (
                  <Badge variant="outline" className="mt-2 border-warning text-warning">
                    Heuristic signature detected, but NOT a confirmed model attack
                  </Badge>
                )}
              </div>
              
              <div className="pt-4 border-t border-border">
                <h4 className="text-sm font-medium mb-2">Heuristic Stage Evidence:</h4>
                {latestForecast.stage_evidence && latestForecast.stage_evidence.length > 0 ? (
                  <ul className="space-y-2">
                    {latestForecast.stage_evidence.map((ev: any, idx: number) => (
                      <li key={idx} className="text-sm">
                        <Badge variant="secondary" className="mr-2 font-mono text-xs">{ev.feature}</Badge>
                        <span className="text-muted-foreground">{ev.reason}</span>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="text-sm text-muted-foreground italic">NOT OBSERVED / INSUFFICIENT EVIDENCE</p>
                )}
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      <Card className="mt-4">
        <CardHeader>
          <CardTitle>Heuristic Next-Stage Forecast</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid md:grid-cols-2 gap-8">
            <div>
              <p className="text-sm text-muted-foreground">Based on Kill-Chain Progression logic, if an attack were to proceed, the next likely phase is:</p>
              <div className="mt-4 text-3xl font-bold">{latestForecast.forecasted_next_stage || 'N/A'}</div>
              {latestForecast.forecasted_next_stage && (
                <div className="mt-2">
                  <Badge variant="outline">Confidence: {latestForecast.forecast_confidence}</Badge>
                </div>
              )}
            </div>
            <div className="bg-secondary/30 p-4 rounded-md border border-border flex items-start">
              <Database className="w-5 h-5 mr-3 text-muted-foreground flex-shrink-0" />
              <p className="text-xs text-muted-foreground">
                Forecasting relies entirely on hardcoded state-transition rules aligned with MITRE ATT&CK heuristics. It is completely decoupled from the LSTM neural network weights.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
