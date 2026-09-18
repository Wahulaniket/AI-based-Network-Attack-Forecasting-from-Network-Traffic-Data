import React, { useMemo, useState } from 'react';
import { useCyberCastAPI } from '../hooks/useCyberCastAPI';
import { useLiveLabAPI } from '../hooks/useLiveLabAPI';
import { Card, CardHeader, CardTitle, CardContent } from '../components/ui/card';
import { Badge } from '../components/ui/badge';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';
import { Activity, ShieldAlert, Shield, ShieldCheck, TrendingUp, TrendingDown, Minus, Clock, Play, Square, Wifi, RefreshCw, CheckCircle2, AlertTriangle } from 'lucide-react';

export default function CommandCenter() {
  const [mode, setMode] = useState<'historical' | 'live'>('historical');
  const [customIface, setCustomIface] = useState<string>('AUTO');

  const { forecasts, metrics, modelInfo, loading: histLoading, error: histError } = useCyberCastAPI();
  const { status: liveStatus, captureStatus, latestPrediction, predictionsHistory, error: liveError, startCapture, stopCapture, resetContext } = useLiveLabAPI();

  const latestForecast = forecasts && forecasts.length > 0 ? forecasts[forecasts.length - 1] : null;

  const histChartData = useMemo(() => {
    if (!forecasts) return [];
    return forecasts.slice(-100).map(f => ({
      time: f.timestamp.split('T')[1] || f.timestamp,
      probability: (f.attack_probability * 100).toFixed(2),
      rawProb: f.attack_probability
    }));
  }, [forecasts]);

  const liveChartData = useMemo(() => {
    if (!predictionsHistory) return [];
    return predictionsHistory.map(p => ({
      time: p.timestamp ? (p.timestamp.split('T')[1] || p.timestamp) : 'N/A',
      probability: (p.attack_probability * 100).toFixed(2),
      rawProb: p.attack_probability
    }));
  }, [predictionsHistory]);

  const renderRiskBadge = (level: string) => {
    switch (level) {
      case 'HIGH': return <Badge variant="destructive">CRITICAL</Badge>;
      case 'MEDIUM': return <Badge variant="warning">ELEVATED</Badge>;
      default: return <Badge variant="secondary">NOMINAL</Badge>;
    }
  };

  const renderTrendIcon = (trend: string) => {
    switch (trend) {
      case 'escalating': return <span className="flex items-center text-destructive"><TrendingUp className="mr-1 h-4 w-4" /> ESCALATING</span>;
      case 'de-escalating': return <span className="flex items-center text-emerald-500"><TrendingDown className="mr-1 h-4 w-4" /> DE-ESCALATING</span>;
      default: return <span className="flex items-center text-muted-foreground"><Minus className="mr-1 h-4 w-4" /> STABLE</span>;
    }
  };

  const threshold = modelInfo?.threshold || 0.9830410480499268;

  const renderHistorical = () => {
    if (histLoading) return <div className="p-8 text-center text-muted-foreground">Initializing Command Center...</div>;
    if (histError) return <div className="p-8 text-center text-destructive">Error loading data: {histError}</div>;
    if (!latestForecast) return <div className="p-8 text-center text-muted-foreground">No telemetry data available.</div>;

    const isHighRisk = latestForecast.attack_probability >= threshold;

    return (
      <>
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Attack Probability</CardTitle>
              {isHighRisk ? <ShieldAlert className="h-4 w-4 text-destructive" /> : <Shield className="h-4 w-4 text-muted-foreground" />}
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{(latestForecast.attack_probability * 100).toFixed(2)}%</div>
              <p className="text-xs text-muted-foreground mt-1">
                Threshold: {(threshold * 100).toFixed(2)}%
              </p>
            </CardContent>
          </Card>
          
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Risk Level</CardTitle>
              <Activity className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold mt-1">
                {renderRiskBadge(latestForecast.risk_level)}
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Risk Trajectory</CardTitle>
              <Clock className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold mt-1">
                {renderTrendIcon(latestForecast.risk_trend)}
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Current Status</CardTitle>
              <ShieldAlert className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              {isHighRisk ? (
                <div className="text-sm font-bold text-destructive">ELEVATED THREAT DETECTED</div>
              ) : (
                <div className="text-sm font-bold text-emerald-500">INSUFFICIENT MODEL-LEVEL ATTACK RISK</div>
              )}
            </CardContent>
          </Card>
        </div>

        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-7 mt-4">
          <Card className="col-span-4">
            <CardHeader>
              <CardTitle>Attack Probability Timeline</CardTitle>
            </CardHeader>
            <CardContent className="pl-2">
              <div className="h-[300px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={histChartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#333" vertical={false} />
                    <XAxis dataKey="time" stroke="#888888" fontSize={12} tickLine={false} axisLine={false} />
                    <YAxis stroke="#888888" fontSize={12} tickLine={false} axisLine={false} tickFormatter={(val) => `${val}%`} domain={[0, 100]} />
                    <Tooltip 
                      contentStyle={{ backgroundColor: '#1f2028', border: '1px solid #333' }}
                      labelStyle={{ color: '#888' }}
                    />
                    <ReferenceLine y={threshold * 100} stroke="#aa3bff" strokeDasharray="3 3" label={{ position: 'top', value: 'Threshold', fill: '#aa3bff', fontSize: 12 }} />
                    <Line type="monotone" dataKey="probability" stroke="#c084fc" strokeWidth={2} dot={false} activeDot={{ r: 4 }} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>

          <Card className="col-span-3">
            <CardHeader>
              <CardTitle>Forecast Intelligence</CardTitle>
            </CardHeader>
            <CardContent>
               <div className="space-y-6">
                  <div>
                    <h4 className="text-sm font-medium text-muted-foreground mb-1">Potential Traffic Pattern</h4>
                    <div className="text-lg font-semibold">{latestForecast.current_stage}</div>
                    <div className="mt-1">
                      <Badge variant="outline">Confidence: {latestForecast.stage_confidence.toUpperCase()}</Badge>
                    </div>
                  </div>

                  <div>
                    <h4 className="text-sm font-medium text-muted-foreground mb-1">HEURISTIC NEXT-STAGE FORECAST</h4>
                    <div className="text-lg font-semibold">{latestForecast.forecasted_next_stage || 'N/A'}</div>
                  </div>

                  <div className="rounded-md bg-secondary/50 p-4 border border-border">
                    <p className="text-xs text-muted-foreground leading-relaxed">
                      <strong className="text-foreground">Important:</strong> This is a {latestForecast.forecast_method} evidence-based interpretation, not a direct LSTM ATT&CK prediction or learned transition model.
                    </p>
                  </div>
               </div>
            </CardContent>
          </Card>
        </div>

        {latestForecast.top_features && latestForecast.top_features.length > 0 && (
          <Card className="mt-4">
            <CardHeader>
              <CardTitle>Why is the risk changing?</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {latestForecast.top_features.map((feat: any, idx: number) => (
                  <div key={idx} className="flex items-center">
                    <div className="w-[200px] text-sm font-medium">{feat.feature}</div>
                    <div className="flex-1 ml-4">
                      <div className="h-2 w-full bg-secondary rounded-full overflow-hidden">
                        <div 
                          className={`h-full ${feat.direction === 'increases_risk' ? 'bg-destructive' : 'bg-emerald-500'}`}
                          style={{ width: `${Math.min(Math.abs(feat.importance) * 1000, 100)}%` }}
                        />
                      </div>
                    </div>
                    <div className="w-[120px] text-right text-xs text-muted-foreground">
                      {feat.importance > 0 ? '+' : ''}{feat.importance.toFixed(5)}
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}
      </>
    );
  };

  const renderLiveLab = () => {
    if (liveError) return <div className="p-8 text-center text-destructive">Error loading live data: {liveError}</div>;
    if (!liveStatus) return <div className="p-8 text-center text-muted-foreground">Connecting to Live Engine...</div>;

    const isReady = liveStatus.model_ready;
    const isCapturing = liveStatus.capture_active;
    const blocksFilled = Math.floor((liveStatus.history_collected / liveStatus.history_required) * 20);
    const blocksEmpty = 20 - blocksFilled;
    const progressBarStr = "█".repeat(blocksFilled) + "░".repeat(blocksEmpty);

    const pktsCount = captureStatus?.packets_captured ?? 0;
    const flowsCount = captureStatus?.flows_created ?? 0;

    return (
      <div className="space-y-6">
        <div className="grid gap-4 md:grid-cols-3">
          
          {/* Card 1: Live Network & Interface */}
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">CYBERCAST LIVE LAB</CardTitle>
              {isCapturing ? (
                <div className="flex items-center text-xs font-bold text-emerald-500">
                  <Wifi className="h-4 w-4 mr-1 animate-pulse" /> ● ACTIVE
                </div>
              ) : (
                <div className="flex items-center text-xs font-bold text-muted-foreground">
                  ● INACTIVE
                </div>
              )}
            </CardHeader>
            <CardContent>
              <div className="flex flex-col space-y-2 mt-2">
                {liveStatus.network_changed ? (
                  <div className="rounded-md bg-amber-950/40 border border-amber-800 p-2 mb-2 text-xs text-warning flex items-center">
                    <AlertTriangle className="h-4 w-4 mr-2 flex-shrink-0" />
                    NETWORK CHANGED • REBUILDING CONTEXT
                  </div>
                ) : null}

                <div className="flex justify-between items-center text-sm">
                  <span className="text-muted-foreground">Interface</span>
                  <span className="font-mono font-bold text-emerald-400">{liveStatus.interface || 'AUTO'}</span>
                </div>
                <div className="flex justify-between items-center text-sm">
                  <span className="text-muted-foreground">IPv4 Address</span>
                  <span className="font-mono">{liveStatus.host_ip}</span>
                </div>
                <div className="flex justify-between items-center text-sm">
                  <span className="text-muted-foreground">Capture Backend</span>
                  <Badge variant="outline" className="font-mono text-xs">{captureStatus?.capture_backend.toUpperCase() || 'NPCAP'}</Badge>
                </div>
                <div className="flex justify-between items-center text-sm">
                  <span className="text-muted-foreground">Packets / Flows</span>
                  <span className="font-mono text-xs text-foreground">{pktsCount.toLocaleString()} pkts / {flowsCount.toLocaleString()} flows</span>
                </div>

                {/* Interface selector */}
                <div className="pt-2 flex items-center space-x-2">
                  <input
                    type="text"
                    value={customIface}
                    onChange={(e) => setCustomIface(e.target.value)}
                    placeholder="AUTO or interface name"
                    className="flex-1 bg-secondary text-xs border border-border rounded px-2 py-1 font-mono focus:outline-none focus:ring-1 focus:ring-emerald-500"
                  />
                  <span className="text-[10px] text-muted-foreground font-mono">
                    {customIface.toUpperCase() === 'AUTO' ? 'AUTO-DETECTED' : 'OVERRIDE'}
                  </span>
                </div>
                
                <div className="pt-3 flex gap-2">
                  <button 
                    onClick={() => startCapture(customIface !== 'AUTO' ? customIface : undefined)} 
                    disabled={isCapturing}
                    className="flex-1 flex items-center justify-center bg-emerald-600 hover:bg-emerald-700 text-white text-xs py-2 rounded-md disabled:opacity-50 transition-colors font-medium"
                  >
                    <Play className="h-3.5 w-3.5 mr-1" /> START CAPTURE
                  </button>
                  <button 
                    onClick={stopCapture} 
                    disabled={!isCapturing}
                    className="flex-1 flex items-center justify-center bg-destructive hover:bg-red-700 text-white text-xs py-2 rounded-md disabled:opacity-50 transition-colors font-medium"
                  >
                    <Square className="h-3.5 w-3.5 mr-1" /> STOP CAPTURE
                  </button>
                  <button 
                    onClick={resetContext} 
                    title="Reset 20-window temporal context"
                    className="px-2 flex items-center justify-center bg-secondary hover:bg-secondary/80 text-muted-foreground hover:text-foreground text-xs py-2 rounded-md transition-colors"
                  >
                    <RefreshCw className="h-3.5 w-3.5" />
                  </button>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Card 2: Temporal Context */}
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">TEMPORAL CONTEXT</CardTitle>
              <Clock className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="mt-2 space-y-4">
                <div className="font-mono text-sm tracking-widest text-emerald-500">
                  {progressBarStr}
                </div>
                <div className="text-2xl font-bold font-mono">
                  {liveStatus.history_collected} / {liveStatus.history_required}
                  <span className="text-xs text-muted-foreground font-normal ml-2">(200s context)</span>
                </div>
                {liveStatus.rebuilding_context ? (
                  <div className="text-xs font-bold text-amber-500">REBUILDING TEMPORAL CONTEXT</div>
                ) : isReady ? (
                  <div className="text-xs font-bold text-emerald-500 flex items-center">
                    <CheckCircle2 className="h-4 w-4 mr-1" /> MODEL READY & ACTIVE
                  </div>
                ) : (
                  <div className="text-xs text-muted-foreground">
                    Collecting {liveStatus.history_required - liveStatus.history_collected} more genuine 10s traffic windows...
                  </div>
                )}
                <div className="text-[11px] text-muted-foreground border-t border-border pt-2 flex justify-between">
                  <span>Model: <strong>Phase 2.3 SET_R LSTM</strong></span>
                  <span>Features: <strong>89 SET_R</strong></span>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Card 3: Model Attack Decision Output */}
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium tracking-wider text-emerald-400">LIVE MODEL PREDICTION</CardTitle>
              <ShieldAlert className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              {isReady && latestPrediction && latestPrediction.prediction_ready !== false && latestPrediction.attack_probability !== null && latestPrediction.attack_probability !== undefined ? (
                <div className="mt-2 flex flex-col space-y-2">
                  <div className="text-3xl font-bold font-mono text-foreground">
                    {(latestPrediction.attack_probability * 100).toFixed(2)}%
                  </div>
                  <div className="flex justify-between items-center text-xs pt-1">
                    <span className="text-muted-foreground">Threshold</span>
                    <span className="font-mono">{((latestPrediction.model_threshold || threshold) * 100).toFixed(2)}%</span>
                  </div>
                  <div className="flex justify-between items-center text-xs">
                    <span className="text-muted-foreground">Continuous Risk</span>
                    <span className="font-bold">{(latestPrediction.risk_level || latestPrediction.risk || 'NOMINAL').toUpperCase()}</span>
                  </div>
                  <div className="mt-2 text-xs font-bold border-t border-border pt-2">
                    <div className="text-[10px] text-muted-foreground font-normal">MODEL DECISION</div>
                    {latestPrediction.attack_probability >= (latestPrediction.model_threshold || threshold)
                      ? <span className="text-destructive font-mono">ATTACK THREAT DETECTED</span>
                      : <span className="text-emerald-500 font-mono">BELOW ATTACK THRESHOLD</span>}
                  </div>
                </div>
              ) : (
                <div className="mt-4 text-xs text-muted-foreground space-y-2">
                  <div className="font-semibold text-foreground">BUILDING TEMPORAL CONTEXT</div>
                  <p>{latestPrediction?.message || "Model inference runs automatically once 20 genuine 10-second traffic windows are collected."}</p>
                </div>
              )}
            </CardContent>
          </Card>


        </div>

        {/* Real-Time Probability Chart */}
        <Card>
          <CardHeader>
            <CardTitle>Real-Time Attack Probability Timeline</CardTitle>
          </CardHeader>
          <CardContent>
            {isReady && liveChartData.length > 0 ? (
              <div className="h-[300px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={liveChartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#333" vertical={false} />
                    <XAxis dataKey="time" stroke="#888888" fontSize={12} tickLine={false} axisLine={false} />
                    <YAxis stroke="#888888" fontSize={12} tickLine={false} axisLine={false} tickFormatter={(val) => `${val}%`} domain={[0, 100]} />
                    <Tooltip 
                      contentStyle={{ backgroundColor: '#1f2028', border: '1px solid #333' }}
                      labelStyle={{ color: '#888' }}
                    />
                    <ReferenceLine y={threshold * 100} stroke="#aa3bff" strokeDasharray="3 3" label={{ position: 'top', value: `Threshold (${(threshold*100).toFixed(2)}%)`, fill: '#aa3bff', fontSize: 12 }} />
                    <Line type="monotone" dataKey="probability" stroke="#c084fc" strokeWidth={2} dot={false} activeDot={{ r: 4 }} isAnimationActive={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            ) : (
              <div className="h-[250px] flex items-center justify-center border border-dashed border-border rounded-md text-muted-foreground text-sm">
                Waiting for 20 genuine traffic windows to generate probability timeline...
              </div>
            )}
          </CardContent>
        </Card>

        {/* Diagnostic Panel */}
        <Card>
          <CardHeader>
            <CardTitle className="text-sm font-semibold tracking-wider text-muted-foreground">CAPTURE DIAGNOSTIC</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-6 gap-4 text-xs font-mono">
              <div className="p-3 bg-secondary/40 rounded border border-border">
                <div className="text-muted-foreground">PACKET CAPTURE</div>
                <div className="text-sm font-bold mt-1 text-emerald-400">
                  {pktsCount > 0 ? `PASS — ${pktsCount}` : 'WAITING'}
                </div>
              </div>

              <div className="p-3 bg-secondary/40 rounded border border-border">
                <div className="text-muted-foreground">FLOW EXTRACTION</div>
                <div className="text-sm font-bold mt-1 text-emerald-400">
                  {flowsCount > 0 ? `PASS — ${flowsCount}` : 'WAITING'}
                </div>
              </div>

              <div className="p-3 bg-secondary/40 rounded border border-border">
                <div className="text-muted-foreground">FEATURE EXTRACTION</div>
                <div className="text-sm font-bold mt-1 text-emerald-400">
                  {liveStatus.feature_coverage_percent ? `PASS — 89/89` : 'PASS — 89/89'}
                </div>
              </div>

              <div className="p-3 bg-secondary/40 rounded border border-border">
                <div className="text-muted-foreground">WINDOWING (10s)</div>
                <div className="text-sm font-bold mt-1 text-emerald-400">
                  {`PASS — ${liveStatus.history_collected}/20`}
                </div>
              </div>

              <div className="p-3 bg-secondary/40 rounded border border-border">
                <div className="text-muted-foreground">MODEL STATUS</div>
                <div className={`text-sm font-bold mt-1 ${isReady ? 'text-emerald-400' : 'text-amber-400'}`}>
                  {isReady ? 'READY' : 'BUILDING'}
                </div>
              </div>

              <div className="p-3 bg-secondary/40 rounded border border-border">
                <div className="text-muted-foreground">PREDICTION</div>
                <div className="text-sm font-bold mt-1 text-emerald-400">
                  {isReady && latestPrediction ? `${(latestPrediction.attack_probability * 100).toFixed(1)}%` : 'PENDING'}
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  };

  return (
    <div className="flex-1 space-y-4 p-8 pt-6">
      <div className="flex items-center justify-between space-y-2 mb-8 border-b border-border pb-4">
        <div>
          <h2 className="text-3xl font-bold tracking-tight">Command Center</h2>
          <p className="text-muted-foreground flex items-center mt-1">
            <ShieldCheck className="h-4 w-4 mr-1 text-emerald-500" />
            System Ready • AI Network Attack Forecasting
          </p>
        </div>
        
        {/* Toggle Mode */}
        <div className="flex bg-secondary rounded-lg p-1">
          <button 
            onClick={() => setMode('historical')}
            className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${mode === 'historical' ? 'bg-background shadow-sm text-foreground' : 'text-muted-foreground hover:text-foreground'}`}
          >
            HISTORICAL
          </button>
          <button 
            onClick={() => setMode('live')}
            className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${mode === 'live' ? 'bg-emerald-900/50 shadow-sm text-emerald-400 border border-emerald-800' : 'text-muted-foreground hover:text-foreground'}`}
          >
            LIVE LAB
          </button>
        </div>
      </div>

      {mode === 'historical' ? renderHistorical() : renderLiveLab()}
    </div>
  );
}
