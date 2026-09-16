import React, { useState, useEffect } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '../components/ui/card';
import { Activity, ShieldAlert, ShieldCheck, Clock, Shield } from 'lucide-react';

const API_BASE = 'http://localhost:8000/api';

export default function NetworkTraffic() {
  const [trafficState, setTrafficState] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch(`${API_BASE}/traffic-state`)
      .then(res => res.json())
      .then(data => {
        setTrafficState(data);
        setLoading(false);
      })
      .catch(err => {
        setError(err.message);
        setLoading(false);
      });
  }, []);

  if (loading) return <div className="p-8 text-center text-muted-foreground">Loading network state...</div>;
  if (error) return <div className="p-8 text-center text-destructive">Error loading data: {error}</div>;
  if (!trafficState) return <div className="p-8 text-center text-muted-foreground">No traffic state available.</div>;

  return (
    <div className="flex-1 space-y-4 p-8 pt-6">
      <div className="flex items-center justify-between space-y-2 mb-8">
        <div>
          <h2 className="text-3xl font-bold tracking-tight">Network Traffic</h2>
          <p className="text-muted-foreground mt-1 flex items-center">
            <Activity className="w-4 h-4 mr-1 text-emerald-500" />
            Aggregated Processing State
          </p>
        </div>
      </div>

      <div className="rounded-md bg-secondary/50 p-4 border border-border mb-6">
        <p className="text-sm text-muted-foreground leading-relaxed">
          <strong className="text-foreground">Data Availability:</strong> CyberCast Phase 2.3 records finalized predictions and explainability attributions. Raw individual PCAP packet/flow features (e.g., flow bytes/s) are not permanently stored in the inference artifact (`forecast_predictions.json`) beyond the top contributing ML features. We are displaying aggregated state metrics derived from the available model artifacts.
        </p>
      </div>

      <div className="grid gap-4 md:grid-cols-3">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Analyzed Windows</CardTitle>
            <Clock className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{trafficState.total_analyzed_windows?.toLocaleString()}</div>
            <p className="text-xs text-muted-foreground mt-1">Number of 10s intervals processed</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">High Risk Windows</CardTitle>
            <ShieldAlert className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-destructive">{trafficState.high_risk_windows?.toLocaleString()}</div>
            <p className="text-xs text-muted-foreground mt-1">Windows exceeding attack threshold</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Escalating Windows</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{trafficState.escalating_windows?.toLocaleString()}</div>
            <p className="text-xs text-muted-foreground mt-1">Positive causal risk trajectories</p>
          </CardContent>
        </Card>
      </div>

      <Card className="mt-4">
         <CardHeader>
            <CardTitle>Timeline Span</CardTitle>
         </CardHeader>
         <CardContent>
            {trafficState.timeline_span ? (
              <div className="flex gap-8">
                 <div>
                    <p className="text-sm text-muted-foreground">Start Recorded Activity</p>
                    <p className="font-mono mt-1 text-lg">{trafficState.timeline_span.start?.replace('T', ' ')}</p>
                 </div>
                 <div>
                    <p className="text-sm text-muted-foreground">End Recorded Activity</p>
                    <p className="font-mono mt-1 text-lg">{trafficState.timeline_span.end?.replace('T', ' ')}</p>
                 </div>
              </div>
            ) : (
              <p className="text-muted-foreground italic">Timeline span unavailable.</p>
            )}
         </CardContent>
      </Card>
    </div>
  );
}
