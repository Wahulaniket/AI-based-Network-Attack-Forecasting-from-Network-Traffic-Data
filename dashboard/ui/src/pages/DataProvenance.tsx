import React from 'react';
import { useCyberCastAPI } from '../hooks/useCyberCastAPI';
import { Card, CardHeader, CardTitle, CardContent } from '../components/ui/card';
import { Database, ShieldAlert, GitCommit, FileJson, BrainCircuit } from 'lucide-react';

export default function DataProvenance() {
  const { modelInfo, loading, error } = useCyberCastAPI();

  if (loading) return <div className="p-8 text-center text-muted-foreground">Loading provenance data...</div>;
  if (error) return <div className="p-8 text-center text-destructive">Error loading data: {error}</div>;

  return (
    <div className="flex-1 space-y-4 p-8 pt-6">
      <div className="flex items-center justify-between space-y-2 mb-8">
        <div>
          <h2 className="text-3xl font-bold tracking-tight">Data & Model Provenance</h2>
          <p className="text-muted-foreground mt-1 flex items-center">
            <Database className="w-4 h-4 mr-1 text-emerald-500" />
            Artifact Integrity Verification
          </p>
        </div>
      </div>

      <div className="rounded-md bg-destructive/10 p-6 border border-destructive/20 mb-8">
        <h3 className="text-lg font-bold text-destructive flex items-center mb-2">
          <ShieldAlert className="w-5 h-5 mr-2" />
          STRICT READ-ONLY GUARANTEE
        </h3>
        <ul className="list-disc pl-6 space-y-2 text-foreground font-medium">
          <li>NO MODEL RETRAINING</li>
          <li>NO LIVE INFERENCE</li>
          <li>READ-ONLY RECORDED OUTPUTS</li>
        </ul>
        <p className="text-sm text-muted-foreground mt-4">
          This dashboard operates exclusively as a presentation layer over the frozen Phase 2.3 outputs. It does not load PyTorch, modify weights, or generate new inferences.
        </p>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <BrainCircuit className="w-5 h-5 mr-2 text-primary" />
              Model Architecture
            </CardTitle>
          </CardHeader>
          <CardContent>
            <dl className="space-y-4 text-sm">
              <div className="grid grid-cols-3 border-b border-border pb-2">
                <dt className="text-muted-foreground font-medium">Architecture</dt>
                <dd className="col-span-2 font-mono">{modelInfo?.model_name || 'Phase 2.3 LSTM'}</dd>
              </div>
              <div className="grid grid-cols-3 border-b border-border pb-2">
                <dt className="text-muted-foreground font-medium">Feature Set</dt>
                <dd className="col-span-2 font-mono">{modelInfo?.feature_set || 'SET_R'}</dd>
              </div>
              <div className="grid grid-cols-3 border-b border-border pb-2">
                <dt className="text-muted-foreground font-medium">Input Features</dt>
                <dd className="col-span-2 font-mono">{modelInfo?.features_count || 89}</dd>
              </div>
              <div className="grid grid-cols-3 border-b border-border pb-2">
                <dt className="text-muted-foreground font-medium">History Windows</dt>
                <dd className="col-span-2 font-mono">{modelInfo?.history_windows || 20}</dd>
              </div>
              <div className="grid grid-cols-3 border-b border-border pb-2">
                <dt className="text-muted-foreground font-medium">Window Size</dt>
                <dd className="col-span-2 font-mono">{modelInfo?.window_size_seconds || 10} seconds</dd>
              </div>
              <div className="grid grid-cols-3 border-b border-border pb-2">
                <dt className="text-muted-foreground font-medium">Total Context</dt>
                <dd className="col-span-2 font-mono">{modelInfo?.context_seconds || 200} seconds</dd>
              </div>
              <div className="grid grid-cols-3">
                <dt className="text-muted-foreground font-medium">Decision Threshold</dt>
                <dd className="col-span-2 font-mono text-primary font-bold">{modelInfo?.threshold?.toFixed(8) || '0.98304104'}</dd>
              </div>
            </dl>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <FileJson className="w-5 h-5 mr-2 text-primary" />
              Source Artifacts
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-6">
              <div>
                <h4 className="font-semibold text-sm mb-2 flex items-center">
                  <GitCommit className="w-4 h-4 mr-2 text-muted-foreground" />
                  Frozen Model
                </h4>
                <p className="text-xs font-mono text-muted-foreground bg-secondary/50 p-2 rounded border border-border">
                  models/phase2_3/champion_lstm.pt
                </p>
              </div>
              
              <div>
                <h4 className="font-semibold text-sm mb-2 flex items-center">
                  <GitCommit className="w-4 h-4 mr-2 text-muted-foreground" />
                  Blind Test Metrics
                </h4>
                <p className="text-xs font-mono text-muted-foreground bg-secondary/50 p-2 rounded border border-border">
                  results/phase2_3/champion_metrics.json
                </p>
              </div>

              <div>
                <h4 className="font-semibold text-sm mb-2 flex items-center">
                  <GitCommit className="w-4 h-4 mr-2 text-muted-foreground" />
                  Forecast & Explainability Data
                </h4>
                <p className="text-xs font-mono text-muted-foreground bg-secondary/50 p-2 rounded border border-border">
                  results/inference/forecast_predictions.json
                </p>
              </div>

              <div>
                <h4 className="font-semibold text-sm mb-2 flex items-center">
                  <GitCommit className="w-4 h-4 mr-2 text-muted-foreground" />
                  Feature Configurations
                </h4>
                <p className="text-xs font-mono text-muted-foreground bg-secondary/50 p-2 rounded border border-border">
                  results/phase2_3/feature_sets.json
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
