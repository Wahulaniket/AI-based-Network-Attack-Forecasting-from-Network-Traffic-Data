import React from 'react';
import { useCyberCastAPI } from '../hooks/useCyberCastAPI';
import { Card, CardHeader, CardTitle, CardContent } from '../components/ui/card';
import { ShieldCheck, Database, Target, BrainCircuit, Activity } from 'lucide-react';
import { Badge } from '../components/ui/badge';

export default function ModelPerformance() {
  const { metrics, modelInfo, loading, error } = useCyberCastAPI();

  if (loading) return <div className="p-8 text-center text-muted-foreground">Loading performance metrics...</div>;
  if (error) return <div className="p-8 text-center text-destructive">Error loading data: {error}</div>;
  if (!metrics) return <div className="p-8 text-center text-muted-foreground">No metrics available.</div>;

  const m = metrics.Test_Metrics || {};

  return (
    <div className="flex-1 space-y-4 p-8 pt-6">
      <div className="flex items-center justify-between space-y-2 mb-8">
        <div>
          <h2 className="text-3xl font-bold tracking-tight">Model Performance</h2>
          <p className="text-muted-foreground flex items-center mt-1">
            <ShieldCheck className="h-4 w-4 mr-1 text-emerald-500" />
            Frozen Offline Model Performance
          </p>
        </div>
      </div>

      <div className="rounded-md bg-secondary/50 p-4 border border-border mb-6">
        <p className="text-sm text-muted-foreground leading-relaxed">
          <strong className="text-foreground">Important:</strong> These are the official <Badge variant="secondary" className="mx-1">FROZEN PHASE 2.3 BLIND TEST</Badge> metrics. They are loaded strictly from the `champion_metrics.json` artifact. These metrics reflect the true offline test performance and are never recalculated from live or replay data.
        </p>
      </div>

      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">PR-AUC</CardTitle>
            <Target className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{m['PR-AUC'] ? m['PR-AUC'].toFixed(4) : 'N/A'}</div>
            <p className="text-xs text-muted-foreground mt-1">Primary performance metric</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">ROC-AUC</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{m['ROC-AUC'] ? m['ROC-AUC'].toFixed(4) : 'N/A'}</div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">F1 Score</CardTitle>
            <Target className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{m['F1'] ? m['F1'].toFixed(4) : 'N/A'}</div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">FPR</CardTitle>
            <ShieldCheck className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{m['FPR'] ? m['FPR'].toFixed(4) : 'N/A'}</div>
            <p className="text-xs text-muted-foreground mt-1">False Positive Rate</p>
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-4 md:grid-cols-2 mt-4">
        <Card>
          <CardHeader>
            <CardTitle>Detailed Metrics</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex justify-between border-b border-border pb-2">
                <span className="font-medium text-muted-foreground">Precision</span>
                <span>{m['Precision'] ? m['Precision'].toFixed(4) : 'N/A'}</span>
              </div>
              <div className="flex justify-between border-b border-border pb-2">
                <span className="font-medium text-muted-foreground">Recall</span>
                <span>{m['Recall'] ? m['Recall'].toFixed(4) : 'N/A'}</span>
              </div>
              <div className="flex justify-between border-b border-border pb-2">
                <span className="font-medium text-muted-foreground">Accuracy</span>
                <span>{m['Accuracy'] ? m['Accuracy'].toFixed(4) : 'N/A'}</span>
              </div>
              <div className="flex justify-between border-b border-border pb-2">
                <span className="font-medium text-muted-foreground">True Positives (TP)</span>
                <span>{m['TP'] ?? 'N/A'}</span>
              </div>
              <div className="flex justify-between border-b border-border pb-2">
                <span className="font-medium text-muted-foreground">True Negatives (TN)</span>
                <span>{m['TN'] ?? 'N/A'}</span>
              </div>
              <div className="flex justify-between border-b border-border pb-2">
                <span className="font-medium text-muted-foreground">False Positives (FP)</span>
                <span>{m['FP'] ?? 'N/A'}</span>
              </div>
              <div className="flex justify-between pb-2">
                <span className="font-medium text-muted-foreground">False Negatives (FN)</span>
                <span>{m['FN'] ?? 'N/A'}</span>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Model Provenance</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex items-center">
                <BrainCircuit className="w-5 h-5 mr-3 text-muted-foreground" />
                <div>
                  <div className="font-medium">{modelInfo?.model_name || 'Phase 2.3 LSTM'}</div>
                  <div className="text-sm text-muted-foreground">Architecture</div>
                </div>
              </div>
              <div className="flex items-center mt-4">
                <Database className="w-5 h-5 mr-3 text-muted-foreground" />
                <div>
                  <div className="font-medium">{modelInfo?.feature_set || 'SET_R'}</div>
                  <div className="text-sm text-muted-foreground">Feature Configuration ({modelInfo?.features_count || 89} features)</div>
                </div>
              </div>
              <div className="flex items-center mt-4">
                <Activity className="w-5 h-5 mr-3 text-muted-foreground" />
                <div>
                  <div className="font-medium">{modelInfo?.history_windows || 20} windows × {modelInfo?.window_size_seconds || 10}s</div>
                  <div className="text-sm text-muted-foreground">Temporal Context ({modelInfo?.context_seconds || 200}s)</div>
                </div>
              </div>
              <div className="flex items-center mt-4">
                <Target className="w-5 h-5 mr-3 text-muted-foreground" />
                <div>
                  <div className="font-medium">{modelInfo?.threshold?.toFixed(4) || '0.9830'}</div>
                  <div className="text-sm text-muted-foreground">Decision Threshold</div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
