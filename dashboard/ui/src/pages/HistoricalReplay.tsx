import React, { useState, useEffect, useRef } from 'react';
import { useCyberCastAPI } from '../hooks/useCyberCastAPI';
import { useChunkedReplay } from '../hooks/useChunkedReplay';
import { Card, CardHeader, CardTitle, CardContent } from '../components/ui/card';
import { Badge } from '../components/ui/badge';
import { Play, Pause, RotateCcw, FastForward, Rewind, Activity, ShieldAlert, Shield, Clock } from 'lucide-react';

export default function HistoricalReplay() {
  const { modelInfo } = useCyberCastAPI();
  const { currentChunk, offset, loading, error, hasMore, nextChunk, prevChunk } = useChunkedReplay(100);
  
  const [localIndex, setLocalIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeed] = useState(1000); // ms per frame
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    if (isPlaying) {
      timerRef.current = setInterval(() => {
        setLocalIndex(prev => {
          if (prev >= currentChunk.length - 1) {
            // Need to go to next chunk if available
            if (hasMore) {
              nextChunk();
              return 0; // reset local index for new chunk
            } else {
              setIsPlaying(false);
              return prev;
            }
          }
          return prev + 1;
        });
      }, speed);
    } else if (timerRef.current) {
      clearInterval(timerRef.current);
    }
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [isPlaying, speed, currentChunk, hasMore, nextChunk]);

  // When chunk changes from nextChunk/prevChunk via button clicks, reset index safely
  useEffect(() => {
    if (!isPlaying && localIndex >= currentChunk.length) {
      setLocalIndex(0);
    }
  }, [currentChunk, isPlaying, localIndex]);

  const togglePlay = () => setIsPlaying(!isPlaying);
  const reset = () => { setIsPlaying(false); setLocalIndex(0); if (offset > 0) prevChunk(); /* not full reset, just local */ };
  const stepForward = () => { 
    if (localIndex < currentChunk.length - 1) {
      setLocalIndex(c => c + 1);
    } else if (hasMore) {
      nextChunk();
      setLocalIndex(0);
    }
  };
  const stepBack = () => { 
    if (localIndex > 0) {
      setLocalIndex(c => c - 1);
    } else if (offset > 0) {
      prevChunk();
      setLocalIndex(99); // Go to end of previous chunk
    }
  };

  if (error) return <div className="p-8 text-center text-destructive">Error: {error}</div>;

  const currentForecast = currentChunk[localIndex];
  const threshold = modelInfo?.threshold || 0.98;
  const isHighRisk = currentForecast ? currentForecast.attack_probability >= threshold : false;

  return (
    <div className="flex-1 space-y-4 p-8 pt-6">
      <div className="flex items-center justify-between space-y-2 mb-8">
        <div>
          <h2 className="text-3xl font-bold tracking-tight flex items-center">
            Historical Replay 
            <Badge variant="outline" className="ml-4 text-emerald-500 border-emerald-500">
              RECORDED MODEL OUTPUTS
            </Badge>
          </h2>
          <p className="text-muted-foreground mt-1 text-sm">
            This module replays chronologically recorded predictions in efficient chunks. No new inference is running.
          </p>
        </div>
      </div>

      <Card className="mb-6 bg-secondary/20">
        <CardContent className="p-6 flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-4">
            <button onClick={stepBack} disabled={offset === 0 && localIndex === 0} className="p-2 rounded-full hover:bg-secondary disabled:opacity-50">
              <Rewind className="w-5 h-5" />
            </button>
            <button onClick={togglePlay} className="p-3 bg-primary text-primary-foreground rounded-full hover:bg-primary/80">
              {isPlaying ? <Pause className="w-6 h-6" /> : <Play className="w-6 h-6 ml-1" />}
            </button>
            <button onClick={stepForward} disabled={!hasMore && localIndex === currentChunk.length - 1} className="p-2 rounded-full hover:bg-secondary disabled:opacity-50">
              <FastForward className="w-5 h-5" />
            </button>
            <button onClick={reset} className="p-2 rounded-full hover:bg-secondary">
              <RotateCcw className="w-5 h-5" />
            </button>
          </div>
          <div className="flex items-center gap-4 text-sm">
            <span className="text-muted-foreground">
              Global Record: {offset + localIndex + 1} {loading && '(Loading...)'}
            </span>
            <div className="px-4 py-2 bg-background border border-border rounded-md font-mono flex items-center min-w-[200px]">
              <Clock className="w-4 h-4 mr-2 text-muted-foreground" />
              {currentForecast?.timestamp.replace('T', ' ') || '-----'}
            </div>
          </div>
          <div className="flex gap-2">
            {[1000, 500, 100].map(s => (
              <button 
                key={s} 
                onClick={() => setSpeed(s)}
                className={`px-3 py-1 rounded-md text-xs font-semibold ${speed === s ? 'bg-primary text-primary-foreground' : 'bg-secondary text-secondary-foreground hover:bg-secondary/80'}`}
              >
                {1000/s}x
              </button>
            ))}
          </div>
        </CardContent>
      </Card>

      {!currentForecast ? (
         <div className="p-8 text-center text-muted-foreground">Loading chunk data...</div>
      ) : (
        <>
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
            <Card className={isHighRisk ? 'border-destructive/50 shadow-[0_0_15px_rgba(220,38,38,0.2)]' : ''}>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Recorded Probability</CardTitle>
                {isHighRisk ? <ShieldAlert className="h-4 w-4 text-destructive" /> : <Shield className="h-4 w-4 text-muted-foreground" />}
              </CardHeader>
              <CardContent>
                <div className={`text-3xl font-bold ${isHighRisk ? 'text-destructive' : ''}`}>
                  {(currentForecast.attack_probability * 100).toFixed(2)}%
                </div>
              </CardContent>
            </Card>
            
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Risk Label</CardTitle>
                <Activity className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold mt-1">
                  <Badge variant={currentForecast.risk_level === 'HIGH' ? 'destructive' : 'secondary'}>
                    {currentForecast.risk_level}
                  </Badge>
                </div>
              </CardContent>
            </Card>

            <Card className="md:col-span-2">
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Intelligence Interpretation</CardTitle>
              </CardHeader>
              <CardContent>
                {isHighRisk ? (
                  <div className="flex gap-4 items-center">
                    <div className="text-xl font-bold text-destructive">ELEVATED</div>
                    <div className="border-l border-border pl-4">
                      <div className="text-sm text-muted-foreground">Potential Pattern</div>
                      <div className="font-medium">{currentForecast.current_stage}</div>
                    </div>
                  </div>
                ) : (
                  <div className="flex gap-4 items-center">
                    <div className="text-xl font-bold text-emerald-500 flex flex-col">
                      <span className="text-sm tracking-tighter">INSUFFICIENT MODEL-LEVEL ATTACK RISK</span>
                    </div>
                    {currentForecast.current_stage !== "UNKNOWN / INSUFFICIENT EVIDENCE" && (
                      <div className="border-l border-border pl-4 text-xs max-w-[200px]">
                        <span className="font-medium text-warning">Heuristic Evidence:</span> {currentForecast.current_stage} (Not a confirmed attack)
                      </div>
                    )}
                  </div>
                )}
              </CardContent>
            </Card>
          </div>

          <Card className="mt-4">
            <CardHeader>
              <CardTitle>Heuristic Next-Stage Forecast</CardTitle>
            </CardHeader>
            <CardContent>
              {currentForecast.forecasted_next_stage ? (
                <div>
                  <p className="text-lg font-semibold">{currentForecast.forecasted_next_stage}</p>
                  <p className="text-sm text-muted-foreground mt-2">
                    This relies on static kill-chain progression logic and observed causal risk trajectories. It is NOT a learned transition model.
                  </p>
                </div>
              ) : (
                <p className="text-muted-foreground italic">Insufficient evidence to forecast next stage.</p>
              )}
            </CardContent>
          </Card>
        </>
      )}
    </div>
  );
}
