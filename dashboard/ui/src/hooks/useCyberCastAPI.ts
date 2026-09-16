import { useState, useEffect } from 'react';

const API_BASE = 'http://localhost:8000/api';

export interface Forecast {
  timestamp: string;
  window_start: string;
  window_end: string;
  attack_probability: number;
  model_threshold: number;
  binary_prediction: number;
  risk_level: string;
  current_stage: string;
  stage_confidence: string;
  stage_evidence: any[];
  forecasted_next_stage: string | null;
  forecast_confidence: string;
  forecast_method: string;
  probability_delta: number | null;
  risk_trend: string;
  top_features: any[];
  temporal_evidence: any[];
}

export function useCyberCastAPI() {
  const [forecasts, setForecasts] = useState<Forecast[]>([]);
  const [metrics, setMetrics] = useState<any>(null);
  const [modelInfo, setModelInfo] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        const [forecastsRes, metricsRes, modelInfoRes] = await Promise.all([
          fetch(`${API_BASE}/forecasts?limit=1000`),
          fetch(`${API_BASE}/metrics`),
          fetch(`${API_BASE}/model-info`)
        ]);

        if (forecastsRes.ok) {
          const forecastsData = await forecastsRes.json();
          setForecasts(forecastsData);
        }
        
        if (metricsRes.ok) {
          const metricsData = await metricsRes.json();
          setMetrics(metricsData);
        }

        if (modelInfoRes.ok) {
          const modelInfoData = await modelInfoRes.json();
          setModelInfo(modelInfoData);
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unknown error occurred');
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  return { forecasts, metrics, modelInfo, loading, error };
}
