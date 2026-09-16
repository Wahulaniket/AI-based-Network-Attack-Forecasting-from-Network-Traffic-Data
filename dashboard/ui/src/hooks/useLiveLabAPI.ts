import { useState, useEffect, useCallback } from 'react';

const API_BASE = 'http://localhost:8000/api/live';

export interface LiveStatus {
  mode: string;
  capture_active: boolean;
  capture_backend?: string;
  interface: string;
  interface_index: number;
  host_ip: string;
  previous_interface: string | null;
  previous_host_ip: string | null;
  accepted_interface: string;
  accepted_host_ip: string;
  network_changed: boolean;
  rebuilding_context: boolean;
  model_name: string;
  feature_set: string;
  window_seconds: number;
  history_required: number;
  history_collected: number;
  model_ready: boolean;
  feature_coverage_percent?: number;
  feature_compatibility_status?: string;
}

export interface CaptureStatus {
  capture_backend: string;
  interface: string;
  interface_description?: string;
  interface_index: number;
  current_ipv4: str;
  capture_active: boolean;
  packets_captured: number;
  bytes_captured: number;
  flows_created: number;
  packets_per_sec: number;
  last_packet_time?: string;
  error?: string;
}

export interface LivePrediction {
  timestamp: string;
  window_start: string;
  window_end: string;
  attack_probability: number;
  model_threshold: number;
  prediction: number;
  risk: string;
  history_collected: number;
  flow_count: number;
  packet_count: number;
  total_bytes: number;
  status?: string;
}

export function useLiveLabAPI() {
  const [status, setStatus] = useState<LiveStatus | null>(null);
  const [captureStatus, setCaptureStatus] = useState<CaptureStatus | null>(null);
  const [latestPrediction, setLatestPrediction] = useState<LivePrediction | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [predictionsHistory, setPredictionsHistory] = useState<LivePrediction[]>([]);

  const fetchStatus = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/status`);
      if (res.ok) {
        const data = await res.json();
        setStatus(data);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    }
  }, []);

  const fetchCaptureStatus = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/capture-status`);
      if (res.ok) {
        const data = await res.json();
        setCaptureStatus(data);
      }
    } catch (err) {
      console.error(err);
    }
  }, []);

  const fetchLatest = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/latest`);
      if (res.ok) {
        const data = await res.json();
        if (!data.status || data.timestamp) {
          setLatestPrediction(data);
          setPredictionsHistory(prev => {
            const exists = prev.find(p => p.timestamp === data.timestamp);
            if (exists) return prev;
            const newHistory = [...prev, data];
            return newHistory.slice(-50);
          });
        }
      }
    } catch (err) {
      console.error(err);
    }
  }, []);

  useEffect(() => {
    const interval = setInterval(() => {
      fetchStatus();
      fetchCaptureStatus();
      fetchLatest();
    }, 2000);
    
    fetchStatus();
    fetchCaptureStatus();
    fetchLatest();

    return () => clearInterval(interval);
  }, [fetchStatus, fetchCaptureStatus, fetchLatest]);

  const startCapture = async (ifaceOverride?: string) => {
    try {
      const url = ifaceOverride ? `${API_BASE}/start?interface=${encodeURIComponent(ifaceOverride)}` : `${API_BASE}/start`;
      await fetch(url, { method: 'POST' });
      fetchStatus();
      fetchCaptureStatus();
    } catch (err) {
      console.error(err);
    }
  };

  const stopCapture = async () => {
    try {
      await fetch(`${API_BASE}/stop`, { method: 'POST' });
      fetchStatus();
      fetchCaptureStatus();
    } catch (err) {
      console.error(err);
    }
  };

  const resetContext = async () => {
    try {
      await fetch(`${API_BASE}/reset`, { method: 'POST' });
      setPredictionsHistory([]);
      setLatestPrediction(null);
      fetchStatus();
    } catch (err) {
      console.error(err);
    }
  };

  return { 
    status, 
    captureStatus,
    latestPrediction, 
    predictionsHistory,
    error, 
    startCapture, 
    stopCapture,
    resetContext
  };
}
