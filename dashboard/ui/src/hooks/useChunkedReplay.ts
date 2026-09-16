import { useState, useEffect } from 'react';
import type { Forecast } from './useCyberCastAPI';

const API_BASE = 'http://localhost:8000/api';

export function useChunkedReplay(chunkSize = 100) {
  const [currentChunk, setCurrentChunk] = useState<Forecast[]>([]);
  const [offset, setOffset] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hasMore, setHasMore] = useState(true);

  // Load the first chunk or when offset changes
  useEffect(() => {
    let active = true;
    const fetchChunk = async () => {
      setLoading(true);
      try {
        const res = await fetch(`${API_BASE}/forecasts?limit=${chunkSize}&offset=${offset}`);
        if (res.ok) {
          const data = await res.json();
          if (active) {
            setCurrentChunk(data);
            setHasMore(data.length === chunkSize);
          }
        } else {
          if (active) setError("Failed to fetch chunk");
        }
      } catch (err: any) {
        if (active) setError(err.message);
      } finally {
        if (active) setLoading(false);
      }
    };
    
    fetchChunk();
    return () => { active = false; };
  }, [offset, chunkSize]);

  const nextChunk = () => {
    if (hasMore) setOffset(prev => prev + chunkSize);
  };

  const prevChunk = () => {
    if (offset > 0) setOffset(prev => Math.max(0, prev - chunkSize));
  };

  return {
    currentChunk,
    offset,
    loading,
    error,
    hasMore,
    nextChunk,
    prevChunk
  };
}
