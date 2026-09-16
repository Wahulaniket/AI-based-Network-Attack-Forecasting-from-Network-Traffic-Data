import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Sidebar } from './components/layout/Sidebar';
import CommandCenter from './pages/CommandCenter';
import HistoricalReplay from './pages/HistoricalReplay';
import ModelPerformance from './pages/ModelPerformance';
import AttackIntelligence from './pages/AttackIntelligence';
import Explainability from './pages/Explainability';
import NetworkTraffic from './pages/NetworkTraffic';
import AttackTimeline from './pages/AttackTimeline';
import DataProvenance from './pages/DataProvenance';

function Layout({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex min-h-screen w-full bg-background">
      <Sidebar />
      <main className="flex-1 flex flex-col overflow-auto h-screen">
        {children}
      </main>
    </div>
  );
}

export default function App() {
  return (
    <Router>
      <Layout>
        <Routes>
          <Route path="/" element={<CommandCenter />} />
          <Route path="/traffic" element={<NetworkTraffic />} />
          <Route path="/timeline" element={<AttackTimeline />} />
          <Route path="/intelligence" element={<AttackIntelligence />} />
          <Route path="/explainability" element={<Explainability />} />
          <Route path="/replay" element={<HistoricalReplay />} />
          <Route path="/performance" element={<ModelPerformance />} />
          <Route path="/provenance" element={<DataProvenance />} />
        </Routes>
      </Layout>
    </Router>
  );
}
