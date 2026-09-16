import React from 'react';
import { NavLink } from 'react-router-dom';
import { LayoutDashboard, History, Shield, BrainCircuit, PlaySquare, Settings, FileSearch, Activity, Database, Clock } from 'lucide-react';
import { cn } from '../../lib/utils';

export function Sidebar() {
  const routes = [
    { name: 'Command Center', path: '/', icon: LayoutDashboard },
    { name: 'Network Traffic', path: '/traffic', icon: Activity },
    { name: 'Attack Timeline', path: '/timeline', icon: Clock },
    { name: 'ATT&CK Intelligence', path: '/intelligence', icon: Shield },
    { name: 'Explainability', path: '/explainability', icon: FileSearch },
    { name: 'Historical Replay', path: '/replay', icon: PlaySquare },
    { name: 'Model Performance', path: '/performance', icon: BrainCircuit },
    { name: 'Data Provenance', path: '/provenance', icon: Database },
  ];

  return (
    <div className="flex flex-col w-64 border-r border-border bg-card text-card-foreground">
      <div className="p-6">
        <h1 className="text-xl font-bold tracking-tight">CYBERCAST</h1>
        <p className="text-xs text-muted-foreground mt-1 tracking-wider uppercase">Command Center</p>
      </div>

      <div className="flex-1 px-4 space-y-2">
        {routes.map(r => (
          <NavLink
            key={r.path}
            to={r.path}
            className={({ isActive }) => 
              cn(
                "flex items-center gap-3 px-3 py-2 rounded-md text-sm transition-colors",
                isActive ? "bg-secondary text-secondary-foreground font-medium" : "text-muted-foreground hover:bg-secondary/50 hover:text-foreground"
              )
            }
          >
            <r.icon className="w-4 h-4" />
            {r.name}
          </NavLink>
        ))}
      </div>

      <div className="p-6 border-t border-border">
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <Settings className="w-4 h-4" />
          <span>v2.3 (Frozen Metrics)</span>
        </div>
      </div>
    </div>
  );
}
