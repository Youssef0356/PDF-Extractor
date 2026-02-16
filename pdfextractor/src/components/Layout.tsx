import React from 'react';
import Sidebar from './Sidebar';

interface LayoutProps {
  children: React.ReactNode;
  activeView: string;
  onViewChange: (view: string) => void;
}

const Layout: React.FC<LayoutProps> = ({ children, activeView, onViewChange }) => {
  return (
    <div className="flex h-screen bg-slate-50 font-sans">
      <Sidebar activeView={activeView} onViewChange={onViewChange} />
      <div className="flex-1 flex flex-col ml-72 overflow-hidden relative">
        {/* Top/Header Area - subtle separation */}
        <header className="h-16 bg-white/80 backdrop-blur-md border-b border-gray-200/60 sticky top-0 z-40 flex items-center justify-between px-8">
          <div className="flex items-center gap-2">
            <span className="bg-blue-100 text-blue-700 text-xs font-semibold px-2.5 py-0.5 rounded-full border border-blue-200">
              v1.0.0
            </span>
            <span className="text-gray-400 text-sm">|</span>
            <span className="text-sm font-medium text-gray-500">PDF Processing Unit</span>
          </div>
          <div className="flex items-center gap-4">
            {/* Add topbar actions if needed */}
          </div>
        </header>

        <main className="flex-1 overflow-auto p-8 scroll-smooth">
          <div className="max-w-7xl mx-auto animate-fade-in-up">
            {children}
          </div>
        </main>
      </div>
    </div>
  );
};

export default Layout;
