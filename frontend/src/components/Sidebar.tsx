import React from 'react';
import mxlab from '../assets/mx_lab.png';
import { FileText } from 'lucide-react';

interface SidebarProps {
  activeView: string;
  onViewChange: (view: string) => void;
}

const Sidebar: React.FC<SidebarProps> = ({ activeView, onViewChange }) => {
  const menuItems = [
    {
      id: 'pdf-extractor',
      label: 'PDF Extractor',
      icon: FileText
    }
  ];

  return (
    <div className="w-72 h-screen bg-gradient-to-b from-slate-50 to-gray-100 shadow-xl flex flex-col fixed left-0 top-0 z-50">
      {/* Header */}
      <div className="p-6 border-b border-blue-200 bg-gradient-to-br from-blue-500 via-blue-600 to-indigo-700 shadow-lg">
        <div className="flex items-center gap-4">
          <div className="w-12 h-12 bg-white/10 rounded-xl flex items-center justify-center backdrop-blur-sm ring-1 ring-white/20 shadow-inner">
            <img src={mxlab} alt="MXLAB" className="w-8 h-8 object-contain" />
          </div>
          <div>
            <h1 className="font-bold text-white text-lg tracking-tight drop-shadow-sm">MX Lab</h1>
            <p className="text-blue-100 text-xs font-medium">Tools Suite</p>
          </div>
        </div>
      </div>

      {/* Menu Items */}
      <nav className="flex-1 p-4 overflow-y-auto">
        <ul className="space-y-2">
          {menuItems.map((item) => {
            const isActive = activeView === item.id;
            return (
              <li key={item.id}>
                <button
                  onClick={() => onViewChange(item.id)}
                  className={`w-full flex items-center gap-3 px-4 py-3.5 rounded-xl text-left transition-all duration-300 group ${isActive
                    ? 'bg-gradient-to-r from-gray-800 to-gray-700 text-white shadow-lg shadow-gray-200 ring-1 ring-gray-700/50 transform scale-[1.02]'
                    : 'text-slate-600 hover:bg-white hover:text-blue-600 hover:shadow-md hover:shadow-gray-100 hover:-translate-y-0.5'
                    }`}
                >
                  <div className={`p-2 rounded-lg transition-colors duration-300 ${isActive ? 'bg-white/10' : 'bg-gray-100 group-hover:bg-blue-50 text-slate-500 group-hover:text-blue-600'
                    }`}>
                    <item.icon className="w-5 h-5" />
                  </div>
                  <span className="font-medium text-sm tracking-wide">{item.label}</span>

                  {isActive && (
                    <div className="ml-auto w-1.5 h-1.5 rounded-full bg-blue-400 shadow-[0_0_8px_rgba(96,165,250,0.6)] animate-pulse" />
                  )}
                </button>
              </li>
            );
          })}
        </ul>
      </nav>

      {/* Footer */}
      <div className="p-4 border-t border-gray-200 bg-gray-50/50 backdrop-blur-sm">
        <div className="flex items-center gap-3 px-2 py-2 rounded-xl hover:bg-white hover:shadow-sm transition-all cursor-pointer group">
          <div className="w-9 h-9 rounded-full bg-gradient-to-tr from-blue-500 to-indigo-500 flex items-center justify-center text-white font-semibold text-sm shadow-md group-hover:scale-110 transition-transform">
            U
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-sm font-semibold text-gray-700 truncate">Utilisateur</p>
            <p className="text-xs text-gray-500 truncate">Admin Access</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Sidebar;