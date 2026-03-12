import React, { useState, useCallback } from 'react';

interface PDFFile {
    file: File;
    id: string;
    status: 'pending' | 'processing' | 'completed' | 'error';
    progress: number;
}

interface PDFDropZoneProps {
    onFilesSelect: (files: File[]) => void;
    queuedFiles: PDFFile[];
    onRemoveFile: (id: string) => void;
    onExtractAll: () => void;
    isExtracting: boolean;
}

const PDFDropZone: React.FC<PDFDropZoneProps> = ({ 
    onFilesSelect, 
    queuedFiles, 
    onRemoveFile, 
    onExtractAll,
    isExtracting
}) => {
    const [isDragging, setIsDragging] = useState(false);

    const handleDragEnter = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();
        setIsDragging(true);
    }, []);

    const handleDragLeave = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();
        setIsDragging(false);
    }, []);

    const handleDragOver = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();
    }, []);

    const handleDrop = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();
        setIsDragging(false);

        const files = Array.from(e.dataTransfer.files).filter(f => f.type === 'application/pdf');
        if (files.length > 0) {
            onFilesSelect(files);
        }
    }, [onFilesSelect]);

    const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
        const files = Array.from(e.target.files || []).filter(f => f.type === 'application/pdf');
        if (files.length > 0) {
            onFilesSelect(files);
        }
    }, [onFilesSelect]);

    const handleClick = () => {
        if (!isExtracting) {
            document.getElementById('file-input')?.click();
        }
    };

    return (
        <div className="space-y-4">
            <div
                className={`
                    relative w-full p-8
                    bg-white/80 border-2 border-dashed border-slate-200 rounded-2xl
                    cursor-pointer transition-all duration-300
                    hover:border-blue-400 hover:bg-slate-50/50
                    ${isDragging ? 'border-blue-500 bg-blue-50/50 scale-[1.01]' : ''}
                    ${isExtracting ? 'opacity-50 cursor-not-allowed' : ''}
                `}
                onDragEnter={handleDragEnter}
                onDragLeave={handleDragLeave}
                onDragOver={handleDragOver}
                onDrop={handleDrop}
                onClick={handleClick}
            >
                <input
                    id="file-input"
                    type="file"
                    multiple
                    accept=".pdf,application/pdf"
                    onChange={handleFileInput}
                    className="hidden"
                    disabled={isExtracting}
                />

                <div className="flex flex-col items-center justify-center text-center py-4">
                    <div className={`mb-4 p-3 rounded-xl bg-blue-50 text-blue-500 transition-transform duration-300 ${isDragging ? 'scale-110' : ''}`}>
                        <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                            <path strokeLinecap="round" strokeLinejoin="round" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                        </svg>
                    </div>
                    <h3 className="text-sm font-semibold text-slate-700">
                        {queuedFiles.length > 0 ? 'Ajouter plus de PDFs' : 'Déposer vos PDFs ici'}
                    </h3>
                    <p className="text-xs text-slate-400 mt-1">ou cliquez pour parcourir</p>
                </div>

                {isDragging && (
                    <div className="absolute inset-0 bg-blue-500/10 rounded-2xl flex items-center justify-center backdrop-blur-[1px]">
                        <p className="text-blue-600 font-bold text-lg animate-bounce">Déposez les fichiers !</p>
                    </div>
                )}
            </div>

            {queuedFiles.length > 0 && (
                <div className="bg-white border border-slate-200 rounded-2xl overflow-hidden shadow-sm">
                    <div className="px-4 py-3 border-bottom border-slate-100 bg-slate-50/50 flex items-center justify-between">
                        <span className="text-[11px] font-bold uppercase tracking-wider text-slate-500">File d'attente ({queuedFiles.length})</span>
                        {!isExtracting && (
                            <button 
                                onClick={(e) => { e.stopPropagation(); onExtractAll(); }}
                                className="px-3 py-1 bg-blue-600 text-white text-[11px] font-bold rounded-full hover:bg-blue-700 transition-colors flex items-center gap-1.5"
                            >
                                <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
                                </svg>
                                Extraire tout
                            </button>
                        )}
                    </div>
                    <div className="divide-y divide-slate-100 max-h-[300px] overflow-y-auto">
                        {queuedFiles.map((pf) => (
                            <div key={pf.id} className="px-4 py-3 flex items-center gap-3 hover:bg-slate-50 transition-colors group">
                                <div className="p-2 rounded-lg bg-slate-100 text-slate-400">
                                    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                                    </svg>
                                </div>
                                <div className="flex-1 min-w-0">
                                    <div className="flex items-center justify-between gap-2">
                                        <p className="text-[13px] font-medium text-slate-700 truncate">{pf.file.name}</p>
                                        {!isExtracting && (
                                            <button 
                                                onClick={() => onRemoveFile(pf.id)}
                                                className="text-slate-300 hover:text-red-500 transition-colors opacity-0 group-hover:opacity-100"
                                            >
                                                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                                                </svg>
                                            </button>
                                        )}
                                    </div>
                                    <div className="mt-1 flex items-center gap-2">
                                        <span className="text-[10px] text-slate-400">{(pf.file.size / 1024 / 1024).toFixed(2)} MB</span>
                                        {pf.status !== 'pending' && (
                                            <>
                                                <div className="h-1 flex-1 bg-slate-100 rounded-full overflow-hidden">
                                                    <div 
                                                        className={`h-full transition-all duration-500 ${pf.status === 'error' ? 'bg-red-500' : 'bg-blue-500'}`}
                                                        style={{ width: `${pf.progress}%` }}
                                                    />
                                                </div>
                                                <span className={`text-[10px] font-bold ${pf.status === 'error' ? 'text-red-500' : 'text-blue-500'}`}>
                                                    {pf.status === 'completed' ? '✓' : pf.status === 'error' ? '!' : `${pf.progress}%`}
                                                </span>
                                            </>
                                        )}
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
};

export default PDFDropZone;
