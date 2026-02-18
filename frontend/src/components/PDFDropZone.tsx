import React, { useState, useCallback } from 'react';

interface PDFDropZoneProps {
    onFileSelect: (file: File) => void;
}

const PDFDropZone: React.FC<PDFDropZoneProps> = ({ onFileSelect }) => {
    const [isDragging, setIsDragging] = useState(false);
    const [selectedFile, setSelectedFile] = useState<File | null>(null);

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

        const files = e.dataTransfer.files;
        if (files && files.length > 0) {
            const file = files[0];
            if (file.type === 'application/pdf') {
                setSelectedFile(file);
                onFileSelect(file);
            } else {
                alert('Please drop a PDF file');
            }
        }
    }, [onFileSelect]);

    const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
        const files = e.target.files;
        if (files && files.length > 0) {
            const file = files[0];
            if (file.type === 'application/pdf') {
                setSelectedFile(file);
                onFileSelect(file);
            } else {
                alert('Please select a PDF file');
            }
        }
    }, [onFileSelect]);

    const handleClick = () => {
        document.getElementById('file-input')?.click();
    };

    return (
        <div
            className={`
                relative w-full max-w-[700px] min-h-[450px] p-12
                bg-gradient-to-br from-white/95 to-gray-50/95
                border-3 border-dashed border-indigo-500/30 rounded-3xl
                cursor-pointer transition-all duration-400 ease-[cubic-bezier(0.4,0,0.2,1)]
                backdrop-blur-md shadow-[0_10px_40px_rgba(0,0,0,0.08)]
                hover:border-indigo-500/60 hover:-translate-y-1 hover:shadow-[0_20px_60px_rgba(99,102,241,0.15)]
                ${isDragging ? '!border-indigo-500 bg-gradient-to-br from-indigo-500/10 to-violet-500/10 !scale-[1.02] shadow-[0_20px_60px_rgba(99,102,241,0.25)]' : ''}
                ${selectedFile ? '!border-emerald-500/50 bg-gradient-to-br from-emerald-500/5 to-emerald-600/5' : ''}
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
                accept=".pdf,application/pdf"
                onChange={handleFileInput}
                style={{ display: 'none' }}
            />

            <div className="flex flex-col items-center justify-center min-h-[350px] text-center">
                {selectedFile ? (
                    <>
                        <div className="mb-8 animate-scale-in">
                            <svg width="64" height="64" viewBox="0 0 64 64" fill="none">
                                <circle cx="32" cy="32" r="32" fill="url(#successGradient)" />
                                <path d="M20 32l8 8 16-16" stroke="white" strokeWidth="4" strokeLinecap="round" strokeLinejoin="round" />
                                <defs>
                                    <linearGradient id="successGradient" x1="0" y1="0" x2="64" y2="64">
                                        <stop offset="0%" stopColor="#10b981" />
                                        <stop offset="100%" stopColor="#059669" />
                                    </linearGradient>
                                </defs>
                            </svg>
                        </div>
                        <h3 className="text-3xl font-bold mb-3 bg-gradient-to-br from-emerald-500 to-emerald-600 bg-clip-text text-transparent">File Selected!</h3>
                        <p className="text-xl font-semibold text-gray-800 my-2 break-all max-w-full">{selectedFile.name}</p>
                        <p className="text-base text-gray-500 my-1">{(selectedFile.size / 1024 / 1024).toFixed(2)} MB</p>
                    </>
                ) : (
                    <>
                        <div className="mb-8 animate-float">
                            <svg width="80" height="80" viewBox="0 0 80 80" fill="none">
                                <rect x="20" y="10" width="40" height="50" rx="4" stroke="url(#iconGradient)" strokeWidth="3" fill="none" />
                                <path d="M30 25h20M30 35h20M30 45h15" stroke="url(#iconGradient)" strokeWidth="3" strokeLinecap="round" />
                                <path d="M40 55v10m-5-5l5 5 5-5" stroke="url(#iconGradient)" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" />
                                <defs>
                                    <linearGradient id="iconGradient" x1="20" y1="10" x2="60" y2="70">
                                        <stop offset="0%" stopColor="#0084ffff" />
                                        <stop offset="100%" stopColor="#5cb8f6ff" />
                                    </linearGradient>
                                </defs>
                            </svg>
                        </div>
                        <h3 className="text-3xl font-bold mb-3 bg-gradient-to-br from-blue-600 to-blue-300 bg-clip-text text-transparent">Drop your PDF here</h3>
                        <p className="text-lg text-gray-500 mb-8 font-medium">or click to browse</p>
                        <div className="flex flex-col items-center gap-3 mt-6">
                            <span className="bg-gradient-to-br from-blue-500 to-blue-300 text-white px-6 py-2 rounded-full font-semibold text-sm tracking-wide shadow-lg shadow-blue-500/30">PDF</span>
                            <span className="text-sm text-gray-400">Maximum file size: 50MB</span>
                        </div>
                    </>
                )}
            </div>

            {isDragging && (
                <div className="absolute inset-0 bg-gradient-to-br from-indigo-500/95 to-violet-500/95 rounded-3xl flex items-center justify-center animate-fade-in">
                    <div className="flex flex-col items-center gap-6">
                        <svg width="100" height="100" viewBox="0 0 100 100" fill="none">
                            <circle cx="50" cy="50" r="45" stroke="white" strokeWidth="4" strokeDasharray="8 8" />
                            <path d="M50 30v40m-15-15l15 15 15-15" stroke="white" strokeWidth="4" strokeLinecap="round" strokeLinejoin="round" />
                        </svg>
                        <p className="text-white text-3xl font-bold m-0 drop-shadow-md">Drop it!</p>
                    </div>
                </div>
            )}
        </div>
    );
};

export default PDFDropZone;
