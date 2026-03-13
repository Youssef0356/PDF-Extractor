import { useState, useCallback } from 'react';
import PDFDropZone from './PDFDropZone';
import EquipmentForm from './EquipmentForm';
import { extractFromPDF } from '../services/api';
import type { EquipmentData } from '../services/api';

interface PDFFile {
    file: File;
    id: string;
    status: 'pending' | 'processing' | 'completed' | 'error';
    progress: number;
}

function EquipmentManagement() {
    const [extractedData, setExtractedData] = useState<EquipmentData | null>(null);
    const [confidence, setConfidence] = useState<Record<string, number> | null>(null);
    const [docType, setDocType] = useState<string>('');
    const [queuedFiles, setQueuedFiles] = useState<PDFFile[]>([]);
    const [isExtracting, setIsExtracting] = useState(false);
    const [globalStatus, setGlobalStatus] = useState<string>('');

    const handleFilesSelect = useCallback((files: File[]) => {
        const newFiles = files.map(file => ({
            file,
            id: Math.random().toString(36).substring(7),
            status: 'pending' as const,
            progress: 0
        }));
        setQueuedFiles(prev => [...prev, ...newFiles]);
    }, []);

    const removeFile = useCallback((id: string) => {
        setQueuedFiles(prev => prev.filter(f => f.id !== id));
    }, []);

    const handleFormReset = useCallback(() => {
        setExtractedData(null);
        setConfidence(null);
        setDocType('');
        setQueuedFiles([]);
        setGlobalStatus('');
    }, []);

    const extractAll = useCallback(async () => {
        if (queuedFiles.length === 0 || isExtracting) return;

        setIsExtracting(true);
        setExtractedData(null);
        setConfidence(null);
        setDocType('');

        const filesToProcess = [...queuedFiles];
        
        for (let i = 0; i < filesToProcess.length; i++) {
            const pf = filesToProcess[i];
            
            // Update status to processing
            setQueuedFiles(prev => prev.map(f => 
                f.id === pf.id ? { ...f, status: 'processing', progress: 10 } : f
            ));
            setGlobalStatus(`Analyse de ${pf.file.name} (${i + 1}/${filesToProcess.length})...`);

            try {
                // Simulated progress steps
                const progressInterval = setInterval(() => {
                    setQueuedFiles(prev => prev.map(f => {
                        if (f.id === pf.id && f.progress < 90) {
                            return { ...f, progress: f.progress + 5 };
                        }
                        return f;
                    }));
                }, 500);

                const response = await extractFromPDF(pf.file);
                clearInterval(progressInterval);

                if (response.success && response.data) {
                    setDocType(response.doc_context?.doc_type || '');
                    setQueuedFiles(prev => prev.map(f => 
                        f.id === pf.id ? { ...f, status: 'completed', progress: 100 } : f
                    ));
                    
                    // Issue 1: Merge data based on confidence scores
                    setExtractedData(prevData => {
                        if (!prevData) return response.data!;
                        
                        const newData = { ...prevData };
                        const newConfidence = response.confidence || {};
                        
                        setConfidence(prevConf => {
                            const mergedConf = { ...(prevConf || {}) };
                            
                            // Iterate over all fields in the new data
                            Object.keys(response.data!).forEach(key => {
                                const k = key as keyof EquipmentData;
                                const curConf = mergedConf[k] || 0;
                                const nextConf = newConfidence[k] || 0;
                                
                                // Only override if new confidence is strictly higher
                                if (nextConf > curConf) {
                                    (newData as any)[k] = response.data![k];
                                    mergedConf[k] = nextConf;
                                }
                            });
                            
                            return mergedConf;
                        });
                        
                        return newData;
                    });
                } else {
                    setQueuedFiles(prev => prev.map(f => 
                        f.id === pf.id ? { ...f, status: 'error', progress: 100 } : f
                    ));
                }
            } catch (err) {
                setQueuedFiles(prev => prev.map(f => 
                    f.id === pf.id ? { ...f, status: 'error', progress: 100 } : f
                ));
            }
        }

        setIsExtracting(false);
        setGlobalStatus('Traitement terminé');
    }, [queuedFiles, isExtracting]);

    return (
        <div className="space-y-6">
            {/* Page Header */}
            <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                    <div className="p-2.5 rounded-xl bg-blue-500 shadow-sm text-white">
                        <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                            <path strokeLinecap="round" strokeLinejoin="round" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.066 2.573c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.573 1.066c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.066-2.573c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                            <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                        </svg>
                    </div>
                    <div>
                        <h1 className="text-xl font-bold text-slate-800">Gestion d'Équipement</h1>
                        <p className="text-xs text-slate-400">Importez des documents techniques et extrayez les données</p>
                    </div>
                </div>
                {isExtracting && (
                    <div className="flex items-center gap-3 px-4 py-2 bg-blue-50 rounded-full border border-blue-100 animate-pulse">
                        <div className="w-2 h-2 rounded-full bg-blue-500" />
                        <span className="text-[11px] font-bold text-blue-700 uppercase tracking-wider">{globalStatus}</span>
                    </div>
                )}
            </div>

            {/* Two-column layout */}
            <div className="grid grid-cols-1 xl:grid-cols-[380px_1fr] gap-8 items-start">
                {/* Left: PDF Drop Zone & Queue */}
                <div className="space-y-6">
                    <div>
                        <h2 className="text-[11px] font-bold text-slate-400 uppercase tracking-widest mb-3 flex items-center gap-2">
                            <span className="w-1 h-4 rounded-full bg-blue-500"></span>
                            Documents Source
                        </h2>
                        <PDFDropZone 
                            onFilesSelect={handleFilesSelect} 
                            queuedFiles={queuedFiles}
                            onRemoveFile={removeFile}
                            onExtractAll={extractAll}
                            isExtracting={isExtracting}
                        />
                    </div>
                </div>

                {/* Right: Equipment Form */}
                <div className="bg-white border border-slate-200 rounded-3xl p-8 shadow-sm">
                    <div className="mb-6 flex items-center justify-between">
                        <div>
                            <h2 className="text-[11px] font-bold text-slate-400 uppercase tracking-widest mb-1 flex items-center gap-2">
                                <span className="w-1 h-4 rounded-full bg-blue-500"></span>
                                Fiche Technique
                            </h2>
                            <p className="text-xs text-slate-400">
                                {extractedData 
                                    ? 'Données extraites par l\'IA — vérifiez les champs avec un badge orange' 
                                    : 'En attente d\'un document pour l\'extraction automatique'}
                            </p>
                        </div>
                    </div>
                    <EquipmentForm
                        extractedData={extractedData}
                        confidence={confidence}
                        isProcessing={isExtracting}
                        docType={docType}
                        onReset={handleFormReset}
                    />
                </div>
            </div>
        </div>
    );
}

export default EquipmentManagement;
