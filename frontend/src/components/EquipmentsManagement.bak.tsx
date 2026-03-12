import { useState, useCallback } from 'react';
import PDFDropZone from './PDFDropZone';
import EquipmentForm from './EquipmentForm';
import { extractFromPDF } from '../services/api';
import type { EquipmentData } from '../services/api';

type ExtractionStatus = 'idle' | 'uploading' | 'processing' | 'success' | 'error';

function EquipmentManagement() {
    const [extractedData, setExtractedData] = useState<EquipmentData | null>(null);
    const [confidence, setConfidence] = useState<Record<string, number> | null>(null);
    const [status, setStatus] = useState<ExtractionStatus>('idle');
    const [statusMessage, setStatusMessage] = useState('');

    const handleFileSelect = useCallback(async (file: File) => {
        setStatus('uploading');
        setStatusMessage('Envoi du PDF au serveur...');
        setExtractedData(null);
        setConfidence(null);

        try {
            setStatus('processing');
            setStatusMessage('Analyse IA en cours... Cela peut prendre quelques minutes.');

            const response = await extractFromPDF(file);

            if (response.success && response.data) {
                setExtractedData(response.data);
                setConfidence(response.confidence ?? null);
                setStatus('success');
                setStatusMessage(
                    `Extraction réussie en ${response.processing_time_seconds?.toFixed(1)}s`
                );
            } else {
                setStatus('error');
                setStatusMessage(response.message || 'Erreur lors de l\'extraction');
            }

        } catch (err) {
            setStatus('error');
            setStatusMessage(
                err instanceof Error
                    ? `Erreur: ${err.message}. Vérifiez que le serveur backend est en cours d'exécution.`
                    : 'Erreur inconnue'
            );
        }
    }, []);

    return (
        <div className="space-y-6">
            {/* Page Header */}
            <div className="flex items-center gap-3">
                <div className="p-2.5 rounded-xl bg-blue-500 shadow-sm">
                    <svg className="w-6 h-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.066 2.573c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.573 1.066c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.066-2.573c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                        <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                    </svg>
                </div>
                <div>
                    <h1 className="text-xl font-bold text-gray-800">Gestion d'Équipement</h1>
                    <p className="text-sm text-gray-400">Importez un document et remplissez les informations</p>
                </div>
            </div>

            {/* Two-column layout */}
            <div className="grid grid-cols-1 xl:grid-cols-[minmax(0,1fr)_minmax(0,1.5fr)] gap-6 items-start">
                {/* Left: PDF Drop Zone */}
                <div className="sticky top-6">
                    <div className="mb-3">
                        <h2 className="text-sm font-semibold text-gray-600 flex items-center gap-2">
                            <span className="w-1 h-5 rounded-full bg-blue-500"></span>
                            Document PDF
                        </h2>
                        <p className="text-xs text-gray-400 mt-1 ml-3">Glissez-déposez ou cliquez pour importer</p>
                    </div>
                    <PDFDropZone onFileSelect={handleFileSelect} />

                    {/* Status Indicator */}
                    {status !== 'idle' && (
                        <div className={`mt-3 p-3 rounded-lg flex items-center gap-2 ${status === 'uploading' || status === 'processing'
                            ? 'bg-blue-50 border border-blue-200'
                            : status === 'success'
                                ? 'bg-green-50 border border-green-200'
                                : 'bg-red-50 border border-red-200'
                            }`}>
                            {(status === 'uploading' || status === 'processing') && (
                                <svg className="w-4 h-4 text-blue-600 animate-spin" fill="none" viewBox="0 0 24 24">
                                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                                </svg>
                            )}
                            {status === 'success' && (
                                <svg className="w-4 h-4 text-green-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                                    <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                                </svg>
                            )}
                            {status === 'error' && (
                                <svg className="w-4 h-4 text-red-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                                    <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                                </svg>
                            )}
                            <p className={`text-sm font-medium ${status === 'uploading' || status === 'processing'
                                ? 'text-blue-700'
                                : status === 'success'
                                    ? 'text-green-700'
                                    : 'text-red-700'
                                }`}>
                                {statusMessage}
                            </p>
                        </div>
                    )}
                </div>

                {/* Right: Equipment Form */}
                <div>
                    <div className="mb-3">
                        <h2 className="text-sm font-semibold text-gray-600 flex items-center gap-2">
                            <span className="w-1 h-5 rounded-full bg-blue-500"></span>
                            Fiche Équipement
                        </h2>
                        <p className="text-xs text-gray-400 mt-1 ml-3">
                            {extractedData
                                ? 'Champs remplis automatiquement par l\'IA — vérifiez et ajustez'
                                : 'Remplissez les champs librement'}
                        </p>
                    </div>
                    <EquipmentForm
                        extractedData={extractedData}
                        confidence={confidence}
                        isProcessing={status === 'uploading' || status === 'processing'}
                    />
                </div>
            </div>
        </div>
    );
}

export default EquipmentManagement;