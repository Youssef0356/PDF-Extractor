import React, { useState, useCallback } from 'react';
import PDFDropZone from './PDFDropZone';
import EquipmentForm from './EquipmentForm';

function EquipmentManagement() {
    const [selectedFile, setSelectedFile] = useState<File | null>(null);

    const handleFileSelect = useCallback((file: File) => {
        console.log('File selected:', file);
        setSelectedFile(file);
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
                    {selectedFile && (
                        <div className="mt-3 p-3 bg-green-50 border border-green-200 rounded-lg flex items-center gap-2">
                            <svg className="w-4 h-4 text-green-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                                <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                            </svg>
                            <p className="text-sm font-medium text-green-700 truncate">{selectedFile.name}</p>
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
                        <p className="text-xs text-gray-400 mt-1 ml-3">Remplissez les champs progressivement</p>
                    </div>
                    <EquipmentForm />
                </div>
            </div>
        </div>
    );
}

export default EquipmentManagement;