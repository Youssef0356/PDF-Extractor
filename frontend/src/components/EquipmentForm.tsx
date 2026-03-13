import React, { useState, useEffect, useMemo } from 'react';
import { 
    fetchSchemaOptions, 
    extractFromPDF, 
    submitCorrections 
} from '../services/api';
import type { 
    InstrumentData,
    SchemaOptions,
    CorrectionBatch,
    CorrectionRecord
} from '../services/api';

// ---------------------------------------------------------------------------
// Form Layout Sections (V2)
// ---------------------------------------------------------------------------

interface FormField {
    key: keyof InstrumentData;
    label: string;
    type: 'select' | 'text' | 'number' | 'boolean' | 'multi-select' | 'custom';
    placeholder?: string;
    options?: string[];
    dependsOn?: string;
    condition?: (values: InstrumentData) => boolean;
}

interface FormSection {
    title: string;
    fields: FormField[];
}

const SECTIONS: FormSection[] = [
    {
        title: "Identification de l'instrument",
        fields: [
            { key: 'category', label: 'Catégorie', type: 'select' },
            { key: 'typeMesure', label: 'Type de Mesure', type: 'select', condition: (v) => v.category === 'Transmetteur/Capteur' },
            { key: 'typeActionneur', label: 'Type d\'Actionneur', type: 'select', condition: (v) => v.category === 'Actionneur' },
            { key: 'code', label: 'Code ISA', type: 'select' },
            { key: 'technologie', label: 'Technologie', type: 'select', condition: (v) => v.category === 'Transmetteur/Capteur' },
            { key: 'codeTechnologie', label: 'Code Technologie', type: 'text' },
            { key: 'marque', label: 'Marque', type: 'select' },
            { key: 'référence', label: 'Référence', type: 'text' },
        ]
    },
    {
        title: "Plage de Mesure",
        fields: [
            { key: 'plageMesureMin', label: 'Minimum', type: 'number', placeholder: '0.0' },
            { key: 'plageMesureMax', label: 'Maximum', type: 'number', placeholder: '100.0' },
            { key: 'plageMesureUnite', label: 'Unité', type: 'text', placeholder: 'bar, m3/h, °C...' },
            { key: 'precision', label: 'Précision', type: 'text', placeholder: '±0.5%...' },
        ]
    },
    {
        title: "Signal & Raccordement",
        fields: [
            { key: 'signalSortie', label: 'Signal de Sortie', type: 'select', condition: (v) => v.category === 'Transmetteur/Capteur' },
            { key: 'hart', label: 'HART', type: 'boolean', condition: (v) => v.category === 'Transmetteur/Capteur' },
            { key: 'alimentation', label: 'Alimentation', type: 'select', condition: (v) => v.category === 'Transmetteur/Capteur' },
            { key: 'communication', label: 'Communication', type: 'select', condition: (v) => v.category === 'Transmetteur/Capteur' && v.alimentation !== 'boucle' },
            { key: 'nombreFils', label: 'Nombre de fils', type: 'select', condition: (v) => v.category === 'Transmetteur/Capteur' },
            { key: 'repérageArmoire', label: 'Repérage Armoire', type: 'text' },
        ]
    },
    {
        title: "Actionneur - Spécifications",
        fields: [
            { key: 'typeVérin', label: 'Type de Vérin', type: 'select', condition: (v) => v.category === 'Actionneur' },
            { key: 'typeActionneurSpécial', label: 'Actionneur Spécial', type: 'select', condition: (v) => v.category === 'Actionneur' },
            { key: 'positionSécurité', label: 'Position de Sécurité', type: 'select', condition: (v) => v.category === 'Actionneur' },
            { key: 'courseMM', label: 'Course (mm)', type: 'number', condition: (v) => v.category === 'Actionneur' },
            { key: 'forceN', label: 'Force (N)', type: 'number', condition: (v) => v.category === 'Actionneur' },
            { key: 'pressionAlimentationBar', label: 'Pression Alim (bar)', type: 'number', condition: (v) => v.category === 'Actionneur' },
        ]
    },
    {
        title: "Seuil & TOR",
        fields: [
            { key: 'sortieTOR', label: 'Sortie TOR', type: 'boolean' },
            { key: 'seuil', label: 'Valeur de seuil', type: 'number', condition: (v) => !!v.sortieTOR },
            { key: 'seuilUnite', label: 'Unité seuil', type: 'text', condition: (v) => !!v.sortieTOR },
        ]
    },
    {
        title: "Installation & Matériaux",
        fields: [
            { key: 'indiceIP', label: 'Indice IP', type: 'select' },
            { key: 'certificats', label: 'Certificats', type: 'multi-select' },
            { key: 'températureProcess', label: 'Température Process', type: 'text', placeholder: '-40...+120°C' },
            { key: 'matériauMembrane', label: 'Matériau Membrane', type: 'select', condition: (v) => v.category === 'Transmetteur/Capteur' && ['Pression', 'Niveau'].includes(v.typeMesure || '') },
        ]
    }
];

// Helper to get all field keys
const ALL_FIELD_KEYS = SECTIONS.flatMap(s => s.fields.map(f => f.key));

interface EquipmentFormProps {
    initialData?: InstrumentData | null;
    confidence?: Record<string, number> | null;
    isProcessing?: boolean;
    onReset?: () => void;
}

const EquipmentForm = ({ 
    initialData, 
    confidence: initialConfidence, 
    isProcessing: externalProcessing,
    onReset 
}: EquipmentFormProps) => {
    // --- State ---
    const [options, setOptions] = useState<SchemaOptions | null>(null);
    const [values, setValues] = useState<InstrumentData>({});
    const [aiValues, setAiValues] = useState<InstrumentData>({});
    const [confidence, setConfidence] = useState<Record<string, number>>({});
    const [isInternalProcessing, setIsInternalProcessing] = useState(false);
    const [pdfId, setPdfId] = useState<string | null>(null);
    
    const isProcessing = externalProcessing || isInternalProcessing;
    
    // UI state
    const [showCorrectionModal, setShowCorrectionModal] = useState(false);
    const [pendingCorrections, setPendingCorrections] = useState<CorrectionRecord[]>([]);
    const [showSavedPopup, setShowSavedPopup] = useState(false);

    // --- Load Options ---
    useEffect(() => {
        fetchSchemaOptions().then(setOptions).catch(err => console.error("Failed to load options", err));
    }, []);

    // --- Sync with external props ---
    useEffect(() => {
        if (initialData) {
            setValues(initialData);
            setAiValues(initialData);
        }
        if (initialConfidence) {
            setConfidence(initialConfidence as Record<string, number>);
        }
    }, [initialData, initialConfidence]);

    // --- Action Handlers ---
    const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (!file) return;

        setIsInternalProcessing(true);
        
        try {
            const res = await extractFromPDF(file);
            if (res.success && res.data) {
                setAiValues(res.data);
                setValues(res.data);
                setConfidence(res.confidence || {});
                setPdfId(res.meta?.pdf_id || null);
            }
        } catch (err) {
            console.error(err);
        } finally {
            setIsInternalProcessing(false);
        }
    };

    const handleChange = (key: keyof InstrumentData, value: any) => {
        setValues(prev => ({ ...prev, [key]: value }));
    };

    const handleReset = () => {
        setValues({});
        setAiValues({});
        setConfidence({});
        setPdfId(null);
        if (onReset) onReset();
    };

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        
        // Detect corrections
        const corrections: CorrectionRecord[] = [];
        ALL_FIELD_KEYS.forEach(key => {
            const userVal = values[key];
            const aiVal = aiValues[key];
            
            // Compare stringified for simple match
            if (JSON.stringify(userVal) !== JSON.stringify(aiVal)) {
                corrections.push({
                    field_name: key as string,
                    ai_extracted_value: aiVal,
                    user_corrected_value: userVal,
                    accepted: false
                });
            }
        });

        if (corrections.length > 0) {
            setPendingCorrections(corrections);
            setShowCorrectionModal(true);
        } else {
            finalizeSubmission([]);
        }
    };

    const finalizeSubmission = async (corrections: CorrectionRecord[]) => {
        if (pdfId) {
            const batch: CorrectionBatch = {
                pdf_id: pdfId,
                category: values.category,
                typeMesure: values.typeMesure,
                corrections
            };
            await submitCorrections(batch);
        }
        setShowCorrectionModal(false);
        setShowSavedPopup(true);
        setTimeout(() => setShowSavedPopup(false), 3000);
    };

    // --- Computed ---
    const dashboardMetrics = useMemo(() => {
        let confirmed = 0;
        let uncertain = 0;
        let empty = 0;

        ALL_FIELD_KEYS.forEach(k => {
            if (!values[k]) empty++;
            else if (confidence[k as string] >= 0.90) confirmed++;
            else uncertain++;
        });

        return { confirmed, uncertain, empty, total: ALL_FIELD_KEYS.length };
    }, [values, confidence]);

    // --- Render Helpers ---
    const getOptionsForField = (field: FormField): string[] => {
        if (!options) return [];
        switch (field.key) {
            case 'category': return options.categories;
            case 'typeMesure': return options.typesMesure;
            case 'typeActionneur': return options.typesActionneur;
            case 'code': return options.codes[values.typeMesure || values.typeActionneur || ''] || [];
            case 'technologie': return options.technologies[values.typeMesure || ''] || [];
            case 'signalSortie': return options.signals;
            case 'alimentation': return options.powers;
            case 'communication': return options.communications;
            case 'marque': return options.brands;
            case 'indiceIP': return options.indicesIP;
            case 'matériauMembrane': return options.materials;
            case 'typeVérin': return options.actuatorTypes;
            case 'typeActionneurSpécial': return options.specialActuatorTypes;
            case 'positionSécurité': return options.safetyPositions;
            case 'nombreFils': return ['2', '3', '4', '5'];
            default: return field.options || [];
        }
    };

    const renderField = (field: FormField) => {
        if (field.condition && !field.condition(values)) return null;

        const val = values[field.key];
        const hasAiResult = aiValues[field.key] !== undefined;
        const isModified = JSON.stringify(val) !== JSON.stringify(aiValues[field.key]);
        const conf = confidence[field.key as string] || 0;

        const inputBaseClass = "w-full px-4 py-2.5 rounded-xl border-2 transition-all duration-200 outline-none text-sm font-medium ";
        const borderClass = isModified 
            ? "border-amber-200 focus:border-amber-400 bg-amber-50/30" 
            : (hasAiResult ? "border-blue-100 focus:border-blue-500 bg-blue-50/10" : "border-slate-100 focus:border-blue-500 bg-white");

        const label = (
            <div className="flex items-center justify-between mb-1.5 px-1">
                <span className="text-[11px] font-bold uppercase tracking-wider text-slate-400">{field.label}</span>
                {hasAiResult && (
                    <div className={`text-[10px] font-bold px-1.5 py-0.5 rounded-full ${conf >= 0.9 ? 'bg-emerald-50 text-emerald-600' : 'bg-amber-50 text-amber-600'}`}>
                        {Math.round(conf * 100)}%
                    </div>
                )}
            </div>
        );

        switch (field.type) {
            case 'select':
                const opts = getOptionsForField(field);
                return (
                    <div key={field.key} className="space-y-1">
                        {label}
                        <select 
                            className={inputBaseClass + borderClass}
                            value={String(val || '')}
                            onChange={(e) => handleChange(field.key, e.target.value)}
                        >
                            <option value="">Sélectionner...</option>
                            {opts.map(o => <option key={o} value={o}>{o}</option>)}
                        </select>
                    </div>
                );
            case 'number':
                return (
                    <div key={field.key} className="space-y-1">
                        {label}
                        <input 
                            type="number" 
                            step="any"
                            className={inputBaseClass + borderClass}
                            value={val === undefined ? '' : String(val)}
                            placeholder={field.placeholder}
                            onChange={(e) => handleChange(field.key, e.target.value === '' ? undefined : parseFloat(e.target.value))}
                        />
                    </div>
                );
            case 'boolean':
                return (
                    <div key={field.key} className="flex items-center justify-between p-3 rounded-xl border-2 border-slate-50 bg-slate-50/30">
                        <span className="text-xs font-bold text-slate-600">{field.label}</span>
                        <button 
                            type="button"
                            onClick={() => handleChange(field.key, !val)}
                            className={`w-12 h-6 rounded-full transition-all duration-300 relative ${val ? 'bg-blue-600' : 'bg-slate-300'}`}
                        >
                            <div className={`absolute top-1 w-4 h-4 bg-white rounded-full transition-all duration-300 ${val ? 'left-7' : 'left-1'}`} />
                        </button>
                    </div>
                );
            case 'multi-select':
                const currentVals = Array.isArray(val) ? val : [];
                const certs = ['ATEX', 'IECEx', 'SIL', 'SIL 2', 'SIL 3', 'FM', 'CSA'];
                return (
                    <div key={field.key} className="space-y-1 col-span-1 md:col-span-2">
                        {label}
                        <div className="flex flex-wrap gap-2">
                            {certs.map(c => (
                                <button
                                    key={c}
                                    type="button"
                                    onClick={() => {
                                        const next = currentVals.includes(c) ? currentVals.filter(v => v !== c) : [...currentVals, c];
                                        handleChange(field.key, next);
                                    }}
                                    className={`px-3 py-1.5 rounded-lg text-xs font-bold border-2 transition-all ${currentVals.includes(c) ? 'bg-blue-600 border-blue-600 text-white shadow-md' : 'bg-white border-slate-100 text-slate-400 hover:border-slate-200'}`}
                                >
                                    {c}
                                </button>
                            ))}
                        </div>
                    </div>
                );
            default:
                return (
                    <div key={field.key} className="space-y-1">
                        {label}
                        <input 
                            type="text" 
                            className={inputBaseClass + borderClass}
                            value={String(val || '')}
                            placeholder={field.placeholder}
                            onChange={(e) => handleChange(field.key, e.target.value)}
                        />
                    </div>
                );
        }
    };

    return (
        <div className="max-w-6xl mx-auto p-6 space-y-8 animate-in fade-in duration-700">
            {/* Header: Upload & Stats */}
            <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-6 bg-white p-8 rounded-3xl shadow-xl shadow-slate-200/50 border border-slate-100">
                <div className="space-y-2">
                    <h1 className="text-3xl font-black text-slate-900 tracking-tight">PDF Extractor <span className="text-blue-600">V2</span></h1>
                    <p className="text-slate-400 font-medium">Extractions intelligentes pour instrumentation industrielle</p>
                </div>
                
                <div className="flex items-center gap-4 w-full md:w-auto">
                    <label className={`
                        flex-1 md:flex-none px-8 py-4 rounded-2xl font-bold text-sm cursor-pointer transition-all active:scale-95
                        flex items-center justify-center gap-3 shadow-lg shadow-blue-500/20
                        ${isProcessing ? 'bg-slate-100 text-slate-400 grayscale cursor-not-allowed' : 'bg-blue-600 text-white hover:bg-blue-700 hover:shadow-blue-500/30'}
                    `}>
                        {isProcessing ? (
                            <div className="w-5 h-5 border-2 border-slate-300 border-t-blue-600 rounded-full animate-spin" />
                        ) : (
                            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" /></svg>
                        )}
                        <span>{isProcessing ? 'Analyse en cours...' : 'Charger une Datasheet'}</span>
                        <input type="file" className="hidden" accept=".pdf" onChange={handleFileChange} disabled={isProcessing} />
                    </label>
                </div>
            </div>

            {/* Dashboard & Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
                {/* Left: Form Content */}
                <form onSubmit={handleSubmit} className="lg:col-span-8 space-y-6">
                    {SECTIONS.map(section => (
                        <div key={section.title} className="bg-white/70 backdrop-blur-sm p-8 rounded-3xl border border-white shadow-sm space-y-6">
                            <h3 className="text-sm font-black text-slate-800 uppercase tracking-widest border-l-4 border-blue-600 pl-4">{section.title}</h3>
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-x-6 gap-y-5">
                                {section.fields.map(field => renderField(field))}
                            </div>
                        </div>
                    ))}
                    
                    <div className="sticky bottom-6 flex justify-end gap-4">
                        <button 
                            type="button"
                            onClick={handleReset}
                            className="px-8 py-5 bg-white text-slate-400 border-2 border-slate-100 rounded-2xl font-black text-sm hover:bg-slate-50 transition-all active:scale-95"
                        >
                            RÉINITIALISER
                        </button>
                        <button 
                            type="submit"
                            className="px-12 py-5 bg-slate-900 text-white rounded-2xl font-black text-lg shadow-2xl shadow-slate-900/40 hover:bg-slate-800 transition-all active:scale-95 flex items-center gap-3 group"
                        >
                            ENREGISTRER L'ÉQUIPEMENT
                            <svg className="w-5 h-5 group-hover:translate-x-1 transition-transform" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M13 7l5 5m0 0l-5 5m5-5H6" /></svg>
                        </button>
                    </div>
                </form>

                {/* Right: Status Dashboard */}
                <div className="lg:col-span-4 space-y-6">
                    <div className="bg-slate-900 rounded-3xl p-8 text-white space-y-8 sticky top-6">
                        <div className="space-y-2">
                            <span className="text-[10px] font-black uppercase tracking-[0.2em] text-slate-500">Progression globale</span>
                            <div className="text-4xl font-black">{Math.round((dashboardMetrics.confirmed + dashboardMetrics.uncertain) / dashboardMetrics.total * 100)}%</div>
                            <div className="h-2 bg-white/10 rounded-full overflow-hidden">
                                <div 
                                    className="h-full bg-blue-500 transition-all duration-1000" 
                                    style={{ width: `${(dashboardMetrics.confirmed + dashboardMetrics.uncertain) / dashboardMetrics.total * 100}%` }} 
                                />
                            </div>
                        </div>

                        <div className="space-y-4">
                            <div className="flex items-center justify-between p-4 bg-white/5 rounded-2xl border border-white/5">
                                <div className="flex items-center gap-3">
                                    <div className="w-2 h-2 rounded-full bg-emerald-500" />
                                    <span className="text-xs font-bold text-slate-300">Confirmés (AI)</span>
                                </div>
                                <span className="font-black">{dashboardMetrics.confirmed}</span>
                            </div>
                            <div className="flex items-center justify-between p-4 bg-white/5 rounded-2xl border border-white/5">
                                <div className="flex items-center gap-3">
                                    <div className="w-2 h-2 rounded-full bg-amber-500" />
                                    <span className="text-xs font-bold text-slate-300">À Réviser</span>
                                </div>
                                <span className="font-black">{dashboardMetrics.uncertain}</span>
                            </div>
                            <div className="flex items-center justify-between p-4 bg-white/5 rounded-2xl border border-white/5">
                                <div className="flex items-center gap-3">
                                    <div className="w-2 h-2 rounded-full bg-red-500" />
                                    <span className="text-xs font-bold text-slate-300">Vides</span>
                                </div>
                                <span className="font-black text-red-500">{dashboardMetrics.empty}</span>
                            </div>
                        </div>

                        <div className="pt-4 border-t border-white/10">
                            <div className="flex items-center gap-4">
                                <div className="w-10 h-10 rounded-xl bg-blue-600/20 flex items-center justify-center text-blue-500">
                                    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
                                </div>
                                <p className="text-[10px] leading-relaxed text-slate-400 font-medium italic">
                                    L'IA apprend de vos corrections. Les champs en <span className="text-amber-500">orange</span> ont un score de confiance modéré.
                                </p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* Correction Modal */}
            {showCorrectionModal && (
                <div className="fixed inset-0 bg-slate-900/60 backdrop-blur-md z-50 flex items-center justify-center p-4 overflow-y-auto">
                    <div className="bg-white rounded-[40px] shadow-2xl w-full max-w-2xl overflow-hidden border border-slate-100 flex flex-col max-h-[90vh]">
                        <div className="px-10 py-8 border-b border-slate-50 flex items-center justify-between shrink-0">
                            <div>
                                <h3 className="text-2xl font-black text-slate-900 leading-tight">Vérification Finale</h3>
                                <p className="text-sm text-slate-400 font-bold uppercase tracking-wider mt-1">{pendingCorrections.length} correction(s) détectée(s)</p>
                            </div>
                            <button onClick={() => setShowCorrectionModal(false)} className="p-3 hover:bg-slate-50 rounded-2xl text-slate-400 transition-colors">
                                <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M6 18L18 6M6 6l12 12" /></svg>
                            </button>
                        </div>

                        <div className="p-10 overflow-y-auto space-y-8">
                            {pendingCorrections.map((corr, idx) => (
                                <div key={corr.field_name} className="space-y-4 p-8 rounded-3xl bg-slate-50 border border-slate-100 relative">
                                    <div className="absolute -top-3 left-8 px-3 py-1 bg-slate-900 text-[9px] font-black text-white rounded-lg uppercase tracking-widest">{corr.field_name}</div>
                                    
                                    <div className="flex items-center gap-6">
                                        <div className="flex-1 space-y-1">
                                            <span className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Valeur IA</span>
                                            <div className="font-bold text-sm text-red-500 line-through truncate">{String(corr.ai_extracted_value || 'vide')}</div>
                                        </div>
                                        <div className="w-8 h-8 rounded-full bg-white flex items-center justify-center text-slate-300 shadow-sm border border-slate-100">
                                            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M14 5l7 7m0 0l-7 7m7-7H3" /></svg>
                                        </div>
                                        <div className="flex-1 space-y-1">
                                            <span className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Votre Valeur</span>
                                            <div className="font-bold text-sm text-emerald-600 truncate">{String(corr.user_corrected_value || 'vide')}</div>
                                        </div>
                                    </div>

                                    <div className="pt-4 border-t border-slate-200/50">
                                        <textarea 
                                            placeholder="Pourquoi cette valeur ? (Optionnel)"
                                            className="w-full bg-white px-5 py-4 rounded-2xl border-2 border-transparent focus:border-blue-500/20 text-sm outline-none transition-all resize-none min-h-[100px] shadow-inner font-medium"
                                            value={corr.rule || ''}
                                            onChange={(e) => {
                                                const next = [...pendingCorrections];
                                                next[idx].rule = e.target.value;
                                                setPendingCorrections(next);
                                            }}
                                        />
                                    </div>
                                </div>
                            ))}
                        </div>

                        <div className="p-10 bg-slate-50 border-t border-slate-100 flex items-center justify-between shrink-0">
                            <button onClick={() => finalizeSubmission(pendingCorrections)} className="text-sm font-black text-slate-400 hover:text-slate-600 transition-colors">IGNORER LES RÈGLES</button>
                            <button 
                                onClick={() => finalizeSubmission(pendingCorrections)}
                                className="px-10 py-5 bg-blue-600 text-white rounded-2xl font-black text-sm shadow-xl shadow-blue-600/30 hover:bg-blue-700 active:scale-95 transition-all"
                            >
                                CONFIRMER ET SAUVEGARDER
                            </button>
                        </div>
                    </div>
                </div>
            )}

            {/* Notifications */}
            {showSavedPopup && (
                <div className="fixed bottom-12 left-1/2 -translate-x-1/2 z-[60] animate-in slide-in-from-bottom-8 duration-500">
                    <div className="bg-emerald-600 text-white px-10 py-5 rounded-[24px] shadow-2xl flex items-center gap-4 border border-emerald-500/50">
                        <div className="w-8 h-8 rounded-full bg-white/20 flex items-center justify-center">
                            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={4} d="M5 13l4 4L19 7" /></svg>
                        </div>
                        <div className="space-y-0.5">
                            <p className="text-sm font-black">Enregistré !</p>
                            <p className="text-[10px] font-bold opacity-80 uppercase tracking-widest">Base de connaissances mise à jour</p>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default EquipmentForm;
