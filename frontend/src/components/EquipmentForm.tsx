import React, { useState, useMemo, useEffect } from 'react';
import { API_BASE } from '../services/api';
import type { EquipmentData } from '../services/api';

// --- Field definitions organized by section ---
interface SelectField {
    key: string;
    label: string;
    type: 'select';
    options: string[];
}

interface TextField {
    key: string;
    label: string;
    type: 'text';
    placeholder: string;
}

interface RangeField {
    key: string;
    label: string;
    type: 'range';
    placeholderMin: string;
    placeholderMax: string;
}

interface FileField {
    key: string;
    label: string;
    type: 'file';
    accept: string;
}

interface DynamicField {
    key: string;
    label: string;
    type: 'dynamic';
    dependsOn: string;
    placeholder: string;
}

interface DateField {
    key: string;
    label: string;
    type: 'date';
}

type FormField = SelectField | TextField | RangeField | FileField | DynamicField | DateField;

interface Section {
    title: string;
    icon: string;
    fields: FormField[];
}

const OPTIONAL_FIELDS = new Set<string>([
    'plageMesure',
    'technologie',
    'nbFils',
    'alimentation',
    'reperage',
    'communication',
    'sortiesAlarme',
    'nomAlarme',
    'typeAlarme',
    'seuilAlarme',
    'uniteAlarme',
    'relaisAssocie',
    'reference',
    'image',
    'datasheet',
    'dateCalibration',
    'indiceIP',
]);

const SECTIONS: Section[] = [
    {
        title: 'General Information',
        icon: '1',
        fields: [
            {
                key: 'equipmentName',
                label: 'equipment_name',
                type: 'text',
                placeholder: 'equipment name',
            },
            {
                key: 'categorie',
                label: 'categorie_equipement',
                type: 'select',
                options: ['Transmetteur', 'Actionneur', 'Capteurs', 'Automate', 'IHM', 'Autre'],
            }
        ],
    },
    {
        title: 'Type de mesure',
        icon: '2',
        fields: [
            {
                key: 'typeMesure',
                label: 'type_mesure',
                type: 'select',
                options: ['Debit', 'Niveau', 'Pression', 'Temperature', 'Autre'],
            },
            {
                key: 'technologie',
                label: 'technologie',
                type: 'select',
                options: [
                    'Electromagnetique',
                    'Magnetique',
                    'Hydraulique',
                    'Pneumatique',
                    'Numerique',
                    'Piezo-resistif',
                    'Electronique',
                    'Section variable',
                    'TFT Tactile',
                    'Tactile LCD',
                    'Autre',
                ],
            },
            {
                key: 'plageMesure',
                label: 'plage_de_mesure',
                type: 'range',
                placeholderMin: 'min',
                placeholderMax: 'max',
            }
        ],
    },
    {
        title: 'Signal',
        icon: '3',
        fields: [
            {
                key: 'typeSignal',
                label: 'type_signal',
                type: 'select',
                options: ['4-20mA', '0-20mA', '0-10V', '0-5V', '0-10V (AI)', '24V DC (DI/DO)', 'Autre'],
            },
            {
                key: 'nbFils',
                label: 'nb_fils',
                type: 'select',
                options: ['1', '2', '4', '2 fils', '4 fils', 'Autre'],
            },
            {
                key: 'alimentation',
                label: 'alimentation',
                type: 'select',
                options: ['None', '24V DC', '220V AC', 'Autre'],
            },
            {
                key: 'reperage',
                label: 'reperage_signal',
                type: 'dynamic',
                dependsOn: 'nbFils',
                placeholder: 'fil',
            },
        ],
    },
    {
        title: 'Communication',
        icon: '4',
        fields: [
            {
                key: 'communication',
                label: 'communication',
                type: 'select',
                options: [
                    'HART',
                    'Modbus RTU',
                    'Modbus TCP',
                    'Modbus TCP/IP',
                    'PROFIBUS DP',
                    'Profibus PA',
                    'Foundation Fieldbus',
                    'Profinet',
                    'Ethernet',
                    'RS-232',
                    'RS-485',
                    'Mitsubishi MC TCP/IP',
                    'Autre',
                ],
            },
        ],
    },
    {
        title: 'Indice IP',
        icon: '5',
        fields: [
            {
                key: 'indiceIP',
                label: 'indice_ip',
                type: 'text',
                placeholder: 'IP67',
            },
        ],
    },
    {
        title: 'Sorties Alarme',
        icon: '6',
        fields: [
            {
                key: 'nomAlarme',
                label: 'nom',
                type: 'text',
                placeholder: 'nom',
            },
            {
                key: 'typeAlarme',
                label: 'type',
                type: 'text',
                placeholder: 'Type',
            },
            {
                key: 'seuilAlarme',
                label: 'seuil',
                type: 'text',
                placeholder: '0',
            },
            {
                key: 'uniteAlarme',
                label: 'unite',
                type: 'text',
                placeholder: 'unit',
            },
            {
                key: 'relaisAssocie',
                label: 'relais_associe',
                type: 'text',
                placeholder: '',
            },
        ],
    },
    {
        title: 'Manufacturer & Files',
        icon: '7',
        fields: [
            {
                key: 'marque',
                label: 'marque',
                type: 'text',
                placeholder: 'marque',
            },
            {
                key: 'modele',
                label: 'modele',
                type: 'text',
                placeholder: 'modele',
            },
            {
                key: 'reference',
                label: 'ref',
                type: 'text',
                placeholder: 'ref',
            },
            {
                key: 'dateCalibration',
                label: 'date_calibration',
                type: 'date',
            },
            {
                key: 'datasheet',
                label: 'Datasheet',
                type: 'file',
                accept: '.pdf,application/pdf',
            },
            {
                key: 'image',
                label: 'Image',
                type: 'file',
                accept: 'image/*',
            },
        ],
    },
];

const ALL_FIELD_KEYS = SECTIONS.flatMap((s) => s.fields.map((f) => f.key));

type FormValues = Record<string, string>;

interface EquipmentFormProps {
    extractedData?: EquipmentData | null;
    confidence?: Record<string, number> | null;
    isProcessing?: boolean;
    docType?: string;
}

function EquipmentForm({ extractedData, confidence, isProcessing = false, docType = '' }: EquipmentFormProps) {

    const [values, setValues] = useState<FormValues>(() => {
        const init: FormValues = {};
        ALL_FIELD_KEYS.forEach((k) => (init[k] = ''));
        return init;
    });

    const [collapsedSections, setCollapsedSections] = useState<Set<string>>(new Set());
    const [aiFilledFields, setAiFilledFields] = useState<Set<string>>(new Set());
    // Store custom values inputted when "Autre" is selected
    const [customValues, setCustomValues] = useState<FormValues>({});

    const [prevValues, setPrevValues] = useState<FormValues>({});
    const [isTyping, setIsTyping] = useState<Set<string>>(new Set());

    // Auto-fill form when extractedData arrives
    useEffect(() => {
        if (!extractedData) return;

        const newValues: FormValues = { ...values };
        const newCustomValues: FormValues = { ...customValues };
        const filledKeys = new Set<string>();

        const simpleFields = [
            'equipmentName',
            'categorie', 'typeMesure', 'technologie', 'typeSignal',
            'nbFils', 'alimentation', 'reperage', 'communication',
            'indiceIP', 'marque', 'modele', 'reference', 'dateCalibration',
        ] as const;

        // Helper to find field definition
        const getFieldDef = (key: string): FormField | undefined => {
            for (const section of SECTIONS) {
                const found = section.fields.find(f => f.key === key);
                if (found) return found;
            }
            return undefined;
        };

        const changedKeys: string[] = [];

        for (const key of simpleFields) {
            const val = extractedData[key as keyof EquipmentData];
            if (val != null && val !== '') {
                const strVal = String(val);
                const fieldDef = getFieldDef(key);

                let finalVal = strVal;
                if (fieldDef?.type === 'select') {
                    if (!fieldDef.options.includes(strVal)) {
                        newValues[key] = 'Autre';
                        newCustomValues[key] = strVal;
                        finalVal = 'Autre';
                    } else {
                        newValues[key] = strVal;
                    }
                } else {
                    newValues[key] = strVal;
                }
                
                if (values[key] !== finalVal) {
                    changedKeys.push(key);
                }
                filledKeys.add(key);
            }
        }

        // Handle plageMesure (range field stored as "min|max")
        if (extractedData.plageMesure) {
            const pm = extractedData.plageMesure;
            const min = pm.min != null ? String(pm.min) : '';
            const max = pm.max != null ? String(pm.max) : '';
            if (min || max) {
                const val = `${min}|${max}`;
                if (values['plageMesure'] !== val) changedKeys.push('plageMesure');
                newValues['plageMesure'] = val;
                filledKeys.add('plageMesure');
            }
        }

        // Handle sortiesAlarme
        if (extractedData.sortiesAlarme && extractedData.sortiesAlarme.length > 0) {
            const alarm = extractedData.sortiesAlarme[0];
            const alarmFields = [
                { key: 'nomAlarme', val: alarm.nomAlarme },
                { key: 'typeAlarme', val: alarm.typeAlarme },
                { key: 'seuilAlarme', val: alarm.seuilAlarme != null ? String(alarm.seuilAlarme) : null },
                { key: 'uniteAlarme', val: alarm.uniteAlarme },
                { key: 'relaisAssocie', val: alarm.relaisAssocie }
            ];

            alarmFields.forEach(({ key, val }) => {
                if (val) {
                    if (values[key] !== val) changedKeys.push(key);
                    newValues[key] = val;
                    filledKeys.add(key);
                }
            });
        }

        // Phase 4: Typewriter effect simulation
        if (changedKeys.length > 0) {
            setPrevValues(values);
            setIsTyping(new Set(changedKeys));
            
            // Staggered finish
            changedKeys.forEach((key, index) => {
                setTimeout(() => {
                    setIsTyping(prev => {
                        const next = new Set(prev);
                        next.delete(key);
                        return next;
                    });
                }, 300 + (index * 50));
            });
        }

        setValues(newValues);
        setCustomValues(newCustomValues);
        setAiFilledFields(filledKeys);
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [extractedData]);

    const toggleSection = (title: string) => {
        setCollapsedSections((prev) => {
            const next = new Set(prev);
            if (next.has(title)) {
                next.delete(title);
            } else {
                next.add(title);
            }
            return next;
        });
    };

    // Compute which fields are enabled
    const enabledFields = useMemo(() => {
        return new Set<string>(ALL_FIELD_KEYS);
    }, []);

    const handleChange = (key: string, value: string) => {
        setValues((prev) => {
            return { ...prev, [key]: value };
        });
    };

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();

        // Build the list of corrections (AI value ≠ what user kept)
        const corrections: Array<{ field: string; ai_value: string; correct_value: string }> = [];

        for (const key of ALL_FIELD_KEYS) {
            // Only check fields the AI actually filled
            if (!aiFilledFields.has(key)) continue;
            if (!extractedData) continue;

            let aiValue: unknown = (extractedData as any)?.[key];

            // Special-case: range field stored as "min|max" in the form
            if (key === 'plageMesure') {
                const pm = (extractedData as any)?.plageMesure;
                const min = pm?.min != null ? String(pm.min) : '';
                const max = pm?.max != null ? String(pm.max) : '';
                aiValue = `${min}|${max}`;
            }

            // Special-case: sortiesAlarme is an array in extractedData, but the form uses flat keys
            if (['nomAlarme', 'typeAlarme', 'seuilAlarme', 'uniteAlarme', 'relaisAssocie'].includes(key)) {
                const alarm0 = (extractedData as any)?.sortiesAlarme?.[0];
                aiValue = alarm0?.[key];
                if (aiValue != null && key === 'seuilAlarme') {
                    aiValue = String(aiValue);
                }
            }

            const currentValue =
                values[key] === 'Autre' && customValues[key]
                    ? customValues[key]
                    : values[key];

            // Normalize both to string for comparison
            const aiStr = aiValue != null ? String(aiValue) : '';
            const currentStr = currentValue ?? '';

            if (aiStr !== '' && aiStr !== currentStr) {
                corrections.push({
                    field: key,
                    ai_value: aiStr,
                    correct_value: currentStr,
                });
            }
        }

        // Send corrections to backend (fire and forget, don't block save)
        if (corrections.length > 0) {
            fetch(`${API_BASE}/feedback`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ corrections, doc_type: docType || '' }),
            }).catch(console.error);
        }

        // Merge custom values back in for submission
        const finalValues = { ...values };
        for (const key of Object.keys(finalValues)) {
            if (finalValues[key] === 'Autre' && customValues[key]) {
                finalValues[key] = customValues[key];
            }
        }
        console.log('Form submitted with corrections detected:', corrections.length);
        console.log('Final values:', finalValues);
    };

    // Compute progress
    const filledCount = ALL_FIELD_KEYS.filter((k) => values[k] !== '').length;
    const progressPercent = Math.round((filledCount / ALL_FIELD_KEYS.length) * 100);

    const renderField = (field: FormField) => {
        const isEnabled = enabledFields.has(field.key);
        const isAiFilled = aiFilledFields.has(field.key);
        const typing = isTyping.has(field.key);
        const isOptional = OPTIONAL_FIELDS.has(field.key);
        
        const hasAiResult = Boolean(extractedData) || Boolean(confidence);
        const currentValue = values[field.key];
        const isEmpty = !currentValue;
        const fieldConfidence = confidence?.[field.key];

        const isModified = isAiFilled && extractedData && (function() {
            const key = field.key;
            const aiVal = extractedData[key as keyof EquipmentData];
            if (aiVal == null) return false;
            
            if (field.type === 'range') {
                const pm = aiVal as any;
                const min = pm.min != null ? String(pm.min) : '';
                const max = pm.max != null ? String(pm.max) : '';
                return currentValue !== `${min}|${max}`;
            }
            
            if (field.type === 'select' && currentValue === 'Autre') {
                return customValues[key] !== String(aiVal);
            }
            
            return currentValue !== String(aiVal);
        })();

        const isPlaceholder = field.key === 'seuilAlarme' && currentValue === '0' && values['uniteAlarme'] === 's';
        const effectivelyEmpty = isEmpty || isPlaceholder || fieldConfidence === 0;

        const barColorClass = !hasAiResult || effectivelyEmpty
            ? 'bg-slate-200'
            : (typeof fieldConfidence === 'number')
                ? (fieldConfidence >= 0.90 ? 'bg-green-600' : 'bg-yellow-500')
                : 'bg-yellow-500';

        const bgTintClass = !hasAiResult || effectivelyEmpty
            ? ''
            : (typeof fieldConfidence === 'number')
                ? (fieldConfidence >= 0.90 ? 'bg-green-50/30' : 'bg-yellow-50/30')
                : 'bg-yellow-50/30';

        const wrapperClass = `
            flex-1 min-w-0 transition-all duration-200 relative
            ${isEnabled ? 'opacity-100' : 'opacity-40 pointer-events-none'}
            ${typing ? 'animate-pulse' : ''}
        `;

        const inputClass = `
            w-full px-3 py-2 rounded-lg border text-sm
            transition-all duration-150 outline-none pl-4
            ${isEnabled
                ? `${bgTintClass} border-slate-200 text-slate-900 hover:border-blue-400 focus:border-blue-500 focus:ring-2 focus:ring-blue-500/15`
                : 'border-gray-200 bg-gray-50 text-gray-400 cursor-not-allowed'
            }
        `;

        switch (field.type) {
            case 'select': {
                const isAutre = values[field.key] === 'Autre';
                return (
                    <div key={field.key} className={wrapperClass}>
                        <label className="flex items-center justify-between text-[11px] font-medium uppercase tracking-wider text-slate-400 mb-1">
                            <div className="flex items-center gap-1.5">
                                <span>{field.label}</span>
                                {isModified && (
                                    <span className="px-1 py-0 rounded bg-slate-100 text-slate-500 text-[8px] font-bold">Modifié</span>
                                )}
                            </div>
                            {hasAiResult && !isEmpty && typeof fieldConfidence === 'number' && (
                                <div className={`flex items-center gap-1 px-1.5 py-0.5 rounded-full text-[9px] font-bold border ${
                                    fieldConfidence >= 0.90 
                                        ? 'bg-green-50 text-green-700 border-green-200' 
                                        : 'bg-yellow-50 text-yellow-700 border-yellow-200'
                                }`}>
                                    {fieldConfidence >= 0.90 ? '✓ ' : ''}{Math.round(fieldConfidence * 100)}%
                                </div>
                            )}
                        </label>
                        <div className="relative group">
                            {hasAiResult && (
                                <div className={`absolute left-0 top-0 bottom-0 w-1 rounded-l-lg z-10 ${barColorClass}`} />
                            )}
                            <select
                                className={`${inputClass} ${isAutre ? 'w-1/3' : 'w-full'}`}
                                disabled={!isEnabled}
                                required={!isOptional}
                                value={values[field.key]}
                                onChange={(e) => handleChange(field.key, e.target.value)}
                            >
                                <option value="">{field.label}</option>
                                {field.options.map((opt) => (
                                    <option key={opt} value={opt}>
                                        {opt}
                                    </option>
                                ))}
                            </select>
                            {isAutre && (
                                <input
                                    type="text"
                                    className={`${inputClass} w-2/3 border-blue-300 ring-1 ring-blue-100 bg-blue-50/30`}
                                    disabled={!isEnabled}
                                    required={!isOptional}
                                    placeholder="Précisez..."
                                    value={customValues[field.key] || ''}
                                    onChange={(e) => setCustomValues(prev => ({ ...prev, [field.key]: e.target.value }))}
                                    autoFocus
                                />
                            )}
                        </div>
                    </div>
                );
            }

            case 'text':
                return (
                    <div key={field.key} className={wrapperClass}>
                        <label className="flex items-center justify-between text-[11px] font-medium uppercase tracking-wider text-slate-400 mb-1">
                            <div className="flex items-center gap-1.5">
                                <span>{field.label}</span>
                                {isModified && (
                                    <span className="px-1 py-0 rounded bg-slate-100 text-slate-500 text-[8px] font-bold">Modifié</span>
                                )}
                            </div>
                            {hasAiResult && !isEmpty && typeof fieldConfidence === 'number' && (
                                <div className={`flex items-center gap-1 px-1.5 py-0.5 rounded-full text-[9px] font-bold border ${
                                    fieldConfidence >= 0.90 
                                        ? 'bg-green-50 text-green-700 border-green-200' 
                                        : 'bg-yellow-50 text-yellow-700 border-yellow-200'
                                }`}>
                                    {fieldConfidence >= 0.90 ? '✓ ' : ''}{Math.round(fieldConfidence * 100)}%
                                </div>
                            )}
                        </label>
                        <div className="relative group">
                            {hasAiResult && (
                                <div className={`absolute left-0 top-0 bottom-0 w-1 rounded-l-lg z-10 ${barColorClass}`} />
                            )}
                            <input
                                type="text"
                                className={inputClass}
                                disabled={!isEnabled}
                                required={!isOptional}
                                placeholder={field.placeholder}
                                value={values[field.key]}
                                onChange={(e) => handleChange(field.key, e.target.value)}
                            />
                        </div>
                    </div>
                );

            case 'range':
                return (
                    <div key={field.key} className={wrapperClass}>
                        <label className="flex items-center justify-between text-[11px] font-medium uppercase tracking-wider text-slate-400 mb-1">
                            <div className="flex items-center gap-1.5">
                                <span>{field.label}</span>
                                {isModified && (
                                    <span className="px-1 py-0 rounded bg-slate-100 text-slate-500 text-[8px] font-bold">Modifié</span>
                                )}
                            </div>
                            {hasAiResult && !isEmpty && typeof fieldConfidence === 'number' && (
                                <div className={`flex items-center gap-1 px-1.5 py-0.5 rounded-full text-[9px] font-bold border ${
                                    fieldConfidence >= 0.90 
                                        ? 'bg-green-50 text-green-700 border-green-200' 
                                        : 'bg-yellow-50 text-yellow-700 border-yellow-200'
                                }`}>
                                    {fieldConfidence >= 0.90 ? '✓ ' : ''}{Math.round(fieldConfidence * 100)}%
                                </div>
                            )}
                        </label>
                        <div className="relative group">
                            {hasAiResult && (
                                <div className={`absolute left-0 top-0 bottom-0 w-1 rounded-l-lg z-10 ${barColorClass}`} />
                            )}
                            <div className="flex gap-2 items-center">
                                <input
                                    type="text"
                                    className={inputClass}
                                    disabled={!isEnabled}
                                    required={!isOptional}
                                    placeholder={field.placeholderMin}
                                    value={values[field.key]?.split('|')[0] || ''}
                                    onChange={(e) => {
                                        const max = values[field.key]?.split('|')[1] || '';
                                        handleChange(field.key, `${e.target.value}|${max}`);
                                    }}
                                />
                                <input
                                    type="text"
                                    className={inputClass}
                                    disabled={!isEnabled}
                                    required={!isOptional}
                                    placeholder={field.placeholderMax}
                                    value={values[field.key]?.split('|')[1] || ''}
                                    onChange={(e) => {
                                        const min = values[field.key]?.split('|')[0] || '';
                                        handleChange(field.key, `${min}|${e.target.value}`);
                                    }}
                                />
                            </div>
                        </div>
                    </div>
                );

            case 'file':
                return (
                    <div key={field.key} className={wrapperClass}>
                        <label className="flex items-center justify-between text-[11px] font-medium uppercase tracking-wider text-slate-400 mb-1">
                            <div className="flex items-center gap-1.5">
                                <span>{field.label}</span>
                                {isModified && (
                                    <span className="px-1 py-0 rounded bg-slate-100 text-slate-500 text-[8px] font-bold">Modifié</span>
                                )}
                            </div>
                            {hasAiResult && !isEmpty && typeof fieldConfidence === 'number' && (
                                <div className={`flex items-center gap-1 px-1.5 py-0.5 rounded-full text-[9px] font-bold border ${
                                    fieldConfidence >= 0.90 
                                        ? 'bg-green-50 text-green-700 border-green-200' 
                                        : 'bg-yellow-50 text-yellow-700 border-yellow-200'
                                }`}>
                                    {fieldConfidence >= 0.90 ? '✓ ' : ''}{Math.round(fieldConfidence * 100)}%
                                </div>
                            )}
                        </label>
                        <div className="relative group">
                            {hasAiResult && (
                                <div className={`absolute left-0 top-0 bottom-0 w-1 rounded-l-lg z-10 ${barColorClass}`} />
                            )}
                            <div
                                className={`
                                    relative flex items-center gap-2 px-3 py-2 rounded-lg border border-dashed
                                    transition-all duration-150
                                    ${isEnabled
                                        ? 'border-gray-300 bg-white hover:border-blue-400 cursor-pointer'
                                        : 'border-gray-200 bg-gray-50 cursor-not-allowed'
                                    }
                                `}
                            >
                                <svg className={`w-4 h-4 ${isEnabled ? 'text-gray-400' : 'text-gray-300'}`} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                                    <path strokeLinecap="round" strokeLinejoin="round" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                                </svg>
                                <span className={`text-sm truncate ${isEnabled ? 'text-gray-500' : 'text-gray-400'}`}>
                                    {values[field.key] || field.label.toLowerCase()}
                                </span>
                                <input
                                    type="file"
                                    accept={field.accept}
                                    disabled={!isEnabled}
                                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                                    onChange={(e) => {
                                        const file = e.target.files?.[0];
                                        if (file) handleChange(field.key, file.name);
                                    }}
                                />
                            </div>
                        </div>
                    </div>
                );

            case 'dynamic': {
                const depValue = values[field.dependsOn];
                const count = parseInt(depValue, 10);
                const inputCount = isNaN(count) || count <= 0 ? 1 : count;
                const dynamicValues = (values[field.key] || '').split('|');

                return (
                    <div key={field.key} className={wrapperClass}>
                        <label className="flex items-center justify-between text-[11px] font-medium uppercase tracking-wider text-slate-400 mb-1">
                            <div className="flex items-center gap-1.5">
                                <span>{field.label}</span>
                                {isModified && (
                                    <span className="px-1 py-0 rounded bg-slate-100 text-slate-500 text-[8px] font-bold">Modifié</span>
                                )}
                            </div>
                            {hasAiResult && !isEmpty && typeof fieldConfidence === 'number' && (
                                <div className={`flex items-center gap-1 px-1.5 py-0.5 rounded-full text-[9px] font-bold border ${
                                    fieldConfidence >= 0.90 
                                        ? 'bg-green-50 text-green-700 border-green-200' 
                                        : 'bg-yellow-50 text-yellow-700 border-yellow-200'
                                }`}>
                                    {fieldConfidence >= 0.90 ? '✓ ' : ''}{Math.round(fieldConfidence * 100)}%
                                </div>
                            )}
                        </label>
                        <div className="relative group">
                            {hasAiResult && (
                                <div className={`absolute left-0 top-0 bottom-0 w-1 rounded-l-lg z-10 ${barColorClass}`} />
                            )}
                            <div className="flex flex-wrap gap-2">
                                {Array.from({ length: inputCount }, (_, i) => (
                                    <input
                                        key={`${field.key}_${i}`}
                                        type="text"
                                        className={inputClass}
                                        style={{ flex: '1 1 0', minWidth: '80px' }}
                                        disabled={!isEnabled}
                                        placeholder={`${field.placeholder} ${i + 1}`}
                                        value={dynamicValues[i] || ''}
                                        onChange={(e) => {
                                            const newDynamic = [...dynamicValues];
                                            while (newDynamic.length < inputCount) newDynamic.push('');
                                            newDynamic[i] = e.target.value;
                                            handleChange(field.key, newDynamic.join('|'));
                                        }}
                                    />
                                ))}
                            </div>
                        </div>
                    </div>
                );
            }

            case 'date':
                return (
                    <div key={field.key} className={wrapperClass}>
                        <label className="flex items-center justify-between text-[11px] font-medium uppercase tracking-wider text-slate-400 mb-1">
                            <div className="flex items-center gap-1.5">
                                <span>{field.label}</span>
                                {isModified && (
                                    <span className="px-1 py-0 rounded bg-slate-100 text-slate-500 text-[8px] font-bold">Modifié</span>
                                )}
                            </div>
                            {hasAiResult && !isEmpty && typeof fieldConfidence === 'number' && (
                                <div className={`flex items-center gap-1 px-1.5 py-0.5 rounded-full text-[9px] font-bold border ${
                                    fieldConfidence >= 0.90 
                                        ? 'bg-green-50 text-green-700 border-green-200' 
                                        : 'bg-yellow-50 text-yellow-700 border-yellow-200'
                                }`}>
                                    {fieldConfidence >= 0.90 ? '✓ ' : ''}{Math.round(fieldConfidence * 100)}%
                                </div>
                            )}
                        </label>
                        <div className="relative group">
                            {hasAiResult && (
                                <div className={`absolute left-0 top-0 bottom-0 w-1 rounded-l-lg z-10 ${barColorClass}`} />
                            )}
                            <input
                                type="date"
                                className={inputClass}
                                disabled={!isEnabled}
                                required={!isOptional}
                                value={values[field.key]}
                                onChange={(e) => handleChange(field.key, e.target.value)}
                            />
                        </div>
                    </div>
                );
        }
    };

    // Calculate quality counts
    const qualityMetrics = useMemo(() => {
        if (!confidence) return null;
        let confirmed = 0;
        let uncertain = 0;
        let empty = 0;

        ALL_FIELD_KEYS.forEach(key => {
            if (!values[key]) {
                empty++;
            } else {
                const conf = confidence[key];
                if (typeof conf === 'number' && conf >= 0.90) confirmed++;
                else uncertain++;
            }
        });

        return { confirmed, uncertain, empty, total: ALL_FIELD_KEYS.length };
    }, [confidence, values]);

    return (
        <form onSubmit={handleSubmit} className="space-y-4">
            {/* Field Completion Dashboard */}
            {qualityMetrics && (
                <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm mb-6">
                    <div className="flex items-center justify-between mb-3">
                        <span className="text-xs font-bold uppercase tracking-widest text-slate-400">Qualité d'extraction</span>
                        <span className="text-xs font-mono font-medium text-slate-500">{filledCount}/{qualityMetrics.total} champs</span>
                    </div>
                    <div className="flex h-1.5 w-full rounded-full overflow-hidden bg-slate-100 mb-4">
                        <div className="h-full bg-green-500 transition-all duration-500" style={{ width: `${(qualityMetrics.confirmed / qualityMetrics.total) * 100}%` }} />
                        <div className="h-full bg-yellow-400 transition-all duration-500" style={{ width: `${(qualityMetrics.uncertain / qualityMetrics.total) * 100}%` }} />
                        <div className="h-full bg-red-500 transition-all duration-500" style={{ width: `${(qualityMetrics.empty / qualityMetrics.total) * 100}%` }} />
                    </div>
                    <div className="flex gap-4">
                        <div className="flex items-center gap-1.5">
                            <div className="w-2 h-2 rounded-full bg-green-500" />
                            <span className="text-[11px] font-semibold text-slate-600">{qualityMetrics.confirmed} Confirmés</span>
                        </div>
                        <div className="flex items-center gap-1.5">
                            <div className="w-2 h-2 rounded-full bg-yellow-400" />
                            <span className="text-[11px] font-semibold text-slate-600">{qualityMetrics.uncertain} À réviser</span>
                        </div>
                        <div className="flex items-center gap-1.5">
                            <div className="w-2 h-2 rounded-full bg-red-500" />
                            <span className="text-[11px] font-semibold text-slate-600">{qualityMetrics.empty} Vides</span>
                        </div>
                    </div>
                </div>
            )}

            {!qualityMetrics && (
                <div className="h-2 bg-gray-100 rounded-full overflow-hidden">
                    <div
                        className="h-full bg-blue-500 transition-all duration-500"
                        style={{ width: `${progressPercent}%` }}
                    />
                </div>
            )}

            <div className="space-y-5">
                {SECTIONS.map((section) => {
                    const isCollapsed = collapsedSections.has(section.title);
                    return (
                        <div key={section.title} className="bg-white/90 border border-slate-200 rounded-2xl shadow-sm overflow-hidden transition-all duration-200">
                            <button
                                type="button"
                                onClick={() => toggleSection(section.title)}
                                className="w-full px-6 py-4 flex items-center justify-between bg-white hover:bg-slate-50 transition-colors"
                            >
                                <div className="flex items-center gap-3">
                                    <span className="text-sm font-semibold tracking-tight text-slate-700">{section.title}</span>
                                </div>
                                <span className={`transform transition-transform duration-200 ${isCollapsed ? 'rotate-180' : ''}`}>
                                    <svg className="w-5 h-5 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                                    </svg>
                                </span>
                            </button>

                            {!isCollapsed && (
                                <div className="px-6 py-5 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-x-6 gap-y-4">
                                    {section.fields.map((field) => renderField(field))}
                                </div>
                            )}
                        </div>
                    );
                })}
            </div>

            <div className="sticky bottom-0 bg-white/80 backdrop-blur-md border-t border-slate-200 px-8 py-4 flex items-center justify-between z-20 -mx-8 -mb-8 rounded-b-3xl">
                <div className="flex items-center gap-6">
                    <div className="flex items-center gap-2">
                        <div className="relative w-10 h-10 flex items-center justify-center">
                            <svg className="w-10 h-10 transform -rotate-90">
                                <circle cx="20" cy="20" r="18" stroke="currentColor" strokeWidth="3" fill="transparent" className="text-slate-100" />
                                <circle cx="20" cy="20" r="18" stroke="currentColor" strokeWidth="3" fill="transparent" strokeDasharray={113} strokeDashoffset={113 - (113 * progressPercent) / 100} className="text-blue-500 transition-all duration-1000" />
                            </svg>
                            <span className="absolute text-[10px] font-bold text-slate-700">{progressPercent}%</span>
                        </div>
                        <div className="flex flex-col">
                            <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">Progression</span>
                            <span className="text-xs font-semibold text-slate-600">{filledCount}/{ALL_FIELD_KEYS.length} champs</span>
                        </div>
                    </div>
                    {qualityMetrics && qualityMetrics.empty > 0 && (
                        <div className="px-3 py-1 bg-red-50 rounded-full border border-red-100 flex items-center gap-1.5">
                            <div className="w-1.5 h-1.5 rounded-full bg-red-500" />
                            <span className="text-[10px] font-bold text-red-600 uppercase tracking-wider">{qualityMetrics.empty} champs requis vides</span>
                        </div>
                    )}
                </div>
                
                <button
                    type="submit"
                    className="
                        px-8 py-3 rounded-xl font-bold text-white text-sm
                        bg-blue-600 hover:bg-blue-700
                        shadow-lg shadow-blue-500/25 hover:shadow-blue-500/40
                        active:scale-[0.98]
                        transition-all duration-150
                        disabled:opacity-50 disabled:grayscale disabled:cursor-not-allowed
                        flex items-center gap-2
                    "
                    disabled={isProcessing}
                >
                    {isProcessing ? (
                        <>
                            <svg className="w-4 h-4 animate-spin" viewBox="0 0 24 24">
                                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                            </svg>
                            Extraction...
                        </>
                    ) : (
                        <>
                            Enregistrer l'équipement
                            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                            </svg>
                        </>
                    )}
                </button>
            </div>
        </form>
    );
}

export default EquipmentForm;
