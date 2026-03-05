import React, { useState, useMemo, useEffect } from 'react';
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
        title: 'Classe',
        icon: '5',
        fields: [
            {
                key: 'classe',
                label: 'classe',
                type: 'select',
                options: ['Classe A', 'Classe B', 'A', 'B', 'Autre'],
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

// Build the ordered list of all field keys for progressive enabling
const ALL_FIELD_KEYS = SECTIONS.flatMap((s) => s.fields.map((f) => f.key));

type FormValues = Record<string, string>;

interface EquipmentFormProps {
    extractedData?: EquipmentData | null;
    isProcessing?: boolean;
}

function EquipmentForm({ extractedData, isProcessing = false }: EquipmentFormProps) {
    const [values, setValues] = useState<FormValues>(() => {
        const init: FormValues = {};
        ALL_FIELD_KEYS.forEach((k) => (init[k] = ''));
        return init;
    });

    const [collapsedSections, setCollapsedSections] = useState<Set<string>>(new Set());
    const [aiFilledFields, setAiFilledFields] = useState<Set<string>>(new Set());
    // Store custom values inputted when "Autre" is selected
    const [customValues, setCustomValues] = useState<FormValues>({});

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
            'classe', 'marque', 'modele', 'reference', 'dateCalibration',
        ] as const;

        // Helper to find field definition
        const getFieldDef = (key: string): FormField | undefined => {
            for (const section of SECTIONS) {
                const found = section.fields.find(f => f.key === key);
                if (found) return found;
            }
            return undefined;
        };

        for (const key of simpleFields) {
            const val = extractedData[key];
            if (val != null && val !== '') {
                const strVal = String(val);
                const fieldDef = getFieldDef(key);

                if (fieldDef?.type === 'select') {
                    // Check if the extracted value is in the options
                    if (!fieldDef.options.includes(strVal)) {
                        // It's a custom value
                        newValues[key] = 'Autre';
                        newCustomValues[key] = strVal;
                    } else {
                        newValues[key] = strVal;
                    }
                } else {
                    newValues[key] = strVal;
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
                newValues['plageMesure'] = `${min}|${max}`;
                filledKeys.add('plageMesure');
            }
        }

        // Handle sortiesAlarme
        if (extractedData.sortiesAlarme && extractedData.sortiesAlarme.length > 0) {
            const alarm = extractedData.sortiesAlarme[0];
            if (alarm.nomAlarme) { newValues['nomAlarme'] = alarm.nomAlarme; filledKeys.add('nomAlarme'); }
            if (alarm.typeAlarme) { newValues['typeAlarme'] = alarm.typeAlarme; filledKeys.add('typeAlarme'); }
            if (alarm.seuilAlarme != null) { newValues['seuilAlarme'] = String(alarm.seuilAlarme); filledKeys.add('seuilAlarme'); }
            if (alarm.uniteAlarme) { newValues['uniteAlarme'] = alarm.uniteAlarme; filledKeys.add('uniteAlarme'); }
            if (alarm.relaisAssocie) { newValues['relaisAssocie'] = alarm.relaisAssocie; filledKeys.add('relaisAssocie'); }
        }

        // eslint-disable-next-line @typescript-eslint/no-unused-vars
        setValues(newValues);
        // eslint-disable-next-line @typescript-eslint/no-unused-vars
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

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        // Merge custom values back in for submission
        const finalValues = { ...values };
        for (const key of Object.keys(finalValues)) {
            if (finalValues[key] === 'Autre' && customValues[key]) {
                finalValues[key] = customValues[key];
            }
        }
        console.log('Form submitted:', finalValues);
    };

    // Compute progress
    const filledCount = ALL_FIELD_KEYS.filter((k) => values[k] !== '').length;
    const progressPercent = Math.round((filledCount / ALL_FIELD_KEYS.length) * 100);

    const renderField = (field: FormField) => {
        const isEnabled = enabledFields.has(field.key);

        const isAiFilled = aiFilledFields.has(field.key);

        const wrapperClass = `
            flex-1 min-w-0 transition-all duration-200
            ${isEnabled ? 'opacity-100' : 'opacity-40 pointer-events-none'}
        `;

        const inputClass = `
            w-full px-3 py-2 rounded-lg border text-sm
            transition-all duration-150 outline-none
            ${isEnabled
                ? 'border-gray-300 bg-white text-gray-800 hover:border-blue-400 focus:border-blue-500 focus:ring-2 focus:ring-blue-500/15'
                : 'border-gray-200 bg-gray-50 text-gray-400 cursor-not-allowed'
            }
        `;

        switch (field.type) {
            case 'select': {
                const isAutre = values[field.key] === 'Autre';
                return (
                    <div key={field.key} className={wrapperClass}>
                        <label className={`block text-xs font-medium mb-1 ${isAiFilled ? 'text-blue-600' : 'text-gray-500'}`}>
                            {isAiFilled && <span className="mr-1">🤖</span>}{field.label}
                        </label>
                        <div className="flex gap-2">
                            <select
                                className={`${inputClass} ${isAutre ? 'w-1/3' : 'w-full'}`}
                                disabled={!isEnabled}
                                required
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
                                    required
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
                        <label className={`block text-xs font-medium mb-1 ${isAiFilled ? 'text-blue-600' : 'text-gray-500'}`}>
                            {isAiFilled && <span className="mr-1">🤖</span>}{field.label}
                        </label>
                        <input
                            type="text"
                            className={inputClass}
                            disabled={!isEnabled}
                            required
                            placeholder={field.placeholder}
                            value={values[field.key]}
                            onChange={(e) => handleChange(field.key, e.target.value)}
                        />
                    </div>
                );

            case 'range':
                return (
                    <div key={field.key} className={wrapperClass}>
                        <label className={`block text-xs font-medium mb-1 ${isAiFilled ? 'text-blue-600' : 'text-gray-500'}`}>
                            {isAiFilled && <span className="mr-1">🤖</span>}{field.label}
                        </label>
                        <div className="flex gap-2 items-center">
                            <input
                                type="text"
                                className={inputClass}
                                disabled={!isEnabled}
                                required
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
                                required
                                placeholder={field.placeholderMax}
                                value={values[field.key]?.split('|')[1] || ''}
                                onChange={(e) => {
                                    const min = values[field.key]?.split('|')[0] || '';
                                    handleChange(field.key, `${min}|${e.target.value}`);
                                }}
                            />
                        </div>
                    </div>
                );

            case 'file':
                return (
                    <div key={field.key} className={wrapperClass}>
                        <label className={`block text-xs font-medium mb-1 ${isAiFilled ? 'text-blue-600' : 'text-gray-500'}`}>
                            {isAiFilled && <span className="mr-1">🤖</span>}{field.label}
                        </label>
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
                );

            case 'dynamic': {
                // Get the number of inputs from the field this depends on
                const depValue = values[field.dependsOn];
                const count = parseInt(depValue, 10);
                const inputCount = isNaN(count) || count <= 0 ? 1 : count;
                // Store values as pipe-separated
                const dynamicValues = (values[field.key] || '').split('|');

                return (
                    <div key={field.key} className={`w-full ${isEnabled ? 'opacity-100' : 'opacity-40 pointer-events-none'} transition-all duration-200`}>
                        <label className={`block text-xs font-medium mb-1 ${isAiFilled ? 'text-blue-600' : 'text-gray-500'}`}>
                            {isAiFilled && <span className="mr-1">🤖</span>}{field.label}
                        </label>
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
                                        // Ensure array is long enough
                                        while (newDynamic.length < inputCount) newDynamic.push('');
                                        newDynamic[i] = e.target.value;
                                        handleChange(field.key, newDynamic.join('|'));
                                    }}
                                />
                            ))}
                        </div>
                    </div>
                );
            }

            case 'date':
                return (
                    <div key={field.key} className={wrapperClass}>
                        <label className={`block text-xs font-medium mb-1 ${isAiFilled ? 'text-blue-600' : 'text-gray-500'}`}>
                            {isAiFilled && <span className="mr-1">🤖</span>}{field.label}
                        </label>
                        <input
                            type="date"
                            className={inputClass}
                            disabled={!isEnabled}
                            required
                            value={values[field.key]}
                            onChange={(e) => handleChange(field.key, e.target.value)}
                        />
                    </div>
                );
        }
    };

    return (
        <form onSubmit={handleSubmit} className="space-y-4">
            {/* Progress Bar */}
            <div className="h-2 bg-gray-100 rounded-full overflow-hidden">
                <div
                    className="h-full bg-blue-500 rounded-full transition-all duration-500 ease-out"
                    style={{ width: `${progressPercent}%` }}
                />
            </div>

            {/* Sections */}
            <div className={`space-y-4 transition-opacity duration-300 ${isProcessing ? 'opacity-50 pointer-events-none' : 'opacity-100'}`}>
                {isProcessing && (
                    <div className="absolute inset-0 z-10 flex flex-col items-center justify-center bg-white/40 backdrop-blur-[2px] rounded-xl">
                        <div className="relative">
                            <div className="w-16 h-16 border-4 border-blue-200 rounded-full animate-pulse blur-[2px]"></div>
                            <div className="absolute inset-0 w-16 h-16 border-4 border-blue-500 rounded-full border-t-transparent animate-spin"></div>
                            <div className="absolute inset-0 flex items-center justify-center">
                                <span className="text-2xl">🤖</span>
                            </div>
                        </div>
                        <p className="mt-4 font-semibold text-blue-700 animate-pulse">Extraction par l'IA en cours...</p>
                        <p className="text-xs text-blue-500 mt-1 text-center px-6">Analyse sémantique et regex des champs du document.</p>
                    </div>
                )}

                {SECTIONS.map((section) => {
                    const isCollapsed = collapsedSections.has(section.title);

                    return (
                        <div
                            key={section.title}
                            className="bg-gray-50/80 rounded-xl border border-gray-200 overflow-hidden relative"
                        >
                            {/* Section Header */}
                            <button
                                type="button"
                                onClick={() => toggleSection(section.title)}
                                className="w-full flex items-center justify-between px-5 py-3 hover:bg-gray-100/60 transition-colors"
                            >
                                <div className="flex items-center gap-3">
                                    <span className="w-6 h-6 rounded-md bg-blue-500 text-white text-xs font-bold flex items-center justify-center">
                                        {section.icon}
                                    </span>
                                    <h3 className="text-sm font-semibold text-gray-700">{section.title}</h3>
                                </div>
                                <svg
                                    className={`w-4 h-4 text-gray-400 transition-transform duration-200 ${isCollapsed ? '' : 'rotate-180'}`}
                                    fill="none"
                                    viewBox="0 0 24 24"
                                    stroke="currentColor"
                                    strokeWidth={2}
                                >
                                    <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
                                </svg>
                            </button>

                            {/* Section Fields — horizontal row layout */}
                            {!isCollapsed && (
                                <div className="px-5 pb-4 pt-1">
                                    <div className="flex flex-wrap gap-4">
                                        {section.fields.map((field) => renderField(field))}
                                    </div>
                                </div>
                            )}
                        </div>
                    );
                })}
            </div>

            {/* Submit Button */}
            <div className="flex justify-end pt-2">
                <button
                    type="submit"
                    className="
                        px-6 py-2.5 rounded-lg font-semibold text-white text-sm
                        bg-blue-500 hover:bg-blue-600
                        shadow-sm hover:shadow-md
                        active:bg-blue-700
                        transition-all duration-150
                        disabled:opacity-50 disabled:cursor-not-allowed
                    "
                >
                    Enregistrer l'équipement
                </button>
            </div>
        </form>
    );
}

export default EquipmentForm;