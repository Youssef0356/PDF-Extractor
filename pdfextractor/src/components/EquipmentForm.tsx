import React, { useState, useMemo } from 'react';

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

type FormField = SelectField | TextField | RangeField | FileField;

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
                key: 'categorie',
                label: 'categorie_equipement',
                type: 'select',
                options: ['Transmetteur', 'Actionneur', 'Autre'],
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
                options: ['Débit', 'Niveau', 'Pression', 'Température', 'Autre'],
            },
            {
                key: 'technologie',
                label: 'technologie',
                type: 'select',
                options: ['Électromagnétique', 'Hydraulique', 'Autre'],
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
                options: ['4-20mA', '0-20mA', '0-10V', '0-5V', 'Autre'],
            },
            {
                key: 'nbFils',
                label: 'nb_fils',
                type: 'select',
                options: ['1', '2', '4', 'Autre'],
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
                type: 'text',
                placeholder: 'reperage_signal',
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
                options: ['HART', 'Modbus RTU', 'Modbus TCP', 'PROFIBUS DP', 'Autre'],
            },
            {
                key: 'classe',
                label: 'classe',
                type: 'select',
                options: ['Classe A', 'Classe B', 'Autre'],
            }
        ],
    },
    {
        title: 'Sorties Alarme',
        icon: '5',
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
        icon: '6',
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

function EquipmentForm() {
    const [values, setValues] = useState<FormValues>(() => {
        const init: FormValues = {};
        ALL_FIELD_KEYS.forEach((k) => (init[k] = ''));
        return init;
    });

    const [collapsedSections, setCollapsedSections] = useState<Set<string>>(new Set());

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
        const enabled = new Set<string>();
        enabled.add(ALL_FIELD_KEYS[0]);

        for (let i = 1; i < ALL_FIELD_KEYS.length; i++) {
            const prevKey = ALL_FIELD_KEYS[i - 1];
            const currentKey = ALL_FIELD_KEYS[i];

            if (currentKey === 'typeMesure' && values['categorie'] === 'Actionneur') {
                continue;
            }

            let effectivePrev = prevKey;
            if (prevKey === 'typeMesure' && values['categorie'] === 'Actionneur') {
                effectivePrev = 'categorie';
            }

            if (values[effectivePrev] !== '') {
                enabled.add(currentKey);
            } else {
                break;
            }
        }

        return enabled;
    }, [values]);

    const handleChange = (key: string, value: string) => {
        setValues((prev) => {
            const next = { ...prev, [key]: value };
            const idx = ALL_FIELD_KEYS.indexOf(key);
            for (let i = idx + 1; i < ALL_FIELD_KEYS.length; i++) {
                next[ALL_FIELD_KEYS[i]] = '';
            }
            return next;
        });
    };

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        console.log('Form submitted:', values);
    };

    // Compute progress
    const filledCount = ALL_FIELD_KEYS.filter((k) => values[k] !== '').length;
    const progressPercent = Math.round((filledCount / ALL_FIELD_KEYS.length) * 100);

    const renderField = (field: FormField) => {
        const isEnabled = enabledFields.has(field.key);

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
            case 'select':
                return (
                    <div key={field.key} className={wrapperClass}>
                        <label className="block text-xs font-medium text-gray-500 mb-1">
                            {field.label}
                        </label>
                        <select
                            className={inputClass}
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
                    </div>
                );

            case 'text':
                return (
                    <div key={field.key} className={wrapperClass}>
                        <label className="block text-xs font-medium text-gray-500 mb-1">
                            {field.label}
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
                        <label className="block text-xs font-medium text-gray-500 mb-1">
                            {field.label}
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
                        <label className="block text-xs font-medium text-gray-500 mb-1">
                            {field.label}
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
            {SECTIONS.map((section) => {
                const isCollapsed = collapsedSections.has(section.title);

                return (
                    <div
                        key={section.title}
                        className="bg-gray-50/80 rounded-xl border border-gray-200 overflow-hidden"
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