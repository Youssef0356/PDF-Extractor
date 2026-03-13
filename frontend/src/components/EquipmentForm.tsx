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
// STATIC FALLBACK OPTIONS — sourced from arborescence_usine.xlsx
// These are used when the backend /schema/options endpoint is unavailable.
// ---------------------------------------------------------------------------
const STATIC_OPTIONS = {
    categories: ['Transmetteur/Capteur', 'Actionneur'],

    typesMesure: ['Pression', 'Débit', 'Température', 'Niveau', 'Analyse procédé'],

    typesActionneur: [
        'Vanne de régulation',
        'Vanne ON/OFF',
        'Électrovanne',
        'Vanne de sécurité',
        'Moteur électrique',
        'Variateur et entraînement',
        'Vérin industriel',
        'Actionneur spécial',
    ],

    codes: {
        'Débit':            ['FT', 'FI', 'FQ', 'FS', 'FSH', 'FSL'],
        'Niveau':           ['LT', 'LI', 'LG', 'LS', 'LSH', 'LSL'],
        'Pression':         ['PT', 'PI', 'PG', 'PS', 'PDT', 'PDI', 'DP'],
        'Température':      ['TT', 'TI', 'TG', 'TS', 'TSH', 'TSL'],
        'Analyse procédé':  ['AT', 'AI', 'pHT', 'O2T', 'COT', 'CO2T'],
        'Vanne de régulation':        ['CV', 'CV-BF', 'CV-GL', 'CV-DP'],
        'Vanne ON/OFF':               ['XV', 'MOV', 'AOV'],
        'Électrovanne':               ['SOV', 'EV'],
        'Vanne de sécurité':          ['SDV', 'ESD', 'PSV', 'PRV', 'BDV'],
        'Moteur électrique':          ['MTR', 'VSD'],
        'Variateur et entraînement':  ['VFD', 'DRV'],
        'Vérin industriel':           ['CYL', 'HCY'],
        'Actionneur spécial':         ['ACT'],
    } as Record<string, string[]>,

    technologies: {
        'Débit':        ['Electromagnétique', 'Ultrason', 'À turbine', 'Rotamètre', 'Coriolis', 'Vortex', 'Pression différentielle'],
        'Niveau':       ['Radar', 'Ultrason', 'Capacitif', 'Flotteur à tige', 'À pression hydrostatique', 'Radiométrique (gamma)'],
        'Pression':     ['Relative', 'Différentielle', 'Absolue'],
        'Température':  ['Thermocouple', 'RTD / PT100', 'Infrarouge', 'Bilame'],
        'Analyse procédé': ['pH', 'Conductivité', 'O2', 'COT', 'Turbidité'],
    } as Record<string, string[]>,

    typeActionneurDetail: {
        'Vanne de régulation': [
            'Pneumatique à membrane', 'Pneumatique à piston', 'Pneumatique rotatif',
            'Pneumatique quart de tour', 'Électrique multitour', 'Électrique quart de tour',
            'Servomoteur', 'Hydraulique linéaire', 'Hydraulique rotatif',
        ],
        'Vanne ON/OFF':           ['Pneumatique', 'Électrique', 'Hydraulique'],
        'Électrovanne':           ['2/2 voies', '4/2 voies', '5/3 voies', 'Directe'],
        'Vanne de sécurité':      ['Ressort', 'Pilotée'],
        'Moteur électrique':      ['Asynchrone', 'Synchrone', 'DC', 'Servomoteur'],
        'Variateur et entraînement': ['VFD', 'Servo drive', 'Soft starter', 'Drive vectoriel'],
        'Vérin industriel': [
            'Pneumatique simple effet', 'Pneumatique double effet', 'Pneumatique sans tige',
            'Pneumatique guidé', 'Pneumatique compact', 'Pneumatique rotatif',
            'Hydraulique simple effet', 'Hydraulique double effet', 'Hydraulique téléscopique',
        ],
        'Actionneur spécial': [
            'Piézoélectrique', 'Magnétique', 'Thermique', 'Électromagnétique', 'Linéaire électrique',
        ],
    } as Record<string, string[]>,

    typeVanne: {
        'Vanne de régulation': [
            'Vanne papillon', 'Vanne globe', 'Vanne bille', 'Vanne à membrane',
            'Vanne à pointeau', 'Vanne segment de bille', 'Vanne 3 voies', 'Vanne 4 voies',
            'Vanne double siège', 'Vanne angle', 'Vanne cryogénique',
        ],
        'Vanne ON/OFF': [
            'Vanne papillon ON/OFF', 'Vanne bille ON/OFF', 'Vanne guillotine',
            'Vanne à membrane', 'Vanne à tiroir', 'Vanne à clapet', 'Vanne à opercule',
        ],
        'Électrovanne': [
            'Électrovanne 2/2 voies', 'Électrovanne 4/2 voies',
            'Électrovanne 5/3 voies', 'Électrovanne directe',
        ],
    } as Record<string, string[]>,

    signals:          ['0-20mA', '4-20mA', '0-5V', '0-10V', '-/+5V', '-/+10V', '24VC', '24VAC', '3-15psi', 'NAMUR', 'Tout ou rien'],
    signalsCommande:  ['0-20mA', '4-20mA', '0-5V', '0-10V', '-/+5V', '-/+10V', '24VC', '24VAC', '3-15psi'],
    powers:           ['Boucle (2 fils)', '24VDC', '24VAC', '12-30VDC', '85-264VAC', '220VAC', 'Autre'],
    communications:   ['non', 'HART', 'Modbus RTU', 'Modbus TCP', 'Profibus DP', 'Profibus PA', 'Foundation Fieldbus', 'Profinet', 'Ethernet', 'RS-232', 'RS-485', 'NFC', 'Autre'],
    nombreFils:       ['2 fils', '3 fils', '4 fils', '5 fils'],
    brands: [
        'Siemens', 'Endress+Hauser', 'Emerson (Rosemount)', 'Emerson (Fisher)',
        'KROHNE', 'ABB', 'Yokogawa', 'Foxboro (Invensys)', 'Schneider Electric',
        'WIKA', 'VEGA Grieshaber', 'Baumer', 'SICK', 'ifm efector',
        'Turck', 'Pepperl+Fuchs', 'Danfoss', 'HARTING',
        'Samson', 'Flowserve', 'Metso (Neles)', 'Bürkert', 'ASCO',
        'Festo', 'SMC', 'Rotork', 'Auma', 'Honeywell', 'Autre',
    ],
    indicesIP:        ['IP54', 'IP65', 'IP66', 'IP67', 'IP68', 'NEMA 4', 'NEMA 4X', 'NEMA 6'],
    materials:        ['316L', '316 SS', 'Hastelloy C-276', 'PTFE', 'Céramique', 'Tantale', 'Monel', 'Autre'],
    certificats:      ['ATEX', 'IECEx', 'SIL 2', 'SIL 3', 'CE', 'FM', 'CSA', 'UL', 'non'],
    safetyPositions:  ['Fail Open (FO)', 'Fail Close (FC)', 'Fail Last (FL)'],
    applications:     ['Air comprimé', 'Gaz', 'Eau', 'Vapeur', 'Huile hydraulique', 'Fluide chimique', 'Autre'],
};

// ---------------------------------------------------------------------------
// Form layout definition
// ---------------------------------------------------------------------------
interface FormField {
    key: string;
    label: string;
    type: 'select' | 'text' | 'number' | 'boolean' | 'multi-select' | 'file';
    placeholder?: string;
    condition?: (v: Record<string, any>) => boolean;
    span?: 'full';
}

interface FormSection {
    title: string;
    icon: string;
    condition?: (v: Record<string, any>) => boolean;
    fields: FormField[];
}

const SECTIONS: FormSection[] = [
    {
        title: 'Identification',
        icon: '🏷️',
        fields: [
            { key: 'category',           label: 'Catégorie',              type: 'select' },
            { key: 'typeMesure',         label: 'Type de Mesure',         type: 'select', condition: v => v.category === 'Transmetteur/Capteur' },
            { key: 'typeActionneur',     label: "Type d'Actionneur",      type: 'select', condition: v => v.category === 'Actionneur' },
            { key: 'code',               label: 'Code ISA',               type: 'select' },
            { key: 'typeVanne',          label: 'Type de vanne',          type: 'select', condition: v => ['Vanne de régulation','Vanne ON/OFF','Électrovanne'].includes(v.typeActionneur) },
            { key: 'technologie',        label: 'Technologie',            type: 'select', condition: v => v.category === 'Transmetteur/Capteur' },
            { key: 'codeTechnologie',    label: 'Code Technologie',       type: 'text' },
            { key: 'marque',             label: 'Marque',                 type: 'select' },
            { key: 'modele',             label: 'Modèle',                 type: 'text',   placeholder: 'Ex: H250 M40' },
            { key: 'référence',          label: 'Référence',              type: 'text',   placeholder: 'Ex: 7MF4433-1DA02' },
        ],
    },
    {
        title: 'Plage de Mesure',
        icon: '📏',
        condition: v => v.category === 'Transmetteur/Capteur',
        fields: [
            { key: 'plageMesureMin',   label: 'Plage min',   type: 'number', placeholder: '0' },
            { key: 'plageMesureMax',   label: 'Plage max',   type: 'number', placeholder: '100' },
            { key: 'plageMesureUnite', label: 'Unité',       type: 'text',   placeholder: 'bar, m³/h, °C…' },
            { key: 'precision',        label: 'Précision',   type: 'text',   placeholder: '±0.5%' },
        ],
    },
    {
        title: 'Signal & Câblage',
        icon: '🔌',
        fields: [
            { key: 'signalSortie',    label: 'Signal de sortie',    type: 'select', condition: v => v.category === 'Transmetteur/Capteur' },
            { key: 'signalCommande',  label: 'Signal de commande',  type: 'select', condition: v => v.category === 'Actionneur' },
            { key: 'hart',            label: 'HART',                type: 'boolean' },
            { key: 'nombreFils',      label: 'Nombre de fils',      type: 'select' },
            { key: 'alimentation',    label: 'Alimentation',        type: 'select' },
            { key: 'communication',   label: 'Communication',       type: 'select', condition: v => v.alimentation !== 'Boucle (2 fils)' },
            { key: 'repérageArmoire', label: 'Repérage Armoire',   type: 'text' },
        ],
    },
    {
        title: 'Actionneur — Spécifications',
        icon: '⚙️',
        condition: v => v.category === 'Actionneur',
        fields: [
            { key: 'typeActionneurDetail',    label: "Type d'actionneur (détail)",  type: 'select' },
            { key: 'positionSécurité',        label: 'Position de sécurité',        type: 'select' },
            { key: 'application',             label: 'Application / fluide',        type: 'select' },
            { key: 'courseMM',                label: 'Course (mm)',                 type: 'number' },
            { key: 'forceN',                  label: 'Force (N)',                   type: 'number' },
            { key: 'pressionAlimentationBar', label: 'Pression alim. (bar)',        type: 'number' },
        ],
    },
    {
        title: 'Seuil & TOR',
        icon: '🔔',
        fields: [
            { key: 'sortieTOR',  label: 'Sortie TOR',      type: 'boolean' },
            { key: 'seuil',      label: 'Valeur de seuil', type: 'number', condition: v => !!v.sortieTOR },
            { key: 'seuilUnite', label: 'Unité seuil',     type: 'text',   condition: v => !!v.sortieTOR },
        ],
    },
    {
        title: 'Installation & Matériaux',
        icon: '🛡️',
        fields: [
            { key: 'indiceIP',           label: 'Indice IP',           type: 'select' },
            { key: 'certificats',        label: 'Certificats',         type: 'multi-select', span: 'full' },
            { key: 'températureProcess', label: 'Température Process', type: 'text',   placeholder: '-40…+120°C' },
            {
                key: 'matériauMembrane', label: 'Matériau Membrane', type: 'select',
                condition: v => v.category === 'Transmetteur/Capteur' && ['Pression', 'Niveau'].includes(v.typeMesure || ''),
            },
        ],
    },
    {
        title: 'Documentation',
        icon: '📄',
        fields: [
            { key: 'datasheetUrl', label: 'Datasheet (PDF)', type: 'file', span: 'full' },
        ],
    },
];

const ALL_KEYS = SECTIONS.flatMap(s => s.fields.map(f => f.key));

// ---------------------------------------------------------------------------
// Option resolver — backend first, static fallback
// ---------------------------------------------------------------------------
function resolveOptions(key: string, vals: Record<string, any>, bo: SchemaOptions | null): string[] {
    switch (key) {
        case 'category':           return bo?.categories      ?? STATIC_OPTIONS.categories;
        case 'typeMesure':         return bo?.typesMesure     ?? STATIC_OPTIONS.typesMesure;
        case 'typeActionneur':     return bo?.typesActionneur ?? STATIC_OPTIONS.typesActionneur;
        case 'code': {
            const parent = vals.typeMesure || vals.typeActionneur || '';
            return (bo?.codes ?? STATIC_OPTIONS.codes)[parent] ?? [];
        }
        case 'typeVanne':          return STATIC_OPTIONS.typeVanne[vals.typeActionneur || ''] ?? [];
        case 'technologie': {
            const parent = vals.typeMesure || '';
            return (bo?.technologies ?? STATIC_OPTIONS.technologies)[parent] ?? [];
        }
        case 'typeActionneurDetail': return STATIC_OPTIONS.typeActionneurDetail[vals.typeActionneur || ''] ?? [];
        case 'signalSortie':
        case 'signalCommande':    return bo?.signals        ?? STATIC_OPTIONS.signals;
        case 'alimentation':      return bo?.powers         ?? STATIC_OPTIONS.powers;
        case 'communication':     return bo?.communications ?? STATIC_OPTIONS.communications;
        case 'nombreFils':        return STATIC_OPTIONS.nombreFils;
        case 'marque':            return bo?.brands         ?? STATIC_OPTIONS.brands;
        case 'indiceIP':          return bo?.indicesIP      ?? STATIC_OPTIONS.indicesIP;
        case 'matériauMembrane':  return bo?.materials      ?? STATIC_OPTIONS.materials;
        case 'positionSécurité':  return bo?.safetyPositions ?? STATIC_OPTIONS.safetyPositions;
        case 'application':       return STATIC_OPTIONS.applications;
        default:                  return [];
    }
}

// Confidence colour helpers
const confColor = (c: number) => {
    if (c >= 0.9) return { bg: 'bg-emerald-100', text: 'text-emerald-700', dot: 'bg-emerald-500' };
    if (c >= 0.6) return { bg: 'bg-amber-100',   text: 'text-amber-700',   dot: 'bg-amber-400'  };
    return              { bg: 'bg-red-100',       text: 'text-red-600',     dot: 'bg-red-400'    };
};

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------
interface EquipmentFormProps {
    initialData?: Record<string, any> | null;
    confidence?: Record<string, number> | null;
    isProcessing?: boolean;
    onReset?: () => void;
}

const EquipmentForm = ({ initialData, confidence: initConf, isProcessing: extProc, onReset }: EquipmentFormProps) => {
    const [backendOpts, setBackendOpts]   = useState<SchemaOptions | null>(null);
    const [optsError, setOptsError]       = useState(false);
    const [vals, setVals]                 = useState<Record<string, any>>({});
    const [aiVals, setAiVals]             = useState<Record<string, any>>({});
    const [conf, setConf]                 = useState<Record<string, number>>({});
    const [internalProc, setInternalProc] = useState(false);
    const [pdfId, setPdfId]               = useState<string | null>(null);
    const [showModal, setShowModal]       = useState(false);
    const [pendingCorr, setPendingCorr]   = useState<CorrectionRecord[]>([]);
    const [savedPopup, setSavedPopup]     = useState(false);

    const isProc = extProc || internalProc;

    // Attempt to load backend options non-blocking; static fallback always works
    useEffect(() => {
        fetchSchemaOptions()
            .then(setBackendOpts)
            .catch(() => setOptsError(true));
    }, []);

    useEffect(() => {
        if (initialData) { setVals(initialData); setAiVals(initialData); }
        if (initConf)    setConf(initConf as Record<string, number>);
    }, [initialData, initConf]);

    const set = (key: string, value: any) => setVals(prev => ({ ...prev, [key]: value }));

    const handlePDFUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (!file) return;
        setInternalProc(true);
        try {
            const res = await extractFromPDF(file);
            if (res.success && res.data) {
                setAiVals(res.data as any);
                setVals(res.data as any);
                setConf((res.confidence || {}) as Record<string, number>);
                setPdfId(res.meta?.pdf_id || null);
            }
        } finally { setInternalProc(false); }
    };

    const handleReset = () => {
        setVals({}); setAiVals({}); setConf({}); setPdfId(null);
        onReset?.();
    };

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        const corrs: CorrectionRecord[] = ALL_KEYS
            .filter(k => JSON.stringify(vals[k]) !== JSON.stringify(aiVals[k]))
            .map(k => ({ field_name: k, ai_extracted_value: aiVals[k], user_corrected_value: vals[k], accepted: false }));
        if (corrs.length > 0) { setPendingCorr(corrs); setShowModal(true); }
        else finalize([]);
    };

    const finalize = async (corrs: CorrectionRecord[]) => {
        if (pdfId) {
            const batch: CorrectionBatch = {
                pdf_id: pdfId,
                category: vals.category,
                typeMesure: vals.typeMesure,
                corrections: corrs,
            };
            await submitCorrections(batch);
        }
        setShowModal(false);
        setSavedPopup(true);
        setTimeout(() => setSavedPopup(false), 3000);
    };

    // Progress metrics
    const metrics = useMemo(() => {
        let filled = 0, confirmed = 0, uncertain = 0;
        ALL_KEYS.forEach(k => {
            const v = vals[k];
            const ok = v !== undefined && v !== null && v !== '' && !(Array.isArray(v) && v.length === 0);
            if (ok) {
                filled++;
                const c = conf[k] || 0;
                if (c >= 0.9) confirmed++;
                else if (c >= 0.01) uncertain++;
            }
        });
        return { filled, confirmed, uncertain, empty: ALL_KEYS.length - filled, pct: Math.round(filled / ALL_KEYS.length * 100) };
    }, [vals, conf]);

    // ── Field renderer ──────────────────────────────────────────────────────
    const renderField = (field: FormField) => {
        if (field.condition && !field.condition(vals)) return null;

        const v       = vals[field.key];
        const aiV     = aiVals[field.key];
        const hasAi   = aiV !== undefined && aiV !== null && aiV !== '';
        const modified = JSON.stringify(v) !== JSON.stringify(aiV);
        const c        = conf[field.key] || 0;
        const colors   = hasAi ? confColor(c) : null;

        const spanClass = field.span === 'full' ? 'sm:col-span-2 lg:col-span-3' : '';

        const base   = 'w-full px-3 py-2 rounded-lg border text-sm font-medium outline-none transition-all duration-150 ';
        const border = modified
            ? 'border-amber-300 bg-amber-50 focus:border-amber-400'
            : hasAi
                ? 'border-blue-200 bg-blue-50/40 focus:border-blue-400'
                : 'border-slate-200 bg-white focus:border-blue-400';

        const labelRow = (
            <div className="flex items-center justify-between mb-1">
                <label className="text-[10px] font-bold uppercase tracking-widest text-slate-500">
                    {field.label}
                </label>
                {hasAi && (
                    <span className={`inline-flex items-center gap-1 text-[9px] font-bold px-1.5 py-0.5 rounded-full ${colors!.bg} ${colors!.text}`}>
                        <span className={`w-1.5 h-1.5 rounded-full ${colors!.dot}`} />
                        {Math.round(c * 100)}%
                    </span>
                )}
            </div>
        );

        if (field.type === 'boolean') {
            return (
                <div key={field.key} className={`flex items-center justify-between px-3 py-2 rounded-lg border border-slate-200 bg-white ${spanClass}`}>
                    <span className="text-[10px] font-bold uppercase tracking-widest text-slate-500">{field.label}</span>
                    <button type="button" onClick={() => set(field.key, !v)}
                        className={`relative w-10 h-5 rounded-full transition-colors duration-200 ${v ? 'bg-blue-600' : 'bg-slate-300'}`}>
                        <span className={`absolute top-0.5 w-4 h-4 bg-white rounded-full shadow transition-all duration-200 ${v ? 'left-5' : 'left-0.5'}`} />
                    </button>
                </div>
            );
        }

        if (field.type === 'multi-select') {
            const selected: string[] = Array.isArray(v) ? v : [];
            return (
                <div key={field.key} className={spanClass}>
                    {labelRow}
                    <div className="flex flex-wrap gap-1.5 p-2 rounded-lg border border-slate-200 bg-white min-h-[36px]">
                        {STATIC_OPTIONS.certificats.map(opt => {
                            const active = selected.includes(opt);
                            return (
                                <button key={opt} type="button"
                                    onClick={() => set(field.key, active ? selected.filter(x => x !== opt) : [...selected, opt])}
                                    className={`text-[10px] font-bold px-2 py-0.5 rounded-md border transition-all ${active
                                        ? 'bg-blue-600 text-white border-blue-600'
                                        : 'bg-white text-slate-500 border-slate-200 hover:border-blue-300'}`}>
                                    {opt}
                                </button>
                            );
                        })}
                    </div>
                </div>
            );
        }

        if (field.type === 'file') {
            return (
                <div key={field.key} className={spanClass}>
                    {labelRow}
                    <label className="inline-flex items-center gap-2 px-3 py-2 rounded-lg border-2 border-dashed border-slate-300 bg-slate-50 text-xs font-bold text-slate-500 cursor-pointer hover:border-blue-300 hover:text-blue-600 transition-all w-full">
                        <UploadIcon />
                        {v ? String(v).split('/').pop() : 'Cliquer pour joindre un PDF'}
                        <input type="file" accept=".pdf" className="hidden"
                            onChange={e => set(field.key, e.target.files?.[0]?.name || '')} />
                    </label>
                </div>
            );
        }

        if (field.type === 'select') {
            const opts = resolveOptions(field.key, vals, backendOpts);
            const needsParent = opts.length === 0 && ['code', 'technologie', 'typeVanne', 'typeActionneurDetail'].includes(field.key);
            return (
                <div key={field.key} className={spanClass}>
                    {labelRow}
                    <select className={base + border + ' cursor-pointer'}
                        value={String(v ?? '')}
                        onChange={e => set(field.key, e.target.value || undefined)}>
                        <option value="">— sélectionner —</option>
                        {opts.map(o => <option key={o} value={o}>{o}</option>)}
                    </select>
                    {needsParent && (
                        <p className="text-[9px] text-slate-400 mt-0.5 pl-1">Sélectionnez le type parent d'abord</p>
                    )}
                </div>
            );
        }

        // text / number
        return (
            <div key={field.key} className={spanClass}>
                {labelRow}
                <input
                    type={field.type === 'number' ? 'number' : 'text'}
                    step={field.type === 'number' ? 'any' : undefined}
                    className={base + border}
                    value={v === undefined || v === null ? '' : String(v)}
                    placeholder={field.placeholder}
                    onChange={e => {
                        const raw = e.target.value;
                        set(field.key, field.type === 'number'
                            ? (raw === '' ? undefined : parseFloat(raw))
                            : (raw || undefined));
                    }}
                />
            </div>
        );
    };

    const visibleSections = SECTIONS.filter(s => !s.condition || s.condition(vals));

    // ── Render ──────────────────────────────────────────────────────────────
    return (
        <div className="space-y-3">

            {/* ── Top progress bar ─────────────────────────────────────── */}
            <div className="bg-white border border-slate-100 rounded-2xl p-4 shadow-sm">
                <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2 flex-wrap">
                        <span className="text-xs font-black text-slate-700">Extraction IA</span>
                        {isProc && (
                            <span className="flex items-center gap-1 text-[10px] font-bold text-blue-600 bg-blue-50 px-2 py-0.5 rounded-full">
                                <Spinner /> Analyse…
                            </span>
                        )}
                        {optsError && (
                            <span className="text-[9px] text-amber-600 bg-amber-50 border border-amber-200 px-2 py-0.5 rounded-full font-bold">
                                ⚠ Backend hors ligne — options intégrées actives
                            </span>
                        )}
                    </div>
                    <span className="text-2xl font-black tabular-nums text-slate-800">{metrics.pct}%</span>
                </div>

                <div className="h-2 bg-slate-100 rounded-full overflow-hidden mb-2.5">
                    <div className="h-full rounded-full transition-all duration-700"
                        style={{
                            width: `${metrics.pct}%`,
                            background: metrics.pct >= 80
                                ? 'linear-gradient(90deg,#10b981,#059669)'
                                : metrics.pct >= 40
                                    ? 'linear-gradient(90deg,#3b82f6,#2563eb)'
                                    : 'linear-gradient(90deg,#94a3b8,#64748b)',
                        }}
                    />
                </div>

                <div className="flex items-center gap-3 flex-wrap">
                    <Pill color="bg-emerald-100 text-emerald-700" dot="bg-emerald-500">{metrics.confirmed} confirmés</Pill>
                    <Pill color="bg-amber-100 text-amber-700"   dot="bg-amber-400"  >{metrics.uncertain} incertains</Pill>
                    <Pill color="bg-slate-100 text-slate-500"   dot="bg-slate-300"  >{metrics.empty} vides</Pill>
                    <div className="ml-auto">
                        <label className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-lg border text-[11px] font-bold cursor-pointer transition-all
                            ${isProc
                                ? 'border-slate-200 text-slate-400 bg-slate-50 cursor-not-allowed'
                                : 'border-blue-200 text-blue-600 bg-blue-50 hover:bg-blue-100'}`}>
                            {isProc ? <><Spinner /> Analyse…</> : <><UploadIcon /> Charger Datasheet</>}
                            <input type="file" accept=".pdf" className="hidden" onChange={handlePDFUpload} disabled={isProc} />
                        </label>
                    </div>
                </div>
            </div>

            {/* ── Form sections ─────────────────────────────────────────── */}
            <form onSubmit={handleSubmit} className="space-y-3">
                {visibleSections.map(section => (
                    <div key={section.title} className="bg-white border border-slate-100 rounded-2xl shadow-sm overflow-hidden">
                        <div className="flex items-center gap-2 px-5 py-3 border-b border-slate-50 bg-slate-50/60">
                            <span className="text-base">{section.icon}</span>
                            <h3 className="text-[11px] font-black uppercase tracking-widest text-slate-600">{section.title}</h3>
                        </div>
                        <div className="p-4 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-x-4 gap-y-3">
                            {section.fields.map(f => renderField(f))}
                        </div>
                    </div>
                ))}

                <div className="flex items-center justify-end gap-2 pt-1 pb-2">
                    <button type="button" onClick={handleReset}
                        className="px-4 py-2 text-xs font-bold text-slate-500 border border-slate-200 rounded-lg hover:bg-slate-50 transition-all">
                        Réinitialiser
                    </button>
                    <button type="submit"
                        className="inline-flex items-center gap-2 px-5 py-2 text-xs font-bold text-white bg-slate-900 rounded-lg hover:bg-slate-700 transition-all">
                        <SaveIcon /> Enregistrer l'équipement
                    </button>
                </div>
            </form>

            {/* ── Correction modal ──────────────────────────────────────── */}
            {showModal && (
                <div className="fixed inset-0 bg-slate-900/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
                    <div className="bg-white rounded-2xl shadow-2xl w-full max-w-lg overflow-hidden border border-slate-100 flex flex-col max-h-[85vh]">
                        <div className="px-6 py-4 border-b border-slate-100 flex items-center justify-between shrink-0">
                            <div>
                                <h3 className="text-base font-black text-slate-900">Vérification Finale</h3>
                                <p className="text-[10px] text-slate-400 font-bold uppercase tracking-widest mt-0.5">
                                    {pendingCorr.length} correction(s) détectée(s)
                                </p>
                            </div>
                            <button onClick={() => setShowModal(false)} className="p-2 hover:bg-slate-50 rounded-lg text-slate-400">
                                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M6 18L18 6M6 6l12 12"/>
                                </svg>
                            </button>
                        </div>

                        <div className="p-6 overflow-y-auto space-y-4 flex-1">
                            {pendingCorr.map((corr, idx) => (
                                <div key={corr.field_name} className="p-4 rounded-xl bg-slate-50 border border-slate-100 space-y-3">
                                    <span className="inline-block text-[9px] font-black text-white bg-slate-800 px-2 py-0.5 rounded-md uppercase tracking-widest">
                                        {corr.field_name}
                                    </span>
                                    <div className="flex items-center gap-3 text-xs">
                                        <div className="flex-1">
                                            <p className="text-[9px] font-bold text-slate-400 uppercase mb-0.5">IA</p>
                                            <p className="font-bold text-red-500 line-through truncate">{String(corr.ai_extracted_value ?? '—')}</p>
                                        </div>
                                        <svg className="w-4 h-4 text-slate-400 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6"/>
                                        </svg>
                                        <div className="flex-1">
                                            <p className="text-[9px] font-bold text-slate-400 uppercase mb-0.5">Vous</p>
                                            <p className="font-bold text-emerald-600 truncate">{String(corr.user_corrected_value ?? '—')}</p>
                                        </div>
                                    </div>
                                    <textarea rows={2} placeholder="Règle optionnelle…"
                                        className="w-full px-3 py-2 rounded-lg border border-slate-200 text-xs resize-none outline-none focus:border-blue-400"
                                        value={corr.rule || ''}
                                        onChange={e => {
                                            const next = [...pendingCorr];
                                            next[idx] = { ...next[idx], rule: e.target.value };
                                            setPendingCorr(next);
                                        }} />
                                </div>
                            ))}
                        </div>

                        <div className="px-6 py-4 border-t border-slate-100 flex items-center justify-between shrink-0">
                            <button onClick={() => finalize(pendingCorr)} className="text-xs font-bold text-slate-400 hover:text-slate-600 transition-colors">
                                Ignorer les règles
                            </button>
                            <button onClick={() => finalize(pendingCorr)}
                                className="px-5 py-2 bg-blue-600 text-white rounded-lg text-xs font-bold hover:bg-blue-700 transition-all">
                                Confirmer et sauvegarder
                            </button>
                        </div>
                    </div>
                </div>
            )}

            {/* ── Saved notification ────────────────────────────────────── */}
            {savedPopup && (
                <div className="fixed bottom-8 left-1/2 -translate-x-1/2 z-[60]">
                    <div className="bg-emerald-600 text-white px-5 py-3 rounded-xl shadow-2xl flex items-center gap-3 border border-emerald-500/50">
                        <div className="w-6 h-6 rounded-full bg-white/20 flex items-center justify-center">
                            <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7"/>
                            </svg>
                        </div>
                        <div>
                            <p className="text-xs font-black">Enregistré !</p>
                            <p className="text-[9px] font-bold opacity-70 uppercase tracking-widest">Base de connaissances mise à jour</p>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

// ---------------------------------------------------------------------------
// Reusable atoms
// ---------------------------------------------------------------------------
const Pill = ({ color, dot, children }: { color: string; dot: string; children: React.ReactNode }) => (
    <span className={`inline-flex items-center gap-1.5 text-[10px] font-bold px-2 py-0.5 rounded-full ${color}`}>
        <span className={`w-1.5 h-1.5 rounded-full ${dot}`} /> {children}
    </span>
);

const Spinner = () => (
    <svg className="w-3 h-3 animate-spin" viewBox="0 0 24 24" fill="none">
        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z"/>
    </svg>
);

const UploadIcon = () => (
    <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12"/>
    </svg>
);

const SaveIcon = () => (
    <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M8 7H5a2 2 0 00-2 2v9a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-3m-1 4l-3 3m0 0l-3-3m3 3V4"/>
    </svg>
);

export default EquipmentForm;
