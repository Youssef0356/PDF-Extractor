"""
Centralized dropdown options, enum maps, and taxonomy constants for Schema V2.

All UI dropdowns, LLM allowed-values, and normalization aliases
reference this single source of truth.
"""

# ---------------------------------------------------------------------------
# Category (Dropdown 1)
# ---------------------------------------------------------------------------

CATEGORY_OPTIONS = [
    "Transmetteur/Capteur",
    "Actionneur",
]

# ---------------------------------------------------------------------------
# Type (Dropdown 2) — depends on category
# ---------------------------------------------------------------------------

TYPE_MESURE_OPTIONS = [
    "Pression",
    "Débit",
    "Température",
    "Niveau",
    "Analyse procédé",
]

TYPE_ACTIONNEUR_OPTIONS = [
    "Vanne de régulation",
    "Vanne ON/OFF",
    "Actionneur mécanique",
    "Régulation & contrôle",
    "Équipement auxiliaire",
]

# ---------------------------------------------------------------------------
# Equipment Code (Dropdown 3) — depends on Dropdown 2
# ---------------------------------------------------------------------------

CODE_MAP: dict[str, list[str]] = {
    # Transmetteur / Capteur
    "Pression":          ["PT", "PI", "PG", "PS", "PDT", "PDI"],
    "Débit":             ["FT", "FI", "FQ", "FS", "FSH", "FSL"],
    "Température":       ["TT", "TI", "TG", "TS", "TSH", "TSL"],
    "Niveau":            ["LT", "LI", "LG", "LS", "LSH", "LSL"],
    "Analyse procédé":   ["AT", "AI", "pHT", "O2T", "COT", "CO2T"],
    # Actionneur
    "Vanne de régulation":    ["CV", "PCV", "FCV", "LCV", "TCV"],
    "Vanne ON/OFF":           ["XV", "MOV", "AOV", "SOV", "SDV"],
    "Actionneur mécanique":   ["ACT", "CYL", "HCY", "MTR", "VSD"],
    "Régulation & contrôle":  ["PIC", "TIC", "FIC", "LIC", "PLC", "DCS"],
    "Équipement auxiliaire":  ["PSV", "PRV", "BDV", "DMP", "FIL"],
}

# Flat set of ALL valid codes (for validation)
ALL_CODES: set[str] = set()
for _codes in CODE_MAP.values():
    ALL_CODES.update(_codes)

# ---------------------------------------------------------------------------
# ISA first-letter → typeMesure / category mapping
# ---------------------------------------------------------------------------

FIRST_LETTER_TO_TYPE: dict[str, str] = {
    "F": "Débit",
    "L": "Niveau",
    "P": "Pression",
    "T": "Température",
    "A": "Analyse procédé",
}

FIRST_LETTER_TO_CATEGORY: dict[str, str] = {
    "F": "Transmetteur/Capteur",
    "L": "Transmetteur/Capteur",
    "P": "Transmetteur/Capteur",
    "T": "Transmetteur/Capteur",
    "A": "Transmetteur/Capteur",
    "C": "Actionneur",
    "X": "Actionneur",
    "M": "Actionneur",
    "D": "Actionneur",
    "H": "Actionneur",
    "V": "Actionneur",
    "B": "Actionneur",
}

# Reverse map: code → typeMesure or typeActionneur
CODE_TO_TYPE: dict[str, str] = {}
for _type_key, _codes in CODE_MAP.items():
    for _c in _codes:
        CODE_TO_TYPE[_c] = _type_key

# ---------------------------------------------------------------------------
# Technology (Dropdown) — depends on typeMesure
# ---------------------------------------------------------------------------

TECHNOLOGY_MAP: dict[str, list[str]] = {
    "Débit": [
        "Électromagnétique",
        "Ultrason",
        "À turbine",
        "Rotamètre",
        "Coriolis",
        "Vortex",
        "Pression différentielle",
        "Autre",
    ],
    "Niveau": [
        "Radar",
        "Ultrason",
        "Capacitif",
        "Flotteur à tige",
        "À pression hydrostatique",
        "Radiométrique (gamma)",
        "Autre",
    ],
    "Pression": [
        "Relative",
        "Différentielle",
        "Autre",
    ],
    "Température": [
        "Thermocouple",
        "RTD (Pt100/Pt1000)",
        "Infrarouge",
        "Bimétallique",
        "Autre",
    ],
    "Analyse procédé": [
        "Électrochimique",
        "Optique",
        "Spectroscopique",
        "Autre",
    ],
}

# Flat set of ALL valid technologies
ALL_TECHNOLOGIES: set[str] = set()
for _techs in TECHNOLOGY_MAP.values():
    ALL_TECHNOLOGIES.update(_techs)

# ---------------------------------------------------------------------------
# Signal de sortie
# ---------------------------------------------------------------------------

SIGNAL_SORTIE_OPTIONS = [
    "4-20mA",
    "0-20mA",
    "0-5V",
    "0-10V",
    "-/+5V",
    "-/+10V",
    "Autre",
]

# ---------------------------------------------------------------------------
# Nombre de fils
# ---------------------------------------------------------------------------

NOMBRE_FILS_OPTIONS = [2, 3, 4, 5]

# ---------------------------------------------------------------------------
# Alimentation
# ---------------------------------------------------------------------------

ALIMENTATION_OPTIONS = [
    "boucle",
    "24VDC",
    "24VAC",
    "220VAC",
    "Autre",
]

# ---------------------------------------------------------------------------
# Communication
# ---------------------------------------------------------------------------

COMMUNICATION_OPTIONS = [
    "non",
    "HART",
    "Modbus",
    "Profibus",
    "Profinet",
    "NFC",
    "Autre",
]

# ---------------------------------------------------------------------------
# Marque (manufacturer)
# ---------------------------------------------------------------------------

MARQUE_OPTIONS = [
    "Siemens",
    "Endress+Hauser",
    "Emerson (Rosemount)",
    "KROHNE",
    "ABB",
    "Yokogawa",
    "Foxboro (Invensys)",
    "Schneider Electric",
    "WIKA",
    "VEGA Grieshaber",
    "Baumer",
    "SICK",
    "ifm efector",
    "Turck",
    "Pepperl+Fuchs",
    "Danfoss",
    "HARTING",
    "Autre",
]

# ---------------------------------------------------------------------------
# Certificats (multi-select)
# ---------------------------------------------------------------------------

CERTIFICATS_OPTIONS = [
    "ATEX",
    "IECEx",
    "SIL",
    "SIL 2",
    "SIL 3",
    "FM",
    "CSA",
]

# ---------------------------------------------------------------------------
# Indice IP
# ---------------------------------------------------------------------------

INDICE_IP_OPTIONS = [
    "IP65",
    "IP66",
    "IP67",
    "IP68",
    "IP69K",
    "NEMA4X",
]

# ---------------------------------------------------------------------------
# Matériau membrane
# ---------------------------------------------------------------------------

MATERIAU_MEMBRANE_OPTIONS = [
    "316L",
    "Hastelloy",
    "PTFE",
    "Céramique",
    "Autre",
]

# ---------------------------------------------------------------------------
# Actionneur-specific options
# ---------------------------------------------------------------------------

TYPE_VERIN_OPTIONS = [
    "Pneumatique simple effet",
    "Pneumatique double effet",
    "Hydraulique simple effet",
    "Hydraulique double effet",
    "Autre",
]

TYPE_ACTIONNEUR_SPECIAL_OPTIONS = [
    "Piézoélectrique",
    "Magnétique",
    "Thermique",
    "Électromagnétique",
    "Linéaire électrique",
]

POSITION_SECURITE_OPTIONS = [
    "Fail Open",
    "Fail Close",
    "Fail Last",
]

# ---------------------------------------------------------------------------
# Fields hidden per category (conditional visibility)
# ---------------------------------------------------------------------------

ACTIONNEUR_HIDDEN_FIELDS = {
    "signalSortie",
    "hart",
    "nombreFils",
    "alimentation",
    "communication",
    "matériauMembrane",
}

# Fields that only show when sortieTOR = True
SORTIE_TOR_FIELDS = {"seuil", "seuilUnite"}

# Fields that only show when alimentation ≠ boucle
COMMUNICATION_VISIBLE_CONDITION = "alimentation_not_boucle"

# Fields that only show for Pression or Niveau
MATERIAU_MEMBRANE_TYPES = {"Pression", "Niveau"}

# ---------------------------------------------------------------------------
# Manual-only fields (AI never fills these)
# ---------------------------------------------------------------------------

MANUAL_ONLY_FIELDS = {
    "référence",
    "repérageArmoire",
    "datasheetUrl",
    "seuilUnite",
    "codeTechnologie",
}
