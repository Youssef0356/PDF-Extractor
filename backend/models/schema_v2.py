"""Pydantic models for Schema V2 — the complete instrument/actionneur schema.

This replaces the old schema.py (now archived as schema_v1_archive.py).
"""

from pydantic import BaseModel, Field
from typing import Optional, Any


class BaseInstrument(BaseModel):
    """Base fields shared by ALL instruments (Transmetteur/Capteur AND Actionneur)."""

    # --- Identity (AI fills these) ---
    category: Optional[str] = Field(
        None,
        description="Transmetteur/Capteur | Actionneur",
    )
    typeMesure: Optional[str] = Field(
        None,
        description="Pression | Débit | Température | Niveau | Analyse procédé",
    )
    code: Optional[str] = Field(
        None,
        description="ISA instrument code: PT, FT, TT, LT, AT, CV, XV, etc.",
    )
    technologie: Optional[str] = Field(
        None,
        description="Measurement technology (depends on typeMesure)",
    )
    codeTechnologie: Optional[str] = Field(
        None,
        description="Technology code (manual only, AI never fills)",
    )

    # --- Measurement range ---
    plageMesureMin: Optional[float] = Field(None, description="Measurement range minimum")
    plageMesureMax: Optional[float] = Field(None, description="Measurement range maximum")
    plageMesureUnite: Optional[str] = Field(None, description="Measurement range unit")

    # --- Signal ---
    signalSortie: Optional[str] = Field(
        None,
        description="4-20mA | 0-20mA | 0-5V | 0-10V | -/+5V | -/+10V | Autre",
    )
    hart: Optional[bool] = Field(None, description="HART protocol support: true | false")

    # --- Wiring ---
    nombreFils: Optional[int] = Field(None, description="Number of wires: 2 | 3 | 4 | 5")
    alimentation: Optional[str] = Field(
        None,
        description="boucle | 24VDC | 24VAC | 220VAC | Autre",
    )

    # --- Communication (shown only if alimentation ≠ boucle) ---
    communication: Optional[str] = Field(
        None,
        description="non | HART | Modbus | Profibus | Profinet | NFC | Autre",
    )

    # --- TOR (conditional block) ---
    sortieTOR: Optional[bool] = Field(None, description="TOR output: true | false")
    seuil: Optional[float] = Field(
        None,
        description="Threshold value (shown only if sortieTOR = true)",
    )
    seuilUnite: Optional[str] = Field(
        None,
        description="Threshold unit (manual only, AI never fills)",
    )

    # --- Plant tag (manual) ---
    repérageArmoire: Optional[str] = Field(
        None,
        description="Cabinet/panel tag (manual only, AI never fills)",
    )

    # --- Performance ---
    precision: Optional[str] = Field(
        None,
        description='Accuracy, e.g. "±0.5%", "0.1°C"',
    )

    # --- Manufacturer ---
    marque: Optional[str] = Field(None, description="Manufacturer brand name")
    référence: Optional[str] = Field(
        None,
        description="Product reference (manual only, AI never fills)",
    )

    # --- Certifications (multi-select) ---
    certificats: Optional[list[str]] = Field(
        None,
        description='Certifications list: ["ATEX", "IECEx", "SIL"] etc.',
    )

    # --- Protection ---
    indiceIP: Optional[str] = Field(
        None,
        description="IP65 | IP66 | IP67 | IP68 | NEMA4X ...",
    )

    # --- Process conditions ---
    températureProcess: Optional[str] = Field(
        None,
        description='Process temperature range, e.g. "-40...+300°C"',
    )
    matériauMembrane: Optional[str] = Field(
        None,
        description="316L | Hastelloy | PTFE | Céramique | Autre",
    )

    # --- Datasheet ---
    datasheetUrl: Optional[str] = Field(
        None,
        description="Datasheet file path (manual only, user uploads)",
    )


class ActionneurInstrument(BaseInstrument):
    """Extended schema for Actionneur-category instruments."""

    typeActionneur: Optional[str] = Field(
        None,
        description=(
            "Vanne de régulation | Vanne ON/OFF | Actionneur mécanique | "
            "Régulation & contrôle | Équipement auxiliaire"
        ),
    )
    typeVérin: Optional[str] = Field(
        None,
        description=(
            "Pneumatique simple effet | Pneumatique double effet | "
            "Hydraulique simple effet | Hydraulique double effet | Autre"
        ),
    )
    typeActionneurSpécial: Optional[str] = Field(
        None,
        description=(
            "Piézoélectrique | Magnétique | Thermique | "
            "Électromagnétique | Linéaire électrique"
        ),
    )
    positionSécurité: Optional[str] = Field(
        None,
        description="Fail Open | Fail Close | Fail Last",
    )
    courseMM: Optional[float] = Field(None, description="Stroke in mm")
    forceN: Optional[float] = Field(None, description="Force in N")
    pressionAlimentationBar: Optional[float] = Field(
        None,
        description="Supply pressure in bar",
    )


class ExtractionResponse(BaseModel):
    """API response for PDF extraction (V2)."""

    success: bool
    data: Optional[dict[str, Any]] = None
    confidence: Optional[dict[str, float]] = None
    message: str = ""
    processing_time_seconds: Optional[float] = None
    meta: Optional[dict[str, Any]] = None
    evidence: Optional[dict[str, list[dict[str, Any]]]] = None
    doc_context: Optional[dict[str, Any]] = None
    timings_ms: Optional[dict[str, float]] = None


# ---------------------------------------------------------------------------
# Field metadata for semantic search + LLM prompts (V2)
# ---------------------------------------------------------------------------
# Each field has:
#   - description:      human-readable description
#   - allowed_values:   list of allowed enum values, or None for open fields
#   - search_queries:   queries used by the vector store to retrieve chunks
#   - ai_fills:         True if AI should attempt extraction, False if manual-only
# ---------------------------------------------------------------------------

FIELD_DESCRIPTIONS_V2: dict[str, dict] = {

    "category": {
        "description": "Top-level category: Transmetteur/Capteur or Actionneur",
        "allowed_values": ["Transmetteur/Capteur", "Actionneur"],
        "search_queries": [
            "sensor transmitter actuator valve controller",
            "capteur transmetteur actionneur vanne régulateur",
            "this instrument is a transmitter actuator",
            "type of device equipment category",
        ],
        "ai_fills": True,
    },

    "typeMesure": {
        "description": "Physical quantity measured: Pression | Débit | Température | Niveau | Analyse procédé",
        "allowed_values": ["Pression", "Débit", "Température", "Niveau", "Analyse procédé"],
        "search_queries": [
            "measurement type flow level pressure temperature",
            "type de mesure débit niveau pression température",
            "this instrument measures",
            "functional principle application",
        ],
        "ai_fills": True,
    },

    "code": {
        "description": "ISA instrument code (PT, FT, TT, LT, AT, CV, XV...)",
        "allowed_values": None,  # validated against CODE_MAP dynamically
        "search_queries": [
            "instrument code PT FT TT LT transmitter indicator",
            "transmetteur indicateur manomètre pressostat",
            "equipment designation type ISA tag",
            "instrument tag FT PT LT TT AT CV XV",
        ],
        "ai_fills": True,
    },

    "technologie": {
        "description": "Measurement/actuation technology",
        "allowed_values": None,  # validated against TECHNOLOGY_MAP dynamically
        "search_queries": [
            "measurement principle technology electromagnetic radar coriolis",
            "principe de mesure technologie électromagnétique radar ultrason",
            "operating principle sensor type",
        ],
        "ai_fills": True,
    },

    "codeTechnologie": {
        "description": "Technology code (manual only)",
        "allowed_values": None,
        "search_queries": [],
        "ai_fills": False,
    },

    "plageMesureMin": {
        "description": "Measurement range minimum value",
        "allowed_values": None,
        "search_queries": [
            "measuring range plage de mesure min max",
            "span range flow rate level pressure",
            "0 to 100 measurement span",
        ],
        "ai_fills": True,
    },

    "plageMesureMax": {
        "description": "Measurement range maximum value",
        "allowed_values": None,
        "search_queries": [
            "measuring range plage de mesure min max",
            "span range flow rate level pressure",
            "0 to 100 measurement span",
        ],
        "ai_fills": True,
    },

    "températureProcess": {
        "description": 'Process temperature range, e.g. "-40...+300°C"',
        "allowed_values": None,
        "search_queries": [
            "process temperature operating temperature range",
            "température du produit température de service",
            "temperature limits min max operating",
        ],
        "ai_fills": True,
    },

    "plageMesureUnite": {
        "description": "Measurement range unit, e.g. bar, m3/h, °C, Pa",
        "allowed_values": None,
        "search_queries": [
            "measuring range unit plage de mesure unité",
            "bar mbar Pa kPa L/h m³/h °C",
        ],
        "ai_fills": True,
    },

    "signalSortie": {
        "description": "Output signal: 4-20mA | 0-20mA | 0-5V | 0-10V | -/+5V | -/+10V | Autre",
        "allowed_values": ["4-20mA", "0-20mA", "0-5V", "0-10V", "-/+5V", "-/+10V", "Autre"],
        "search_queries": [
            "output signal 4-20mA signal de sortie",
            "analog output current voltage signal",
            "sortie analogique courant tension",
        ],
        "ai_fills": True,
    },

    "hart": {
        "description": "HART protocol support (true/false)",
        "allowed_values": [True, False],
        "search_queries": [
            "HART communication protocol HART 5 HART 7",
            "highway addressable remote transducer",
        ],
        "ai_fills": True,
    },

    "nombreFils": {
        "description": "Number of wires: 2 | 3 | 4 | 5",
        "allowed_values": [2, 3, 4, 5],
        "search_queries": [
            "2-wire 4-wire wiring loop powered",
            "nombre de fils raccordement 2 fils 4 fils",
            "two wire four wire connection",
        ],
        "ai_fills": True,
    },

    "alimentation": {
        "description": "Power supply: boucle | 24VDC | 24VAC | 220VAC | Autre",
        "allowed_values": ["boucle", "24VDC", "24VAC", "220VAC", "Autre"],
        "search_queries": [
            "power supply voltage 24VDC 24VAC 220VAC loop powered",
            "alimentation tension boucle de courant",
            "supply voltage operating voltage",
        ],
        "ai_fills": True,
    },

    "communication": {
        "description": "Communication protocol: non | HART | Modbus | Profibus | Profinet | NFC | Autre",
        "allowed_values": ["non", "HART", "Modbus", "Profibus", "Profinet", "NFC", "Autre"],
        "search_queries": [
            "communication protocol HART Modbus PROFIBUS Profinet fieldbus",
            "protocole de communication numérique",
            "digital interface fieldbus",
        ],
        "ai_fills": True,
    },

    "sortieTOR": {
        "description": "TOR/relay output present (true/false)",
        "allowed_values": [True, False],
        "search_queries": [
            "relay output alarm contact limit switch NAMUR",
            "sortie TOR relais contact sec alarme seuil",
            "binary output switching output",
        ],
        "ai_fills": True,
    },

    "seuil": {
        "description": "Switching threshold value (only if sortieTOR = true)",
        "allowed_values": None,
        "search_queries": [
            "switching threshold setpoint alarm limit value",
            "seuil de commutation valeur limite",
            "trip point alarm setpoint",
        ],
        "ai_fills": True,
    },

    "seuilUnite": {
        "description": "Threshold unit (manual only, AI never fills)",
        "allowed_values": None,
        "search_queries": [],
        "ai_fills": False,
    },

    "repérageArmoire": {
        "description": "Cabinet/panel tag (manual only)",
        "allowed_values": None,
        "search_queries": [],
        "ai_fills": False,
    },

    "precision": {
        "description": 'Accuracy / precision, e.g. "±0.5%"',
        "allowed_values": None,
        "search_queries": [
            "accuracy precision measurement uncertainty",
            "précision exactitude incertitude de mesure",
            "measurement error ± percent",
        ],
        "ai_fills": True,
    },

    "marque": {
        "description": "Manufacturer brand name",
        "allowed_values": None,  # validated against MARQUE_OPTIONS with Autre fallback
        "search_queries": [
            "manufacturer brand fabricant marque",
            "Siemens KROHNE Endress+Hauser ABB Yokogawa",
            "copyright logo manufacturer name",
        ],
        "ai_fills": True,
    },

    "référence": {
        "description": "Product reference / order number (manual only, AI never fills)",
        "allowed_values": None,
        "search_queries": [],
        "ai_fills": False,
    },

    "certificats": {
        "description": "Certifications list: ATEX, IECEx, SIL, FM, CSA",
        "allowed_values": None,  # multi-select, validated individually
        "search_queries": [
            "ATEX IECEx SIL certification hazardous area",
            "certificat homologation zone explosive",
            "explosion proof intrinsically safe",
        ],
        "ai_fills": True,
    },

    "indiceIP": {
        "description": "IP protection class: IP65, IP66, IP67, IP68, NEMA4X",
        "allowed_values": None,
        "search_queries": [
            "IP protection class ingress protection NEMA",
            "indice de protection IP67 IP68 étanchéité",
            "enclosure rating weatherproof",
        ],
        "ai_fills": True,
    },

    "matériauMembrane": {
        "description": "Wetted parts material: 316L | Hastelloy | PTFE | Céramique | Autre",
        "allowed_values": ["316L", "Hastelloy", "PTFE", "Céramique", "Autre"],
        "search_queries": [
            "wetted parts material diaphragm membrane",
            "matériau membrane pièces en contact produit",
            "316L Hastelloy PTFE ceramic wetted material",
        ],
        "ai_fills": True,
    },

    "datasheetUrl": {
        "description": "Datasheet file path (manual only, user uploads)",
        "allowed_values": None,
        "search_queries": [],
        "ai_fills": False,
    },

    # --- Actionneur-specific fields ---
    "typeActionneur": {
        "description": (
            "Vanne de régulation | Vanne ON/OFF | Actionneur mécanique | "
            "Régulation & contrôle | Équipement auxiliaire"
        ),
        "allowed_values": [
            "Vanne de régulation", "Vanne ON/OFF", "Actionneur mécanique",
            "Régulation & contrôle", "Équipement auxiliaire",
        ],
        "search_queries": [
            "control valve on-off valve actuator controller",
            "vanne de régulation vanne tout-ou-rien actionneur",
            "safety valve relief valve equipment type",
        ],
        "ai_fills": True,
    },

    "typeVérin": {
        "description": (
            "Pneumatique simple effet | Pneumatique double effet | "
            "Hydraulique simple effet | Hydraulique double effet | Autre"
        ),
        "allowed_values": [
            "Pneumatique simple effet", "Pneumatique double effet",
            "Hydraulique simple effet", "Hydraulique double effet", "Autre",
        ],
        "search_queries": [
            "actuator type pneumatic hydraulic single double acting",
            "vérin pneumatique hydraulique simple double effet",
            "cylinder actuator spring return",
        ],
        "ai_fills": True,
    },

    "typeActionneurSpécial": {
        "description": "Piézoélectrique | Magnétique | Thermique | Électromagnétique | Linéaire électrique",
        "allowed_values": [
            "Piézoélectrique", "Magnétique", "Thermique",
            "Électromagnétique", "Linéaire électrique",
        ],
        "search_queries": [
            "piezoelectric magnetic thermal electromagnetic linear actuator",
            "piézoélectrique magnétique thermique linéaire électrique",
        ],
        "ai_fills": True,
    },

    "positionSécurité": {
        "description": "Fail-safe position: Fail Open | Fail Close | Fail Last",
        "allowed_values": ["Fail Open", "Fail Close", "Fail Last"],
        "search_queries": [
            "fail open fail close fail last safety position",
            "position de sécurité défaut ouvert fermé",
            "failure mode spring return air fail",
        ],
        "ai_fills": True,
    },

    "courseMM": {
        "description": "Actuator stroke in mm",
        "allowed_values": None,
        "search_queries": [
            "stroke travel length course mm",
            "course actionneur vérin mm",
            "piston stroke distance",
        ],
        "ai_fills": True,
    },

    "forceN": {
        "description": "Actuator force in N",
        "allowed_values": None,
        "search_queries": [
            "force thrust output N Newton",
            "force actionneur poussée N",
            "actuator force torque",
        ],
        "ai_fills": True,
    },

    "pressionAlimentationBar": {
        "description": "Supply pressure in bar",
        "allowed_values": None,
        "search_queries": [
            "supply pressure air pressure bar",
            "pression alimentation air bar",
            "pneumatic supply pressure operating pressure",
        ],
        "ai_fills": True,
    },
}

# Convenience: set of all field names that the AI should attempt to extract
AI_EXTRACTABLE_FIELDS = {k for k, v in FIELD_DESCRIPTIONS_V2.items() if v.get("ai_fills", True)}

# Fields specific to Transmetteur/Capteur (hidden for Actionneur)
TRANSMETTEUR_ONLY_FIELDS = {
    "signalSortie", "hart", "nombreFils", "alimentation",
    "communication", "matériauMembrane",
}

# Fields specific to Actionneur (hidden for Transmetteur/Capteur)
ACTIONNEUR_ONLY_FIELDS = {
    "typeActionneur", "typeVérin", "typeActionneurSpécial",
    "positionSécurité", "courseMM", "forceN", "pressionAlimentationBar",
}
