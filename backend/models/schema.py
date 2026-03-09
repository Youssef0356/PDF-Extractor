"""Pydantic models for the equipment form schema and API responses."""

from pydantic import BaseModel, Field
from typing import Optional, Any


class PlageMesure(BaseModel):
    min: Optional[float] = None
    max: Optional[float] = None
    unite: Optional[str] = None


class SortieAlarme(BaseModel):
    nomAlarme: Optional[str] = None
    typeAlarme: Optional[str] = None       # "Haut" | "Bas" | "Défaut"
    seuilAlarme: Optional[float] = None
    uniteAlarme: Optional[str] = None
    relaisAssocie: Optional[str] = None


class EquipmentSchema(BaseModel):
    """The target JSON schema that the AI must fill."""

    equipmentName: Optional[str] = None

    categorie: Optional[str] = Field(
        None,
        description=(
            "Pressure transmitter | Pressure gauge | Differential pressure transmitter | "
            "Temperature transmitter | Temperature sensor | Temperature indicator | "
            "Flowmeter | Electromagnetic flowmeter | Coriolis flowmeter | Vortex flowmeter | Ultrasonic flowmeter | "
            "Variable area flowmeter | Turbine flowmeter | Positive displacement meter | Pitot tube / annubar | Orifice plate | "
            "Level transmitter | Level switch | Radar level sensor | Ultrasonic level sensor | Float level switch | Capacitive level sensor | "
            "Vibrating fork / tuning fork | Density meter | Viscosity meter | "
            "pH sensor / transmitter | Conductivity sensor | Dissolved oxygen sensor | Turbidity sensor | Gas analyzer | Flame detector | Gas detector | "
            "Moisture analyzer | Particle counter | Chromatograph | "
            "Control valve | Valve positioner | Pneumatic actuator | Electric actuator | Solenoid valve | Safety relief valve | Butterfly valve | "
            "Pump | Variable speed drive | "
            "PLC | DCS controller | PID controller | Remote I/O | Safety system (SIS) | "
            "HMI panel | Field indicator | Paperless recorder | SCADA / RTU | "
            "Signal isolator / barrier | Power supply | Signal converter | "
            "Weighing system | Vibration sensor | Position sensor | Speed sensor | Torque sensor | Noise / acoustic sensor | Autre"
        ),
    )

    typeMesure: Optional[str] = Field(
        None,
        description="Debit | Niveau | Pression | Temperature | Autre",
    )

    technologie: Optional[str] = Field(
        None,
        description=(
            "Electromagnetique | Piezo-resistif | Ultrasonique | Coriolis | Vortex | "
            "Radar | Capacitif | Magnetique | Hydraulique | Pneumatique | "
            "Section variable (flotteur) | Numerique | Electronique | "
            "TFT Tactile | Tactile LCD | Autre"
        ),
    )

    plageMesure: Optional[PlageMesure] = None

    typeSignal: Optional[str] = Field(
        None,
        description="4-20mA | 0-20mA | 0-10V | 0-5V | 0-10V (AI) | 24V DC (DI/DO) | Autre",
    )

    nbFils: Optional[str] = Field(
        None,
        description="2 fils | 4 fils | Autre",
        # Removed "1" — no standard single-wire instrument signal exists.
        # Removed bare "2" / "4" — normalise to "2 fils" / "4 fils" in _normalize_value.
    )

    alimentation: Optional[str] = Field(
        None,
        description=(
            "24V DC | 12-30V DC | 14-30V DC | 85-264V AC | 220V AC | "
            "Loop powered (2 fils) | Autre"
            # Wide open — real datasheets have many supply ranges.
            # The LLM returns the verbatim value; _normalize_value canonicalises common ones.
        ),
    )

    reperage: Optional[str] = None

    communication: Optional[str] = Field(
        None,
        description=(
            "HART | Modbus RTU | Modbus TCP | Modbus TCP/IP | "
            "PROFIBUS DP | Profibus PA | Foundation Fieldbus | "
            "Profinet | Ethernet | RS-232 | RS-485 | "
            "Mitsubishi MC TCP/IP | Autre"
        ),
    )

    classeProtection: Optional[str] = Field(
        None,
        description="IP protection class, e.g. IP67, IP68, NEMA 4X",
        # Renamed from "classe" — "classe" was ambiguous (safety class vs IP class).
        # If you need IEC safety class, add a separate "classeSurete" field.
    )

    sortiesAlarme: Optional[list[SortieAlarme]] = None

    marque: Optional[str] = None      # Manufacturer brand (e.g. "Siemens", "Krohne")
    modele: Optional[str] = None      # Model designation (e.g. "H250 M40", "P320")
    reference: Optional[str] = None  # Order / part number (e.g. "7MF4433-1DA02")
    dateCalibration: Optional[str] = None  # Calibration interval, e.g. "12 mois"


class ExtractionResponse(BaseModel):
    """API response for PDF extraction."""
    success: bool
    data: Optional[EquipmentSchema] = None
    message: str = ""
    processing_time_seconds: Optional[float] = None
    meta: Optional[dict[str, Any]] = None
    evidence: Optional[dict[str, list[dict[str, Any]]]] = None
    doc_context: Optional[dict[str, Any]] = None
    timings_ms: Optional[dict[str, float]] = None


# ---------------------------------------------------------------------------
# Field metadata for semantic search + LLM prompts
# ---------------------------------------------------------------------------
# Rules:
#  - allowed_values must EXACTLY match the Pydantic field description values.
#  - Open fields (marque, modele, reference, reperage, alimentation, equipmentName,
#    plageMesure, sortiesAlarme, dateCalibration) have allowed_values = None.
#  - search_queries are used by the vector store to retrieve relevant chunks.
# ---------------------------------------------------------------------------

FIELD_DESCRIPTIONS: dict[str, dict] = {

    "equipmentName": {
        "description": "Commercial product name / designation (human-readable, NOT an order number)",
        "allowed_values": None,
        "search_queries": [
            "equipment name designation",
            "product name",
            "device name",
            "nom équipement désignation",
            "instrument name",
        ],
    },

    "categorie": {
        "description": (
            "Category of the equipment. Pick ONE exact taxonomy label. "
            "Use Autre only if none fits."
        ),
        "allowed_values": [
            "Pressure transmitter",
            "Pressure gauge",
            "Differential pressure transmitter",
            "Temperature transmitter",
            "Temperature sensor",
            "Temperature indicator",
            "Flowmeter",
            "Electromagnetic flowmeter",
            "Coriolis flowmeter",
            "Vortex flowmeter",
            "Ultrasonic flowmeter",
            "Variable area flowmeter",
            "Turbine flowmeter",
            "Positive displacement meter",
            "Pitot tube / annubar",
            "Orifice plate",
            "Level transmitter",
            "Level switch",
            "Radar level sensor",
            "Ultrasonic level sensor",
            "Float level switch",
            "Capacitive level sensor",
            "Vibrating fork / tuning fork",
            "Density meter",
            "Viscosity meter",
            "pH sensor / transmitter",
            "Conductivity sensor",
            "Dissolved oxygen sensor",
            "Turbidity sensor",
            "Gas analyzer",
            "Flame detector",
            "Gas detector",
            "Moisture analyzer",
            "Particle counter",
            "Chromatograph",
            "Control valve",
            "Valve positioner",
            "Pneumatic actuator",
            "Electric actuator",
            "Solenoid valve",
            "Safety relief valve",
            "Butterfly valve",
            "Pump",
            "Variable speed drive",
            "PLC",
            "DCS controller",
            "PID controller",
            "Remote I/O",
            "Safety system (SIS)",
            "HMI panel",
            "Field indicator",
            "Paperless recorder",
            "SCADA / RTU",
            "Signal isolator / barrier",
            "Power supply",
            "Signal converter",
            "Weighing system",
            "Vibration sensor",
            "Position sensor",
            "Speed sensor",
            "Torque sensor",
            "Noise / acoustic sensor",
            "Autre",
        ],
        "search_queries": [
            "instrument type category equipment",
            "transmitter gauge flowmeter level switch valve actuator PLC HMI",
            "type d'instrument catégorie équipement",
            "control valve positioner actuator pneumatic electric",
            "variable area flowmeter rotameter",
        ],
    },

    "typeMesure": {
        "description": "Physical quantity being measured. One of: Debit, Niveau, Pression, Temperature, Autre",
        # KEPT: these 5 are exhaustive for the domain; open-ended would reduce consistency.
        "allowed_values": ["Debit", "Niveau", "Pression", "Temperature", "Autre"],
        "search_queries": [
            "measured variable grandeur mesurée",
            "flow measurement débit",
            "pressure pression",
            "level niveau",
            "temperature",
            "domaine d'application mesure",
        ],
    },

    "technologie": {
        "description": (
            "Measurement or actuation technology. "
            "One of: Electromagnetique, Piezo-resistif, Ultrasonique, Coriolis, Vortex, "
            "Radar, Capacitif, Magnetique, Hydraulique, Pneumatique, "
            "Section variable (flotteur), Numerique, Electronique, TFT Tactile, Tactile LCD, Autre"
        ),
        # FIXED: added Ultrasonique, Coriolis, Vortex, Radar, Capacitif, Section variable (flotteur).
        # REMOVED: bare "Section variable" — always qualify with "(flotteur)" to avoid ambiguity.
        "allowed_values": [
            "Electromagnetique",
            "Piezo-resistif",
            "Ultrasonique",
            "Coriolis",
            "Vortex",
            "Radar",
            "Capacitif",
            "Magnetique",
            "Hydraulique",
            "Pneumatique",
            "Section variable (flotteur)",
            "Numerique",
            "Electronique",
            "TFT Tactile",
            "Tactile LCD",
            "Autre",
        ],
        "search_queries": [
            "principe de mesure measurement principle",
            "section variable flotteur rotameter variable area",
            "électromagnétique electromagnetic",
            "piézo-résistif piezo resistive",
            "ultrasonique ultrasonic",
            "coriolis vortex radar capacitif",
            "technologie principe fonctionnement",
        ],
    },

    "plageMesure": {
        "description": "Measurement range: {min, max, unite}. Only numeric min/max ranges — NOT ratios or rangeability.",
        "allowed_values": None,
        "search_queries": [
            "plage de mesure étendue measuring range",
            "débit min max Qmin Qmax",
            "pression min max",
            "température min max",
            "niveau min max",
            "operating range span",
        ],
    },

    "typeSignal": {
        "description": "Output signal type. One of: 4-20mA, 0-20mA, 0-10V, 0-5V, 0-10V (AI), 24V DC (DI/DO), Autre",
        "allowed_values": ["4-20mA", "0-20mA", "0-10V", "0-5V", "0-10V (AI)", "24V DC (DI/DO)", "Autre"],
        "search_queries": [
            "output signal type sortie analogique",
            "4-20 mA analog output",
            "0-10 V output",
            "digital input output DI DO",
            "signal de sortie type",
        ],
    },

    "nbFils": {
        "description": "Number of wires for electrical connection. One of: 2 fils, 4 fils, Autre",
        # FIXED: removed bare "1", "2", "4" — always normalise to "2 fils" / "4 fils".
        "allowed_values": ["2 fils", "4 fils", "Autre"],
        "search_queries": [
            "2-wire 4-wire wiring connection",
            "2 fils 4 fils raccordement",
            "loop powered two wire",
            "nombre de fils câblage",
        ],
    },

    "alimentation": {
        "description": "Power supply voltage/type as stated in the document (e.g. 24V DC, 14-30V DC, 220V AC, loop powered)",
        # OPEN: real datasheets have too many supply ranges to enumerate.
        # _normalize_value handles common aliases (Vdc→V DC, etc.).
        "allowed_values": None,
        "search_queries": [
            "power supply alimentation tension",
            "supply voltage Vdc Vac",
            "24V DC 220V AC",
            "loop powered 2 fils",
            "auxiliary power",
        ],
    },

    "reperage": {
        "description": "Plant tag / instrument tag identifier (e.g. FT-101, PT-202A) — site-specific, not a model number",
        "allowed_values": None,
        "search_queries": [
            "tag number repérage instrument tag",
            "P&ID tag identification signal",
            "loop tag FT PT LT TT",
        ],
    },

    "communication": {
        "description": (
            "Digital communication protocol. "
            "One of: HART, Modbus RTU, Modbus TCP, Modbus TCP/IP, PROFIBUS DP, Profibus PA, "
            "Foundation Fieldbus, Profinet, Ethernet, RS-232, RS-485, Mitsubishi MC TCP/IP, Autre"
        ),
        "allowed_values": [
            "HART",
            "Modbus RTU",
            "Modbus TCP",
            "Modbus TCP/IP",
            "PROFIBUS DP",
            "Profibus PA",
            "Foundation Fieldbus",
            "Profinet",
            "Ethernet",
            "RS-232",
            "RS-485",
            "Mitsubishi MC TCP/IP",
            "Autre",
        ],
        "search_queries": [
            "communication protocol protocole",
            "HART fieldbus",
            "Modbus PROFIBUS Profinet",
            "Foundation Fieldbus Profibus PA",
            "Ethernet RS-485 RS-232",
            "digital communication interface",
        ],
    },

    "classeProtection": {
        "description": "IP / NEMA protection class (e.g. IP67, IP68, NEMA 4X)",
        # FIXED: renamed from "classe" which was ambiguous.
        # OPEN: too many valid IP codes to enumerate.
        "allowed_values": None,
        "search_queries": [
            "IP protection class degré protection",
            "IP67 IP68 NEMA enclosure rating",
            "ingress protection indice de protection",
        ],
    },

    "sortiesAlarme": {
        "description": (
            "Alarm/relay outputs — only when alarm name, threshold value, AND unit are ALL explicitly stated. "
            "Each item: {nomAlarme, typeAlarme (Haut|Bas|Défaut), seuilAlarme, uniteAlarme, relaisAssocie}"
        ),
        "allowed_values": None,
        "search_queries": [
            "alarm output sortie alarme seuil",
            "relay threshold setpoint relais seuil",
            "limit detector high low fault alarm",
            "NAMUR alarm relay contact",
        ],
    },

    "marque": {
        "description": "Manufacturer / brand name (e.g. Siemens, Krohne, Endress+Hauser, Yokogawa, ABB)",
        "allowed_values": None,
        "search_queries": [
            "manufacturer brand fabricant marque",
            "Siemens Krohne Endress Hauser Yokogawa ABB Emerson SAMSON",
            "constructeur fournisseur",
        ],
    },

    "modele": {
        "description": "Model name or designation (e.g. H250 M40, SITRANS P320)",
        "allowed_values": None,
        "search_queries": [
            "model designation modèle",
            "product family series",
            "type désignation produit",
            "type 3271 servomoteur",
        ],
    },

    "reference": {
        "description": "Manufacturer order / part / article number (e.g. 6ES7 315-2EH14-0AB0, 7MF4433-1DA02)",
        "allowed_values": None,
        "search_queries": [
            "order number part number référence",
            "article number numéro de commande",
            "ordering data référence produit",
            "6ES7 7MF FMR",
        ],
    },

    "dateCalibration": {
        "description": "Calibration interval / frequency expressed as '<N> mois' (e.g. '12 mois', '6 mois')",
        "allowed_values": None,
        "search_queries": [
            "calibration interval frequency périodicité étalonnage",
            "recalibration period schedule",
            "tous les N mois every N months",
            "calibration due date interval",
        ],
    },
}