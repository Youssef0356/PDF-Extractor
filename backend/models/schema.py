"""
Pydantic models for the equipment form schema and API responses.
"""
from pydantic import BaseModel, Field
from typing import Optional


class PlageMesure(BaseModel):
    min: Optional[float] = None
    max: Optional[float] = None
    unite: Optional[str] = None


class SortieAlarme(BaseModel):
    nomAlarme: Optional[str] = None
    typeAlarme: Optional[str] = None
    seuilAlarme: Optional[float] = None
    uniteAlarme: Optional[str] = None
    relaisAssocie: Optional[str] = None


class EquipmentSchema(BaseModel):
    """The target JSON schema that the AI must fill."""
    equipmentName: Optional[str] = None
    categorie: Optional[str] = Field(
        None, description="Transmetteur | Actionneur | Autre"
    )
    typeMesure: Optional[str] = Field(
        None, description="Debit | Niveau | Pression | Temperature | Autre"
    )
    technologie: Optional[str] = Field(
        None, description="Electromagnetique | Hydraulique | Autre"
    )
    plageMesure: Optional[PlageMesure] = None

    typeSignal: Optional[str] = Field(
        None, description="4-20mA | 0-20mA | 0-10V | 0-5V | Autre"
    )
    nbFils: Optional[str] = Field(
        None, description="1 | 2 | 4 | Autre"
    )
    alimentation: Optional[str] = Field(
        None, description="None | 24V DC | 220V AC | Autre"
    )
    reperage: Optional[str] = None

    communication: Optional[str] = Field(
        None, description="HART | Modbus RTU | Modbus TCP | PROFIBUS DP | Autre"
    )

    classe: Optional[str] = Field(
        None, description="Classe A | Classe B | Autre"
    )

    sortiesAlarme: Optional[list[SortieAlarme]] = None

    marque: Optional[str] = None
    modele: Optional[str] = None
    reference: Optional[str] = None
    dateCalibration: Optional[str] = None


class ExtractionResponse(BaseModel):
    """API response for PDF extraction."""
    success: bool
    data: Optional[EquipmentSchema] = None
    message: str = ""
    processing_time_seconds: Optional[float] = None


# -- Field metadata for semantic search prompts ----------------------
FIELD_DESCRIPTIONS = {
    "equipmentName": {
        "description": "Equipment name / designation (human-readable name)",
        "allowed_values": None,
        "search_queries": [
            "equipment name",
            "designation",
            "device name",
            "product name",
            "nom de l'equipement",
            "désignation",
        ],
    },
    "categorie": {
        "description": "Category of the equipment (Transmetteur, Actionneur, or Autre)",
        "allowed_values": ["Transmetteur", "Actionneur", "Autre"],
        "search_queries": [
            "equipment type transmitter",
            "pressure transmitter",
            "transmitter",
            "instrument type",
            "categorie equipement transmetteur actionneur",
            "type d'instrument",
        ],
    },
    "typeMesure": {
        "description": "Type of measurement (Debit, Niveau, Pression, Temperature, or Autre)",
        "allowed_values": ["Debit", "Niveau", "Pression", "Temperature", "Autre"],
        "search_queries": [
            "measured variable",
            "measurement type",
            "pressure",
            "differential pressure",
            "absolute pressure",
            "gauge pressure",
            "flow",
            "level",
            "temperature",
            "type de mesure debit niveau pression temperature",
            "grandeur mesurée",
        ],
    },
    "technologie": {
        "description": "Technology used (Electromagnetique, Hydraulique,Pneumatique or Autre)",
        "allowed_values": ["Electromagnetique", "Hydraulique","Pneumatique", "Autre"],
        "search_queries": [
            "principe de mesure",
            "measurement principle",
            "operating principle",
            "section variable",
            "débitmètre à section variable",
            "rotamètre",
            "variable area flowmeter",
            "technologie electromagnetique hydraulique pneumatique",
        ],
    },
    "plageMesure": {
        "description": "Measurement range with minimum, maximum, and unit",
        "allowed_values": None,
        "search_queries": [
            "plage de mesure",
            "étendue de mesure",
            "measuring range",
            "measurement range",
            "plage de débit",
            "débit min",
            "débit max",
            "Qmin Qmax",
            "débitmètre plage",
        ],
    },
    "typeSignal": {
        "description": "Output signal type (4-20mA, 0-20mA, 0-10V, 0-5V, or Autre)",
        "allowed_values": ["4-20mA", "0-20mA", "0-10V", "0-5V", "Autre"],
        "search_queries": [
            "output signal",
            "output",
            "4-20 mA",
            "4 ... 20 mA",
            "0-10 V",
            "0-20 mA",
            "analog output",
            "type de signal sortie",
        ],
    },
    "nbFils": {
        "description": "Number of wires (1, 2, 4, or Autre)",
        "allowed_values": ["1", "2", "4", "Autre"],
        "search_queries": [
            "2-wire",
            "4-wire",
            "two-wire",
            "four-wire",
            "wiring",
            "electrical connection",
            "nombre de fils cablage",
        ],
    },
    "alimentation": {
        "description": "Power supply (None, 24V DC, 220V AC, or Autre)",
        "allowed_values": ["None", "24V DC", "220V AC", "Autre"],
        "search_queries": [
            "power supply",
            "supply voltage",
            "power requirements",
            "24 V DC",
            "24V DC",
            "220 V AC",
            "alimentation tension",
        ],
    },
    "reperage": {
        "description": "Tag or label identifier for the signal",
        "allowed_values": None,
        "search_queries": [
            "tag number identifier label reperage",
            "reperage identification signal",
            "instrument tag",
        ],
    },
    "communication": {
        "description": "Communication protocol (HART, Modbus RTU, Modbus TCP, PROFIBUS DP, or Autre)",
        "allowed_values": ["HART", "Modbus RTU", "Modbus TCP", "PROFIBUS DP", "Autre"],
        "search_queries": [
            "HART",
            "communication protocol",
            "fieldbus",
            "Modbus",
            "PROFIBUS",
            "protocole communication",
        ],
    },
    "classe": {
        "description": "Equipment class (Classe A, Classe B, or Autre)",
        "allowed_values": ["Classe A", "Classe B", "Autre"],
        "search_queries": [
            "equipment class classification safety",
            "classe equipement",
            "safety integrity level class",
        ],
    },
    "sortiesAlarme": {
        "description": "Alarm outputs with name, type, threshold, unit, and associated relay",
        "allowed_values": None,
        "search_queries": [
            "alarm output threshold relay",
            "sortie alarme seuil relais",
            "alarm setpoint warning fault",
        ],
    },
    "marque": {
        "description": "Manufacturer brand name",
        "allowed_values": None,
        "search_queries": [
            "manufacturer",
            "brand",
            "Siemens",
            "marque fabricant constructeur",
        ],
    },
    "modele": {
        "description": "Model name or number",
        "allowed_values": None,
        "search_queries": [
            "model",
            "model designation",
            "type",
            "SITRANS P320",
            "P320",
            "modele numero produit",
        ],
    },
    "reference": {
        "description": "Product reference or part number",
        "allowed_values": None,
        "search_queries": [
            "order number",
            "ordering data",
            "part number",
            "reference number",
            "7MF",
            "reference numero de commande",
        ],
    },
    "dateCalibration": {
        "description": "Calibration date",
        "allowed_values": None,
        "search_queries": [
            "calibration date factory calibration",
            "date de calibration etalonnage",
            "last calibrated date",
        ],
    },
}
