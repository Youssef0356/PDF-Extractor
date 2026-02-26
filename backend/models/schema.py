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
    "categorie": {
        "description": "Category of the equipment (Transmetteur, Actionneur, or Autre)",
        "allowed_values": ["Transmetteur", "Actionneur", "Autre"],
        "search_queries": [
            "equipment category type transmitter actuator",
            "categorie equipement transmetteur actionneur",
            "type of instrument device",
        ],
    },
    "typeMesure": {
        "description": "Type of measurement (Debit, Niveau, Pression, Temperature, or Autre)",
        "allowed_values": ["Debit", "Niveau", "Pression", "Temperature", "Autre"],
        "search_queries": [
            "measurement type flow level pressure temperature",
            "type de mesure debit niveau pression temperature",
            "measured variable process variable",
        ],
    },
    "technologie": {
        "description": "Technology used (Electromagnetique, Hydraulique, or Autre)",
        "allowed_values": ["Electromagnetique", "Hydraulique", "Autre"],
        "search_queries": [
            "technology electromagnetic hydraulic measurement principle",
            "technologie electromagnetique hydraulique principe de mesure",
            "sensing technology sensor type",
        ],
    },
    "plageMesure": {
        "description": "Measurement range with minimum, maximum, and unit",
        "allowed_values": None,
        "search_queries": [
            "measurement range min max span",
            "plage de mesure etendue",
            "operating range scale",
        ],
    },
    "typeSignal": {
        "description": "Output signal type (4-20mA, 0-20mA, 0-10V, 0-5V, or Autre)",
        "allowed_values": ["4-20mA", "0-20mA", "0-10V", "0-5V", "Autre"],
        "search_queries": [
            "output signal type 4-20mA voltage current",
            "type de signal sortie",
            "analog output signal",
        ],
    },
    "nbFils": {
        "description": "Number of wires (1, 2, 4, or Autre)",
        "allowed_values": ["1", "2", "4", "Autre"],
        "search_queries": [
            "number of wires wire configuration 2-wire 4-wire",
            "nombre de fils cablage",
            "wiring connection type",
        ],
    },
    "alimentation": {
        "description": "Power supply (None, 24V DC, 220V AC, or Autre)",
        "allowed_values": ["None", "24V DC", "220V AC", "Autre"],
        "search_queries": [
            "power supply voltage 24V DC 220V AC",
            "alimentation tension",
            "supply voltage power requirements",
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
            "communication protocol HART Modbus PROFIBUS",
            "protocole communication",
            "digital communication fieldbus",
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
            "manufacturer brand maker",
            "marque fabricant constructeur",
            "made by company",
        ],
    },
    "modele": {
        "description": "Model name or number",
        "allowed_values": None,
        "search_queries": [
            "model name number product",
            "modele numero produit",
            "model designation",
        ],
    },
    "reference": {
        "description": "Product reference or part number",
        "allowed_values": None,
        "search_queries": [
            "reference number part number order code",
            "reference numero de commande",
            "catalog number SKU",
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
