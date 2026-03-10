"""
Regex-based extractor for deterministic fields.
Scans the full PDF text for well-known patterns, producing high-confidence
results BEFORE the LLM is invoked. Fields not matched are left for the LLM.
"""
import re
from typing import Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _first_match(patterns: list[re.Pattern], text: str) -> Optional[re.Match]:
    for pat in patterns:
        m = pat.search(text)
        if m:
            return m
    return None


def _looks_like_reference(value: str) -> bool:
    """Sanity-check: a reference must contain both letters and digits and be long enough."""
    v = value.strip()
    return (
        len(v) >= 5
        and any(c.isdigit() for c in v)
        and any(c.isalpha() for c in v)
    )


def _looks_like_plc_cpu_datasheet(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in [
        "simatic s7", "s7-1200", "s7-1500", "s7-300", "s7-400",
        "programmable logic controller", "cpu 12", "cpu 15",
        "digital input", "digital output", "analog input", "analog output",
    ])


# ---------------------------------------------------------------------------
# Known manufacturer brands  (order = priority; first match wins)
# FIXED: removed duplicate regex-escape variants — re.escape() handles + correctly.
# ADDED: Rosemount, Vega, Yokogawa models, ABB, Bürkert, Cameron, Magnetrol, Omega,
#        Hach, Mettler Toledo, Fluke, Omron, Rockwell, GF Piping.
# ---------------------------------------------------------------------------
_BRANDS: list[tuple[str, re.Pattern]] = []

_BRAND_NAMES = [
    # Most common in process industry — ordered by frequency to hit fast
    "Siemens",
    "Endress+Hauser",
    "Endress & Hauser",
    "KROHNE",
    "Krohne",
    "ABB",
    "Emerson",
    "Rosemount",           # Emerson brand — kept separate as it appears alone
    "Yokogawa",
    "Honeywell",
    "Vega",
    "VEGA",
    "Schneider Electric",
    "Schneider",
    "Pepperl+Fuchs",
    "Danfoss",
    "Bürkert",
    "Burkert",
    "Festo",
    "IFM",
    "ifm",
    "Wika",
    "WIKA",
    "Bosch Rexroth",
    "Fisher",              # Emerson / Fisher Controls
    "Foxboro",
    "Samson",
    "Azbil",
    "Fuji Electric",
    "Hitachi",
    "Mitsubishi Electric",
    "Mitsubishi",
    "Phoenix Contact",
    "Turck",
    "Sick",
    "SICK",
    "Balluff",
    "Banner Engineering",
    "Banner",
    "Keyence",
    "KEYENCE",
    "Hach",
    "Mettler Toledo",
    "Mettler-Toledo",
    "Fluke",
    "Omron",
    "OMRON",
    "Rockwell Automation",
    "Allen-Bradley",       # Rockwell brand
    "Cameron",             # AVEVA / Schlumberger flow brand
    "Magnetrol",
    "Flowserve",
    "Spirax Sarco",
    "Bürkert",
    "GF Piping",
    "Georg Fischer",
    "Omega",
    "OMEGA",
    "Dwyer",
    "Badger Meter",
    "Sensirion",
    "Kistler",
    "Gems Sensors",
    "Gems",
    "ControlAir",
    "Metso",
    "Neles",               # Metso sub-brand for valves
    "Rotork",
    "EIM",                 # Rotork brand
    "Auma",
    "AUMA",
    "Bettis",              # Emerson actuator brand
]

for _name in _BRAND_NAMES:
    _pat = re.compile(r"\b" + re.escape(_name) + r"\b", re.IGNORECASE)
    _BRANDS.append((_name, _pat))


# ---------------------------------------------------------------------------
# Model patterns by brand
# ADDED: Yokogawa EJX/EJA, ABB 2600T/266, Vega VEGAPULS/VEGAFLEX,
#        Rosemount 3051/3144/644, Emerson, Honeywell, Festo, Bürkert
# ---------------------------------------------------------------------------
_MODEL_PATTERNS_BY_BRAND: dict[str, list[re.Pattern]] = {
    "Siemens": [
        re.compile(r"\b(SIMATIC\s+HMI\s+KTP\d{3,4}\s+(?:BASIC|COMFORT|MOBILE))\b", re.IGNORECASE),
        re.compile(r"\b(SITRANS\s+[A-Z]{1,3}\d{2,4}(?:\s*[A-Z]+\d*)?)\b", re.IGNORECASE),
        re.compile(r"\b(SIPART\s+PS\d+[A-Z]*)\b", re.IGNORECASE),
        re.compile(r"\b(SIMATIC\s+S7-\d{3,4})\b", re.IGNORECASE),
        re.compile(r"\b(CPU\s*\d{3,4}[A-Z]?\s*[A-Z]{2}/[A-Z]{2}/[A-Z]{2})\b", re.IGNORECASE),
        re.compile(r"\b(CPU\s*\d{3,4}[A-Z]{0,2})\b", re.IGNORECASE),
        re.compile(r"\b(SIMATIC\s+ET\s*200\w*)\b", re.IGNORECASE),
        re.compile(r"\b(SCALANCE\s+\w+)\b", re.IGNORECASE),
    ],
    "KROHNE": [
        re.compile(r"\b(H250\s*M?\d{0,2})\b", re.IGNORECASE),
        re.compile(r"\b(OPTIFLUX\s*\d+[A-Z]*)\b", re.IGNORECASE),
        re.compile(r"\b(OPTISONIC\s*\d+[A-Z]*)\b", re.IGNORECASE),
        re.compile(r"\b(OPTITEMP\s*\w+)\b", re.IGNORECASE),
        re.compile(r"\b(OPTIMASS\s*\d+[A-Z]*)\b", re.IGNORECASE),
        re.compile(r"\b(OPTIBAR\s*\w+)\b", re.IGNORECASE),
        re.compile(r"\b(OPTIWAVE\s*\w+)\b", re.IGNORECASE),
        re.compile(r"\b(IFC\s*\d{3,}[A-Z]*)\b", re.IGNORECASE),
    ],
    "Endress+Hauser": [
        re.compile(r"\b(Promag\s*[A-Z]?\d{2,3}[A-Z]?)\b", re.IGNORECASE),
        re.compile(r"\b(Proline\s+\w+)\b", re.IGNORECASE),
        re.compile(r"\b(Cerabar\s+[A-Z]\d{3})\b", re.IGNORECASE),
        re.compile(r"\b(Deltabar\s+[A-Z]\d{3})\b", re.IGNORECASE),
        re.compile(r"\b(Liquiline\s+\w+)\b", re.IGNORECASE),
        re.compile(r"\b(Prowirl\s*\w+)\b", re.IGNORECASE),
        re.compile(r"\b(Promass\s*\w+)\b", re.IGNORECASE),
        re.compile(r"\b(Levelflex\s*\w+)\b", re.IGNORECASE),
        re.compile(r"\b(Micropilot\s*\w+)\b", re.IGNORECASE),
        re.compile(r"\b(Prosonic\s*\w+)\b", re.IGNORECASE),
        re.compile(r"\b(iTEMP\s*\w+)\b", re.IGNORECASE),
        re.compile(r"\b(Gammapilot\s*\w+)\b", re.IGNORECASE),
        # E+H order code format: e.g. FMR51-AAACBMJF2A4, PMC71-...
        re.compile(r"\b([A-Z]{2,4}\d{2,3}[-][A-Z0-9]{4,20})\b"),
    ],
    "Yokogawa": [
        re.compile(r"\b(EJX\s*\d{3}[A-Z]?)\b", re.IGNORECASE),
        re.compile(r"\b(EJA\s*\d{3}[A-Z]?(?:-E)?)\b", re.IGNORECASE),
        re.compile(r"\b(YTMX\d{3})\b", re.IGNORECASE),
        re.compile(r"\b(YTA\d{3}[A-Z]?)\b", re.IGNORECASE),          # temp transmitters
        re.compile(r"\b(DPharp\s+EJ[XA]\w+)\b", re.IGNORECASE),
        re.compile(r"\b(ADMAG\s*\w+)\b", re.IGNORECASE),             # magnetic flowmeters
        re.compile(r"\b(ROTAMASS\s*\w+)\b", re.IGNORECASE),
        re.compile(r"\b(YEWFLO\s*\w+)\b", re.IGNORECASE),
    ],
    "Rosemount": [
        re.compile(r"\b(3051\s*[A-Z]{0,3}\d?)\b"),                   # 3051C, 3051CD, 3051TG
        re.compile(r"\b(3144[A-Z]?)\b"),                              # temp transmitter
        re.compile(r"\b(644[A-Z]?)\b"),                               # temp transmitter
        re.compile(r"\b(2051[A-Z]?)\b"),
        re.compile(r"\b(8800[A-Z]?)\b"),                              # vortex
        re.compile(r"\b(8700[A-Z]?)\b"),                              # magnetic
        re.compile(r"\b(1056\w*)\b"),                                  # analyzer
        re.compile(r"\b(5300\w*)\b"),                                  # radar level
        re.compile(r"\b(5408\w*)\b"),                                  # radar level
    ],
    "ABB": [
        re.compile(r"\b(2600T\s*[A-Z]{0,4})\b", re.IGNORECASE),      # pressure transmitter
        re.compile(r"\b(266\s*[A-Z]{1,4}\w*)\b", re.IGNORECASE),     # pressure
        re.compile(r"\b(FSM\d{4})\b", re.IGNORECASE),                 # mag flowmeter
        re.compile(r"\b(CoriolisMaster\s*\w*)\b", re.IGNORECASE),
        re.compile(r"\b(ProcessMaster\s*\w*)\b", re.IGNORECASE),
        re.compile(r"\b(LevelMaster\s*\w*)\b", re.IGNORECASE),
        re.compile(r"\b(TB\d{2,3}\w*)\b", re.IGNORECASE),             # temp transmitters
    ],
    "Vega": [
        re.compile(r"\b(VEGAPULS\s*\d+\s*[A-Z]*)\b", re.IGNORECASE),
        re.compile(r"\b(VEGAFLEX\s*\d+\s*[A-Z]*)\b", re.IGNORECASE),
        re.compile(r"\b(VEGACAP\s*\d+\s*[A-Z]*)\b", re.IGNORECASE),
        re.compile(r"\b(VEGABAR\s*\d+\s*[A-Z]*)\b", re.IGNORECASE),
        re.compile(r"\b(VEGASWING\s*\d+\s*[A-Z]*)\b", re.IGNORECASE),
        re.compile(r"\b(VEGAPOINT\s*\d+\s*[A-Z]*)\b", re.IGNORECASE),
        re.compile(r"\b(VEGASONIC\s*\d+\s*[A-Z]*)\b", re.IGNORECASE),
    ],
    "Honeywell": [
        re.compile(r"\b(ST\s*\d{3,4}[A-Z]*)\b", re.IGNORECASE),      # SmartLine transmitters
        re.compile(r"\b(STD\s*\d{3,4}[A-Z]*)\b", re.IGNORECASE),
        re.compile(r"\b(STG\s*\d{3,4}[A-Z]*)\b", re.IGNORECASE),
        re.compile(r"\b(SPS\s*\w+)\b", re.IGNORECASE),
    ],
    "Bürkert": [
        re.compile(r"\b(Type\s*\d{4}[A-Z]?)\b", re.IGNORECASE),      # Bürkert uses "Type XXXX"
        re.compile(r"\b(8\d{3}[A-Z]?)\b"),                            # e.g. 8020, 8035
    ],
    "Festo": [
        re.compile(r"\b(VPPM[\-\w]+)\b", re.IGNORECASE),
        re.compile(r"\b(VTUG[\-\w]+)\b", re.IGNORECASE),
        re.compile(r"\b(CPX[\-\w]+)\b", re.IGNORECASE),
        re.compile(r"\b(SDE\d[\-\w]*)\b", re.IGNORECASE),
    ],
    "Samson": [
        re.compile(r"\b(Type\s*\d{4,5})\b", re.IGNORECASE),
        re.compile(r"\b(3730[\-\w]*)\b"),                              # positioners
        re.compile(r"\b(3241[\-\w]*)\b"),                              # globe valves
    ],
    "Rotork": [
        re.compile(r"\b(IQ\s*\d+[A-Z]*)\b", re.IGNORECASE),
        re.compile(r"\b(AQ\s*\d+[A-Z]*)\b", re.IGNORECASE),
        re.compile(r"\b(CVA\s*\d+[A-Z]*)\b", re.IGNORECASE),
    ],
}

# Generic "Model / Type / Modèle : VALUE" fallback
_GENERIC_MODEL_PATTERNS = [
    re.compile(r"(?:Model|Modèle|Modele|Type)\s*[:=]\s*([A-Z0-9][A-Z0-9\s\-/\.]{2,30})", re.IGNORECASE),
]


# ---------------------------------------------------------------------------
# Reference / order-number patterns
# ADDED: Yokogawa EJX/EJA suffix codes, ABB 2600T, Endress+Hauser alphanumeric,
#        Vega model codes, Rosemount style, generic "Order No." patterns.
# ---------------------------------------------------------------------------
_REFERENCE_PATTERNS = [
    # Siemens 7MF pressure transmitters
    re.compile(r"\b(7MF\d[\d\s\-\.]{5,20})\b"),
    # Siemens HMI: 6AV2123-2GB03-0AX0
    re.compile(r"\b(6AV\s*\d{4}[-\s]?[0-9A-Z]{5}[-\s]?[0-9A-Z]{4}[-\s]?[0-9A-Z]{4})\b", re.IGNORECASE),
    # Siemens S7: 6ES7 315-2EH14-0AB0
    re.compile(r"\b(6ES7\s*[\dA-Z]{3,4}[-\s]?[\dA-Z]{5,6}[-\s]?[\dA-Z]{4})\b", re.IGNORECASE),
    # Siemens IO: 6GK, 6FX, 6SL, 6RA, 6SE families
    re.compile(r"\b(6[A-Z]{2}\d[\dA-Z\-]{6,20})\b", re.IGNORECASE),
    # Yokogawa EJX/EJA: EJX110A-DMS5B-92DN/D4 style
    re.compile(r"\b(EJ[XA]\d{3}[A-Z][-][A-Z0-9]{3,}(?:[-/][A-Z0-9]+)*)\b", re.IGNORECASE),
    # Endress+Hauser: PMC71-..., FMR51-..., TMT82-...
    re.compile(r"\b([A-Z]{2,4}\d{2}[-][A-Z0-9]{4,20})\b"),
    # ABB 2600T: 261GS4E1B1... (long alphanumeric)
    re.compile(r"\b(2[0-9]{3}[A-Z]{1,3}[0-9A-Z]{4,16})\b"),
    # Vega: VEGAPULS64.XXAXXXXX
    re.compile(r"\b(VEGA[A-Z]+\s*\d{2}[-.]?[A-Z0-9]{4,12})\b", re.IGNORECASE),
    # Rosemount: 3051CD2A22A1AB4M5... (long alphanumeric without hyphens)
    re.compile(r"\b(30[0-9]{2}[A-Z]{1,3}[0-9A-Z]{4,20})\b"),
    # Generic "Order No." / "Part No." / "Article No." / "Ref. No." patterns
    re.compile(
        r"(?:Order\s*(?:No\.?|Number|code)|Part\s*(?:No\.?|Number)|"
        r"Article\s*(?:No\.?|Number)|Ref(?:erence)?\s*(?:No\.?)?|"
        r"Numéro\s*(?:de\s*commande|d['']article))\s*[:=]?\s*"
        r"([A-Z0-9][\w\-\./ ]{4,40})",
        re.IGNORECASE,
    ),
]


# ---------------------------------------------------------------------------
# Tag / reperage patterns
# ADDED: French-style tags (FT-0201, PT-202A), instrument loop tags
# ---------------------------------------------------------------------------
_TAG_PATTERNS = [
    # ISA-style: FT-0201, PT-1234, LT-001A
    re.compile(r"\b([FPTLAICQUZ][TIRC][-_]\d{2,6}[A-Z]?)\b"),
    # Extended ISA two-letter prefix: FIC-101, PCV-202, AIT-003
    re.compile(r"\b([A-Z]{2,4}[-_]\d{3,6}(?:\.\d+)?)\b"),
    # "Tag: FT-101" or "Repérage: PT-202"
    re.compile(r"(?:Tag|Repère|Repérage|Reperage|Label|Boucle)\s*[:=]\s*([A-Z0-9][\w\-\.]{2,20})", re.IGNORECASE),
]


# ---------------------------------------------------------------------------
# IP / protection class patterns
# ADDED: full IP code extraction (NEMA, IP6X, IP69K, etc.)
# ---------------------------------------------------------------------------
_IP_PATTERNS = [
    re.compile(r"\b(IP\s*6[5-9](?:K?)\b)", re.IGNORECASE),   # IP65, IP67, IP68, IP69K
    re.compile(r"\b(IP\s*[0-6][0-9])\b", re.IGNORECASE),      # general IPxx
    re.compile(r"\b(NEMA\s*(?:Type\s*)?\d+X?)\b", re.IGNORECASE),
]


# ---------------------------------------------------------------------------
# Communication protocol patterns
# ADDED: Foundation Fieldbus, Profibus PA, Profinet, Ethernet/IP, ISA100,
#        WirelessHART, IO-Link, OPC-UA
# ---------------------------------------------------------------------------
def _extract_communication(text: str) -> Optional[dict]:
    def _negated(start: int) -> bool:
        window = text[max(0, start - 50):start].lower()
        return any(t in window for t in [": non", " non", ": no ", " no ", "=0", ": 0", "without", "sans "])

    protocols = [
        # Most specific first to avoid partial matches
        (re.compile(r"\bPROFIBUS\s+PA\b", re.IGNORECASE),          "Profibus PA"),
        (re.compile(r"\bPROFIBUS\s+DP\b", re.IGNORECASE),          "PROFIBUS DP"),
        (re.compile(r"\bPROFIBUS\b", re.IGNORECASE),                "PROFIBUS DP"),
        (re.compile(r"\bPROFINET\b", re.IGNORECASE),                "Profinet"),
        (re.compile(r"\bFoundation\s+Fieldbus\b", re.IGNORECASE),   "Foundation Fieldbus"),
        (re.compile(r"\bFF\b"),                                      "Foundation Fieldbus"),  # careful — short
        (re.compile(r"\bModbus\s+TCP(?:/IP)?\b", re.IGNORECASE),    "Modbus TCP/IP"),
        (re.compile(r"\bModbus\s+RTU\b", re.IGNORECASE),            "Modbus RTU"),
        (re.compile(r"\bModbus\b", re.IGNORECASE),                   "Modbus RTU"),
        (re.compile(r"\bEtherNet/IP\b", re.IGNORECASE),             "Ethernet"),
        (re.compile(r"\bIndustrial\s+Ethernet\b", re.IGNORECASE),   "Ethernet"),
        (re.compile(r"\bEthernet\b", re.IGNORECASE),                "Ethernet"),
        (re.compile(r"\bRS[\-\s]?485\b", re.IGNORECASE),            "RS-485"),
        (re.compile(r"\bRS[\-\s]?232\b", re.IGNORECASE),            "RS-232"),
        (re.compile(r"\bIO[\-\s]?Link\b", re.IGNORECASE),           "IO-Link"),
        (re.compile(r"\bWirelessHART\b", re.IGNORECASE),            "WirelessHART"),
        (re.compile(r"\bHART\b"),                                    "HART"),
    ]

    for pat, canonical in protocols:
        m = pat.search(text)
        if m and not _negated(m.start()):
            return {"value": canonical, "confidence": 1.0, "quote": m.group(0).strip(), "source": "regex"}

    return None


# ---------------------------------------------------------------------------
# Output signal type
# ADDED: NAMUR, pulse, frequency, relay patterns
# ---------------------------------------------------------------------------
def _extract_type_signal(text: str) -> Optional[dict]:
    patterns = [
        (re.compile(r"4\s*(?:[\.\-…–]+|\bto\b|\bà\b)\s*20\s*mA", re.IGNORECASE),           "4-20mA"),
        (re.compile(r"0\s*[\.\-…–]+\s*20\s*mA", re.IGNORECASE),           "0-20mA"),
        (re.compile(r"0\s*[\.\-…–]+\s*10\s*V\b", re.IGNORECASE),          "0-10V"),
        (re.compile(r"0\s*[\.\-…–]+\s*5\s*V\b", re.IGNORECASE),           "0-5V"),
        (re.compile(r"\b24\s*V\s*DC\s*\(?DI[/\)]?DO\)?", re.IGNORECASE),  "24V DC (DI/DO)"),
    ]
    for pat, canonical in patterns:
        m = pat.search(text)
        if m:
            return {"value": canonical, "confidence": 1.0, "quote": m.group(0).strip(), "source": "regex"}
    return None


# ---------------------------------------------------------------------------
# Power supply  (alimentation)
# ADDED: loop-powered, wide-range AC/DC supplies, common industrial voltages
# ---------------------------------------------------------------------------
def _extract_alimentation(text: str) -> Optional[dict]:
    # Specific fixed voltages first, then ranges
    patterns = [
        # Loop powered (implies 24V DC 2-wire, no separate supply)
        (re.compile(r"\bloop[\-\s]*powered\b", re.IGNORECASE),                    "Loop powered (2 fils)"),
        (re.compile(r"\balimenté\s+par\s+la\s+boucle\b", re.IGNORECASE),          "Loop powered (2 fils)"),
        # Common fixed voltages
        (re.compile(r"\b24\s*V\s*DC\b", re.IGNORECASE),                           "24V DC"),
        (re.compile(r"\b12\s*V\s*DC\b", re.IGNORECASE),                           "12V DC"),
        (re.compile(r"\b48\s*V\s*DC\b", re.IGNORECASE),                           "48V DC"),
        (re.compile(r"\b230\s*V\s*AC\b", re.IGNORECASE),                          "230V AC"),
        (re.compile(r"\b220\s*V\s*AC\b", re.IGNORECASE),                          "220V AC"),
        (re.compile(r"\b110\s*V\s*AC\b", re.IGNORECASE),                          "110V AC"),
        (re.compile(r"\b115\s*V\s*AC\b", re.IGNORECASE),                          "115V AC"),
        # Range patterns: "10.5 ... 30 V DC", "85 ... 264 V AC"
        (re.compile(r"(\d{1,3}(?:\.\d+)?\s*[\.\-…–]+\s*\d{1,3}(?:\.\d+)?\s*V\s*DC)", re.IGNORECASE), None),
        (re.compile(r"(\d{1,3}(?:\.\d+)?\s*[\.\-…–]+\s*\d{1,3}(?:\.\d+)?\s*V\s*AC)", re.IGNORECASE), None),
    ]
    for pat, canonical in patterns:
        m = pat.search(text)
        if m:
            val = canonical if canonical else m.group(1).strip()
            # Normalize whitespace in range values
            val = re.sub(r"\s+", " ", val)
            return {"value": val, "confidence": 1.0, "quote": m.group(0).strip(), "source": "regex"}
    return None


# ---------------------------------------------------------------------------
# Wire count  (nbFils)
# ADDED: French "2 fils", "4 fils" patterns, 3-wire (RTD)
# ---------------------------------------------------------------------------
def _extract_nb_fils(text: str) -> Optional[dict]:
    patterns = [
        (re.compile(r"\b2[\-\s]*(?:wire|fils?|conducteurs?)\b", re.IGNORECASE),   "2 fils"),
        (re.compile(r"\btwo[\-\s]*wire\b", re.IGNORECASE),                         "2 fils"),
        (re.compile(r"\b4[\-\s]*(?:wire|fils?|conducteurs?)\b", re.IGNORECASE),   "4 fils"),
        (re.compile(r"\bfour[\-\s]*wire\b", re.IGNORECASE),                        "4 fils"),
        (re.compile(r"\b3[\-\s]*(?:wire|fils?|conducteurs?)\b", re.IGNORECASE),   "3 fils"),
        (re.compile(r"\bthree[\-\s]*wire\b", re.IGNORECASE),                       "3 fils"),
    ]
    for pat, canonical in patterns:
        m = pat.search(text)
        if m:
            return {"value": canonical, "confidence": 1.0, "quote": m.group(0).strip(), "source": "regex"}
    return None


# ---------------------------------------------------------------------------
# IP / protection class  (classeProtection)
# ---------------------------------------------------------------------------
def _extract_classe_protection(text: str) -> Optional[dict]:
    for pat in _IP_PATTERNS:
        m = pat.search(text)
        if m:
            value = re.sub(r"\s+", "", m.group(0)).upper()   # "IP 67" → "IP67"
            return {"value": value, "confidence": 1.0, "quote": m.group(0).strip(), "source": "regex"}
    return None


# ---------------------------------------------------------------------------
# Brand (marque)
# ---------------------------------------------------------------------------
def _extract_marque(text: str) -> Optional[dict]:
    for canonical, pat in _BRANDS:
        m = pat.search(text)
        if m:
            return {"value": canonical, "confidence": 1.0, "quote": m.group(0).strip(), "source": "regex"}
    return None


# ---------------------------------------------------------------------------
# Model (modele)
# ---------------------------------------------------------------------------
def _extract_modele(text: str, brand: Optional[str] = None) -> Optional[dict]:
    # Normalise brand key to match dict keys
    brand_key = None
    if brand:
        for key in _MODEL_PATTERNS_BY_BRAND:
            if key.lower() == brand.lower() or brand.lower().startswith(key.lower()):
                brand_key = key
                break

    # Try brand-specific patterns first
    candidate_keys = ([brand_key] if brand_key else []) + [
        k for k in _MODEL_PATTERNS_BY_BRAND if k != brand_key
    ]

    for key in candidate_keys:
        for pat in _MODEL_PATTERNS_BY_BRAND[key]:
            m = pat.search(text)
            if m:
                value = re.sub(r"\s+", " ", m.group(1).strip())
                return {"value": value, "confidence": 1.0, "quote": m.group(0).strip(), "source": "regex"}

    # Generic label fallback
    for pat in _GENERIC_MODEL_PATTERNS:
        m = pat.search(text)
        if m:
            value = re.sub(r"\s+", " ", m.group(1).strip())
            return {"value": value, "confidence": 0.75, "quote": m.group(0).strip(), "source": "regex"}

    return None


# ---------------------------------------------------------------------------
# Reference / order number
# ---------------------------------------------------------------------------
def _extract_reference(text: str) -> Optional[dict]:
    for pat in _REFERENCE_PATTERNS:
        m = pat.search(text)
        if m:
            value = re.sub(r"\s+", " ", m.group(1).strip())
            value = re.sub(r"\s+(?:page|p\.|seite)\s*\d+\b", "", value, flags=re.IGNORECASE).strip()
            if not _looks_like_reference(value):
                continue
            return {"value": value, "confidence": 0.9, "quote": m.group(0).strip(), "source": "regex"}
    return None


# ---------------------------------------------------------------------------
# Equipment name
# ---------------------------------------------------------------------------
def _extract_equipment_name(text: str) -> Optional[dict]:
    patterns = [
        re.compile(r"(?:Equipment|Device|Instrument|Product)\s*Name\s*[:=]\s*([^\n]{3,80})", re.IGNORECASE),
        re.compile(r"(?:Designation|Désignation)\s*[:=]\s*([^\n]{3,80})", re.IGNORECASE),
        re.compile(r"(?:Instrument|Product)\s*[:=]\s*([^\n]{3,80})", re.IGNORECASE),
    ]
    for pat in patterns:
        m = pat.search(text)
        if m:
            value = re.sub(r"\s+", " ", m.group(1).strip())
            if len(value) >= 3:
                return {"value": value, "confidence": 0.85, "quote": m.group(0).strip(), "source": "regex"}
    return None


# ---------------------------------------------------------------------------
# Category  (only when explicitly labeled)
# ---------------------------------------------------------------------------
def _extract_categorie(text: str) -> Optional[dict]:
    pat = re.compile(
        r"(?:Cat(?:égorie|egorie)|Category|Type\s+d['']?instrument)\s*[:=]\s*([^\n]{3,80})",
        re.IGNORECASE,
    )
    m = pat.search(text)
    if m:
        value = re.sub(r"\s+", " ", m.group(1).strip())
        return {"value": value, "confidence": 1.0, "quote": m.group(0).strip(), "source": "regex"}
    return None


# ---------------------------------------------------------------------------
# Measurement type  (keyword frequency heuristic)
# IMPROVED: avoid ambiguity by requiring a score gap between top two candidates.
# ---------------------------------------------------------------------------
def _extract_type_mesure(text: str) -> Optional[dict]:
    t = text.lower()
    scores = {
        "Pression": (
            len(re.findall(r"\bpressure\b", t))
            + len(re.findall(r"\bpression\b", t))
            + 2 * len(re.findall(r"\bdifferential\s+pressure\b", t))
            + 2 * len(re.findall(r"\bgauge\s+pressure\b", t))
            + len(re.findall(r"\bmbar\b|\bpsi\b", t))
        ),
        "Debit": (
            len(re.findall(r"\bflow\b", t))
            + len(re.findall(r"\bdebit\b|\bdébit\b", t))
            + 2 * len(re.findall(r"\bflowmeter\b|\bflow\s*meter\b", t))
            + len(re.findall(r"\bm3/h\b|\bl/h\b|\bgph\b|\bgpm\b", t))
        ),
        "Niveau": (
            len(re.findall(r"\blevel\b", t))
            + len(re.findall(r"\bniveau\b", t))
            + 2 * len(re.findall(r"\blevel\s+transmitt\b", t))
        ),
        "Temperature": (
            len(re.findall(r"\btemperature\b|\btempérature\b", t))
            + 2 * len(re.findall(r"\bthermocouple\b|\bRTD\b|\bPT100\b|\bPT1000\b", text))  # case-sensitive
            + len(re.findall(r"\b°c\b|\b°f\b", t))
        ),
    }

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_type, best_score = sorted_scores[0]
    second_score = sorted_scores[1][1]

    # Require min score of 3 AND clear lead over the runner-up to avoid ties
    if best_score >= 3 and (best_score - second_score) >= 2:
        return {
            "value": best_type,
            "confidence": 0.8,
            "quote": f"{best_type.lower()} keywords: {best_score} hits",
            "source": "regex",
        }
    return None


# ---------------------------------------------------------------------------
# Measurement range  (plageMesure)
# IMPROVED: reject ratio patterns (10:1) and time-unit ranges (ms, µs)
# ---------------------------------------------------------------------------
def _extract_plage_mesure(text: str) -> Optional[dict]:
    # Immediately reject ratio patterns — these are rangeability, not a range
    if re.search(r"\b\d+\s*:\s*\d+\b", text):
        # Only reject if ratio appears near "range" keywords
        ratio_ctx = re.search(
            r"(?:range|plage|étendue|span).{0,80}\b\d+\s*:\s*\d+\b",
            text, re.IGNORECASE | re.DOTALL
        )
        if ratio_ctx:
            return None

    range_pat = re.compile(
        r"(\d+(?:[.,]\d+)?)\s*"              # min value
        r"(?:\.{2,3}|…|–|\-|to|à)\s*"        # separator (added French "à")
        r"(\d+(?:[.,]\d+)?)\s*"              # max value
        r"(L/h|m³/h|m3/h|Nm³/h|t/h|kg/h"   # flow units
        r"|bar|mbar|kPa|MPa|Pa|hPa|psi|inH2O|mmH2O|mmHg"  # pressure units
        r"|°C|°F|K"                          # temperature units
        r"|mm|cm|m|ft|in"                    # level/distance units
        r"|%)\b",                            # percentage
        re.IGNORECASE,
    )

    # Time units to reject
    time_units = re.compile(r"\b(ms|µs|us|μs|nsec|msec)\b", re.IGNORECASE)

    for m in range_pat.finditer(text):
        # Reject time-unit matches
        if time_units.search(m.group(3)):
            continue

        # Check for measuring-range context in surrounding text
        ctx_start = max(0, m.start() - 200)
        context = text[ctx_start:m.start()].lower()
        range_keywords = [
            "measuring range", "span", "plage", "range", "étendue",
            "mesure", "operating range", "working range",
        ]
        if not any(kw in context for kw in range_keywords):
            continue

        try:
            min_val = float(m.group(1).replace(",", "."))
            max_val = float(m.group(2).replace(",", "."))
            unite = m.group(3)
            if min_val >= max_val:  # sanity check
                continue
            return {
                "value": {"min": min_val, "max": max_val, "unite": unite},
                "confidence": 0.85,
                "quote": m.group(0).strip(),
                "source": "regex",
            }
        except ValueError:
            continue

    return None


# ---------------------------------------------------------------------------
# Tag / reperage
# ---------------------------------------------------------------------------
def _extract_reperage(text: str) -> Optional[dict]:
    for pat in _TAG_PATTERNS:
        m = pat.search(text)
        if m:
            value = m.group(1) if pat.groups else m.group(0)
            return {"value": value.strip(), "confidence": 0.9, "quote": m.group(0).strip(), "source": "regex"}
    return None


# ---------------------------------------------------------------------------
# Brand-to-Siemens lock when order number confirms it
# ---------------------------------------------------------------------------
def _lock_siemens_from_reference(full_text: str, results: dict) -> None:
    ref_val = (results.get("reference") or {}).get("value", "")
    if not isinstance(ref_val, str):
        return
    ref_norm = re.sub(r"\s+", "", ref_val).upper()
    if ref_norm.startswith(("6AV", "6ES7", "7MF", "6GK", "6FX", "6SL")):
        results["marque"] = {"value": "Siemens", "confidence": 1.0, "quote": ref_val, "source": "regex"}
        # Try to extract HMI model from text
        hm = re.search(r"\bKTP\d{3,4}\s+(?:BASIC|COMFORT|MOBILE)\b", full_text, re.IGNORECASE)
        if hm and "modele" not in results:
            results["modele"] = {
                "value": f"SIMATIC HMI {hm.group(0).strip()}",
                "confidence": 0.9,
                "quote": hm.group(0).strip(),
                "source": "regex",
            }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def extract_with_regex(full_text: str) -> dict[str, dict]:
    """
    Scan the full PDF text with regex patterns.

    Returns:
        Dict mapping field_name -> {value, confidence, quote, source}
        for each field confidently matched. Unmatched fields are omitted.
    """
    results: dict[str, dict] = {}

    # --- High-confidence structural fields ---
    for field, fn in [
        ("typeSignal",       _extract_type_signal),
        ("communication",    _extract_communication),
        ("alimentation",     _extract_alimentation),
        ("nbFils",           _extract_nb_fils),
        ("classeProtection", _extract_classe_protection),
        ("marque",           _extract_marque),
        ("reperage",         _extract_reperage),
        ("equipmentName",    _extract_equipment_name),
        ("reference",        _extract_reference),
    ]:
        r = fn(full_text)
        if r:
            results[field] = r

    # Model benefits from knowing the brand
    brand = (results.get("marque") or {}).get("value")
    r = _extract_modele(full_text, brand=brand)
    if r:
        results["modele"] = r

    # Lock brand/model when Siemens order number is confirmed
    _lock_siemens_from_reference(full_text, results)

    # --- Heuristic fields (skip for PLC datasheets — too noisy) ---
    if not _looks_like_plc_cpu_datasheet(full_text):
        for field, fn in [
            ("categorie",  _extract_categorie),
            ("typeMesure", _extract_type_mesure),
            ("plageMesure", _extract_plage_mesure),
        ]:
            r = fn(full_text)
            if r:
                results[field] = r

    return results