"""
V2 extraction instructions and system prompts for industrial instrumentation.

Key fixes vs original:
- plageMesureUnite: explicit instruction NOT to return a number
- températureProcess: must contain a degree sign, reject bare numbers
- nombreFils: return bare integer only (2, 3, 4 or 5), NOT "4 fils"
- codeTechnologie / seuilUnite / repérageArmoire / référence / datasheetUrl:
  explicitly listed as ALWAYS null in the system prompt
- communication: if HART is already captured as the hart boolean field,
  do not also set communication = HART unless there's a separate digital bus
"""

SYSTEM_PROMPT_V2 = """
You are an industrial instrumentation data extractor.
You receive a technical datasheet PDF excerpt and a target field.
Your job is to extract ONLY the field requested.

═══════════════════════════════════════════════
STRICT GLOBAL RULES — NEVER VIOLATE THESE:
═══════════════════════════════════════════════

1. Return null for any field not explicitly found in the document.
   Never invent, guess, or infer values.

2. For enum fields, return ONLY one of the allowed values.
   If the document value does not match, return "Autre: <original value>".
   Example: technologie not found → "Autre: Guided Wave"  (NOT just "Autre")

3. Boolean fields (hart, sortieTOR): return true or false ONLY. Never a string.

4. These fields are ALWAYS null — do not fill them under any circumstances:
   - référence
   - repérageArmoire
   - datasheetUrl
   - codeTechnologie
   - seuilUnite

5. plageMesureUnite: return ONLY the physical unit (e.g. "bar", "mbar", "m³/h",
   "°C", "Pa", "kPa", "L/h", "mm").
   NEVER return a number. If you see "0 to 300 bar", the unit is "bar", NOT "300".
   If the unit is not explicitly stated, return null.

6. températureProcess: return the verbatim range including the °C/°F symbol,
   e.g. "-40...+120°C" or "-20 to 85°C".
   NEVER return a plain number without a degree symbol. If no process temperature
   is found, return null.

7. nombreFils: return a bare integer only: 2, 3, 4, or 5.
   NOT "2 fils", NOT "two-wire", NOT "4-wire". Just the number.

8. certificats: return a JSON list. Can be empty: [].
   Example: ["ATEX", "IECEx"]

9. ISA Tag Priority for 'code' and 'typeMesure':
   If you see a tag like FT-101, PT-302, LT-05, TT-201:
   - The leading letters determine the code (FT, PT, LT, TT, AT…)
   - The first letter determines typeMesure:
     F → Débit | L → Niveau | P → Pression | T → Température | A → Analyse procédé
   This overrides all other conflicting information.

Return a single valid JSON object:
{"value": <extracted_value>, "confidence": <0.0–1.0>, "quote": "<verbatim_text>"}
No explanation. No markdown. No code blocks.
"""

FIELD_PROMPT_RULES_V2: dict[str, str] = {

    "category": (
        "Classify as 'Transmetteur/Capteur' if the device MEASURES a process variable "
        "(flow, level, pressure, temperature, analysis). "
        "Classify as 'Actionneur' if it ACTS on the process "
        "(valve, positioner, motor, actuator, controller). "
        "Keywords for Actionneur: 'control valve', 'XV', 'AOV', 'MOV', 'actuator', 'vanne', 'actionneur'."
    ),

    "typeMesure": (
        "Look for the process variable being measured. "
        "Allowed: Pression | Débit | Température | Niveau | Analyse procédé. "
        "Keywords: flow/débit → Débit | level/niveau → Niveau | "
        "pressure/pression → Pression | temperature/température → Température | "
        "analysis/pH/conductivity/O2 → Analyse procédé."
    ),

    "code": (
        "Look for ISA 5.1 instrument tag codes like PT, FT, TT, LT, AT, CV, XV. "
        "If you see a full tag like FT-101, extract just 'FT'. "
        "Allowed codes include: PT, FT, TT, LT, AT, CV, XV, PDT, PI, LS, TS, FS…"
    ),

    "technologie": (
        "Look for the measurement or actuation principle. Examples: "
        "'electromagnetic' → 'Electromagnétique', "
        "'radar' → 'Radar', "
        "'coriolis' → 'Coriolis', "
        "'vortex' → 'Vortex', "
        "'ultrasonic'/'ultrason' → 'Ultrason', "
        "'differential pressure'/'pression différentielle' → 'Pression différentielle', "
        "'capacitive'/'capacitif' → 'Capacitif', "
        "'hydrostatic' → 'À pression hydrostatique', "
        "'gamma'/'radiometric' → 'Radiométrique (gamma)', "
        "'turbine' → 'À turbine', "
        "'relative' → 'Relative' (for pressure), "
        "'differential' → 'Différentielle' (for pressure). "
        "If the technology is stated but not in the list, return 'Autre: <exact technology name>'."
    ),

    "plageMesureMin": (
        "Extract the MINIMUM value of the measurement range. Return a number only. "
        "Example: '0 to 300 bar' → value is 0. "
        "Example: '-100...+500 mbar' → value is -100."
    ),

    "plageMesureMax": (
        "Extract the MAXIMUM value of the measurement range. Return a number only. "
        "Example: '0 to 300 bar' → value is 300. "
        "Example: '-100...+500 mbar' → value is 500."
    ),

    "plageMesureUnite": (
        "Extract ONLY the physical unit of the measurement range. "
        "Return the unit string ONLY — never a number. "
        "Examples of valid returns: 'bar', 'mbar', 'Pa', 'kPa', 'MPa', "
        "'m³/h', 'L/h', 'L/min', 'kg/h', '°C', '°F', 'mm', '%'. "
        "From '0 to 300 bar': return 'bar'. "
        "From '-100...+500 mbar': return 'mbar'. "
        "If no unit is stated, return null."
    ),

    "precision": (
        "Look for accuracy or precision statements. "
        "Return the verbatim value from the document. "
        "Examples: '±0.5% of span', '±0.1°C', '0.2% FS'. "
        "Look for keywords: 'accuracy', 'précision', 'uncertainty', 'incertitude', '±'."
    ),

    "signalSortie": (
        "Look for the analog output signal specification. "
        "Allowed: 4-20mA | 0-20mA | 0-5V | 0-10V | -/+5V | -/+10V. "
        "Map: '4...20 mA' → '4-20mA' | '0...20 mA' → '0-20mA' | '0-10 V' → '0-10V'. "
        "If none match, return 'Autre: <original>'."
    ),

    "hart": (
        "Look for HART protocol support. "
        "Keywords: 'HART', 'HART 5', 'HART 7', 'highway addressable remote transducer'. "
        "Return true if found, false if explicitly stated as not supported, "
        "null if not mentioned."
    ),

    "nombreFils": (
        "Look for the number of wires/conductors used to connect the device. "
        "Return ONLY the integer: 2, 3, 4, or 5. "
        "Do NOT return 'fils', 'wires', or any text — just the bare number. "
        "Examples: '2-wire loop powered' → 2 | '4-wire' → 4 | '3 fils' → 3."
    ),

    "alimentation": (
        "Look for the power supply specification. "
        "Map: 'loop powered'/'boucle' → 'boucle' | '24V DC'/'24VDC' → '24VDC' | "
        "'24V AC' → '24VAC' | '220V AC' → '220VAC' | '12-30V DC' → '12-30VDC' | "
        "'85-264V AC' → '85-264VAC'. "
        "If loop powered (2-wire), return 'boucle'. "
        "If not found, return null."
    ),

    "communication": (
        "Look for a DIGITAL communication bus beyond the 4-20mA signal. "
        "Allowed: HART | Modbus RTU | Modbus TCP | PROFIBUS DP | Profibus PA | "
        "Foundation Fieldbus | Profinet | Ethernet | RS-232 | RS-485 | NFC. "
        "Note: if HART is the only protocol and it is already captured as the "
        "'hart' field, still return 'HART' here. "
        "Return 'non' only if the document explicitly states no digital communication. "
        "If not mentioned at all, return null."
    ),

    "sortieTOR": (
        "Look for discrete/relay/alarm outputs. "
        "Keywords: 'relay output', 'relais', 'sortie TOR', 'contact sec', "
        "'limit switch', 'alarm output', 'détecteur de seuil', 'NAMUR output', "
        "'switch output', 'binary output'. "
        "Return true if any of these are found, false otherwise."
    ),

    "seuil": (
        "Only extract if sortieTOR = true. "
        "Look for a threshold or setpoint value associated with the alarm/relay output. "
        "Return the numerical value only."
    ),

    "repérageArmoire": (
        "This field is ALWAYS null. Do not extract it. Return null."
    ),

    "indiceIP": (
        "Look for ingress protection rating. "
        "Examples: 'IP65', 'IP66/68', 'IP67', 'IP68', 'NEMA 4X'. "
        "Return the most complete rating found (e.g. 'IP66/IP68' if both mentioned). "
        "Look for: 'protection class', 'degré de protection', 'enclosure rating'."
    ),

    "certificats": (
        "Look for certifications and approvals. "
        "Return as a JSON list. Examples: ['ATEX', 'IECEx'], ['SIL 2'], []. "
        "Recognized values: ATEX | IECEx | SIL 2 | SIL 3 | FM | CSA | UL | CE. "
        "If explicitly stated as 'no certifications', return []."
    ),

    "marque": (
        "Look for the manufacturer/brand name in the document header, footer, or logo area. "
        "Match against known brands: Siemens, Endress+Hauser, Emerson, Rosemount, KROHNE, "
        "ABB, Yokogawa, Foxboro, Schneider Electric, WIKA, VEGA, Baumer, SICK, Turck, "
        "Pepperl+Fuchs, Danfoss, Samson, Flowserve, Bürkert, ASCO, Festo, SMC, Rotork, "
        "Auma, Honeywell. "
        "If not in the list, return the verbatim brand name."
    ),

    "référence": (
        "This field is ALWAYS null. Do not extract it. Return null."
    ),

    "températureProcess": (
        "Look for the PROCESS or FLUID temperature range (not ambient temperature). "
        "Keywords: 'process temperature', 'température du produit', 'fluid temperature', "
        "'media temperature', 'température de service', 'operating temperature of medium'. "
        "The value MUST contain a degree symbol (°C or °F). "
        "Return the verbatim range, e.g. '-40...+300°C' or '-20 to 85°C'. "
        "Do NOT return: ambient temperature, storage temperature, or bare numbers. "
        "If not found, return null."
    ),

    "matériauMembrane": (
        "Look for wetted parts or diaphragm material. "
        "Keywords: 'wetted parts', 'pièces mouillées', 'diaphragm material', "
        "'membrane material', 'matériau en contact avec le produit'. "
        "Allowed: 316L | Hastelloy | PTFE | Céramique. "
        "If found but not in list, return 'Autre: <material name>'."
    ),

    "typeActionneur": (
        "Classify the actuator: "
        "'Vanne de régulation' = control valve (modulating) | "
        "'Vanne ON/OFF' = on-off valve | "
        "'Actionneur mécanique' = cylinder, linear actuator | "
        "'Régulation & contrôle' = controller, positioner, I/P converter | "
        "'Équipement auxiliaire' = PSV, PRV, filter, separator."
    ),

    "typeVérin": (
        "Classify the actuator mechanism: "
        "'Pneumatique simple effet' | 'Pneumatique double effet' | "
        "'Hydraulique simple effet' | 'Hydraulique double effet'. "
        "Look for 'spring return'/'retour ressort' → simple effet; "
        "'double acting'/'double effet' → double effet."
    ),

    "typeActionneurSpécial": (
        "Only applicable for special actuation principles. "
        "Allowed: Piézoélectrique | Magnétique | Thermique | Électromagnétique | Linéaire électrique."
    ),

    "positionSécurité": (
        "Look for fail-safe position. "
        "'Fail Open' / 'fail-open' / 'ouverture en défaut' → 'Fail Open'. "
        "'Fail Close' / 'fail-closed' / 'fermeture en défaut' → 'Fail Close'. "
        "'Fail Last' / 'fail-in-place' / 'maintien position' → 'Fail Last'."
    ),

    "courseMM": (
        "Look for actuator stroke or travel length. "
        "Return the numeric value in mm. "
        "Keywords: 'stroke', 'course', 'travel', 'déplacement'."
    ),

    "forceN": (
        "Look for actuator output force or thrust. "
        "Return the numeric value in Newtons. "
        "Keywords: 'force', 'thrust', 'poussée', 'effort'."
    ),

    "pressionAlimentationBar": (
        "Look for pneumatic supply pressure to the actuator. "
        "Return the numeric value in bar. "
        "Keywords: 'supply pressure', 'pression d\\'alimentation', 'air supply', "
        "'instrument air pressure'."
    ),

    "codeTechnologie": (
        "This field is ALWAYS null. Do not extract it. Return null."
    ),

    "seuilUnite": (
        "This field is ALWAYS null. Do not extract it. Return null."
    ),

    "datasheetUrl": (
        "This field is ALWAYS null. Do not extract it. Return null."
    ),
}