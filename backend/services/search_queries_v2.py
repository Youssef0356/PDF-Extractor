"""
V2 multilingual vector search queries for all schema fields.
"""

FIELD_SEARCH_QUERIES = {
    "category": [
        "sensor transmitter actuator valve controller",
        "capteur transmetteur actionneur vanne régulateur",
        "this instrument is a transmitter actuator",
        "type of device equipment category",
    ],
    "typeMesure": [
        "measurement type flow level pressure temperature",
        "type de mesure débit niveau pression température",
        "this instrument measures",
        "functional principle application",
    ],
    "code": [
        "instrument code PT FT TT LT transmitter indicator",
        "transmetteur indicateur manomètre pressostat",
        "equipment designation type ISA tag",
    ],
    "technologie": [
        "measurement principle technology electromagnetic radar coriolis",
        "principe de mesure technologie électromagnétique radar ultrason",
        "operating principle sensor type",
    ],
    "plageMesureMin": [
        "measuring range plage de mesure min",
        "span min range flow rate level pressure",
        "lower limit of measurement",
    ],
    "plageMesureMax": [
        "measuring range plage de mesure max",
        "span max range flow rate level pressure",
        "upper limit of measurement",
    ],
    "plageMesureUnite": [
        "measuring range unit unit of measure",
        "bar mbar Pa kPa L/h m3/h kg/h C F",
    ],
    "signalSortie": [
        "output signal 4-20mA signal de sortie",
        "analog output current voltage signal",
        "sortie analogique courant tension",
    ],
    "hart": [
        "HART communication protocol HART 5 HART 7",
        "highway addressable remote transducer",
    ],
    "nombreFils": [
        "2-wire 4-wire wiring loop powered",
        "nombre de fils raccordement 2 fils 4 fils",
        "two wire four wire connection",
    ],
    "alimentation": [
        "power supply voltage 24VDC 24VAC 220VAC loop powered",
        "alimentation tension boucle de courant",
        "supply voltage operating voltage",
    ],
    "communication": [
        "communication protocol HART Modbus PROFIBUS Profinet fieldbus",
        "protocole de communication numérique",
        "digital interface fieldbus",
    ],
    "sortieTOR": [
        "relay output alarm contact limit switch NAMUR",
        "sortie TOR relais contact sec alarme seuil",
        "binary output switching output",
    ],
    "seuil": [
        "switching threshold setpoint alarm limit value",
        "seuil de commutation valeur limite",
        "trip point alarm setpoint",
    ],
    "precision": [
        "accuracy precision measurement uncertainty",
        "précision exactitude incertitude de mesure",
        "measurement error ± percent",
    ],
    "marque": [
        "manufacturer brand fabricant marque",
        "Siemens KROHNE Endress+Hauser ABB Yokogawa",
        "copyright logo manufacturer name",
    ],
    "certificats": [
        "ATEX IECEx SIL certification hazardous area",
        "certificat homologation zone explosive",
        "explosion proof intrinsically safe",
    ],
    "indiceIP": [
        "IP protection class ingress protection NEMA",
        "indice de protection IP67 IP68 étanchéité",
        "enclosure rating weatherproof",
    ],
    "températureProcess": [
        "process temperature operating temperature range",
        "température du produit température de service",
        "temperature limits min max operating",
    ],
    "matériauMembrane": [
        "wetted parts material diaphragm membrane",
        "matériau membrane pièces en contact produit",
        "316L Hastelloy PTFE ceramic wetted material",
    ],
    # Actionneur specific
    "typeActionneur": [
        "control valve on-off valve actuator controller",
        "vanne de régulation vanne tout-ou-rien actionneur",
        "safety valve relief valve equipment type",
    ],
    "typeVérin": [
        "actuator type pneumatic hydraulic single double acting",
        "vérin pneumatique hydraulique simple double effet",
        "cylinder actuator spring return",
    ],
    "typeActionneurSpécial": [
        "piezoelectric magnetic thermal electromagnetic linear actuator",
        "piézoélectrique magnétique thermique linéaire électrique",
    ],
    "positionSécurité": [
        "fail open fail close fail last safety position",
        "position de sécurité défaut ouvert fermé",
        "failure mode spring return air fail",
    ],
    "courseMM": [
        "stroke travel length course mm",
        "course actionneur vérin mm",
        "piston stroke distance",
    ],
    "forceN": [
        "force thrust output N Newton",
        "force actionneur poussée N",
        "actuator force torque",
    ],
    "pressionAlimentationBar": [
        "supply pressure air pressure bar",
        "pression alimentation air bar",
        "pneumatic supply pressure operating pressure",
    ],
}
