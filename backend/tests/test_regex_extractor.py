import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from services.regex_extractor import extract_with_regex

def test_extract_brand_siemens():
    text = "The manufacturer is Siemens and the order number is 6ES72141AG400XB0."
    res = extract_with_regex(text)
    
    assert "marque" in res
    assert res["marque"]["value"] == "Siemens"
    
    assert "reference" in res
    assert res["reference"]["value"] == "6ES72141AG400XB0"


def test_extract_brand_krohne_model():
    text = "KROHNE OPTIFLUX 4300 C flowmeter."
    res = extract_with_regex(text)
    
    assert "marque" in res
    assert res["marque"]["value"] == "KROHNE"
    
    assert "modele" in res
    assert "OPTIFLUX" in res["modele"]["value"]


def test_extract_signal_and_power():
    text = "Output signal: 4-20mA. Power supply: 24V DC. 2-wire setup."
    res = extract_with_regex(text)
    
    assert "typeSignal" in res
    assert res["typeSignal"]["value"] == "4-20mA"
    
    assert "alimentation" in res
    assert res["alimentation"]["value"] == "24V DC"
    
    assert "nbFils" in res
    assert res["nbFils"]["value"] == "2"

def test_extract_communication():
    text = "The device uses HART protocol."
    res = extract_with_regex(text)
    
    assert "communication" in res
    assert res["communication"]["value"] == "HART"

def test_extract_plage_mesure():
    text = "Measuring range is 0 ... 100 bar."
    res = extract_with_regex(text)
    
    assert "plageMesure" in res
    assert res["plageMesure"]["value"]["min"] == 0.0
    assert res["plageMesure"]["value"]["max"] == 100.0
    assert res["plageMesure"]["value"]["unite"].lower() == "bar"

def test_extract_reperage():
    text = "Tag: PT-102A"
    res = extract_with_regex(text)
    
    assert "reperage" in res
    assert res["reperage"]["value"] == "PT-102A"
