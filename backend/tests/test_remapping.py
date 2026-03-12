
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.llm_extractor import _apply_cross_field_guards

def test_alimentation_to_nbfils_remapping():
    # Case 1: "Loop powered" should remap to 2 fils and clear alimentation
    data = {"alimentation": "Loop powered (2 fils)"}
    result = _apply_cross_field_guards(data)
    assert result.get("nbFils") == "2 fils"
    assert "alimentation" not in result

    # Case 2: Voltage range should NOT clear alimentation
    data = {"alimentation": "12 ... 35 V DC"}
    result = _apply_cross_field_guards(data)
    assert result.get("alimentation") == "12 ... 35 V DC"
    assert result.get("nbFils") is None

    # Case 3: Both wiring and voltage (e.g. "24V DC, 2 wires") 
    data = {"alimentation": "24V DC, 2 wires", "nbFils": None}
    result = _apply_cross_field_guards(data)
    assert result.get("nbFils") == "2 fils"
    # It might keep alimentation if voltage is present
    assert result.get("alimentation") == "24V DC, 2 wires"

def test_nomalarme_validation():
    # Case 1: Real alarm name
    data = {"sortiesAlarme": [{"nomAlarme": "High High", "typeAlarme": "Haut", "seuilAlarme": 90.0}]}
    result = _apply_cross_field_guards(data)
    assert len(result["sortiesAlarme"]) == 1

    # Case 2: Parameter description as alarm name
    data = {"sortiesAlarme": [{"nomAlarme": "Output signal = Damping (63 % of the input variable), adjustable", "seuilAlarme": 0.0}]}
    result = _apply_cross_field_guards(data)
    assert "sortiesAlarme" not in result

    # Case 3: Placeholder values (0.0s)
    data = {"sortiesAlarme": [{"nomAlarme": "Delay", "seuilAlarme": 0.0, "uniteAlarme": "s"}]}
    result = _apply_cross_field_guards(data)
    # "Delay" contains "delay", so it's kept even with 0.0s
    assert len(result["sortiesAlarme"]) == 1
    
    data = {"sortiesAlarme": [{"nomAlarme": "Some random name", "seuilAlarme": 0.0, "uniteAlarme": "s"}]}
    result = _apply_cross_field_guards(data)
    assert "sortiesAlarme" not in result

if __name__ == "__main__":
    test_alimentation_to_nbfils_remapping()
    test_nomalarme_validation()
    print("All backend remapping tests passed!")
