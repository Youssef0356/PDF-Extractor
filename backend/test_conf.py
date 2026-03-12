from services.llm_extractor import _compute_confidence
import json

tests = [
  {
    'field': 'equipmentName',
    'value': 'SITRANS LR150',
    'quote': 'SITRANS LR150',
    'model_conf': 1.0,
    'checks': {
      'non_null': True,
      'expected_type': True,
      'allowed_value': True,
      'quote_supported': True,
      'quote_in_context': True
    }
  },
  {
    'field': 'alimentation',
    'value': '35 V DC',
    'quote': '12 … 35 V DC\nat 20 mA:\n9 … 35 V DC',
    'model_conf': 0.95,
    'checks': {
      'non_null': True,
      'expected_type': True,
      'allowed_value': True,
      'quote_supported': True,
      'quote_in_context': False
    }
  }
]

for t in tests:
    conf = _compute_confidence(
        checks=t['checks'],
        chunks=[{'distance': 0.2}],
        source='llm',
        model_confidence=t['model_conf']
    )
    print(f"{t['field']} calculated conf: {conf}")
