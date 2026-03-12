import json
import glob
import os

files = sorted(glob.glob('../outputs/*_evidence.json'), key=os.path.getmtime, reverse=True)[:2]
for f in files:
    print(f'\n--- {os.path.basename(f)} ---')
    with open(f, encoding='utf-8') as fh:
        data = json.load(fh)
        meta = data.get('llm_meta')
        if not meta:
            print('No llm_meta found.')
            continue
            
        for field in ['equipmentName', 'categorie', 'typeMesure', 'alimentation']:
            m = meta.get(field)
            if not m:
                print(f'{field}: Not in meta')
                continue
            print(f'{field}:')
            print(f'  value: {m.get("value")}')
            print(f'  accepted: {m.get("accepted")}')
            print(f'  confidence: {m.get("confidence")}')
            print(f'  rejection_reason: {m.get("rejection_reason")}')
            raw_str = (m.get("raw") or "")[:150].replace('\n', ' ')
            print(f'  raw: {raw_str}')
            print(f'  quote: {m.get("quote")}')
