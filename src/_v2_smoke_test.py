"""Smoke test for V2 scoring module - run from project root."""
import sys
sys.path.insert(0, 'src')

import pandas as pd
from main_v2 import _build_preindexed_lookup, _score_pair_inline

s1_df = pd.DataFrame({
    'entity_id': ['S1-001', 'S1-002'],
    'business_name': ['Acme Robotics Corp.', 'Starbucks Coffee'],
    'business_address': ['123 Main St, NY', '500 Pine St'],
    'country': ['US', 'US'],
})
s2_df = pd.DataFrame({
    'entity_id': ['S2-001', 'S2-002', 'S2-003'],
    'business_name': ['Acme Robotics Corporation', 'Starbucks', 'Different Co'],
    'business_address': ['123 Main Street, NY', '500 Pine Street', '999 Other St'],
    'country': ['USA', 'US', 'US'],
})

s1_idx = _build_preindexed_lookup(s1_df)
s2_idx = _build_preindexed_lookup(s2_df)

print("s1_idx keys:", list(s1_idx.keys()))
print("s2 country_norm for S2-001:", s2_idx['S2-001']['_country_norm'])

s1row = s1_idx['S1-001']
for cid in ['S2-001', 'S2-002', 'S2-003']:
    crow = s2_idx[cid]
    sc = _score_pair_inline(
        s1row['_name_tok'], s1row['_addr_tok'], s1row['_name_str'], s1row['_country_norm'],
        crow['_name_tok'], crow['_addr_tok'], crow['_name_str'], crow['_country_norm'],
    )
    print(f'  S1-001 vs {cid}: score={sc:.3f}')
print('OK - V2 scoring works')
