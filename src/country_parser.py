"""Per-country address parsing for US, India, France."""
import re
from typing import Dict

US_STATES = {
    'AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN',
    'IA','KS','KY','LA','ME','MD','MA','MI','MN','MS','MO','MT','NE','NV','NH',
    'NJ','NM','NY','NC','ND','OH','OK','OR','PA','RI','SC','SD','TN','TX','UT',
    'VT','VA','WA','WV','WI','WY','DC'
}


def parse_address(addr: str, country: str) -> Dict[str, str]:
    """Extract structured address components per country.
    
    Returns dict with keys: zip/pin/postal, state, city depending on country.
    """
    if not isinstance(addr, str):
        return {}
    
    a = addr.upper()
    result = {}
    
    if country in ("India", "IN", "Bharat"):
        m = re.search(r'\b(\d{6})\b', a)
        if m:
            result["pin"] = m.group(1)
        parts = [p.strip() for p in addr.split(",")]
        if len(parts) >= 2:
            result["city"] = parts[-2].upper()
        if len(parts) >= 3:
            result["state"] = parts[-3].upper()
    
    elif country in ("US", "USA", "United States"):
        m = re.search(r'\b(\d{5})(?:-\d{4})?\b', a)
        if m:
            result["zip"] = m.group(1)
        for tok in re.findall(r'\b[A-Z]{2}\b', a):
            if tok in US_STATES:
                result["state"] = tok
                break
    
    elif country in ("France", "FR"):
        m = re.search(r'\b(0[1-9]|[1-8]\d|9[0-5])\d{3}\b', a)
        if m:
            result["postal"] = m.group(1)
        parts = [p.strip() for p in addr.split(",")]
        if parts:
            result["city"] = parts[-1].upper()
    
    return result


def component_match(c1: Dict, c2: Dict) -> Dict[str, int]:
    """Return 0/1 features for component matches."""
    feats = {}
    for key in ["pin", "zip", "postal"]:
        v1, v2 = c1.get(key), c2.get(key)
        if v1 and v2:
            feats[f"{key}_match"] = int(v1 == v2)
        else:
            feats[f"{key}_match"] = -1
    
    if "city" in c1 and "city" in c2:
        feats["city_match"] = int(c1["city"] == c2["city"])
    else:
        feats["city_match"] = -1
    
    if "state" in c1 and "state" in c2:
        feats["state_match"] = int(c1["state"] == c2["state"])
    else:
        feats["state_match"] = -1
    
    return feats