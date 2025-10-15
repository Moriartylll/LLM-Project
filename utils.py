# utils.py

import os
import re
import json
from datetime import datetime
from typing import List, Optional, Dict

# Read file bytes from various input formats
def read_file_bytes(f):
    if hasattr(f, "read"):
        try:
            data = f.read()
            name = os.path.basename(getattr(f, "name", "uploaded"))
            return data, name
        except Exception:
            pass
    path = getattr(f, "path", None)
    if path and os.path.exists(path):
        with open(path, "rb") as fp:
            data = fp.read()
        return data, os.path.basename(path)
    if isinstance(f, dict) and "path" in f and os.path.exists(f["path"]):
        with open(f["path"], "rb") as fp:
            data = fp.read()
        return data, os.path.basename(f["path"])
    if isinstance(f, (str, os.PathLike)) and os.path.exists(str(f)):
        p = str(f)
        with open(p, "rb") as fp:
            data = fp.read()
        return data, os.path.basename(p)
    raise TypeError(f"Unsupported upload type: {type(f)}; value={repr(f)}")
    

# Convert Swedish price string to float
def to_float(s: Optional[str]) -> Optional[float]:
    if not s:
        return None
    s = s.replace(" ", "").replace("kr", "").replace("KR", "")
    if s.count(",") == 1 and s.count(".") == 0:
        s = s.replace(",", ".")
    elif s.count(",") > 1 and s.count(".") == 0:
        s = s.replace(",", "")
    return float(re.sub(r"[^\d.-]", "", s)) if re.search(r"\d", s) else None


# Normalize date string to ISO format
def normalize_date(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    s2 = s.replace(".", "-").replace("/", "-")
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d-%m-%y"):
        try:
            return datetime.strptime(s2, fmt).date().isoformat()
        except:
            pass
    m = DATE_RE.search(s2)
    if m:
        return normalize_date(m.group(1))
    return None