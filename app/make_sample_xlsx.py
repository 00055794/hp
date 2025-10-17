import pandas as pd
from pathlib import Path

cols = [
    "ROOMS","TOTAL_AREA","FLOOR","TOTAL_FLOORS","FURNITURE","CONDITION","CEILING","MATERIAL","YEAR","LATITUDE","LONGITUDE"
]

rows = [
    {"ROOMS":2, "TOTAL_AREA":60, "FLOOR":5,  "TOTAL_FLOORS":9,  "FURNITURE":"Partial", "CONDITION":"Good",      "CEILING":2.7, "MATERIAL":"Brick",     "YEAR":2015, "LATITUDE":43.238, "LONGITUDE":76.886},
    {"ROOMS":1, "TOTAL_AREA":35, "FLOOR":8,  "TOTAL_FLOORS":12, "FURNITURE":"No",      "CONDITION":"Excellent", "CEILING":2.6, "MATERIAL":"Panel",     "YEAR":2019, "LATITUDE":51.160, "LONGITUDE":71.435},
    {"ROOMS":3, "TOTAL_AREA":85, "FLOOR":3,  "TOTAL_FLOORS":5,  "FURNITURE":"Full",    "CONDITION":"Good",      "CEILING":2.8, "MATERIAL":"Monolithic","YEAR":2010, "LATITUDE":42.315, "LONGITUDE":69.585},
]

df = pd.DataFrame(rows, columns=cols)
out = Path(__file__).parent / "sample_input.xlsx"
df.to_excel(out, index=False)
print(f"Wrote sample file: {out}")
