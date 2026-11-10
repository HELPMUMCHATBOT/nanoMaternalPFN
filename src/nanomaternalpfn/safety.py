"""Non-clinical demonstration of configurable vital-sign flags, not a certified CDS."""
def flags(v):
    warnings=[]
    sbp=v.get('sbp')
    dbp=v.get('dbp')
    hb=v.get('hemoglobin')
    glucose=v.get('glucose_mmol_l')
    hr=v.get('heart_rate')
    if (sbp is not None and sbp >= 160) or (dbp is not None and dbp >= 110):
        warnings.append('severe blood pressure measurement: clinical review required')
    if hb is not None and hb < 7:
        warnings.append('low hemoglobin measurement: clinical review required')
    if sbp is not None and hr is not None and sbp > 0 and hr / sbp >= .9:
        warnings.append('elevated shock-index measurement: clinical review required')
    if glucose is not None and (glucose >= 11.1 or glucose < 2.8):
        warnings.append('extreme glucose measurement: clinical review required')
    return warnings
