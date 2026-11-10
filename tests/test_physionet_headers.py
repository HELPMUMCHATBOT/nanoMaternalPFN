"""Schema and leakage tests using source-faithful PhysioNet WFDB header fixtures."""
import numpy as np
from nanomaternalpfn.data.physionet_headers import (
    _parse_comments,_tpehg_dataframe,_ctu_dataframe,
)


def test_tpehg_parses_open_header_and_protects_outcome():
    raw="""tpehg1007 12 20.000000 35180
#Comments:
#    RecID 1007
#    Gestation 35
#    Rectime 31.3
#    Age 30
#    Parity 0
#    Abortions 0
#    Weight 58
#    Hypertension no
#    Diabetes no
#    Placental_position front
#    Bleeding_first_trimester no
#    Bleeding_second_trimester no
#    Funneling negative
#    Smoker no
"""
    a=_parse_comments(raw)
    assert a["Gestation"]=="35" and a["Rectime"]=="31.3"
    pairs=[("tpehg1007.hea",a),
           ("tpehg1008.hea",{**a,"RecID":"1008","Gestation":"39","Age":"35"})]
    x,y=_tpehg_dataframe(pairs)
    assert list(y)==[1,0]
    assert not {"Gestation","Group","Premature","preterm"}.intersection(x.columns)
    assert "Rectime" in x
    assert x["Hypertension"].iloc[0]==0


def test_ctu_header_uses_maternal_metadata_not_delivery_outcomes():
    raw="""1001 2 4 19200
#----- Additional parameters for record 1001
#-- Outcome measures
#pH           7.14
#Apgar1       6
#-- Fetus/Neonate descriptors
#Gest. weeks  37
#Weight(g)    2660
#-- Maternal (risk-)factors
#Age          32
#Gravidity    1
#Parity       0
#Diabetes     1
#Hypertension 0
#Preeclampsia 0
#-- Delivery descriptors
#Deliv. type  1
"""
    a=_parse_comments(raw)
    assert a["pH"]=="7.14"
    pairs=[("1001.hea",a)]*11+[("1002.hea",{**a,"pH":"7.20","Age":"25","Parity":"1"})]*12
    x,y=_ctu_dataframe(pairs)
    assert len(y)==23 and y.sum()==11
    assert list(x.columns)==["Age","Gravidity","Parity","Diabetes","Hypertension","Preeclampsia"]
    assert "pH" not in x and "Apgar1" not in x
    assert np.isfinite(x.to_numpy()).all()
