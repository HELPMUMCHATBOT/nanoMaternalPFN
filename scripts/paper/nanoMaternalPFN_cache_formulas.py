"""Cache externally verified simple Excel formulas without replacing the formulas.

openpyxl writes formulas but no cached numerical result; this ZIP-level patch
caches values already recomputed by the primary statistical analysis.
"""
import os
from pathlib import Path
import re
import zipfile
import pandas as pd
OUT=Path(os.environ.get('NANO_STATS_OUTPUT','/mnt/data/nanoMaternalPFN_statistics'))
book=OUT/'nanoMaternalPFN_Statistical_Results.xlsx'
lb=pd.read_csv(OUT/'leaderboard_all.csv').set_index('method')
task=pd.read_csv(OUT/'task_auroc_all_models.csv')
difference=lb.loc['adaptive_tiny_3expert','auroc']-lb.loc['tabpfn_v2_default','auroc']
values={
 'xl/worksheets/sheet1.xml':{
   'B6':lb.loc['adaptive_tiny_3expert','auroc'],
   'B7':lb.loc['tabpfn_v2_default','auroc'],
   'B8':difference,
 },
 'xl/worksheets/sheet4.xml':{}
}
for rownum,(_,r) in enumerate(task.iterrows(),2):
 values['xl/worksheets/sheet4.xml'][f'F{rownum}']=r['adaptive_tiny_3expert']-r['tabpfn_v2_default']
 values['xl/worksheets/sheet4.xml'][f'G{rownum}']=r['adaptive_tiny_3expert']-r['adaptive_tiny']
original=book.read_bytes()
tmp=book.with_suffix('.cached.xlsx')
with zipfile.ZipFile(book,'r') as src,zipfile.ZipFile(tmp,'w') as dst:
 for item in src.infolist():
  data=src.read(item.filename)
  if item.filename in values:
   s=data.decode('utf-8')
   for addr,val in values[item.filename].items():
    pat=r'(<c r="'+re.escape(addr)+r'"[^>]*><f>[^<]+</f><v>)(</v>)'
    s,n=re.subn(pat,lambda m:m.group(1)+format(val,'.17g')+m.group(2),s)
    if n!=1:
     raise ValueError(f'Unable to cache one formula for {addr}: {n}')
   data=s.encode('utf-8')
  dst.writestr(item,data)
tmp.replace(book)
from openpyxl import load_workbook
formula=load_workbook(book,read_only=True,data_only=False)
cache=load_workbook(book,read_only=True,data_only=True)
assert formula['Read Me']['B8'].value=='=B6-B7'
assert abs(cache['Read Me']['B8'].value-difference)<1e-12
for i,(_,r) in enumerate(task.iterrows(),2):
 assert abs(cache['Task AUROC'][f'F{i}'].value-(r['adaptive_tiny_3expert']-r['tabpfn_v2_default']))<1e-12
print('Cached',sum(map(len,values.values())),'Excel formulas; verified with data_only=True')
