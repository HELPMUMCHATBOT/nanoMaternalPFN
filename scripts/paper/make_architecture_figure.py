#!/usr/bin/env python3
"""Document the implemented PFN forward path (diagram of code, not performance)."""
from pathlib import Path
import argparse
from html import escape

BOXES=[
    ('Frozen context and queries','X: context rows plus query rows; y known for context only'),
    ('Context-only input encoding','Per-feature normalization from context; missing-value indicators'),
    ('Feature and target tokens','Projected value+missingness; context one-hot and query class prior'),
    ('Repeated bidirectional-attention blocks','Feature self-attention within row; patient queries attend to context keys/values; MLP'),
    ('Query-label decoder','v2 affine-free LayerNorm; query target token; classification logits'),
]

def render(path:Path)->Path:
    width=1060
    height=630
    boxx=120
    boxwidth=820
    h=78
    start=45
    step=109
    pieces=[
      f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="heading desc">',
      '<title id="heading">nanoMaternalPFN v2 inference architecture, derived from model.py</title>',
      '<desc id="desc">Context-only normalization, missingness and target encoding, repeated feature and patient attention, and query-label decoder.</desc>',
      '<rect x="0" y="0" width="1060" height="630" fill="white"/>',
      '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6" fill="none" stroke="#333" stroke-width="1.5"/></marker></defs>',
    ]
    for i,(name,detail) in enumerate(BOXES):
        y=start+i*step
        fill='#f5f5f5' if i%2==0 else '#fff'
        pieces += [f'<rect x="{boxx}" y="{y}" width="{boxwidth}" height="{h}" rx="8" fill="{fill}" stroke="#222" stroke-width="1.2"/>',
                   f'<text x="{boxx+26}" y="{y+30}" font-family="Arial,Helvetica,sans-serif" font-size="20" font-weight="bold" fill="#111">{escape(name)}</text>',
                   f'<text x="{boxx+26}" y="{y+57}" font-family="Arial,Helvetica,sans-serif" font-size="16" fill="#333">{escape(detail)}</text>']
        if i+1<len(BOXES):
            pieces.append(f'<path d="M{width//2} {y+h+3} L{width//2} {y+step-8}" fill="none" stroke="#333" stroke-width="1.8" marker-end="url(#arrow)"/>')
    pieces.append('<text x="530" y="610" text-anchor="middle" font-family="Arial,Helvetica,sans-serif" font-size="13" fill="#555">Source: src/nanomaternalpfn/model.py (pre_ln_v2). Research architecture schematic; not a measured result.</text>')
    pieces.append('</svg>')
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text('\n'.join(pieces)+'\n',encoding='utf-8')
    return path

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    print(render(a.output))
