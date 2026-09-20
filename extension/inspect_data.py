"""Single-pass, read-only audit of the supplied wide CSV and compact curve extraction."""
import argparse,csv,hashlib,json,os
from pathlib import Path
import numpy as np

def parse_row(raw,encoding): return next(csv.reader([raw.decode(encoding).rstrip('\r\n')]))
def f(v):
    try: return float(v)
    except ValueError: return np.nan
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--csv',required=True);ap.add_argument('--out',default=str(Path(__file__).resolve().parents[1]));a=ap.parse_args()
    src=Path(a.csv); out=Path(a.out); small=out/'data_small'; logs=out/'outputs'/'logs';small.mkdir(parents=True,exist_ok=True);logs.mkdir(parents=True,exist_ok=True)
    targets={0,900,1800,2700,3600,4500,5400,6300}; picks=[]; sha=hashlib.sha256(); rows=0; bad_width=0; total=finite=valid=nonmono=dup=0; reasons={'nonfinite':0,'x_gt_100':0,'double_zero':0}; offsets=[]
    with src.open('rb') as fh:
        offset=fh.tell(); head=fh.readline();sha.update(head); header=parse_row(head,'utf-8-sig'); expected=len(header)
        for raw in fh:
            offset=fh.tell();sha.update(raw); fields=parse_row(raw,'utf-8'); rows+=1
            if len(fields)!=expected: bad_width+=1;continue
            vals=np.fromiter((f(v) for v in fields),float,count=expected); x=vals[3::2];z=vals[4::2]; good=np.isfinite(x)&np.isfinite(z); finite+=int(good.sum()); reasons['nonfinite']+=int((~good).sum()); gt=x>100;reasons['x_gt_100']+=int((good&gt).sum()); zero=(x==0)&(z==0);reasons['double_zero']+=int((good&~gt&zero).sum()); keep=good&~gt&~zero; valid+=int(keep.sum());total+=x.size
            xx=x[keep]; nonmono+=int(np.any(np.diff(xx)<0)) if xx.size>1 else 0;dup+=int(np.any(np.diff(xx)==0)) if xx.size>1 else 0
            if (rows-1) in targets:
                picks.append({'row_index':rows-1,'byte_offset':offset-len(raw),'y_line':vals[2],'x':x[keep],'z':z[keep],'pair_columns':np.flatnonzero(keep).astype(int)})
                offsets.append({'row_index':rows-1,'byte_offset':offset-len(raw),'valid_points':int(keep.sum()),'excluded_nonfinite':int((~good).sum()),'excluded_x_gt_100':int((good&gt).sum()),'excluded_double_zero':int((good&~gt&zero).sum())})
    digest=sha.hexdigest();np.savez_compressed(small/'real_curves.npz',curves=np.array(picks,dtype=object))
    audit={'path':str(src),'bytes':src.stat().st_size,'sha256':digest,'encoding':'utf-8-sig header / utf-8 rows','delimiter':',','header_columns':expected,'header_prefix':header[:9],'header_suffix':header[-4:],'rows':rows,'expected_pairs':(expected-3)//2,'bad_width_rows':bad_width,'pair_total':total,'finite_pairs':finite,'valid_pairs':valid,'reasons':reasons,'nonmonotonic_rows':nonmono,'duplicate_x_rows':dup,'selection_rule':'fixed predeclared source row indices 0,900,...,6300; valid rule is finite AND X<=100 AND NOT(X==0 AND Z==0)','representation':'arc_length_to_radius / r(s), because audited X has many nonmonotonic rows; no hard sorting Z(X).','json_mapping':'UNCONFIRMED: label=122 polygon is not named as a defect line and not used for training.','selected_offsets':offsets}
    (small/'source_index.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8');(logs/'csv_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(audit,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
