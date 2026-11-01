"""Archive the GPU-1 optimization evidence and generate consistent summaries."""
import gzip
import json
import math
import re
import shutil
import statistics
import subprocess
from decimal import Decimal
from pathlib import Path

OUT = Path(__file__).resolve().parent / 'rewrite/results/round5_h200_gpu1'
OUT.mkdir(parents=True, exist_ok=True)
TAGS = ['packed','direct','fused','cache','baseline250','gate32','final250','baseline-repeat','optimized250']

def grade(row):
    nums=re.findall(r'[-+]?[0-9][0-9,]*(?:[.][0-9]+)?',row['text'])
    gold=row['answer'].split('####')[-1].strip().replace(',','')
    return bool(nums and Decimal(nums[-1].replace(',',''))==Decimal(gold))

summary={}
for tag in TAGS:
    p=Path('/tmp/bolt-'+tag+'.json')
    if not p.exists():continue
    d=json.loads(p.read_text())
    with gzip.open(OUT/(tag+'.json.gz'),'wt') as f:json.dump(d,f)
    rows=d['eval']
    summary[tag]={
        'startup_seconds':d['startup_seconds'],
        'median_tok_s':{k:statistics.median(x['tok_s'] for x in v) for k,v in d['decode'].items()},
        'median_diverse_tok_s':{k:statistics.median(x['tok_s'] for x in v) for k,v in d.get('decode_diverse',{}).items()},
        'gsm_correct':sum(grade(r) for r in rows),'gsm_n':len(rows),
        'gsm_correct_strict_text':sum(bool((z:=re.findall(r'[-+]?[0-9][0-9,]*(?:[.][0-9]+)?',r['text'])) and z[-1].replace(',','')==r['answer'].split('####')[-1].strip().replace(',','')) for r in rows),
        'truncated':sum(r['finish_reason']=='length' for r in rows),
        'eval_seconds':d['eval_seconds'],
        'linear_backend':d.get('linear_backend'),
        'dense_cache':d.get('dense_cache'),
    }
    log=Path('/tmp/bolt-'+tag+'.log')
    if log.exists():shutil.copyfile(log,OUT/(tag+'.log'))
if 'baseline250' in summary and 'final250' in summary:
    a=json.load(open('/tmp/bolt-baseline250.json'))['eval']
    best=Path('/tmp/bolt-optimized250.json')
    b=json.load(open(best if best.exists() else '/tmp/bolt-final250.json'))['eval']
    changes=[{'question':x['question'],'baseline_correct':grade(x),'optimized_correct':grade(y)} for x,y in zip(a,b)]
    improves=sum(not x['baseline_correct'] and x['optimized_correct'] for x in changes)
    regressions=sum(x['baseline_correct'] and not x['optimized_correct'] for x in changes)
    n=improves+regressions
    p=min(1,2*sum(math.comb(n,k) for k in range(min(improves,regressions)+1))/2**n) if n else 1
    summary['paired_gsm']={'improved':improves,'regressed':regressions,'mcnemar_exact_p':p}
    (OUT/'paired-grades.json').write_text(json.dumps(changes,indent=2))
for name in ['kernel-tests','direct-tests','fused-tests','cache-kernel-tests','singleton-tests','public-cache-parity','e2e-final','e2e-original-check','e2e-exact-batch','real-accuracy-final','dispatch-validated','precommit-singleton','singleton-moe','singleton-sweep','singleton-sweep-small','prefill-warps','prefill-sweep','prefill-splitn','tests-final','public-final','precommit-final','dispatch-final-current','singleton-sweep-final','verification-final','packed-register','packed-register-sweep','packed-moe','moe-threshold','expanded-sweep','expanded-shift','plane-isolate','plane-corrected']:
    p=Path('/tmp/bolt-'+name+'.log')
    if p.exists():shutil.copyfile(p,OUT/(name+'.log'))
summary['hardware']=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,uuid,memory.total,driver_version','--format=csv'],text=True)
summary['baseline_commit']=subprocess.check_output(['git','-C','/home/seigyo/vllmServer/vllm','rev-parse','HEAD'],text=True).strip()
summary['grade_method']='Greedy raw Q/A, first 250 GSM8K test questions, max_tokens=768, final-number exact numeric match. Not an official few-shot lm-eval run.'
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
(OUT/'production.patch').write_bytes(subprocess.check_output(['git','-C','/home/seigyo/vllmServer/vllm','diff']))
print(json.dumps(summary,indent=2))
