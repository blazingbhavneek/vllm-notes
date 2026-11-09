"""GSM8K parity through the live OpenAI-compatible HTTP server."""
import argparse, asyncio, hashlib, json, re, time
from decimal import Decimal
from pathlib import Path
import aiohttp

async def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base-url',default='http://127.0.0.1:8001');ap.add_argument('--count',type=int,default=250);ap.add_argument('--concurrency',type=int,default=32);ap.add_argument('--out',required=True);args=ap.parse_args()
    dataset=Path('/tmp/bolt-gsm8k-test.jsonl');rows=[json.loads(x) for x in dataset.read_text().splitlines()][:args.count]
    semaphore=asyncio.Semaphore(args.concurrency)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=900)) as session:
        async def one(i,row):
            payload={'model':'nemotron-bolt-h200','prompt':f"Q: {row['question']}\nA: Let's think step by step.\n",'temperature':0,'seed':0,'max_tokens':768}
            async with semaphore:
                start=time.perf_counter()
                async with session.post(args.base_url+'/v1/completions',json=payload) as response:
                    body=await response.json()
                    if response.status!=200:raise RuntimeError((response.status,body))
                elapsed=time.perf_counter()-start
            choice=body['choices'][0];text=choice['text'];nums=re.findall(r'[-+]?[0-9][0-9,]*(?:[.][0-9]+)?',text);gold=row['answer'].split('####')[-1].strip().replace(',','');pred=nums[-1].replace(',','') if nums else None
            result={'index':i,'question':row['question'],'gold':gold,'prediction':pred,'correct':bool(pred is not None and Decimal(pred)==Decimal(gold)),'text':text,'finish_reason':choice['finish_reason'],'usage':body['usage'],'latency_s':elapsed,'model':body['model'],'id':body['id']}
            print('ANSWER',i,result['correct'],result['finish_reason'],round(elapsed,2),flush=True)
            return result
        start=time.perf_counter();results=await asyncio.gather(*(one(i,r) for i,r in enumerate(rows)))
    out={'base_url':args.base_url,'dataset_sha256':hashlib.sha256(dataset.read_bytes()).hexdigest(),'concurrency':args.concurrency,'elapsed_s':time.perf_counter()-start,'correct':sum(r['correct'] for r in results),'n':len(results),'truncated':sum(r['finish_reason']=='length' for r in results),'results':results}
    Path(args.out).write_text(json.dumps(out,indent=2));print('SUMMARY',out['correct'],out['n'],'truncated',out['truncated'],flush=True)

if __name__=='__main__':asyncio.run(main())
