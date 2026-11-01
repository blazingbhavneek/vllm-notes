"""Nsight SQLite critical-path accounting; overlaps are unioned, never summed."""
import argparse,collections,json,sqlite3,statistics
from pathlib import Path

def union(intervals):
    out=[]
    for a,b in sorted(intervals):
        if out and a<=out[-1][1]:out[-1]=(out[-1][0],max(out[-1][1],b))
        else:out.append((a,b))
    return out

def analyze(path):
    db=sqlite3.connect(path);db.row_factory=sqlite3.Row
    names=dict(db.execute('select id,value from StringIds'))
    kernels=[dict(r) for r in db.execute('select * from CUPTI_ACTIVITY_KIND_KERNEL order by start')]
    for k in kernels:k['name']=names[k['shortName']]
    runtime={r['correlationId']:dict(r) for r in db.execute('select * from CUPTI_ACTIVITY_KIND_RUNTIME')}
    groups=collections.defaultdict(list)
    for k in kernels:groups[k['correlationId']].append(k)
    graph=[]
    for corr,ks in groups.items():
        api=runtime.get(corr)
        if api and 'GraphLaunch' in names[api['nameId']]:
            start=min(k['start'] for k in ks);end=max(k['end'] for k in ks);merged=union((k['start'],k['end']) for k in ks)
            graph.append({'start':start,'end':end,'duration_ms':(end-start)/1e6,'busy_ms':sum(b-a for a,b in merged)/1e6,'kernel_count':len(ks),'api_ms':(api['end']-api['start'])/1e6,'enqueue_to_start_ms':(start-api['end'])/1e6})
    graph.sort(key=lambda r:r['start'])
    steady=graph[5:-3] if len(graph)>10 else graph
    totals=collections.defaultdict(lambda:{'count':0,'sum_ms':0,'registers':set(),'threads':set(),'shared_bytes':set(),'grids':collections.Counter()})
    for g in steady:
        for k in kernels:
            if g['start']<=k['start'] and k['end']<=g['end']:
                t=totals[k['name']];t['count']+=1;t['sum_ms']+=(k['end']-k['start'])/1e6;t['registers'].add(k['registersPerThread']);t['threads'].add(k['blockX']*k['blockY']*k['blockZ']);t['shared_bytes'].add(k['staticSharedMemory']+k['dynamicSharedMemory']);t['grids'][k['gridX']*k['gridY']*k['gridZ']]+=1
    table=[]
    for name,t in totals.items():
        table.append({'name':name,'calls_per_graph':t['count']/max(1,len(steady)),'summed_ms_per_graph':t['sum_ms']/max(1,len(steady)),'registers':sorted(t['registers']),'threads':sorted(t['threads']),'shared_bytes':sorted(t['shared_bytes']),'common_grid_ctas':t['grids'].most_common(6)})
    table.sort(key=lambda x:-x['summed_ms_per_graph'])
    first,last=kernels[0]['start'],max(k['end'] for k in kernels);merged=union((k['start'],k['end']) for k in kernels)
    between=[]
    for a,b in zip(steady,steady[1:]):
        if b['start']-a['start']<50e6:
            busy=sum(max(0,min(e,b['start'])-max(s,a['end'])) for s,e in merged if e>a['end'] and s<b['start'])
            between.append({'period_ms':(b['start']-a['start'])/1e6,'outside_graph_ms':(b['start']-a['end'])/1e6,'model_kernel_idle_ms':max(0,b['start']-a['end']-busy)/1e6})
    result={'trace':str(path),'kernel_count':len(kernels),'capture_span_ms':(last-first)/1e6,'capture_union_busy_ms':sum(b-a for a,b in merged)/1e6,'graph_count':len(graph),'steady_graph_count':len(steady),'median_graph_duration_ms':statistics.median(g['duration_ms'] for g in steady) if steady else None,'median_graph_busy_ms':statistics.median(g['busy_ms'] for g in steady) if steady else None,'median_graph_idle_ms':statistics.median(g['duration_ms']-g['busy_ms'] for g in steady) if steady else None,'median_graph_api_ms':statistics.median(g['api_ms'] for g in steady) if steady else None,'median_inter_graph':{k:statistics.median(r[k] for r in between) for k in between[0]} if between else None,'kernels':table,'graphs':graph,'scope':'Model-process kernels only. Kernel duration sums include overlap; union busy excludes overlap. Not hardware warp-stall measurements.'}
    Path(str(path)+'.analysis.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ('kernels','graphs')},indent=2))
    for r in table[:12]:print(r)
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('traces',nargs='+');args=ap.parse_args()
    for name in args.traces:analyze(Path(name))
