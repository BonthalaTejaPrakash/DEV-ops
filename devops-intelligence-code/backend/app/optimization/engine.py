import statistics
def analyze_bottlenecks(org, pipeline_id, runs=None):
    if not runs: return {'pipeline_id':pipeline_id,'bottlenecks':[],'recommendations':[],'note':'No runs yet'}
    stages=['checkout_duration','build_duration','test_duration','deployment_duration']; recs=[]
    for s in stages:
        vals=[r[s] for r in runs if s in r]; avg=statistics.mean(vals); p95=sorted(vals)[int(len(vals)*0.95)-1]; total_avg=sum(statistics.mean([r[x] for r in runs]) for x in stages)
        if avg>total_avg*0.4:
            recs.append({'stage':s,'problem':s+' is slow','evidence':f'avg {avg:.0f}s, p95 {p95:.0f}s, n={len(vals)}','solution':'Enable caching' if 'build' in s else 'Run independent tests in parallel' if 'test' in s else 'Optimize deployment steps','expected_benefit':'ESTIMATE - measure after implementation','priority':'P0','status':'Not started'})
    return {'pipeline_id':pipeline_id,'bottlenecks':recs,'recommendations':recs}
