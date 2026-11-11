// Real in-browser ML - mirrors backend/app/ml/training.py
export type Row = Record<string, number|string>;
export function mean(a:number[]){return a.reduce((s,x)=>s+x,0)/(a.length||1)}
export function evaluate(yTrue:number[], yProb:number[]){
  let tp=0,fp=0,tn=0,fn=0;
  yTrue.forEach((y,i)=>{ const p=yProb[i]>=0.5?1:0; if(y===1&&p===1)tp++; else if(y===0&&p===1)fp++; else if(y===0&&p===0)tn++; else fn++; });
  const acc=(tp+tn)/(yTrue.length||1); const prec=tp/(tp+fp||1); const rec=tp/(tp+fn||1); const f1=2*prec*rec/(prec+rec||1);
  const pairs=yTrue.map((y,i)=>({y,p:yProb[i]})).sort((a,b)=>a.p-b.p); let rankSum=0; const nPos=yTrue.filter(Boolean).length, nNeg=yTrue.length-nPos;
  pairs.forEach((d,i)=>{ if(d.y===1) rankSum+=i+1 }); const auc=(rankSum-nPos*(nPos+1)/2)/(nPos*nNeg||1);
  return { accuracy:acc, precision:prec, recall:rec, f1, confusion:{tp,fp,tn,fn}, rocAuc:auc };
}
export function riskFromProb(p:number){ return p<0.35?'LOW':p<=0.65?'MEDIUM':'HIGH'; }
export function trainLogistic(X:number[][], y:number[], epochs=800, lr=0.05){
  const d=X[0].length; let w=new Array(d).fill(0), b=0;
  for(let e=0;e<epochs;e++){ let gw=new Array(d).fill(0), gb=0; X.forEach((row,i)=>{ const z=row.reduce((s,v,j)=>s+v*w[j],b); const p=1/(1+Math.exp(-z)); const err=p-y[i]; row.forEach((v,j)=>gw[j]+=err*v); gb+=err; }); w=w.map((v,j)=>v-lr*gw[j]/X.length-0.001*v); b-=lr*gb/X.length; }
  return { w,b, predict:(row:number[])=>1/(1+Math.exp(-row.reduce((s,v,j)=>s+v*w[j],b))) };
}
export function generateSampleRuns(n=320): Row[] {
  let seed=42; const rnd=()=>{ seed=(seed*1103515245+12345)%2147483648; return seed/2147483648; };
  const pipelines=['web-api','worker','frontend','billing']; const triggers=['push','pull_request','schedule','manual']; const rows:Row[]=[];
  for(let i=0;i<n;i++){ const build=60+rnd()*220, checkout=8+rnd()*22, test=80+rnd()*280, deploy=15+rnd()*55; const prevFail=Math.floor(rnd()*4), retry=rnd()>0.7?Math.floor(rnd()*3):0, changes=Math.floor(rnd()*40); const logit=-2.1+0.004*build+0.003*test+0.62*prevFail+0.55*retry+0.02*changes+(rnd()-0.5)*1.2; const p=1/(1+Math.exp(-logit)); const failed=rnd()<p; const total=80+Math.floor(rnd()*120); const ft=failed?Math.floor(rnd()*8)+1:Math.floor(rnd()*2); rows.push({ run_id:i+2841, pipeline_id:pipelines[i%4], build_duration:Math.round(build), checkout_duration:Math.round(checkout), test_duration:Math.round(test), deployment_duration:Math.round(deploy), total_tests:total, passed_tests:total-ft, failed_tests:ft, previous_failure_count:prevFail, retry_count:retry, deployment_error_count:failed?Math.floor(rnd()*2):0, code_changes_count:changes, trigger_type:triggers[Math.floor(rnd()*4)], execution_status:failed?'Failure':'Success' }); }
  return rows;
}
