"""End-to-end PILOT workflow with disjoint train/validation/test query splits.

Stages: prepare, index, collect, train, calibrate, evaluate (or all).
Produces new measurements; it does not overwrite the archived plotting data.
"""
import argparse
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time

# Set before NumPy/LightGBM imports; command-line environment can override these.
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(key,'1')
import numpy as np
from data import read_bin, read_vectors, write_bin

ROOT = Path(__file__).resolve().parents[1]


def save(path, data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def load(path):return json.loads(path.read_text())


def exact_gt(base, queries, k, block=16384):
    """Blocked exhaustive squared L2, float64 accumulation, row-ID tie breaking."""
    result = np.empty((len(queries),k),dtype='<u4')
    for qs in range(0,len(queries),64):
        q = np.asarray(queries[qs:qs+64],dtype=np.float64)
        best = np.full((len(q),k),np.inf);ids = np.full((len(q),k),-1,dtype=np.int64)
        for start in range(0,len(base),block):
            b = np.asarray(base[start:start+block],dtype=np.float64)
            dist = np.maximum(0,(q*q).sum(1)[:,None]+(b*b).sum(1)[None,:]-2*q@b.T)
            candidates = np.concatenate([best,dist],axis=1)
            candidate_ids = np.concatenate([ids,np.broadcast_to(np.arange(start,start+len(b)),dist.shape)],axis=1)
            # Lexicographic sorting makes integer-distance ties reproducible.
            take = np.lexsort((candidate_ids,candidates),axis=1)[:,:k]
            best = np.take_along_axis(candidates,take,axis=1)
            ids = np.take_along_axis(candidate_ids,take,axis=1)
        result[qs:qs+len(q)] = ids
        print(f'ground truth {qs+len(q)}/{len(queries)}',flush=True)
    return result


class Workflow:
    def __init__(self, config, build_dir):
        self.cfg = load(config);self.base_dir=config.resolve().parent
        self.build = build_dir.resolve();self.work = self.resolve(self.cfg['work_dir']);self.work.mkdir(parents=True,exist_ok=True)
        self.dtype=self.cfg.get('dtype','float32');self.ext='u8bin' if self.dtype=='uint8' else 'fbin'
        self.k=int(self.cfg.get('k',100));self.steps=int(self.cfg.get('probe_steps',100));self.seed=int(self.cfg.get('seed',2025))
        self.counts=[int(self.cfg.get(key,default)) for key,default in [('train_queries',5000),('validation_queries',500),('test_queries',1000)]]
        if min(self.counts)<3 or self.k<1 or self.steps<1:raise ValueError('Invalid split/k/probe settings')
        self.total=sum(self.counts)
        # A changed configuration must not silently reuse another run's outputs.
        record=self.work/'experiment.json'
        frozen=dict(self.cfg,source_config=str(config.resolve()))
        if record.exists() and load(record)!=frozen:raise ValueError('Configuration changed: choose a fresh work_dir')
        save(record,frozen)

    def resolve(self,path):
        p=Path(path);return (p if p.is_absolute() else self.base_dir/p).resolve()

    def native(self,name,values):
        cfg=dict(index=str(self.work/'index.hnsw'),query=str(self.work/f'queries.{self.ext}'),gt=str(self.work/'gt.ibin'),
                 dtype=self.dtype,k=self.k,probe_steps=self.steps,pca_mean=str(self.work/'pca_mean.fbin'),
                 pca_components=str(self.work/'pca_components.fbin'),model=str(self.work/'classifier.txt'),model_metadata=str(self.work/'model_metadata.json'),
                 pca_scores=str(self.work/'pca_scores.fbin'),pca_norms=str(self.work/'pca_norms.fbin'),
                 output=str(self.work/f'{name}.json'),log=str(self.work/f'{name}.log'))
        cfg.update(values);path=self.work/f'{name}.config.json';save(path,cfg)
        env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
        subprocess.run([str(self.build/'pilot_search'),str(path)],check=True,env=env)
        return load(self.work/f'{name}.json')['rows']

    def prepare(self):
        from sklearn.decomposition import PCA
        base=read_bin(self.resolve(self.cfg['base']),self.dtype)
        query=read_bin(self.resolve(self.cfg['query']),self.dtype)
        if len(base)<max(200,self.k) or base.shape[1]<32 or query.shape[1]!=base.shape[1] or len(query)<self.total:
            raise ValueError('Need N>=max(200,k), D>=32 and enough queries for all splits')
        rng=np.random.default_rng(self.seed);ids=rng.permutation(len(query))[:self.total]
        q=np.asarray(query[ids]);write_bin(self.work/f'queries.{self.ext}',q,self.dtype)
        split={'train':list(range(self.counts[0])), 'validation':list(range(self.counts[0],sum(self.counts[:2]))),
               'test':list(range(sum(self.counts[:2]),self.total)), 'source_query_ids':ids.tolist()}
        save(self.work/'splits.json',split)
        if self.cfg.get('ground_truth'):
            source=read_vectors(self.resolve(self.cfg['ground_truth']),'uint32')
            if source.shape[0]!=len(query) or source.shape[1]<self.k or np.max(source)>=len(base):raise ValueError('Ground truth shape/base mismatch')
            gt=np.asarray(source[ids,:self.k],dtype='<u4')
        else:gt=exact_gt(base,q,self.k)
        write_bin(self.work/'gt.ibin',gt,'uint32')
        size=min(int(self.cfg.get('pca_sample',50000)),len(base));prefix=int(self.cfg.get('pca_prefix',min(64,base.shape[1])))
        if size<max(32,prefix) or not 32<=prefix<=base.shape[1]:raise ValueError('PCA prefix/sample out of range')
        sample_ids=rng.choice(len(base),size,replace=False)
        pca=PCA(n_components=prefix,svd_solver='full').fit(np.asarray(base[sample_ids],dtype=np.float32))
        write_bin(self.work/'pca_mean.fbin',pca.mean_[None,:],'<f4');write_bin(self.work/'pca_components.fbin',pca.components_,'<f4')
        save(self.work/'preparation.json',{'base_shape':list(base.shape),'dtype':self.dtype,'pca_sample_ids':sample_ids.tolist(),
             'probe_steps':self.steps,'feature_schema':'clean51','metric':'squared_l2','ground_truth':self.cfg.get('ground_truth','blocked exhaustive float64')})
        # Projection is optional for exact/partial execution, avoiding huge unused assets.
        if 'pca' in self.cfg.get('executors',['exact','partial']):
            import struct
            with (self.work/'pca_scores.fbin').open('wb') as scores,(self.work/'pca_norms.fbin').open('wb') as norms:
                scores.write(struct.pack('<II',len(base),prefix));norms.write(struct.pack('<II',len(base),1))
                for start in range(0,len(base),8192):
                    x=np.asarray(base[start:start+8192],dtype=np.float32)-pca.mean_
                    (x@pca.components_.T).astype('<f4').tofile(scores)
                    np.sum(x*x,axis=1,dtype=np.float32).astype('<f4').tofile(norms)

    def index(self):
        base=str(self.resolve(self.cfg['base']));index=str(self.work/'index.hnsw');log=str(self.work/'index.log')
        if Path(index).exists():raise FileExistsError('Index already exists: reuse later stages or choose a fresh work_dir')
        if self.dtype=='uint8':
            cmd=[str(self.build/'index_uint8'),'--data',base,'--index',index,'--log-file',log,'--seed',str(self.seed),'--threads','1','--checkpoint-every','0',
                 '--M',str(self.cfg.get('M',16)),'--ef-construction',str(self.cfg.get('ef_construction',100))]
        else:
            cmd=[str(self.build/'index_float'),'--path_data',base,'--path_index',index,'--log-file',log,'--random_seed',str(self.seed),
                 '--build_mode','hierarchy','--M',str(self.cfg.get('M',16)),'--ef_construction',str(self.cfg.get('ef_construction',100))]
        subprocess.run(cmd,check=True)

    def collect(self):
        split=load(self.work/'splits.json');ids=split['train']+split['validation']
        self.native('features',{'mode':'features','query_ids':ids})
        grid=[int(e) for e in self.cfg.get('ef_grid',[100,200,400,800,1600,3200])]
        self.native('grid',{'mode':'grid','query_ids':ids,'ef_grid':grid})

    def train(self):
        from lightgbm import LGBMClassifier
        split=load(self.work/'splits.json');features={r['query_id']:r['features'] for r in load(self.work/'features.json')['rows']}
        rows=load(self.work/'grid.json')['rows'];target=float(self.cfg.get('target_recall',.9));grid=sorted({r['ef'] for r in rows})
        min_ef={i:grid[-1] for i in split['train']};reachable=set()
        for r in rows:
            if r['query_id'] in min_ef and r['recall']>=target-1e-9:
                min_ef[r['query_id']]=min(min_ef[r['query_id']],r['ef']);reachable.add(r['query_id'])
        x=np.asarray([features[i] for i in split['train']],dtype=np.float32)
        labels=np.zeros(len(x),dtype=int);ef=np.asarray([min_ef[i] for i in split['train']]);groups=np.array_split(np.argsort(ef,kind='stable'),3)
        for tier,group in enumerate(groups):labels[group]=tier
        model=LGBMClassifier(objective='multiclass',num_class=3,n_estimators=int(self.cfg.get('trees',100)),num_leaves=31,
                             min_child_samples=max(2,min(20,len(x)//20)),random_state=self.seed,n_jobs=1,verbosity=-1)
        model.fit(x,labels);model.booster_.save_model(str(self.work/'classifier.txt'))
        save(self.work/'model_metadata.json',{'probe_steps':self.steps,'feature_schema':'clean51',
             'dimensions':read_bin(self.work/f'queries.{self.ext}',self.dtype).shape[1],'k':self.k})
        pred=model.predict(np.asarray([features[i] for i in split['validation']],dtype=np.float32)).astype(int)
        # Validation selects budgets; no test labels are accessed.
        val={ (r['query_id'],r['ef']):r for r in rows if r['query_id'] in set(split['validation']) }
        best=None
        for caps in itertools.combinations_with_replacement(grid,3):
            selected=[val[(i,caps[t])] for i,t in zip(split['validation'],pred)]
            recall=float(np.mean([r['recall'] for r in selected]));cost=float(np.mean([r['ndc'] for r in selected]))
            if recall>=target-1e-9 and (best is None or cost<best['ndc']):best={'caps':list(caps),'recall':recall,'ndc':cost}
        if best is None:raise RuntimeError('No budget triple reaches validation recall; extend ef_grid in a NEW experiment')
        best.update(feature_schema='clean51',probe_steps=self.steps,train_censored=len(x)-len(reachable),
                    train_tier_boundaries=[int(ef[g].max()) for g in groups],selection='validation full-distance grid; online path verified during calibration')
        save(self.work/'planner.json',best)

    @staticmethod
    def summarize(rows):
        # Median of repeat means; every record includes projection/probe/model/search time.
        runs={}
        for r in rows:runs.setdefault(r['repeat'],[]).append(r)
        return {'recall':float(np.mean([r['recall'] for r in rows])),
                'attainment_090':float(np.mean([r['recall']>=.9-1e-9 for r in rows])),
                'ndc':float(np.mean([r['ndc'] for r in rows])),
                'mean_ms':float(np.median([np.mean([r['ms'] for r in v]) for v in runs.values()])),
                'queries':len(next(iter(runs.values()))),'repeats':len(runs)}

    def calibrate(self):
        split=load(self.work/'splits.json');plan=load(self.work/'planner.json');target=float(self.cfg.get('target_recall',.9))
        common={'mode':'pilot','query_ids':split['validation'],'caps':plan['caps'],'repeats':int(self.cfg.get('repeats',5))}
        rows=self.native('validation_exact',dict(common,executor='exact'));control=self.summarize(rows)
        if control['recall']<target-1e-9:
            # Shared-state search can differ from grid search; validate a conservative cap.
            common['caps']=[max(self.cfg['ef_grid'])]*3
            rows=self.native('validation_exact',dict(common,executor='exact'));control=self.summarize(rows)
            if control['recall']<target-1e-9:raise RuntimeError('Online exact validation misses recall; enlarge ef_grid in a new run')
        candidates=[{'executor':'exact','metrics':control}]
        if 'partial' in self.cfg.get('executors',['exact','partial']):
            opts=dict(executor='partial',block=int(self.cfg.get('partial_block',64)),activation='heap_full')
            rows=self.native('validation_partial',dict(common,**opts));candidates.append(dict(opts,metrics=self.summarize(rows)))
        if 'pca' in self.cfg.get('executors',[]):
            for hard,activation in itertools.product([False,True],['heap_full','combined128']):
                opts=dict(executor='pca',prefix=int(self.cfg['pca_prefix']),margin=1.,hard_only=hard,activation=activation)
                rows=self.native(f'validation_pca_{int(hard)}_{activation}',dict(common,**opts));candidates.append(dict(opts,metrics=self.summarize(rows)))
        feasible=[c for c in candidates if c['metrics']['recall']>=target-1e-9 and control['recall']-c['metrics']['recall']<=.01+1e-9]
        winner=min(feasible,key=lambda c:c['metrics']['mean_ms'])
        selection={'caps':common['caps'],'winner':winner,'candidates':candidates,'selected_on':'validation','probe_steps':self.steps}
        save(self.work/'selection.json',selection)

    def evaluate(self):
        split=load(self.work/'splits.json');selection=load(self.work/'selection.json');winner=dict(selection['winner']);winner.pop('metrics')
        common={'query_ids':split['test'],'repeats':int(self.cfg.get('repeats',5))}
        plain=self.native('test_plain',dict(common,mode='plain',ef=int(self.cfg.get('plain_ef',max(self.cfg['ef_grid'])))))
        pilot=self.native('test_pilot',dict(common,mode='pilot',caps=selection['caps'],**winner))
        control=self.native('test_exact_control',dict(common,mode='pilot',caps=selection['caps'],executor='exact'))
        report={'plain':self.summarize(plain),'pilot':self.summarize(pilot),'exact_control':self.summarize(control),
                'selected_executor':winner,'caps':selection['caps'],'probe_steps':self.steps,
                'timing_scope':'one untimed warmup pass, then single-thread query PCA projection, probe, classification and resumed search; excludes loading and GT scoring',
                'ndc_scope':'full-distance equivalents; excludes query projection and feature arithmetic',
                'protocol':'new disjoint query split; not a replacement for archived paper measurements'}
        report['test_recall_target_met']=report['pilot']['recall']>=float(self.cfg.get('target_recall',.9))-1e-9
        report['test_executor_recall_loss']=report['exact_control']['recall']-report['pilot']['recall']
        save(self.work/'summary.json',report);print(json.dumps(report,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--build-dir',type=Path,default=ROOT/'build')
    parser.add_argument('--stage',choices=['all','prepare','index','collect','train','calibrate','evaluate'],default='all')
    args=parser.parse_args();flow=Workflow(args.config,args.build_dir)
    for stage in ['prepare','index','collect','train','calibrate','evaluate']:
        if args.stage not in ('all',stage):continue
        start=time.time();print('START',stage,flush=True);getattr(flow,stage)();print('DONE',stage,round(time.time()-start,3),'seconds',flush=True)

if __name__=='__main__':main()
