"""Check both dtypes end to end and compare features with the retained collectors."""
import argparse
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import numpy as np
from data import read_bin, write_bin

ROOT=Path(__file__).resolve().parents[1]


def check(build, directory):
    from run import exact_gt
    # Exhaustive reference with a known tie verifies row-ID tie breaking.
    b=np.array([[0.,0.],[1.,0.],[-1.,0.],[4.,0.]],dtype='float32')
    np.testing.assert_array_equal(exact_gt(b,np.array([[0.,0.]],dtype='float32'),3,2),[[0,1,2]])
    for binary in ('index_float','index_uint8'):
        result=subprocess.run([str(build/binary)],capture_output=True,text=True)
        assert result.returncode!=0 and 'required' in (result.stdout+result.stderr).lower()
    for dtype,dim,steps in [('float32',96,100),('uint8',128,100),('float32',960,50),('float32',512,50)]:
        template='toy' if dtype=='float32' else 'toy_uint8'
        name=template+f'_{dim}';ext='fbin' if dtype=='float32' else 'u8bin'
        data=directory/name;work=directory/(name+'_run')
        subprocess.run([sys.executable,str(ROOT/'reproduction/data.py'),'toy','--dtype',dtype,'--output',str(data)],check=True)
        if dim not in (96,128):
            rows=np.random.default_rng(2025).normal(size=(2400,dim)).astype('float32')
            write_bin(data/'base.fbin',rows[:2000]);write_bin(data/'query.fbin',rows[2000:])
        config=json.loads((ROOT/f'configs/{template}.json').read_text())
        config['probe_steps']=steps
        if dim==960:config['pca_prefix']=40
        if dim==512:config['pca_prefix']=256
        config.update(base=str(data/f'base.{ext}'),query=str(data/f'query.{ext}'),work_dir=str(work))
        path=directory/f'{name}.json';path.write_text(json.dumps(config))
        with (directory/f'{name}.log').open('w') as log:
            subprocess.run([sys.executable,str(ROOT/'reproduction/run.py'),'--config',str(path),'--build-dir',str(build)],check=True,stdout=log,stderr=subprocess.STDOUT)
        summary=json.loads((work/'summary.json').read_text())
        assert summary['test_recall_target_met']
        split=json.loads((work/'splits.json').read_text())
        assert not set(split['train'])&set(split['validation']) and not set(split['test'])&(set(split['train'])|set(split['validation']))
        for method in ('plain','pilot','exact_control'):
            assert summary[method]['queries']==80 and summary[method]['repeats']==2
            assert np.isfinite(summary[method]['mean_ms']) and summary[method]['mean_ms']>0
        # Original collector consumes the historical SPCA100 model layout.
        mean=read_bin(work/'pca_mean.fbin');components=read_bin(work/'pca_components.fbin')
        model=work/'legacy_pca.bin'
        with model.open('wb') as f:
            f.write(b'SPCA100\0');f.write(struct.pack('<QII',2000,mean.shape[1],components.shape[0]))
            np.asarray(mean,dtype='<f4').tofile(f);np.asarray(components.T,dtype='<f4').tofile(f)
        reference=work/'legacy_features.json';collector='probe_deep' if dtype=='float32' else 'probe_sift'
        subprocess.run([str(build/collector),str(work/'index.hnsw'),str(work/f'queries.{ext}'),str(model),str(reference),'20'],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        old=json.loads(reference.read_text())['queries']
        if steps==100:
            new=json.loads((work/'features.json').read_text())['rows'][:20]
        else:
            parity=json.loads((work/'features.config.json').read_text())
            parity.update(probe_steps=100,query_ids=list(range(20)),output=str(work/'parity.json'),log=str(work/'parity.log'))
            parity_path=work/'parity.config.json';parity_path.write_text(json.dumps(parity))
            subprocess.run([str(build/'pilot_search'),str(parity_path)],check=True)
            new=json.loads((work/'parity.json').read_text())['rows']
        keep=[i for i in range(59) if i not in range(16,24)]
        np.testing.assert_allclose(np.array([r['features'] for r in old])[:,keep],np.array([r['features'] for r in new]),rtol=1e-5,atol=1e-5)
        # Full-budget search must agree with an exhaustive nearest-distance reference.
        cfg=json.loads((work/'test_pilot.config.json').read_text());cfg.update(caps=[2000]*3,executor='exact',query_ids=list(range(5)),repeats=1)
        cfg.update(output=str(work/'full_exact.json'),log=str(work/'full_exact.log'))
        native=work/'check.config.json';native.write_text(json.dumps(cfg));subprocess.run([str(build/'pilot_search'),str(native)],check=True)
        exact=json.loads((work/'full_exact.json').read_text())['rows']
        base=read_bin(data/f'base.{ext}',dtype);queries=read_bin(work/f'queries.{ext}',dtype);truth=read_bin(work/'gt.ibin','uint32')
        for row in exact:
            qi=row['query_id'];q=np.asarray(queries[qi],dtype=np.float64)
            a=np.sort(np.sum((np.asarray(base[row['neighbors']],dtype=np.float64)-q)**2,axis=1))
            b=np.sort(np.sum((np.asarray(base[truth[qi]],dtype=np.float64)-q)**2,axis=1))
            np.testing.assert_allclose(a,b,rtol=1e-6,atol=1e-6)
            assert row['probe_cached_distances']>0
        # Fused partial sums should preserve the exact result on this fixture.
        cfg.update(executor='partial',block=64,output=str(work/'full_partial.json'),log=str(work/'full_partial.log'))
        native.write_text(json.dumps(cfg));subprocess.run([str(build/'pilot_search'),str(native)],check=True)
        partial=json.loads((work/'full_partial.json').read_text())['rows']
        for a,b in zip(exact,partial):assert set(a['neighbors'])==set(b['neighbors'])
        cfg['probe_steps']=50 if steps==100 else 100;native.write_text(json.dumps(cfg))
        result=subprocess.run([str(build/'pilot_search'),str(native)],capture_output=True,text=True)
        assert result.returncode!=0 and 'configuration mismatch' in result.stderr
        print(dtype,dim,steps, 'PASS: pipeline, clean51 parity, exhaustive distances, partial pruning, metadata guard',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--build-dir',type=Path,default=ROOT/'build');p.add_argument('--output',type=Path)
    args=p.parse_args()
    if args.output:
        args.output.mkdir(parents=True,exist_ok=True);check(args.build_dir.resolve(),args.output.resolve())
    else:
        with tempfile.TemporaryDirectory(prefix='pilot-smoke-') as d:check(args.build_dir.resolve(),Path(d))

if __name__=='__main__':main()
