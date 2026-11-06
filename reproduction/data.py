"""Download official vectors or convert local vectors to the portable bin format."""
import argparse
import ftplib
import json
import os
from pathlib import Path
import shutil
import struct
import tarfile
import urllib.request
import urllib.parse

import numpy as np

SOURCES = {
    'sift': ('https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/', 'u8bin', 128, 'uint8'),
    'deep': ('https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/', 'fbin', 96, 'float32'),
}


def read_bin(path, dtype='float32'):
    path = Path(path)
    with path.open('rb') as f:
        n, d = struct.unpack('<II', f.read(8))
    dt = np.dtype(dtype).newbyteorder('<')
    if not n or not d or path.stat().st_size != 8 + n*d*dt.itemsize:
        raise ValueError(f'Invalid bin header/length: {path}')
    return np.memmap(path, mode='r', offset=8, dtype=dt, shape=(n, d))


def read_vectors(path, dtype='float32'):
    path = Path(path)
    if path.suffix == '.npy':
        a = np.load(path, mmap_mode='r', allow_pickle=False)
        if a.ndim != 2:
            raise ValueError('Expected a matrix')
        return a
    if path.suffix in ('.fvecs', '.ivecs', '.bvecs'):
        with path.open('rb') as f:
            d, = struct.unpack('<I', f.read(4))
        dt = np.dtype({'fvecs':'<f4', 'ivecs':'<u4', 'bvecs':'u1'}[path.suffix[1:]])
        row_size = 4+d*dt.itemsize
        if not d or path.stat().st_size % row_size:
            raise ValueError('Invalid vecs file size')
        rows = np.memmap(path, mode='r', dtype=np.dtype([('dim','<u4'),('values',dt,(d,))]))
        if not np.all(rows['dim'] == d):
            raise ValueError('Inconsistent vecs dimensions')
        return rows['values']
    return read_bin(path, dtype)


def write_bin(path, rows, dtype=None, normalize=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    dt = np.dtype(dtype or rows.dtype).newbyteorder('<')
    if normalize and dt != np.dtype('<f4'):
        raise ValueError('Normalization requires float32 output')
    tmp = path.with_name(path.name+'.tmp')
    with tmp.open('wb') as f:
        f.write(struct.pack('<II', *rows.shape))
        for start in range(0, len(rows), 8192):
            block = np.asarray(rows[start:start+8192], dtype=dt)
            if not np.isfinite(block).all():
                raise ValueError('Non-finite input vectors')
            if normalize:
                block = block / np.maximum(np.linalg.norm(block, axis=1, keepdims=True), 1e-30)
            block.tofile(f)
    tmp.replace(path)


def download(url, dst, size=None):
    """Resume a download, optionally stop after a known byte prefix."""
    dst = Path(dst); dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if size is not None and dst.stat().st_size != size:
            raise ValueError(f'Existing file has wrong size: {dst}')
        return
    part = dst.with_name(dst.name+'.part')
    offset = part.stat().st_size if part.exists() else 0
    if size is not None and offset > size:
        raise ValueError('Oversized partial download')
    if size is not None and offset == size:
        part.replace(dst); return
    if url.startswith('ftp://'):
        address = urllib.parse.urlparse(url)
        with ftplib.FTP(address.hostname, timeout=60) as ftp:
            ftp.login(); ftp.voidcmd('TYPE I')
            total = ftp.size(address.path)
            if total is None or offset > total:
                raise ValueError('Invalid FTP length/partial file')
            if offset < total:
                with part.open('ab') as out:
                    ftp.retrbinary('RETR '+address.path, out.write, blocksize=1 << 20, rest=offset or None)
            if part.stat().st_size != total:
                raise RuntimeError('Incomplete FTP download; rerun to resume')
        part.replace(dst); return
    req = urllib.request.Request(url, headers={'Range': f'bytes={offset}-'+(str(size-1) if size else '')})
    with urllib.request.urlopen(req, timeout=60) as response:
        status = response.status
        if offset and status != 206:
            raise RuntimeError('Server ignored resume range; partial file was not appended')
        if status == 206:
            start = int(response.headers['Content-Range'].split()[1].split('-')[0])
            if start != offset:
                raise RuntimeError('Wrong Content-Range')
        total = size
        if total is None and response.headers.get('Content-Length'):
            total = offset + int(response.headers['Content-Length'])
        with part.open('ab') as out:
            while total is None or offset < total:
                block = response.read(min(8 << 20, total-offset) if total else 8 << 20)
                if not block:
                    break
                out.write(block); offset += len(block)
            if total is not None and offset != total:
                raise RuntimeError('Incomplete download; rerun to resume')
    part.replace(dst)


def download_dataset(args):
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    if args.base_count is not None and args.base_count < 100:
        raise ValueError('base-count must be at least 100')
    if args.dataset in SOURCES:
        prefix, ext, d, dtype = SOURCES[args.dataset]
        n = args.base_count or 100_000_000
        if not 100 <= n <= 1_000_000_000:
            raise ValueError('base-count must be in [100, 1000000000]')
        base = out/f'base.{ext}'; size = 8+n*d*np.dtype(dtype).itemsize
        download(prefix+f'base.1B.{ext}', base, size)
        with base.open('r+b') as f:
            old_n, old_d = struct.unpack('<II', f.read(8))
            if old_d != d or old_n not in (1_000_000_000, n):
                raise ValueError('Unexpected upstream header')
            f.seek(0);f.write(struct.pack('<II', n,d))
        query = out/f'query.{ext}'
        download(prefix+f'query.public.10K.{ext}', query, 8+10000*d*np.dtype(dtype).itemsize)
        read_bin(base,dtype);read_bin(query,dtype)
    elif args.dataset == 'gist':
        archive = out/'gist.tar.gz'
        download('ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz', archive)
        # Copy only known files; never extract arbitrary archive paths.
        with tarfile.open(archive,'r:gz') as tf:
            for wanted in ('gist_base.fvecs','gist_query.fvecs','gist_learn.fvecs'):
                matches = [m for m in tf.getmembers() if m.isfile() and Path(m.name).name == wanted]
                if len(matches) != 1:
                    raise ValueError(f'Missing/ambiguous {wanted}')
                dest = out/wanted
                if not dest.exists():
                    with tf.extractfile(matches[0]) as src, dest.open('wb') as dst:
                        shutil.copyfileobj(src,dst)
        base = out/'base.fbin'; query = out/'query.fbin'
        rows = read_vectors(out/'gist_base.fvecs')
        if args.base_count and args.base_count > len(rows):
            raise ValueError('base-count exceeds GIST source size')
        write_bin(base, rows[:args.base_count] if args.base_count else rows, 'float32')
        # LEARN supplies enough queries for train/validation/test in the new workflow.
        write_bin(query, read_vectors(out/'gist_learn.fvecs'), 'float32')
    else:
        chunks = [0,1,2,3,4,5,6,7,9,10]  # RoarGraph's published order, intentionally excludes 8.
        files = []
        for i in chunks:
            p = out/f'img_emb_{i}.npy'
            download(f'https://the-eye.eu/public/AI/cah/laion400m-met-release/laion400m-embeddings/images/img_emb_{i}.npy',p)
            files.append(p)
        arrays = [np.load(p,mmap_mode='r',allow_pickle=False) for p in files]
        n = args.base_count or 10_000_000
        if n > sum(len(x) for x in arrays) or any(x.ndim != 2 or x.shape[1] != 512 for x in arrays):
            raise ValueError('LAION shard shape/count mismatch')
        base = out/'base.fbin'; tmp = base.with_suffix('.tmp'); remaining = n
        with tmp.open('wb') as f:
            f.write(struct.pack('<II',n,512))
            for rows in arrays:
                for start in range(0,min(len(rows),remaining),8192):
                    count = min(8192,remaining,len(rows)-start)
                    b = np.asarray(rows[start:start+count],dtype='<f4')
                    b = b/np.maximum(np.linalg.norm(b,axis=1,keepdims=True),1e-30)
                    b.tofile(f);remaining -= count
                if not remaining:break
        tmp.replace(base)
        raw = out/'laion.query.10k.fbin'
        download('https://zenodo.org/records/11090378/files/laion.query.10k.fbin',raw)
        query = out/'query.fbin';write_bin(query,read_bin(raw),'float32',normalize=True)
    print(json.dumps({'base':str(base),'query':str(query)},indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    dl = sub.add_parser('download');dl.add_argument('--dataset',choices=['gist','sift','deep','laion'],required=True)
    dl.add_argument('--output',type=Path,required=True);dl.add_argument('--base-count',type=int)
    cv = sub.add_parser('convert');cv.add_argument('--input',type=Path,required=True);cv.add_argument('--output',type=Path,required=True)
    cv.add_argument('--format',choices=['auto','bigann-gt'],default='auto')
    cv.add_argument('--dtype',choices=['float32','uint8','uint32'],default='float32');cv.add_argument('--limit',type=int);cv.add_argument('--normalize',action='store_true')
    toy = sub.add_parser('toy');toy.add_argument('--output',type=Path,required=True);toy.add_argument('--dtype',choices=['float32','uint8'],default='float32')
    args = parser.parse_args()
    if args.command == 'download':download_dataset(args)
    elif args.command == 'convert':
        if args.input.resolve()==args.output.resolve():raise ValueError('Input and output must differ')
        if args.format == 'bigann-gt':
            with args.input.open('rb') as f:
                n,k = struct.unpack('<II',f.read(8))
            if not n or not k or args.input.stat().st_size not in (8+n*k*4,8+n*k*8):
                raise ValueError('Invalid Big ANN ground-truth size')
            rows = np.memmap(args.input,mode='r',offset=8,dtype='<u4',shape=(n,k))
            args.dtype = 'uint32'
        else:
            rows = read_vectors(args.input,args.dtype)
        if args.limit is not None and not 0 < args.limit <= len(rows):raise ValueError('Invalid limit')
        write_bin(args.output,rows[:args.limit],args.dtype,args.normalize)
    else:
        rng = np.random.default_rng(2025);d = 128 if args.dtype=='uint8' else 96
        rows = rng.normal(size=(2400,d)).astype('float32')
        if args.dtype=='uint8':rows=np.clip(rows*35+128,0,255).astype('uint8')
        ext = 'u8bin' if args.dtype=='uint8' else 'fbin'
        write_bin(args.output/f'base.{ext}',rows[:2000]);write_bin(args.output/f'query.{ext}',rows[2000:])
        print(args.output.resolve())

if __name__=='__main__':main()
