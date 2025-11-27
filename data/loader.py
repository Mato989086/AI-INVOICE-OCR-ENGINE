import numpy as np
from typing import List,Dict,Callable,Optional,Iterator
from concurrent.futures import ThreadPoolExecutor
import queue,threading
class DataLoader:
    def __init__(s,ds,batch_size:int=1,shuffle:bool=True,num_workers:int=0,collate_fn:Callable=None,drop_last:bool=False):
        s.ds,s.bs,s.sh,s.nw,s.cf,s.dl=ds,batch_size,shuffle,num_workers,collate_fn or _default_collate,drop_last
        s._idx=[];s._pos=0
    def __iter__(s)->Iterator:
        s._idx=list(range(len(s.ds)))
        if s.sh:np.random.shuffle(s._idx)
        s._pos=0;return s
    def __next__(s)->Dict:
        if s._pos>=len(s._idx):raise StopIteration
        end=min(s._pos+s.bs,len(s._idx))
        if s.dl and end-s._pos<s.bs:raise StopIteration
        batch_idx=s._idx[s._pos:end];s._pos=end
        if s.nw>0:
            with ThreadPoolExecutor(max_workers=s.nw)as ex:items=list(ex.map(s.ds.__getitem__,batch_idx))
        else:items=[s.ds[i]for i in batch_idx]
        return s.cf(items)
    def __len__(s)->int:
        n=len(s.ds)//s.bs;return n if s.dl else n+(1 if len(s.ds)%s.bs else 0)
def _default_collate(items:List[Dict])->Dict:
    if not items:return{}
    keys=items[0].keys();batch={}
    for k in keys:
        vals=[it[k]for it in items]
        if isinstance(vals[0],np.ndarray):
            try:batch[k]=np.stack(vals)
            except:batch[k]=vals
        else:batch[k]=vals
    return batch
def collate_fn(items:List[Dict])->Dict:
    return _default_collate(items)
class _PrefetchLoader:
    def __init__(s,loader:DataLoader,n_prefetch:int=2):s.ld,s.np=loader,n_prefetch;s._q=queue.Queue(n_prefetch);s._stop=False
    def _worker(s):
        try:
            for batch in s.ld:
                if s._stop:break
                s._q.put(batch)
        finally:s._q.put(None)
    def __iter__(s)->Iterator:
        s._stop=False;t=threading.Thread(target=s._worker,daemon=True);t.start()
        while True:
            batch=s._q.get()
            if batch is None:break
            yield batch
    def __len__(s)->int:return len(s.ld)
    def stop(s):s._stop=True
class _BucketLoader:
    def __init__(s,ds,batch_size:int,key_fn:Callable,n_bucket:int=10):
        s.ds,s.bs,s.kf,s.nb=ds,batch_size,key_fn,n_bucket;s._buckets=[[]for _ in range(n_bucket)]
    def __iter__(s)->Iterator:
        idx=list(range(len(s.ds)));np.random.shuffle(idx)
        for i in idx:
            k=s.kf(s.ds[i]);bi=min(int(k*s.nb),s.nb-1);s._buckets[bi].append(i)
            if len(s._buckets[bi])>=s.bs:
                batch=[s.ds[j]for j in s._buckets[bi]];s._buckets[bi]=[];yield _default_collate(batch)
        for bucket in s._buckets:
            if bucket:
                batch=[s.ds[j]for j in bucket];yield _default_collate(batch)
        s._buckets=[[]for _ in range(s.nb)]
