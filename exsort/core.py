"""Independent, validation-ablated exsort changes. Legacy path is exactly preserved.

UNCERTAINTY is a Gaussian mean-absolute-displacement IoU surrogate, not a
calibrated survival probability. Uses existing EV, no extra tuned threshold.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from . import legacy as L
from .vendor.ocsort.association import iou_batch
from .vendor.ocsort.ocsort import KalmanBoxTracker,convert_bbox_to_z,speed_direction

norm_box_to_xyxy=L.norm_box_to_xyxy
xyxy_to_norm_box=L.xyxy_to_norm_box
DET=L.DET;LOW=L.LOW;NMS=L.NMS;IOU=L.IOU
MIN_HITS=L.MIN_HITS;MAX_AGE=L.MAX_AGE;HEAD_FRACTION=L.HEAD_FRACTION;EV=L.EV
OPTS=dict(L.OPTS,admissible=False,lifecycle=False,birth_policy='axis',noise=False,coast='legacy')
ADMISSIBLE=False;LIFECYCLE=False;BIRTH_POLICY='axis';NOISE=False;COAST_MODE='legacy'
MEASURE='raw';LOW_POLICY='legacy';ASSOCIATION='iou'
INIT=0.0  # 0 reuses DET; .4 is TrackTrack's explicit birth threshold.
LOW_OUTPUT='all';OUTPUT_FILTER='none'
TT_MIN_AREA=100.;TT_MAX_ASPECT=1.6
RECOVERY=False;LOST_BIRTH='veto'

def assoc(a,b,scores=None,low=False):
    sim=iou_batch(np.asarray(a,np.float32).reshape(-1,4),np.asarray(b,np.float32).reshape(-1,4))
    threshold=EV if low and LOW_POLICY=='strict' else IOU
    metric=sim.copy()
    if ASSOCIATION=='hmiou':
        a=np.asarray(a,np.float32).reshape(-1,4);b=np.asarray(b,np.float32).reshape(-1,4)
        inter=np.maximum(np.minimum(a[:,None,3],b[None,:,3])-np.maximum(a[:,None,1],b[None,:,1]),0.)
        union=np.maximum(a[:,None,3],b[None,:,3])-np.minimum(a[:,None,1],b[None,:,1])
        metric*=inter/np.maximum(union,np.finfo(np.float32).eps)
    elif ASSOCIATION=='score' and scores is not None:
        metric*=np.asarray(scores)[:,None]
    valid=sim>=threshold
    if not OPTS['admissible']:
        if not sim.size or not valid.any():return []
        return [(int(d),int(t)) for d,t in L.linear_assignment(-metric) if valid[d,t]]
    n,m=sim.shape
    if not n or not m:return []
    # Dummy unmatched columns. Cardinality first, total IoU second; invalid edges forbidden.
    reward=np.zeros((n,m+n),float)
    reward[:,:m]=np.where(valid,1.+sim/(min(n,m)+1.),-1e6)
    rows,cols=linear_sum_assignment(reward,maximize=True)
    return [(int(i),int(j)) for i,j in zip(rows,cols) if j<m and valid[i,j]]

class ConfidenceKalmanBoxTracker(KalmanBoxTracker):
    def update(self,bbox):
        if bbox is None:
            self.kf.update(None);return
        # Same observation bookkeeping as OC-SORT; R passed explicitly so ORU
        # cannot replace the current real measurement's noise during unfreeze.
        if self.last_observation.sum()>=0:
            previous=None
            for i in range(self.delta_t):
                dt=self.delta_t-i
                if self.age-dt in self.observations:
                    previous=self.observations[self.age-dt];break
            if previous is None:previous=self.last_observation
            self.velocity=speed_direction(previous,bbox)
        self.last_observation=bbox
        self.observations[self.age]=bbox
        self.history_observations.append(bbox)
        self.time_since_update=0;self.history=[];self.hits+=1;self.hit_streak+=1
        scale=max(1.-float(bbox[4]),np.finfo(float).eps)
        if OPTS['noise']=='low_only':scale=max(1.,DET/max(float(bbox[4]),np.finfo(float).eps))
        R=self.kf.R*scale
        self.kf.update(convert_bbox_to_z(bbox),R=R)

class Track(L.Track):
    def __init__(self,tid,row,W,H):
        super().__init__(tid,row,W,H);self.high_hits=1
        self.high_box=row[:4].copy();self.since_high=0
        self.prior_box=row[:4].copy()
        if OPTS['noise']:self.kf.__class__=ConfidenceKalmanBoxTracker
    def observe(self,row):
        super().observe(row)
        self.high_hits+=int(row[4]>=DET)
        if row[4]>=DET:self.high_box=row[:4].copy();self.since_high=0
    def out_box(self):
        if self.kf.time_since_update or MEASURE=='raw':return super().out_box()
        b=self.pred()
        if MEASURE=='blend':
            score=float(self.kf.last_observation[4]);b=score*self.box+(1.-score)*b
        return xyxy_to_norm_box(b,self.W,self.H)

class EXSORT(L.EXSORT):
    def __init__(self):
        OPTS.update(admissible=ADMISSIBLE,lifecycle=LIFECYCLE,birth_policy=BIRTH_POLICY,noise=NOISE,coast=COAST_MODE)
        for k in ('DET','LOW','NMS','IOU','MIN_HITS','MAX_AGE','HEAD_FRACTION','EV'):setattr(L,k,globals()[k])
        L.OPTS.clear();L.OPTS.update(OPTS)
        self.legacy=not any((OPTS['admissible'],OPTS['lifecycle'],OPTS['birth_policy']!='axis',OPTS['noise'],OPTS['coast']!='legacy'))
        super().__init__()
    def confirmed(self,t):
        return (t.high_hits if OPTS['lifecycle'] else t.kf.hits)>=MIN_HITS
    def claims(self,b,t):
        if LOST_BIRTH=='defer' and t.kf.time_since_update>0:return False
        policy=OPTS['birth_policy']
        if policy=='axis' or (policy in ('mixed','iom_active') and t.kf.time_since_update>0):return super()._claims(b,t)
        if t.kf.time_since_update>0:return False
        if policy=='iom_active':
            a=np.asarray(b);r=t.box
            inter=float(np.prod(np.maximum(np.minimum(a[2:],r[2:])-np.maximum(a[:2],r[:2]),0.)))
            area=min(float(np.prod(np.maximum(a[2:]-a[:2],0.))),float(np.prod(np.maximum(r[2:]-r[:2],0.))))
            return inter/max(area,np.finfo(float).eps)>=IOU
        return float(iou_batch(np.asarray(b,np.float32)[None],t.box[None])[0,0])>=IOU
    def emission(self,t):
        if not OPTS['emit'] or not t.now or not self._localised(t):return False
        mode=OPTS['coast']
        if mode in ('high_anchor','last_high'):
            if t.since_high>MAX_AGE:return False
            if mode=='last_high':return t.kf.last_observation[4]>=DET
            return float(iou_batch(t.pred()[None],t.high_box[None])[0,0])>=EV
        if mode=='none':return False
        if mode=='budget':return t.kf.time_since_update<=MAX_AGE
        if mode=='uncertainty':
            b=t.pred();w,h=np.maximum(b[2:]-b[:2],1.)
            # Expected absolute displacement for zero-mean Gaussian centre errors.
            dx,dy=np.sqrt(np.maximum(np.diag(t.kf.kf.P)[:2],0.)*2./np.pi)
            overlap=max(w-dx,0.)*max(h-dy,0.)
            stability=overlap/max(2*w*h-overlap,np.finfo(float).eps)
            observed=float(iou_batch(b[None],t.box[None])[0,0])
            return observed*stability>=EV
        return True
    def step(self,dets,W,H,frame_idx,sat=None):
        if self.legacy:return super().step(dets,W,H,frame_idx,sat)
        self.frame+=1
        D=np.asarray([list(norm_box_to_xyxy(d['box'],W,H))+[d['conf']] for d in dets if d['conf']>=DET],np.float32).reshape(-1,5)
        D=D[L._nms_keep(D)]
        low=np.asarray([list(norm_box_to_xyxy(d['box'],W,H))+[d['conf']] for d in dets if LOW<=d['conf']<DET],np.float32).reshape(-1,5)
        for t in self.tracks:t.since_high+=1
        P=np.asarray([t.kf.predict()[0][:4] for t in self.tracks],np.float32).reshape(-1,4)
        for i,t in enumerate(self.tracks):t.prior_box=P[i].copy()
        ud=list(range(len(D)))
        confirmed=[i for i,t in enumerate(self.tracks) if not OPTS['lifecycle'] or self.confirmed(t)]
        tentative=[i for i,t in enumerate(self.tracks) if OPTS['lifecycle'] and not self.confirmed(t)]
        def high_pass(ut):
            nonlocal ud
            for use_last in (False,True):
                if not ud or not ut:break
                refs=np.asarray([self.tracks[i].box if use_last else P[i] for i in ut],np.float32)
                matches=assoc(D[ud,:4],refs,D[ud,4])
                for d,t in matches:self.tracks[ut[t]].observe(D[ud[d]])
                md={d for d,_ in matches};mt={t for _,t in matches}
                ud=[d for i,d in enumerate(ud) if i not in md];ut=[t for i,t in enumerate(ut) if i not in mt]
            return ut
        unmatched=high_pass(confirmed)
        if len(low) and unmatched and OPTS['low']:
            matches=assoc(low[:,:4],[self.tracks[i].pred() for i in unmatched],low[:,4],low=True)
            for d,t in matches:self.tracks[unmatched[t]].observe(low[d])
            mt={t for _,t in matches};unmatched=[t for i,t in enumerate(unmatched) if i not in mt]
        unmatched+=high_pass(tentative)
        for i in unmatched:
            t=self.tracks[i];t.kf.update(None);t.now=False
            if OPTS['explain'] and sat is not None and self.confirmed(t):
                if L._lamp(sat,t.pred(),W,H) is not None:t.now=t.latched=True
            if not t.now:t.unexplained+=1
        if OPTS['lifecycle']:
            self.tracks=[t for t in self.tracks if self.confirmed(t) or t.kf.time_since_update==0]
        if OPTS['birth_policy']!='axis':self.tracks=[t for t in self.tracks if t.alive()]
        for d in ud:
            row=D[d]
            if row[4]<max(DET,INIT):continue
            if OPTS['birth'] and any(self.claims(row[:4],t) for t in self.tracks):continue
            self.n+=1;self.tracks.append(Track(self.n,row,W,H))
        self.tracks=[t for t in self.tracks if t.alive()]
        if RECOVERY:self._inherit()
        out=[t for t in self.tracks if (self.confirmed(t) or (not OPTS['lifecycle'] and self.frame<=MIN_HITS)) and (t.kf.time_since_update==0 or self.emission(t))]
        if LOW_OUTPUT!='all':
            accepted=[]
            for t in out:
                if t.kf.time_since_update or t.kf.last_observation[4]>=DET:
                    accepted.append(t);continue
                ref=t.high_box if LOW_OUTPUT=='high_anchor' else t.prior_box
                if LOW_OUTPUT=='high_anchor' and t.since_high>MAX_AGE:continue
                if float(iou_batch(t.box[None],ref[None])[0,0])>=EV:accepted.append(t)
            out=accepted
        if OUTPUT_FILTER=='tracktrack':
            accepted=[]
            for t in out:
                a,b,c,d=norm_box_to_xyxy(t.out_box(),W,H);w,h=c-a,d-b
                if h>0 and w*h>TT_MIN_AREA and w/h<=TT_MAX_ASPECT:accepted.append(t)
            out=accepted
        for t in self.tracks:
            if self.confirmed(t):t.seen=True
        return out
