"""Deterministic, window-local ECG features. No labels or demographics enter X."""
from __future__ import annotations
import hashlib
import io
import re
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, find_peaks, welch

VERSION = '2.0.0-xqrs-experiment'
FS = 100
WINDOW = 6000
FEATURES = ['ecg_std','ecg_iqr','ecg_range','diff_std','flat_fraction','rail_fraction',
            'power_low','power_mid','power_high','peak_count','bad_rr_fraction',
            'rr_mean','rr_std','rr_median','rr_iqr','rmssd','pnn50',
            'rr_trend','rr_power_slow','rr_power_fast','amp_mean','amp_std','amp_iqr',
            'amp_rmssd','amp_resp_power']
SOS = butter(3, [5, 20], btype='bandpass', fs=FS, output='sos')

class Archive:
    """Read the supplied archive without extracting its contents."""
    def __init__(self, path):
        self.path = Path(path)
        self.z = zipfile.ZipFile(path)
        entries = [n for n in self.z.namelist() if not n.endswith('/')]
        self.names = {Path(n).name:n for n in entries}
        if len(entries) != len(self.names):
            raise ValueError('Archive has ambiguous basenames.')
        self.records = sorted(n[:-4] for n in self.names if re.fullmatch(r'[abcx]\d{2}\.hea',n))
    def read(self, name):
        return self.z.read(self.names[name])
    def signal(self, record):
        if record not in self.records:
            raise ValueError('Unknown record ID.')
        lines = self.read(record+'.hea').decode().splitlines()
        h, s = lines[0].split(), lines[1].split()
        if h[1:3] != ['1','100'] or s[1] != '16':
            raise ValueError('Expected one-channel, 100 Hz WFDB format 16.')
        raw = np.frombuffer(self.read(record+'.dat'),dtype='<i2')
        if len(raw) != int(h[3]): raise ValueError('Signal length mismatch.')
        if int(raw.sum(dtype=np.int64)) % 65536 != int(s[6]) % 65536:
            raise ValueError('Signal checksum mismatch.')
        return (raw.astype(float)-int(s[4]))/float(s[2])
    def annotations(self, record, suffix='apn'):
        name=record+'.'+suffix
        if name not in self.names: return []
        data=self.read(name); pos=0; sample=0; out=[]
        while pos+1<len(data):
            word=int.from_bytes(data[pos:pos+2],'little'); pos+=2
            if word==0: break
            kind,delta=word>>10,word&1023
            if kind==59:
                value=(int.from_bytes(data[pos:pos+2],'little')<<16)|int.from_bytes(data[pos+2:pos+4],'little')
                sample+=value if value<2**31 else value-2**32; pos+=4
            elif kind==63: pos+=delta+delta%2
            elif kind in (60,61,62): pass
            else: sample+=delta; out.append((sample,kind))
        return out
    def metadata(self):
        result={}
        for line in self.read('additional-information.txt').decode().splitlines():
            p=line.split()
            if p and re.fullmatch(r'[abcx]\d{2}',p[0]):
                result[p[0]]=dict(age=int(p[8]),sex=p[9],height=int(p[10]),weight=int(p[11]))
        return result
    def verify(self):
        n=0
        for line in self.read('SHA256SUMS.txt').decode().splitlines():
            expected,name=line.split(); name=Path(name.lstrip('*')).name
            if hashlib.sha256(self.read(name)).hexdigest()!=expected: raise ValueError('Hash mismatch: '+name)
            n+=1
        return n
    def close(self): self.z.close()

def groups_from_metadata(meta):
    """Conservative connected groups, including matches in original test records.

    These are candidate repeat-subject groups, NOT verified patient identities.
    Demographics only define evaluation groups and are never model inputs.
    """
    parent={r:r for r in meta}
    def root(r):
        while parent[r]!=r: r=parent[r]
        return r
    def union(a,b): parent[root(b)]=root(a)
    seen={}
    for r,m in sorted(meta.items()):
        key=tuple(m[k] for k in ('age','sex','height','weight'))
        if key in seen: union(seen[key],r)
        else: seen[key]=r
    if 'c05' in parent and 'c06' in parent: union('c05','c06')
    members={}
    for r in meta: members.setdefault(root(r),[]).append(r)
    return {r:min(members[root(r)]) for r in meta}

def band_power(signal, fs, low, high):
    if len(signal)<8: return np.nan
    f,p=welch(signal,fs=fs,nperseg=min(len(signal),256))
    mask=(f>=low)&(f<high)
    return float(p[mask].sum()*(f[1]-f[0]))

def detect_peaks(x):
    """WFDB XQRS adaptive detection on one complete minute only."""
    from wfdb.processing import xqrs_detect
    filtered=sosfiltfilt(SOS,x)
    peaks=xqrs_detect(sig=np.asarray(x,dtype=float),fs=FS,verbose=False)
    # At 100 Hz the detector's integration maxima can precede/follow the QRS.
    # Align each candidate to a local absolute filtered extremum for RR/amplitude.
    refined=[]
    for peak in peaks:
        lo=max(0,peak-20); hi=min(len(x),peak+21)
        q=lo+int(np.argmax(np.abs(filtered[lo:hi])))
        if q<20 or q>=len(x)-20: continue
        if refined and q-refined[-1]<25:
            if abs(filtered[q])>abs(filtered[refined[-1]]): refined[-1]=q
        else: refined.append(q)
    return np.asarray(refined,dtype=int),filtered


def extract_window(x):
    """Return feature mapping, quality disposition, and peak positions for 60s ECG."""
    x=np.asarray(x,dtype=float)
    if x.shape!=(WINDOW,) or not np.isfinite(x).all():
        raise ValueError('Expected exactly 6,000 finite ECG samples in millivolts.')
    out={k:np.nan for k in FEATURES}
    out.update(ecg_std=float(x.std()),ecg_iqr=float(np.ptp(np.percentile(x,[25,75]))),
               ecg_range=float(np.ptp(x)),diff_std=float(np.diff(x).std()),
               flat_fraction=float(np.mean(np.diff(x)==0)),rail_fraction=float(np.mean(np.abs(x)>=10.23)))
    total=band_power(x,FS,0,50)+1e-12
    for name,lo,hi in [('low',0,0.5),('mid',0.5,15),('high',15,50)]:
        out['power_'+name]=band_power(x,FS,lo,hi)/total
    peaks,filtered=detect_peaks(x)
    rr=np.diff(peaks)/FS
    good=(rr>=0.3)&(rr<=2.0)
    out['peak_count']=len(peaks)
    out['bad_rr_fraction']=float(1-good.mean()) if len(rr) else 1.
    valid=rr[good]
    if len(valid)>=5:
        # RMSSD includes only consecutive pairs of physiologically plausible RR.
        adjacent=good[:-1]&good[1:]; differences=np.diff(rr)[adjacent]
        out.update(rr_mean=float(valid.mean()),rr_std=float(valid.std()),rr_median=float(np.median(valid)),
                   rr_iqr=float(np.ptp(np.percentile(valid,[25,75]))),
                   rmssd=float(np.sqrt(np.mean(differences**2))) if len(differences) else np.nan,
                   pnn50=float(np.mean(np.abs(differences)>0.05)) if len(differences) else np.nan)
        times=peaks[1:][good]/FS
        out['rr_trend']=float(np.polyfit(times,valid,1)[0])
        grid=np.arange(times[0],times[-1],0.25)
        interpolated=np.interp(grid,times,valid)
        out['rr_power_slow']=band_power(interpolated,4,0.04,0.15)
        out['rr_power_fast']=band_power(interpolated,4,0.15,0.4)
    if len(peaks)>=5:
        amplitude=np.abs(filtered[peaks])
        out.update(amp_mean=float(amplitude.mean()),amp_std=float(amplitude.std()),
                   amp_iqr=float(np.ptp(np.percentile(amplitude,[25,75]))),
                   amp_rmssd=float(np.sqrt(np.mean(np.diff(amplitude)**2))))
        grid=np.arange(peaks[0]/FS,peaks[-1]/FS,0.25)
        out['amp_resp_power']=band_power(np.interp(grid,peaks/FS,amplitude),4,0.1,0.4)
    reasons=[]
    if out['ecg_std']<0.01: reasons.append('low_variation')
    if out['rail_fraction']>0.01: reasons.append('clipping')
    if not 25<=len(peaks)<=180: reasons.append('peak_count')
    if out['bad_rr_fraction']>0.2: reasons.append('rr_quality')
    return out,';'.join(reasons),peaks

def record_features(archive,record,labeled_only=False):
    signal=archive.signal(record)
    annotations=archive.annotations(record)
    labels={s:int(c==8) for s,c in annotations}
    if any(c not in (1,8) for _,c in annotations): raise ValueError('Unexpected apnea code.')
    starts=sorted(labels) if labeled_only else range(0,len(signal)-WINDOW+1,WINDOW)
    rows=[]
    for start in starts:
        if start+WINDOW>len(signal): continue
        feat,reason,_=extract_window(signal[start:start+WINDOW])
        rows.append(dict(record=record,start_sample=start,minute=start//WINDOW,
                         label=labels.get(start,np.nan),quality_ok=not bool(reason),quality_reason=reason,**feat))
    return pd.DataFrame(rows)
