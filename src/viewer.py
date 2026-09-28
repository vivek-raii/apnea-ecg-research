"""Small localhost research viewer; no cloud service, upload, or clinical claims."""
import argparse, json, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs
import joblib
from ecg import Archive, record_features
from predict import predict_frame

ROOT=Path(__file__).resolve().parents[1]
PAGE='''<!doctype html><html><head><meta charset="utf-8"><title>Apnea ECG research viewer</title>
<style>body{font:16px system-ui;max-width:1050px;margin:35px auto;padding:20px;color:#172033}button,select{padding:10px;font-size:16px}button{background:#2563eb;color:white;border:0;border-radius:6px}canvas{width:100%;border:1px solid #ddd;margin:20px 0}table{border-collapse:collapse;width:100%}th,td{padding:8px;border-bottom:1px solid #ddd;text-align:right}.note{background:#fff7ed;padding:15px}a{color:#2563eb}</style></head><body>
<h1>ECG apnea research viewer</h1><p class="note">Research prototype. Scores are uncalibrated and do not diagnose sleep apnea. The final model was trained on a/b/c records except c06; results on those records are not independent evaluation.</p>
<p><a href="/report" target="_blank">Open held-out evaluation report</a></p>
<label>Recording <select id="record"></select></label> <button id="run">Predict complete minutes</button>
<p id="status">Loading available recordings…</p><canvas id="chart" width="1000" height="280"></canvas>
<p>X: minute from start. Y: score from 0 to 1. Red dashed line: development-selected threshold. The table shows all predictions; quality warnings indicate unreliable input.</p>
<div style="max-height:500px;overflow:auto"><table><thead><tr><th>Minute</th><th>Score</th><th>Prediction</th><th>Reference</th><th>Signal quality</th></tr></thead><tbody id="rows"></tbody></table></div>
<script>
const status=document.getElementById('status'), rec=document.getElementById('record');
fetch('/records').then(r=>r.json()).then(d=>{for(const r of d.records){let o=document.createElement('option');o.textContent=r;o.value=r;rec.append(o)}rec.value='x01';status.textContent='Choose a recording. First prediction may take a few seconds.'});
document.getElementById('run').onclick=async()=>{let b=document.getElementById('run');b.disabled=true;status.textContent='Extracting ECG features and predicting…';
try{let response=await fetch('/predict?record='+encodeURIComponent(rec.value));let d=await response.json();if(!response.ok)throw Error(d.error);
document.getElementById('rows').replaceChildren();for(let r of d.rows){let tr=document.createElement('tr');for(let v of [r.minute,r.score.toFixed(3),r.prediction?'A':'N',r.label===null?'Unavailable':(r.label?'A':'N'),r.quality_ok?'Accepted':r.quality_reason]){let td=document.createElement('td');td.textContent=v;tr.append(td)}document.getElementById('rows').append(tr)}
let c=document.getElementById('chart'),x=c.getContext('2d');x.clearRect(0,0,c.width,c.height);x.strokeStyle='#ddd';x.fillStyle='#172033';x.font='12px sans-serif';
for(let k=0;k<=4;k++){let y=20+k*60;x.beginPath();x.moveTo(45,y);x.lineTo(980,y);x.stroke();x.fillText((1-k/4).toFixed(2),5,y+4)}
x.strokeStyle='#dc2626';x.setLineDash([5,5]);x.beginPath();x.moveTo(45,260-240*d.threshold);x.lineTo(980,260-240*d.threshold);x.stroke();x.setLineDash([]);x.strokeStyle='#2563eb';x.beginPath();
d.rows.forEach((r,i)=>{let px=45+935*i/Math.max(1,d.rows.length-1),py=260-240*r.score;i?x.lineTo(px,py):x.moveTo(px,py)});x.stroke();x.fillText('0',45,277);x.fillText(String(d.rows.at(-1).minute),950,277);
status.textContent=d.rows.length+' complete minutes. Quality accepted: '+(100*d.rows.filter(r=>r.quality_ok).length/d.rows.length).toFixed(1)+'%. Reference labels: '+(d.rows.some(r=>r.label!==null)?'available (training records)':'unavailable');
}catch(e){status.textContent=e.message}finally{b.disabled=false}};
</script></body></html>'''

def serve(port):
    config=json.loads((ROOT/'config.json').read_text()); bundle=joblib.load(ROOT/'outputs/final_model.joblib')
    a=Archive(config['dataset_zip']); records=a.records; a.close()
    cache={}; lock=threading.Lock()
    class Handler(BaseHTTPRequestHandler):
        def send(self,data,content_type='application/json',status=200):
            raw=data.encode('utf-8'); self.send_response(status); self.send_header('Content-Type',content_type+'; charset=utf-8')
            self.send_header('Content-Length',str(len(raw))); self.send_header('X-Content-Type-Options','nosniff'); self.end_headers(); self.wfile.write(raw)
        def do_GET(self):
            route=urlparse(self.path)
            if route.path=='/': return self.send(PAGE,'text/html')
            if route.path=='/records': return self.send(json.dumps(dict(records=records)))
            if route.path=='/report': return self.send((ROOT/'outputs/results_report.html').read_text(encoding='utf-8'),'text/html')
            if route.path=='/predict':
                record=parse_qs(route.query).get('record',[''])[0]
                if record not in records: return self.send(json.dumps(dict(error='Choose a valid record.')),status=400)
                try:
                    with lock:
                        if record not in cache:
                            a=Archive(config['dataset_zip'])
                            try: f=record_features(a,record)
                            finally: a.close()
                            result=predict_frame(f,bundle)
                            cache[record]=result.to_json(orient='records')
                    return self.send('{"threshold":'+str(bundle['threshold'])+',"rows":'+cache[record]+'}')
                except Exception as e: return self.send(json.dumps(dict(error=str(e))),status=500)
            self.send(json.dumps(dict(error='Not found')),status=404)
    print(f'Open http://127.0.0.1:{port} in your browser. Ctrl+C stops the viewer.',flush=True)
    ThreadingHTTPServer(('127.0.0.1',port),Handler).serve_forever()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8765);serve(p.parse_args().port)
