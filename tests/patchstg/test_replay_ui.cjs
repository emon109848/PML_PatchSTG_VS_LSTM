/* Dependency-free control checks using a minimal DOM/canvas harness.
 * This exercises real replay JavaScript, but is not a browser rendering test.
 * Run: node tests/test_replay_ui.cjs [generated/index.html]
 */
'use strict';
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const artifact=process.argv.slice(2).find(arg=>arg!=='--comparison');
const compareMode=process.argv.includes('--comparison');
const template=fs.readFileSync(artifact||'demos/patchstg_replay.html','utf8');
const source=template.match(/<script>\s*([\s\S]*?)<\/script>/)[1];
const synthetic={history:12,horizon:12,interval_minutes:15,selection:'chronological',calendar_note:'assumed',zero_policy:'exclude zeros',
 provenance:{full_test_metrics:{mae:17.1943}},sensors:[{id:'A',road:'I5',direction:'N',latitude:32,longitude:-117},{id:'B',road:'I8',direction:'E',latitude:33,longitude:-116}],
 observations:Array.from({length:25},(_,i)=>[i,i+10]),predictions:Array.from({length:2},()=>[Array(12).fill(12),Array(12).fill(22)]),
 frames:[{window_index:0,observation_offset:12,issued_at:'2019-01-01T02:45'},{window_index:1,observation_offset:13,issued_at:'2019-01-01T03:00'}]};
if(compareMode){synthetic.models={patchstg:{label:'PatchSTG adaptation',predictions:synthetic.predictions,full_test_metrics:{mae:17.1943}},bigst:{label:'BigST short history (no LSFE/spatial prior)',predictions:Array.from({length:2},()=>[Array(12).fill(14),Array(12).fill(24)]),full_test_metrics:{mae:18.8241}}};delete synthetic.predictions;}
const embedded=template.match(/<script id="replay-data" type="application\/json">([\s\S]*?)<\/script>/)[1];
const data=artifact?JSON.parse(embedded):synthetic;
const comparison=Boolean(data.models&&data.models.bigst);
const patchModel=comparison?data.models.patchstg:{predictions:data.predictions,full_test_metrics:data.provenance.full_test_metrics};
const strokes=[];
const context2d=new Proxy({}, {get:(obj,key)=>key==='stroke'?(()=>strokes.push(obj.strokeStyle)):(obj[key]||(()=>{})),set:(obj,key,value)=>{obj[key]=value;return true;}});
class Element{
 constructor(){this.textContent='';this.value='';this.checked=false;this.children=[];this.listeners={};}
 appendChild(node){this.children.push(node);return node;}
 replaceChildren(){this.children=[];}
 addEventListener(name,callback){this.listeners[name]=callback;}
 setAttribute(name,value){this[name]=value;}
 getBoundingClientRect(){return {width:600,height:330,left:0,top:0};}
 getContext(){return context2d;}
 fire(event){this.listeners[event]({target:this,clientX:20,clientY:20});}
}
const elements={};for(const match of template.matchAll(/id="([^"]+)"/g))elements[match[1]]=new Element();
elements['replay-data'].textContent=JSON.stringify(data);elements.reveal.checked=true;elements.speed.value='600';
const intervals=new Map();let intervalId=0;
const document={getElementById:id=>elements[id],createElement:()=>new Element(),listeners:{},hidden:false,addEventListener(name,cb){this.listeners[name]=cb;}};
const window={devicePixelRatio:1,addEventListener(){},setInterval(cb){intervals.set(++intervalId,cb);return intervalId;},clearInterval(id){intervals.delete(id);}};
vm.runInNewContext(source,{document,window,console});
assert.equal(elements.sensor.children.length,data.sensors.length);
assert.equal(elements['horizon-table'].children.length,12);
assert.match(elements['frame-label'].textContent,/Forecast origin 1 /);
assert.equal(elements['test-mae'].textContent,Number(patchModel.full_test_metrics.mae).toLocaleString('en-US',{minimumFractionDigits:4,maximumFractionDigits:4}));
elements.sensor.value=String(data.sensors.length-1);elements.sensor.fire('change');
assert.match(elements['chart-title'].textContent,new RegExp(data.sensors.at(-1).id));
elements.lead.value='11';elements.lead.fire('change');assert.equal(elements['forecast-label'].textContent,comparison?'PatchSTG forecast · +180 min':'Predicted flow · +180 min');
if(comparison){
 const big=data.models.bigst;const fmt=(n,d)=>Number(n).toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d});
 assert.equal(elements['bigst-forecast'].textContent,fmt(big.predictions[0].at(-1)[11],1));
 assert.equal(elements['bigst-test-mae'].textContent,fmt(big.full_test_metrics.mae,4));
 assert.equal(elements['horizon-table'].children[0].children.length,6);
 const truth=data.observations.slice(data.frames[0].observation_offset,data.frames[0].observation_offset+12).map(row=>row.at(-1));
 const valid=truth.map((v,i)=>({v,i})).filter(x=>x.v!==0);
 for(const [id,model] of [['local-mae',patchModel],['bigst-mae',big]]){const value=valid.reduce((sum,x)=>sum+Math.abs(x.v-model.predictions[0].at(-1)[x.i]),0)/valid.length;assert.equal(elements[id].textContent,valid.length?fmt(value,2):'—');}
 for(const view of ['bigst','patchstg','both']){strokes.length=0;elements['model-view'].value=view;elements['model-view'].fire('change');assert.equal(strokes.includes('#59debb'),view!=='bigst');assert.equal(strokes.includes('#e49aff'),view!=='patchstg');}
 elements['map-model'].value='bigst';elements['map-model'].fire('change');assert.match(elements['network-description'].textContent,/BigST/);
 elements['map-model'].value='patchstg';elements['map-model'].fire('change');
} else if(elements['bigst-forecast-card']) {assert.equal(elements['bigst-forecast-card'].hidden,true);}
elements.reveal.checked=false;elements.reveal.fire('change');assert.equal(elements['local-mae'].textContent,'—');
assert.equal(elements['horizon-table'].children[0].children[comparison?3:2].textContent,'Hidden');
if(comparison)assert.equal(elements['bigst-mae'].textContent,'—');
elements.timeline.value=String(data.frames.length-1);elements.timeline.fire('input');assert.equal(elements.next.disabled,true);
elements.play.fire('click');assert.equal(elements.timeline.value,0);assert.equal(intervals.size,1);
for(let i=0;i<data.frames.length+1;i++)for(const callback of [...intervals.values()])callback();
assert.equal(intervals.size,0);assert.equal(elements.play.textContent,'Play');assert.equal(elements.next.disabled,true);
elements.play.fire('click');document.hidden=true;document.listeners.visibilitychange();assert.equal(intervals.size,0);
elements.previous.fire('click');assert.equal(elements.timeline.value,0);
console.log((comparison?'Comparison':'Single-model')+' replay controls passed: forecasts/errors, sensor/lead, reveal, model lines/map, table, timeline, playback and hidden-tab pause.');
