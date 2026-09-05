const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const {JSDOM} = require('jsdom');
const html = fs.readFileSync('darwintrade/live/static/index.html', 'utf8');
const script = fs.readFileSync('darwintrade/live/static/app.js', 'utf8');
const sample = JSON.parse(fs.readFileSync('darwintrade/live/static/sample-report.json', 'utf8').replace(/^\uFEFF/, ''));
const tick = () => new Promise(resolve => setTimeout(resolve, 15));
async function setup(decide, saved) {
  const dom = new JSDOM(html, {url: 'http://localhost', runScripts: 'outside-only'});
  const w = dom.window;
  w.HTMLElement.prototype.scrollIntoView = function() {};
  const calls = [];
  if (saved) w.localStorage.setItem('darwintrade.session', saved);
  w.fetch = async (url, options) => {
    calls.push({url, options});
    const result = url === '/api/health' ? {llm_configured: true, default_universe: ['AAPL'], max_symbols: 30}
      : url === '/static/sample-report.json' ? sample
      : url.startsWith('/api/sessions/') ? {session_id: saved}
      : await decide(JSON.parse(options.body));
    return {ok: !result.error, status: result.error ? 422 : 200, json: async () => result.error ? {detail: result.error} : result};
  };
  w.eval(script);
  await tick();
  return {dom, w, calls, el: id => w.document.getElementById(id)};
}
function submit(w, el) {
  el('symbols').value = 'AAPL, aapl MSFT';
  el('form').dispatchEvent(new w.Event('submit', {bubbles:true,cancelable:true}));
}
test('sample is labelled and leaves remembered session unchanged', async () => {
  const {dom,w,el,calls} = await setup(null, 'saved-session');
  el('example').click(); await tick();
  assert.match(el('report-title').textContent, /illustrative/);
  assert.equal(el('positions').children.length, 3);
  assert.equal(w.localStorage.getItem('darwintrade.session'), 'saved-session');
  assert.equal(calls.filter(c => c.url === '/api/decide').length, 0);
  dom.window.close();
});
test('continued request deduplicates symbols, omits capital, and renders report', async () => {
  const {dom,w,el,calls} = await setup(async body => ({...sample,sample:false,session_id:body.session_id}), 'saved-session');
  submit(w,el); await tick();
  const body = JSON.parse(calls.find(c => c.url === '/api/decide').options.body);
  assert.deepEqual(body.symbols, ['AAPL','MSFT']);
  assert.equal(body.capital, undefined);
  assert.equal(body.session_id, 'saved-session');
  assert.equal(el('results').hidden, false);
  assert.equal(el('progress').hidden, true);
  assert.equal(el('capital').disabled, true);
  el('reset-session').click();
  assert.equal(w.localStorage.getItem('darwintrade.session'), null);
  assert.equal(el('capital').disabled, false);
  dom.window.close();
});
test('validation errors are readable, stale results hidden, submit restored', async () => {
  const {dom,w,el} = await setup(async () => ({error:[{loc:['body','trade_date'],msg:'Invalid date'}]}));
  el('results').hidden = false;
  submit(w,el); await tick();
  assert.equal(el('error').textContent, 'trade_date: Invalid date');
  assert.equal(el('results').hidden, true);
  assert.equal(el('run').disabled, false);
  assert.equal(el('empty-state').hidden, false);
  dom.window.close();
});
test('duplicate clicks issue one request and report text cannot inject HTML', async () => {
  let finish;
  const {dom,w,el,calls} = await setup(() => new Promise(resolve => { finish=resolve; }));
  submit(w,el); submit(w,el);
  assert.equal(calls.filter(c => c.url === '/api/decide').length, 1);
  finish({...sample,sample:false,session_id:'new-session',positions:[{...sample.positions[0],thesis:'<img src=x onerror=alert(1)>'}]});
  await tick();
  assert.equal(el('positions').querySelector('img'),null);
  assert.match(el('positions').textContent, /<img/);
  assert.equal(w.localStorage.getItem('darwintrade.session'),'new-session');
  dom.window.close();
});
