"""
交互式预测大屏生成器
======================
把 model.json 内嵌进单文件 HTML，浏览器端用 JS 遍历随机森林做推理。
特点：无需服务器、无需联网、答辩现场双击即用。

生成: output/dashboard/predict_dashboard.html
"""

import json
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
with open(PROJECT_ROOT / "config" / "config.yaml", "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

DASH = PROJECT_ROOT / CONFIG["paths"]["dashboard"]
RISK_LEVELS = CONFIG["risk"]["levels"]

HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>小麦产量情景模拟（离线演示）</title>
<style>
  *{margin:0;padding:0;box-sizing:border-box}
  body{font-family:"Microsoft YaHei","PingFang SC",Arial,sans-serif;background:#0a1420;color:#dbe4ee;min-height:100vh;padding:24px}
  .wrap{max-width:1320px;margin:0 auto}
  header{text-align:center;margin-bottom:26px}
  h1{font-size:27px;font-weight:600;letter-spacing:1px;color:#eaf2fa}
  h1 span{color:#4fc3f7}
  .sub{color:#5f7d95;font-size:13px;margin-top:8px}
  .grid{display:grid;grid-template-columns:400px 1fr;gap:20px}
  .card{background:#111e2e;border:1px solid #1e3348;border-radius:10px;padding:20px}
  .card h2{font-size:15px;color:#8fb3cc;font-weight:500;margin-bottom:16px;padding-bottom:10px;border-bottom:1px solid #1e3348;display:flex;justify-content:space-between}
  label{display:block;font-size:13px;color:#8fb3cc;margin:14px 0 6px}
  select{width:100%;background:#0d1826;color:#dbe4ee;border:1px solid #27405a;border-radius:6px;padding:9px;font-size:14px;outline:none}
  .slider-row{display:flex;align-items:center;gap:10px}
  input[type=range]{flex:1;-webkit-appearance:none;height:4px;background:#27405a;border-radius:2px;outline:none}
  input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:14px;height:14px;border-radius:50%;background:#4fc3f7;cursor:pointer}
  .val{width:62px;text-align:right;font-size:13px;color:#4fc3f7;font-variant-numeric:tabular-nums}
  .big{font-size:44px;font-weight:700;color:#4fc3f7;line-height:1;font-variant-numeric:tabular-nums}
  .big small{font-size:14px;color:#8fb3cc;font-weight:400;margin-left:6px}
  .row{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-top:4px}
  .kpi{background:#0d1826;border:1px solid #1e3348;border-radius:8px;padding:14px}
  .kpi .k{font-size:12px;color:#64829b}
  .kpi .v{font-size:20px;font-weight:600;margin-top:6px;font-variant-numeric:tabular-nums}
  .risk{display:inline-block;padding:6px 16px;border-radius:20px;font-size:14px;font-weight:600;margin-top:14px}
  .tag{font-size:12px;padding:2px 9px;border-radius:10px;font-weight:500}
  .bar-item{margin:12px 0}
  .bar-head{display:flex;justify-content:space-between;font-size:12px;color:#7897b0;margin-bottom:5px}
  .bar-bg{height:7px;background:#0d1826;border-radius:4px;overflow:hidden}
  .bar-fill{height:100%;border-radius:4px;transition:width .25s}
  table{width:100%;border-collapse:collapse;font-size:13px;margin-top:6px}
  th{text-align:left;color:#64829b;font-weight:500;padding:8px 6px;border-bottom:1px solid #1e3348;font-size:12px}
  td{padding:7px 6px;border-bottom:1px solid #16273a;font-variant-numeric:tabular-nums}
  tr:hover td{background:#14222f}
  .note{font-size:12px;color:#4d6a80;margin-top:16px;line-height:1.7}
  footer{text-align:center;color:#3d566b;font-size:12px;margin-top:26px}
  .empty{padding:40px;text-align:center;color:#5f7d95;line-height:1.8}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>小麦产量<span>情景模拟</span>（离线演示）</h1>
    <div class="sub">37 个农业县 · 2010-2024 逐日气象 · 随机森林情景预测</div>
  </header>

  <div class="grid">
    <div>
      <div class="card">
        <h2><span>情景设置</span><span id="modelTag"></span></h2>
        <label>选择县</label>
        <select id="county"></select>

        <label>春季降水（3-4月）相对常年 <span style="color:#64829b">— 春旱</span></label>
        <div class="slider-row">
          <input type="range" id="p34" min="-80" max="50" value="0" step="5">
          <span class="val" id="p34v">0 %</span>
        </div>

        <label>灌浆期最高温（5-6月）偏移 <span style="color:#64829b">— 高温逼熟</span></label>
        <div class="slider-row">
          <input type="range" id="dt" min="-4" max="6" value="0" step="0.5">
          <span class="val" id="dtv">0 ℃</span>
        </div>

        <label>干热风日数（5-6月）</label>
        <div class="slider-row">
          <input type="range" id="dhw" min="0" max="15" value="0" step="1">
          <span class="val" id="dhwv">0 天</span>
        </div>

        <label>倒春寒日数（3-4月，最低温&lt;0℃）</label>
        <div class="slider-row">
          <input type="range" id="cold" min="0" max="25" value="0" step="1">
          <span class="val" id="coldv">0 天</span>
        </div>

        <div class="note">拖动滑块模拟不同气象情景，模型实时输出预测单产与气象影响等级。全 0 表示该县多年平均气候。</div>
      </div>
    </div>

    <div>
      <div class="card">
        <h2><span>预测结果</span><span id="baselineTag"></span></h2>
        <div style="display:flex;align-items:flex-end;justify-content:space-between">
          <div>
            <div class="big" id="predYield">—<small>kg/亩</small></div>
            <div style="font-size:13px;color:#8fb3cc;margin-top:8px" id="deviation"></div>
          </div>
          <div class="risk" id="riskTag">—</div>
        </div>
        <div class="row" style="margin-top:22px">
          <div class="kpi"><div class="k">县域常年单产</div><div class="v" id="kMean">—</div></div>
          <div class="kpi"><div class="k">预测单产</div><div class="v" id="kPred">—</div></div>
          <div class="kpi"><div class="k">气象影响得分</div><div class="v" id="kScore">—</div></div>
        </div>
        <div style="margin-top:22px">
          <div style="font-size:13px;color:#8fb3cc;margin-bottom:4px">情景偏离度（相对该县多年平均）</div>
          <div id="factors"></div>
        </div>
      </div>

      <div class="card" style="margin-top:20px">
        <h2><span>各县基准状态</span><span style="color:#64829b;font-size:12px">按气象影响得分降序</span></h2>
        <div id="tableWrap"></div>
      </div>
    </div>
  </div>
  <footer>河南人工智能创新大赛 · 大数据应用赛 · 模型训练组</footer>
</div>

<script>
const MODEL = __MODEL_JSON__;
const RISK_LEVELS = __RISK_JSON__;

function predictRF(x){
  const F = MODEL.features;
  let sum = 0;
  for(const tr of MODEL.trees){
    let i = 0;
    while(tr.cl[i] !== -1){
      const fi = tr.f[i];
      i = (x[fi] <= tr.t[i]) ? tr.cl[i] : tr.cr[i];
    }
    sum += tr.v[i];
  }
  return sum / MODEL.trees.length;
}

function riskScore(pred, mean, hotDry, cold){
  let s = 0;
  const loss = Math.max(0, (1 - pred / mean)) * 100;
  s += Math.min(40, loss * 4);
  s += Math.min(25, hotDry * 2.5);
  s += Math.min(20, cold * 1);
  return Math.min(100, Math.round(s * 10) / 10);
}

function levelOf(score){
  for(const l of RISK_LEVELS){ if(score >= l.min && score < l.max) return l.name; }
  return RISK_LEVELS[RISK_LEVELS.length-1].name;
}

function colorOf(level){
  if(level.indexOf('高') === 0) return '#ef5350';
  if(level.indexOf('偏高') === 0) return '#ffa726';
  if(level.indexOf('中') === 0) return '#ffd54f';
  return '#66bb6a';
}

const stats = MODEL.county_stats;
const names = Object.keys(stats).sort((a,b)=>a.localeCompare(b,'zh'));

function buildFeatures(county, p34, dT, dhw, cold){
  const base = stats[county].features;
  const x = MODEL.features.map(f => base[f]);
  const idx = {};
  MODEL.features.forEach((f,i)=>idx[f]=i);
  const scale = 1 + p34/100;
  x[idx['season_precip']] = base['season_precip'] * scale;
  x[idx['season_precip_34']] = base['season_precip_34'] * scale;
  x[idx['season_precip_56']] = base['season_precip_56'] * (1 + p34/200);
  x[idx['precip_anom']] = base['precip_anom'] + p34/100;
  x[idx['season_precip_norm']] = base['season_precip_norm'] + p34/100;
  x[idx['spi_12']] = base['spi_12'] + p34/100 * 0.8;
  x[idx['season_tmax_56']] = base['season_tmax_56'] + dT;
  x[idx['season_tmax']] = base['season_tmax'] + dT/2;
  x[idx['season_tmax_34']] = base['season_tmax_34'] + dT/2;
  x[idx['hot_days_56']] = base['hot_days_56'] + Math.max(0, Math.round(dT*2)) + Math.round(dhw/2);
  x[idx['hot_dry_days']] = base['hot_dry_days'] + dhw;
  x[idx['hot_days']] = base['hot_days'] + Math.max(0, Math.round(dT*2)) + Math.round(dhw/2);
  x[idx['cold_days_34']] = base['cold_days_34'] + cold;
  x[idx['cold_days']] = base['cold_days'] + cold;
  return x;
}

function renderFactors(county, p34, dT, dhw, cold){
  const items = [
    {name:'春季降水', delta: p34/100, unit:'%', good: p34>=0},
    {name:'灌浆期温度', delta: dT/3, unit:'℃', good: dT<=0},
    {name:'干热风日数', delta: dhw/10, unit:'天', good: dhw===0},
    {name:'倒春寒日数', delta: cold/20, unit:'天', good: cold===0},
  ];
  const html = items.map(it=>{
    const w = Math.min(100, Math.abs(it.delta)*100);
    const c = it.good ? '#66bb6a' : '#ef5350';
    return `<div class="bar-item">
      <div class="bar-head"><span>${it.name}</span>
      <span>${it.delta===0?'基准':(it.delta>0?'+':'')+Math.round(w)+'% 偏离'}</span></div>
      <div class="bar-bg"><div class="bar-fill" style="width:${w}%;background:${c}"></div></div>
    </div>`;
  }).join('');
  document.getElementById('factors').innerHTML = html;
}

function update(){
  const county = document.getElementById('county').value;
  const p34 = +document.getElementById('p34').value;
  const dT = +document.getElementById('dt').value;
  const dhw = +document.getElementById('dhw').value;
  const cold = +document.getElementById('cold').value;

  document.getElementById('p34v').textContent = p34 + ' %';
  document.getElementById('dtv').textContent = dT + ' ℃';
  document.getElementById('dhwv').textContent = dhw + ' 天';
  document.getElementById('coldv').textContent = cold + ' 天';

  const mean = stats[county].mean_yield;
  const x = buildFeatures(county, p34, dT, dhw, cold);
  const norm = predictRF(x);
  const pred = norm * mean;
  const dev = (norm - 1) * 100;
  const score = riskScore(pred, mean, stats[county].features['hot_dry_days'] + dhw,
                          stats[county].features['cold_days_34'] + cold);
  const level = levelOf(score);

  document.getElementById('predYield').innerHTML = pred.toFixed(1) + '<small>kg/亩</small>';
  document.getElementById('deviation').textContent =
    '较常年 ' + (dev>=0?'+':'') + dev.toFixed(1) + '%';
  document.getElementById('kMean').textContent = mean.toFixed(1);
  document.getElementById('kPred').textContent = pred.toFixed(1);
  document.getElementById('kScore').textContent = score.toFixed(1);
  const t = document.getElementById('riskTag');
  t.textContent = level;
  t.style.background = colorOf(level) + '22';
  t.style.color = colorOf(level);
  t.style.border = '1px solid ' + colorOf(level) + '66';
  document.getElementById('baselineTag').textContent = county + ' · 常年 ' + mean.toFixed(0) + ' kg/亩';

  renderFactors(county, p34, dT, dhw, cold);
}

function renderTable(){
  const rows = names.map(c=>{
    const x = buildFeatures(c, 0, 0, 0, 0);
    const mean = stats[c].mean_yield;
    const pred = predictRF(x) * mean;
    const score = riskScore(pred, mean, stats[c].features['hot_dry_days'], stats[c].features['cold_days_34']);
    return {c, pred, score, level: levelOf(score)};
  }).sort((a,b)=>b.score - a.score);

  const html = '<table><tr><th>县</th><th>预测单产</th><th>气象影响得分</th><th>等级</th></tr>' +
    rows.map(r=>`<tr><td>${r.c}</td><td>${r.pred.toFixed(1)}</td><td>${r.score.toFixed(1)}</td>
      <td><span class="tag" style="background:${colorOf(r.level)}22;color:${colorOf(r.level)}">${r.level}</span></td></tr>`).join('') +
    '</table>';
  document.getElementById('tableWrap').innerHTML = html;
}

document.getElementById('county').innerHTML =
  names.map(n=>`<option value="${n}">${n}</option>`).join('');
document.getElementById('modelTag').textContent = MODEL.trees.length + ' 棵树';
['county','p34','dt','dhw','cold'].forEach(id=>{
  document.getElementById(id).addEventListener('input', update);
});
renderTable();
update();
</script>
</body>
</html>
"""


def generate() -> Path:
    model_json = DASH / "model.json"
    if not model_json.exists():
        raise FileNotFoundError(f"缺少 {model_json}，请先运行 src.models.export_model")

    payload = model_json.read_text(encoding="utf-8")
    html = HTML.replace("__MODEL_JSON__", payload) \
               .replace("__RISK_JSON__", json.dumps(RISK_LEVELS, ensure_ascii=False))

    out = DASH / "predict_dashboard.html"
    out.write_text(html, encoding="utf-8")
    print(f"预测大屏已生成 -> {out}  ({out.stat().st_size/1024:.0f} KB)")
    return out


if __name__ == "__main__":
    generate()
