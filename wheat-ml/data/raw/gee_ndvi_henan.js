// ============================================================
// 河南县级 NDVI 抓取（MODIS MOD13Q1，250m，16 天合成）
// 用法：粘贴到 code.earthengine.google.com 的 Script 里，改 ASSET 后点 Run
// ============================================================

// ---------------- 配置区（只改这两行） ----------------
var ASSET = 'projects/ee-你的用户名/assets/henan_counties_wkt'; // 上传 CSV 后得到的 asset 路径
var YEARS = [2018, 2013];   // 先跑这两年验证；验证通过后改成 [2010,2011,...,2024] 全量
// -----------------------------------------------------

var counties = ee.FeatureCollection(ASSET);
var col = ee.ImageCollection('MODIS/061/MOD13Q1').select(['NDVI', 'SummaryQA']);

YEARS.forEach(function (y) {
  // 小麦关键期 3/1 — 6/10，只取这段能大幅减少导出量
  var sub = col.filterDate(ee.Date.fromYMD(y, 3, 1), ee.Date.fromYMD(y, 6, 11));

  var feats = sub.map(function (img) {
    // NDVI 原始为 int16，×0.0001 才是真值
    var nd = img.select('NDVI').multiply(0.0001);
    // SummaryQA: 0=good 1=marginal 2=snow 3=cloudy；只保留 0/1
    nd = nd.updateMask(img.select('SummaryQA').lte(1));
    // 再滤掉 fill 值（原始 -3000 → -0.3）
    nd = nd.updateMask(nd.gte(0));

    var stat = nd.reduceRegions({
      collection: counties,
      reducer: ee.Reducer.mean()
        .combine({ reducer2: ee.Reducer.max(), sharedInputs: true })
        .combine({ reducer2: ee.Reducer.count(), sharedInputs: true }),
      scale: 250,
      tileScale: 4
    });

    return stat.map(function (f) {
      return f.set('image_date', img.date().format('YYYY-MM-DD'));
    });
  }).flatten();

  Export.table.toDrive({
    collection: ee.FeatureCollection(feats),
    description: 'henan_ndvi_' + y,
    folder: 'henan_ndvi',
    fileFormat: 'CSV',
    selectors: ['name', 'image_date', 'NDVI_mean', 'NDVI_max', 'NDVI_count']
  });
});

// 点 Run 后，右上角 Tasks 面板会出现 henan_ndvi_2018 等任务，逐个点 RUN。
// 完成后去 Google Drive 的 henan_ndvi 文件夹下载 CSV。
// 全量 15 年会生成 15 个任务，建议先跑两年确认没问题再全开。
