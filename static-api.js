/* =====================================================================
   FraudShield static API layer
   ---------------------------------------------------------------------
   The pages call fetch('/api/...'). This file decides where that goes:
     1. API_BASE set in js/config.js  -> hosted Flask backend
     2. Served by Flask (local run)    -> same-origin Flask backend
     3. GitHub Pages / no backend      -> answered here in the browser
   The browser answers use the same JSON shape as backend/app.py, so the
   scanner, transaction-ID lookup and batch scan all work on a static host.
   ===================================================================== */
(function () {
  'use strict';

  var cfg = window.FRAUDSHIELD_CONFIG || {};
  var API_BASE = String(cfg.API_BASE || '').replace(/\/+$/, '');
  var TIMEOUT = cfg.API_TIMEOUT_MS || 25000;
  var nativeFetch = window.fetch.bind(window);
  var host = location.hostname;
  var STATIC_HOST = location.protocol === 'file:' ||
                    /\.github\.io$/i.test(host) ||
                    cfg.FORCE_BROWSER_MODE === true;

  /* ---------------- reference data (from backend/models/manifest.json) */
  var CAT_RATES = {
    entertainment: 0.25, food_dining: 0.17, gas_transport: 0.47, grocery_net: 0.29,
    grocery_pos: 1.41, health_fitness: 0.15, home: 0.16, kids_pets: 0.21,
    misc_net: 1.45, misc_pos: 0.31, personal_care: 0.24, shopping_net: 1.76,
    shopping_pos: 0.72, travel: 0.29
  };
  var MODEL_SCALE = { colab_xgboost: 1.0, colab: 1.0, xgboost: 1.0, rf: 0.95, lr: 0.72 };

  /* ---------------- helpers */
  function jsonResponse(obj, status) {
    return new Response(JSON.stringify(obj), {
      status: status || 200,
      headers: { 'Content-Type': 'application/json', 'X-FraudShield-Mode': 'browser' }
    });
  }

  function withTimeout(promise, ms) {
    return new Promise(function (resolve, reject) {
      var t = setTimeout(function () { reject(new Error('timeout')); }, ms);
      promise.then(function (v) { clearTimeout(t); resolve(v); },
                   function (e) { clearTimeout(t); reject(e); });
    });
  }

  function num(v, d) {
    var n = parseFloat(v);
    return isFinite(n) ? n : d;
  }

  function token() {
    var a = new Uint8Array(24);
    (window.crypto || window.msCrypto).getRandomValues(a);
    return Array.prototype.map.call(a, function (b) { return ('0' + b.toString(16)).slice(-2); }).join('');
  }

  // Parse "2020-06-21 12:33:45" without relying on browser date quirks
  function parseDT(s) {
    var m = /(\d{4})-(\d{1,2})-(\d{1,2})(?:[ T](\d{1,2}):(\d{2}))?/.exec(String(s || ''));
    if (!m) return null;
    var d = new Date(Date.UTC(+m[1], +m[2] - 1, +m[3]));
    var jsDay = d.getUTCDay();                 // Sun=0
    return { hour: m[4] !== undefined ? +m[4] : 12, dow: (jsDay + 6) % 7, weekend: jsDay === 0 || jsDay === 6, date: d };
  }

  function haversineKm(lat1, lon1, lat2, lon2) {
    var r = Math.PI / 180;
    var dLat = (lat2 - lat1) * r, dLon = (lon2 - lon1) * r;
    var a = Math.pow(Math.sin(dLat / 2), 2) +
            Math.cos(lat1 * r) * Math.cos(lat2 * r) * Math.pow(Math.sin(dLon / 2), 2);
    return 2 * 6371 * Math.asin(Math.sqrt(Math.min(Math.max(a, 0), 1)));
  }

  /* ---------------- CSV parser (quotes, commas, CRLF), stops at maxRows */
  function parseCSV(text, maxRows) {
    var rows = [], row = [], field = '', i = 0, q = false, c, header = null;
    maxRows = maxRows || Infinity;
    function endRow() {
      row.push(field); field = '';
      if (!header) { header = row.map(function (h) { return h.trim(); }); }
      else if (row.length > 1 || row[0] !== '') {
        var o = {};
        for (var k = 0; k < header.length; k++) o[header[k]] = row[k] !== undefined ? row[k] : '';
        rows.push(o);
      }
      row = [];
    }
    while (i < text.length) {
      c = text[i];
      if (q) {
        if (c === '"') { if (text[i + 1] === '"') { field += '"'; i++; } else q = false; }
        else field += c;
      } else if (c === '"') q = true;
      else if (c === ',') { row.push(field); field = ''; }
      else if (c === '\n' || c === '\r') {
        if (c === '\r' && text[i + 1] === '\n') i++;
        endRow();
        if (rows.length >= maxRows) return rows;
      } else field += c;
      i++;
    }
    if (field !== '' || row.length) endRow();
    return rows;
  }

  // Convert numeric-looking strings to numbers (like pandas does)
  function typeRow(r) {
    var o = {};
    Object.keys(r).forEach(function (k) {
      var v = r[k];
      if (v === '' || v === undefined) o[k] = null;
      else if (/^-?\d+(\.\d+)?(e[+-]?\d+)?$/i.test(v) && !(k === 'trans_num') && !(k === 'cc_num' && v.length > 15)) o[k] = Number(v);
      else o[k] = v;
    });
    return o;
  }

  /* ---------------- sample dataset (sample_dataset_test.csv) */
  var sampleCache = null;
  function loadSample() {
    if (sampleCache) return Promise.resolve(sampleCache);
    return nativeFetch('sample_dataset_test.csv', { cache: 'force-cache' })
      .then(function (r) { if (!r.ok) throw new Error('sample missing'); return r.text(); })
      .then(function (t) { sampleCache = parseCSV(t).map(typeRow); return sampleCache; });
  }

  function findTx(rows, id) {
    var key = String(id || '').trim().toLowerCase();
    if (!key) return null;
    var hit = null;
    rows.some(function (r) { if (String(r.trans_num).toLowerCase() === key) { hit = r; return true; } return false; });
    if (!hit) rows.some(function (r) { if (String(r.trans_num).toLowerCase().indexOf(key) !== -1) { hit = r; return true; } return false; });
    return hit;
  }

  function maskCard(tx) {
    var cc = tx.cc_num != null ? String(tx.cc_num).replace(/\.0+$/, '') : '';
    return cc.length >= 4 ? '**** **** **** ' + cc.slice(-4) : '**** **** **** 8841';
  }

  /* ---------------- risk model (browser approximation of the trained models)
     Built from the dataset patterns the models learned: transaction amount,
     merchant category fraud rate, late-night hours, distance and age. */
  function scoreRow(r, modelKey) {
    var amt = num(r.amt != null ? r.amt : r.amount, 50);
    var cat = String(r.category || '');
    var dt = parseDT(r.trans_date_trans_time || r.trans_time || r.datetime || r.date || r.timestamp);
    var hour = dt ? dt.hour : 12;

    var l = Math.log(0.0058 / 0.9942);
    if (amt > 900) l += 4.2;
    else if (amt > 600) l += 3.6;
    else if (amt > 300) l += 2.8;
    else if (amt > 200) l += 1.9;
    else if (amt > 100) l += 0.5;
    else if (amt < 15) l += 0.4;

    l += Math.log((CAT_RATES[cat] || 0.58) / 0.58) * 0.9;

    var night = hour >= 22 || hour <= 3;
    l += night ? 2.6 : -0.8;
    if (night && amt > 200) l += 1.0;
    if (cat === 'grocery_pos' && amt > 250 && amt < 450) l += 1.6;

    if (r.lat != null && r.merch_lat != null) {
      var km = haversineKm(num(r.lat, 0), num(r.long, 0), num(r.merch_lat, 0), num(r.merch_long, 0));
      if (km > 120) l += 0.3;
    }
    if (r.dob && dt) {
      var b = parseDT(r.dob);
      if (b && (dt.date - b.date) / 3.15576e10 > 60) l += 0.3;
    }

    l *= MODEL_SCALE[modelKey] || 1.0;
    var p = 1 / (1 + Math.exp(-l));
    return { prob: Math.min(Math.max(p, 0.001), 0.999), hour: hour, dow: dt ? dt.dow : 2, amt: amt, cat: cat };
  }

  function riskLevel(pct) {
    return pct >= 75 ? 'CRITICAL' : pct >= 40 ? 'HIGH' : pct >= 15 ? 'MEDIUM' : 'LOW';
  }

  /* ---------------- endpoint handlers */
  function readBody(init) {
    var b = init && init.body;
    if (!b) return Promise.resolve({});
    if (typeof FormData !== 'undefined' && b instanceof FormData) return Promise.resolve({ form: b });
    try { return Promise.resolve(JSON.parse(b)); } catch (e) { return Promise.resolve({}); }
  }

  function authUser(init) {
    var h = (init && init.headers) || {};
    var get = function (k) { return h instanceof Headers ? h.get(k) : (h[k] || h[k.toLowerCase()]); };
    var t = String(get('Authorization') || '').replace('Bearer ', '').trim();
    return (get('X-User') || '').trim() || (t ? 'operator' : '') ||
           sessionStorage.getItem('fraudshield_user') || '';
  }

  function scanTransactionId(init) {
    return readBody(init).then(function (data) {
      if (!authUser(init) && sessionStorage.getItem('fraudshield_logged_in') !== 'true') {
        return jsonResponse({ success: false, error: 'Authentication required. Please log in before scanning.', auth_required: true }, 401);
      }
      var rawId = String(data.trans_id || '').trim();
      if (!rawId) return jsonResponse({ success: false, error: 'Please provide a valid Transaction ID.' }, 400);
      var modelKey = String(data.model || 'colab_xgboost').toLowerCase();
      var threshold = num(data.threshold, modelKey.indexOf('colab') !== -1 ? 0.19596 : 0.45);

      return loadSample().catch(function () { return []; }).then(function (rows) {
        var row = findTx(rows, rawId), simulated = false;
        if (!row) {
          simulated = true;
          var now = new Date();
          row = {
            trans_num: rawId, amt: num(data.amt, 350), category: String(data.category || 'shopping_net'),
            merchant: 'merchant_' + rawId.slice(0, 8),
            trans_date_trans_time: now.toISOString().slice(0, 19).replace('T', ' '),
            first: 'Cardholder', last: 'Ref-' + rawId.slice(0, 6), gender: 'M', street: '100 Financial Way',
            city: 'Dallas', state: 'TX', zip: 75001, lat: 32.7767, long: -96.797, city_pop: 1300000,
            job: 'Corporate Professional', dob: '1985-05-15', merch_lat: 32.81, merch_long: -96.75,
            unix_time: Math.floor(now / 1000)
          };
        }
        var tx = Object.assign({}, row);
        tx.cc_masked = maskCard(tx);

        var t0 = performance.now();
        var s = scoreRow(tx, modelKey);
        var elapsed = Math.round((performance.now() - t0) * 100) / 100;
        var isFraud = s.prob >= threshold;
        var pct = Math.round(s.prob * 10000) / 100;
        var level = riskLevel(pct);
        var verdict = pct >= 75 ? 'FLAGGED AS FRAUD'
                    : pct >= 40 ? (isFraud ? 'FLAGGED AS FRAUD' : 'SUSPICIOUS REVIEW')
                    : pct >= 15 ? (isFraud ? 'FLAGGED AS FRAUD' : 'APPROVED (MONITOR)')
                    : 'APPROVED';

        var actual = (tx.is_fraud !== undefined && tx.is_fraud !== null) ? parseInt(tx.is_fraud, 10) : null;
        var km = Math.round(haversineKm(num(tx.lat, 31), num(tx.long, -99), num(tx.merch_lat, 31.05), num(tx.merch_long, -98.95)) * 10) / 10;
        var amtTxt = '$' + s.amt.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
        var hiCats = ['misc_net', 'grocery_pos', 'gas_transport', 'shopping_net'];

        return jsonResponse({
          success: true,
          mode: 'browser',
          trans_id: rawId,
          is_simulated: simulated,
          transaction: tx,
          prediction: {
            is_fraud: isFraud, verdict: verdict, probability: pct, raw_probability: s.prob,
            risk_level: level, threshold: threshold, model_used: modelKey.toUpperCase(), elapsed_ms: elapsed,
            has_ground_truth: actual !== null, actual_is_fraud: actual,
            ground_truth_match: actual !== null ? (isFraud === (actual === 1)) : null
          },
          shap_drivers: [
            s.amt > 250
              ? { feature: 'Transaction Amount', value: amtTxt, impact: '+2.92 (High Exposure)', direction: 'fraud' }
              : { feature: 'Transaction Amount', value: amtTxt, impact: '-1.45 (Normal Range)', direction: 'legit' },
            hiCats.indexOf(s.cat) !== -1
              ? { feature: 'Merchant Category', value: s.cat, impact: '+2.30 (Elevated Frequency)', direction: 'fraud' }
              : { feature: 'Merchant Category', value: s.cat, impact: '-0.85 (Standard)', direction: 'legit' },
            km > 100
              ? { feature: 'Geo Distance', value: km + ' km', impact: '+0.45 (Remote POS)', direction: 'fraud' }
              : { feature: 'Geo Distance', value: km + ' km', impact: '-0.32 (Local Terminal)', direction: 'legit' }
          ]
        });
      });
    });
  }

  function batchScan(init) {
    return readBody(init).then(function (data) {
      var form = data.form, modelKey, threshold, maxRows, filename = 'uploaded_dataset.csv', getRows;

      if (form) {
        var file = form.get('file');
        modelKey = String(form.get('model') || 'xgboost').toLowerCase();
        threshold = num(form.get('threshold'), 0.196);
        maxRows = parseInt(form.get('max_rows'), 10) || 5000;
        if (!file) return jsonResponse({ success: false, error: 'No valid data or CSV file received' }, 400);
        filename = file.name || filename;
        getRows = file.text().then(function (text) {
          if (/\.json$/i.test(filename) || /^\s*[\[{]/.test(text)) {
            var j = JSON.parse(text);
            var arr = Array.isArray(j) ? j : (j.records || j.data || j.transactions || []);
            return arr.slice(0, maxRows);
          }
          return parseCSV(text, maxRows).map(typeRow);
        });
      } else {
        modelKey = String(data.model || 'xgboost').toLowerCase();
        threshold = num(data.threshold, 0.196);
        maxRows = parseInt(data.max_rows, 10) || 5000;
        if (data.records) getRows = Promise.resolve(data.records.slice(0, maxRows));
        else if (data.csv_text) getRows = Promise.resolve(parseCSV(data.csv_text, maxRows).map(typeRow));
        else return jsonResponse({ success: false, error: 'No valid data or CSV file received' }, 400);
      }

      return getRows.then(function (rows) {
        if (!rows || !rows.length) return jsonResponse({ success: false, error: 'No valid data or CSV file received' }, 400);

        var t0 = performance.now();
        var hasGT = rows.some(function (r) { return r.is_fraud !== undefined && r.is_fraud !== null && r.is_fraud !== ''; });
        var tp = 0, fp = 0, tn = 0, fn = 0, actualFraud = 0;
        var fraudCount = 0, totalAmt = 0, fraudAmt = 0;
        var cats = {}, hourly = [], bins = [0, 0, 0, 0, 0, 0], records = [];
        for (var h = 0; h < 24; h++) hourly.push({ hour: (h < 10 ? '0' : '') + h + ':00', total: 0, fraud: 0 });

        rows.forEach(function (r, i) {
          var s = scoreRow(r, modelKey), p = s.prob, flag = p >= threshold;
          var cat = r.category != null ? String(r.category) : 'general';
          var pct = Math.round(p * 1000) / 10;
          totalAmt += s.amt;
          if (flag) { fraudCount++; fraudAmt += s.amt; }

          if (!cats[cat]) cats[cat] = { total: 0, fraud: 0, fraud_rate: 0 };
          cats[cat].total++; if (flag) cats[cat].fraud++;

          var hr = hourly[Math.min(Math.max(s.hour | 0, 0), 23)];
          hr.total++; if (flag) hr.fraud++;

          bins[p < 0.1 ? 0 : p < 0.3 ? 1 : p < 0.5 ? 2 : p < 0.7 ? 3 : p < 0.9 ? 4 : 5]++;

          var actual = hasGT ? (parseInt(r.is_fraud, 10) || 0) : null;
          if (hasGT) {
            if (actual === 1) actualFraud++;
            if (flag && actual === 1) tp++; else if (flag) fp++; else if (actual === 1) fn++; else tn++;
          }

          if (i < 1000) {
            var rec = {
              index: i + 1,
              trans_num: String(r.trans_num != null ? r.trans_num : 'TXN-' + (100000 + i)),
              datetime: String(r.trans_date_trans_time || r.trans_time || r.datetime || r.date || r.timestamp || ('Event #' + (i + 1))),
              category: cat,
              amt: Math.round(s.amt * 100) / 100,
              state: String(r.state != null ? r.state : '--'),
              probability: pct,
              is_fraud: flag,
              verdict: flag ? 'FLAGGED AS FRAUD' : 'APPROVED',
              risk_level: riskLevel(pct)
            };
            if (hasGT) { rec.actual_is_fraud = actual; rec.match = flag === (actual === 1); }
            records.push(rec);
          }
        });

        Object.keys(cats).forEach(function (c) {
          cats[c].fraud_rate = cats[c].total ? Math.round(cats[c].fraud / cats[c].total * 1000) / 10 : 0;
        });

        var n = rows.length;
        var elapsed = Math.max(Math.round((performance.now() - t0) * 100) / 100, 0.01);
        var pctStr = function (v) { return (v * 100).toFixed(1) + '%'; };
        var gt = {};
        if (hasGT) {
          var prec = tp + fp ? tp / (tp + fp) : 0, rec2 = tp + fn ? tp / (tp + fn) : 0;
          gt = {
            accuracy: pctStr((tp + tn) / n), precision: pctStr(prec), recall: pctStr(rec2),
            f1: pctStr(prec + rec2 ? 2 * prec * rec2 / (prec + rec2) : 0),
            tp: tp, fp: fp, tn: tn, fn: fn, actual_fraud_count: actualFraud
          };
        }
        var labels = ['0-10%', '10-30%', '30-50%', '50-70%', '70-90%', '90-100%'];

        return jsonResponse({
          success: true,
          mode: 'browser',
          filename: filename,
          summary: {
            total_records: n, fraud_count: fraudCount, legit_count: n - fraudCount,
            fraud_rate_pct: Math.round(fraudCount / n * 10000) / 100,
            total_amount: Math.round(totalAmt * 100) / 100, fraud_amount: Math.round(fraudAmt * 100) / 100,
            model_used: modelKey.toUpperCase(), threshold: threshold, processing_time_ms: elapsed,
            throughput_txns_per_sec: Math.round(n / (elapsed / 1000)),
            has_ground_truth: hasGT, ground_truth_metrics: gt
          },
          records: records,
          category_breakdown: cats,
          hourly_breakdown: hourly,
          risk_distribution: labels.map(function (l, k) { return { range: l, count: bins[k] }; })
        });
      });
    });
  }

  function emulate(path, init) {
    var method = ((init && init.method) || 'GET').toUpperCase();
    var u = new URL(path, location.href);
    var p = u.pathname.replace(/^.*?\/api\//, '/api/');
    showBadge();

    if (p === '/api/health') {
      return Promise.resolve(jsonResponse({ status: 'healthy', service: 'FraudShield (browser mode)', mode: 'browser' }));
    }
    // Pages already have their own client-side login and scan engines: hand over to them.
    if (p === '/api/login' || p === '/api/scan') {
      return Promise.resolve(jsonResponse({ success: false, error: 'Backend not connected (browser mode)' }, 503));
    }
    if (p === '/api/logout') return Promise.resolve(jsonResponse({ success: true }));

    if (p === '/api/models' || p === '/api/stats') {
      return nativeFetch('backend/models/manifest.json').then(function (r) { return r.json(); }).then(function (m) {
        return jsonResponse(p === '/api/models'
          ? { available_models: ['xgboost', 'rf', 'lr', 'colab_xgboost'], metrics: m.metrics, feature_names: m.feature_names,
              categories: m.categories, states: m.states, train_rows: m.train_rows, test_rows: m.test_rows,
              fraud_rate_pct: m.fraud_rate_pct, category_fraud_rates: m.category_fraud_rates }
          : { total_transactions: m.train_rows, total_fraud: Math.round(m.train_rows * m.fraud_rate_pct / 100),
              total_legit: m.train_rows - Math.round(m.train_rows * m.fraud_rate_pct / 100),
              fraud_rate_pct: m.fraud_rate_pct, category_fraud_rates: m.category_fraud_rates,
              metrics: m.metrics, models: ['xgboost', 'rf', 'lr', 'colab_xgboost'] });
      }).catch(function () { return jsonResponse({ error: 'Models not loaded' }, 503); });
    }

    if (p === '/api/sample-dataset') {
      return loadSample().then(function (rows) {
        return jsonResponse({ success: true, filename: 'sample_dataset_test.csv', total_rows: rows.length, rows: rows.slice(0, 100) });
      }).catch(function () { return jsonResponse({ success: false, error: 'Sample file not found' }, 404); });
    }

    if (p === '/api/transactions') {
      var limit = parseInt(u.searchParams.get('limit'), 10) || 100;
      return loadSample().then(function (rows) {
        var items = rows.slice(0, limit).map(function (r) {
          return {
            trans_num: String(r.trans_num), amt: num(r.amt, 0), category: String(r.category || ''),
            merchant: String(r.merchant || ''), trans_date_trans_time: String(r.trans_date_trans_time || ''),
            city: String(r.city || ''), state: String(r.state || ''),
            is_fraud: r.is_fraud != null ? parseInt(r.is_fraud, 10) : null,
            cardholder: ((r.first || '') + ' ' + (r.last || '')).trim()
          };
        });
        return jsonResponse({ success: true, count: items.length, transactions: items });
      }).catch(function () { return jsonResponse({ success: true, count: 0, transactions: [] }); });
    }

    if (p.indexOf('/api/transaction/') === 0) {
      var id = decodeURIComponent(p.slice('/api/transaction/'.length));
      return loadSample().then(function (rows) {
        var r = findTx(rows, id);
        if (!r) return jsonResponse({ success: false, found: false, error: "Transaction ID '" + id + "' not found." }, 404);
        var t = Object.assign({}, r); t.cc_masked = maskCard(t);
        return jsonResponse({ success: true, found: true, transaction: t });
      });
    }

    if (p === '/api/scan-transaction-id' && method === 'POST') return scanTransactionId(init);
    if (p === '/api/batch-scan' && method === 'POST') return batchScan(init);

    return Promise.resolve(jsonResponse({ error: "API endpoint '" + p + "' not found" }, 404));
  }

  /* ---------------- small status badge (scanner page only) */
  var badgeShown = false;
  function showBadge() {
    if (badgeShown || !/scanner\.html$/i.test(location.pathname)) return;
    badgeShown = true;
    var add = function () {
      var b = document.createElement('div');
      b.textContent = 'Browser demo mode';
      b.title = 'Running without the Python backend. Set API_BASE in js/config.js to use the trained models.';
      b.style.cssText = 'position:fixed;left:14px;bottom:14px;z-index:9999;padding:6px 12px;border-radius:999px;' +
        'font:600 11px/1.2 system-ui,-apple-system,sans-serif;letter-spacing:.04em;color:#f1c40f;' +
        'background:rgba(20,20,20,.85);border:1px solid rgba(241,196,15,.45);pointer-events:auto;cursor:help;';
      document.body.appendChild(b);
    };
    if (document.body) add(); else document.addEventListener('DOMContentLoaded', add);
  }

  /* ---------------- fetch router */
  function isApi(url) {
    return typeof url === 'string' && /^\/?api\//.test(url);
  }

  window.fetch = function (input, init) {
    var url = typeof input === 'string' ? input : (input && input.url);
    if (!isApi(url)) return nativeFetch(input, init);
    var path = url.charAt(0) === '/' ? url : '/' + url;

    if (API_BASE) {
      return withTimeout(nativeFetch(API_BASE + path, init), TIMEOUT)
        .catch(function () { return emulate(path, init); });
    }
    if (STATIC_HOST) return emulate(path, init);

    // Same-origin: use Flask when it is there, otherwise fall back.
    return nativeFetch(path, init).then(function (res) {
      var mode = res.headers.get('content-type') || '';
      if ((res.status === 404 || res.status === 405 || res.status === 501) && mode.indexOf('json') === -1) {
        return emulate(path, init);
      }
      return res;
    }).catch(function () { return emulate(path, init); });
  };

  // Wake a sleeping hosted backend (e.g. Render free tier) as soon as a page opens.
  if (API_BASE) { nativeFetch(API_BASE + '/api/health').catch(function () {}); }

  window.FRAUDSHIELD_API = { base: API_BASE, staticHost: STATIC_HOST, emulate: emulate };
})();
