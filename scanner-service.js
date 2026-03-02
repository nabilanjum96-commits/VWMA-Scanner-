// ============================================================================
// VWMA CLOUD SCANNER — GitHub Actions Edition
//
// Runs as a single scan pass, sends Telegram alerts, then exits.
// Designed for cron-triggered GitHub Actions (every 5 minutes).
//
// Env vars (set as GitHub Secrets):
//   TELEGRAM_BOT_TOKEN   — Telegram bot API token
//   TELEGRAM_CHAT_ID     — Chat ID to send alerts to
//
// Optional env vars:
//   TIMEFRAMES           — comma-separated (default: 5m)
//   SYMBOLS              — comma-separated (default: all 14 core assets)
//   COOLDOWN_MINUTES     — min between repeat alerts (default: 60)
// ============================================================================

const ccxt = require('ccxt');
const TelegramBot = require('node-telegram-bot-api');
const fs = require('fs');
const path = require('path');

// ── Configuration ─────────────────────────────────────────────────────

const TELEGRAM_BOT_TOKEN = process.env.TELEGRAM_BOT_TOKEN;
const TELEGRAM_CHAT_ID = process.env.TELEGRAM_CHAT_ID;

if (!TELEGRAM_BOT_TOKEN || !TELEGRAM_CHAT_ID) {
  console.error('ERROR: TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set');
  process.exit(1);
}

const CORE_ASSETS = [
  'BTC', 'ETH', 'BNB', 'XRP', 'SOL',
  'DOGE', 'ADA', 'LINK', 'AVAX', 'HBAR',
  'LTC', 'DOT', 'ETC', 'ATOM'
];

const CONFIG = {
  vwmaPeriod: 100,
  sigmaAlert: 3,
  sigmaOuter: 6,
  slopeShort: 10,
  slopeLong: 50,
  bwHistoryBars: 500,
  cooldownMinutes: parseInt(process.env.COOLDOWN_MINUTES) || 60,
  symbols: process.env.SYMBOLS ? process.env.SYMBOLS.split(',').map(s => s.trim()) : CORE_ASSETS,
  timeframes: process.env.TIMEFRAMES ? process.env.TIMEFRAMES.split(',').map(s => s.trim()) : ['5m'],
  quote: 'USDT',
};

const BARS_TO_FETCH = 700;
const COOLDOWN_FILE = path.join(__dirname, 'cooldowns.json');

// ── Exchange + Telegram ───────────────────────────────────────────────

const exchange = new ccxt.binanceusdm({ enableRateLimit: true });
const bot = new TelegramBot(TELEGRAM_BOT_TOKEN, { polling: false });

// ── Cooldown persistence (survives between GitHub Actions runs) ───────

function loadCooldowns() {
  try {
    if (fs.existsSync(COOLDOWN_FILE)) {
      return JSON.parse(fs.readFileSync(COOLDOWN_FILE, 'utf8'));
    }
  } catch (e) {}
  return {};
}

function saveCooldowns(cooldowns) {
  fs.writeFileSync(COOLDOWN_FILE, JSON.stringify(cooldowns));
}

function isOnCooldown(cooldowns, symbol, timeframe, direction) {
  const key = `${symbol}_${timeframe}_${direction}`;
  const last = cooldowns[key];
  if (!last) return false;
  return (Date.now() - last) < CONFIG.cooldownMinutes * 60 * 1000;
}

function setCooldown(cooldowns, symbol, timeframe, direction) {
  cooldowns[`${symbol}_${timeframe}_${direction}`] = Date.now();
  // Prune expired cooldowns
  const cutoff = Date.now() - CONFIG.cooldownMinutes * 60 * 1000;
  for (const [k, v] of Object.entries(cooldowns)) {
    if (v < cutoff) delete cooldowns[k];
  }
}

// ── Math helpers ──────────────────────────────────────────────────────

function calcVWMA(bars) {
  let sumPV = 0, sumV = 0;
  for (let i = 0; i < bars.length; i++) {
    const close = bars[i][4];
    const vol = bars[i][5] || 0;
    sumPV += close * vol;
    sumV += vol;
  }
  return sumV > 0 ? sumPV / sumV : null;
}

function calcSigma(bars, mean) {
  let sumSq = 0;
  for (let i = 0; i < bars.length; i++) {
    const dev = bars[i][4] - mean;
    sumSq += dev * dev;
  }
  return Math.sqrt(sumSq / bars.length);
}

function calcRSI(bars, period = 14) {
  if (bars.length < period + 1) return null;
  let avgGain = 0, avgLoss = 0;
  for (let i = 1; i <= period; i++) {
    const change = bars[i][4] - bars[i - 1][4];
    if (change > 0) avgGain += change;
    else avgLoss -= change;
  }
  avgGain /= period;
  avgLoss /= period;
  for (let i = period + 1; i < bars.length; i++) {
    const change = bars[i][4] - bars[i - 1][4];
    avgGain = (avgGain * (period - 1) + (change > 0 ? change : 0)) / period;
    avgLoss = (avgLoss * (period - 1) + (change < 0 ? -change : 0)) / period;
  }
  if (avgLoss === 0) return 100;
  const rs = avgGain / avgLoss;
  return Math.round((100 - 100 / (1 + rs)) * 100) / 100;
}

function rankPercentile(val, history) {
  if (val === null || history.length < 10) return null;
  const below = history.filter(v => v <= val).length;
  return Math.round((below / history.length) * 100);
}

// ── Core analysis ─────────────────────────────────────────────────────

function analyze(ohlcv) {
  const period = CONFIG.vwmaPeriod;
  if (!ohlcv || ohlcv.length < period + 2) return null;

  const n = ohlcv.length;
  const window = ohlcv.slice(n - period, n);
  const vwma = calcVWMA(window);
  if (vwma === null) return null;

  const sigma = calcSigma(window, vwma);
  if (sigma === 0) return null;

  const currentPrice = ohlcv[n - 1][4];
  const priceZScore = (currentPrice - vwma) / sigma;

  const upper3s = vwma + CONFIG.sigmaAlert * sigma;
  const lower3s = vwma - CONFIG.sigmaAlert * sigma;
  const upper6s = vwma + CONFIG.sigmaOuter * sigma;
  const lower6s = vwma - CONFIG.sigmaOuter * sigma;

  const spread3to6 = (CONFIG.sigmaOuter - CONFIG.sigmaAlert) * sigma;
  const spreadPct = (spread3to6 / vwma) * 100;

  // Historical percentiles
  const historicalSpreads = [];
  const historicalSlopeShort = [];
  const historicalSlopeLong = [];

  for (let t = period; t < n; t++) {
    const w = ohlcv.slice(t - period, t);
    const v = calcVWMA(w);
    if (v === null) continue;
    const s = calcSigma(w, v);
    historicalSpreads.push(((CONFIG.sigmaOuter - CONFIG.sigmaAlert) * s / v) * 100);

    if (t >= period + CONFIG.slopeShort) {
      const wPrev = ohlcv.slice(t - period - CONFIG.slopeShort, t - CONFIG.slopeShort);
      const vPrev = calcVWMA(wPrev);
      if (vPrev !== null && vPrev > 0) historicalSlopeShort.push(((v - vPrev) / vPrev) * 100);
    }
    if (t >= period + CONFIG.slopeLong) {
      const wPrev = ohlcv.slice(t - period - CONFIG.slopeLong, t - CONFIG.slopeLong);
      const vPrev = calcVWMA(wPrev);
      if (vPrev !== null && vPrev > 0) historicalSlopeLong.push(((v - vPrev) / vPrev) * 100);
    }
  }

  const spreadPercentile = rankPercentile(spreadPct, historicalSpreads.slice(-CONFIG.bwHistoryBars));

  let slopeShort = null, slopeLong = null;
  if (n >= period + CONFIG.slopeShort) {
    const prev = calcVWMA(ohlcv.slice(n - period - CONFIG.slopeShort, n - CONFIG.slopeShort));
    if (prev !== null && prev > 0) slopeShort = ((vwma - prev) / prev) * 100;
  }
  if (n >= period + CONFIG.slopeLong) {
    const prev = calcVWMA(ohlcv.slice(n - period - CONFIG.slopeLong, n - CONFIG.slopeLong));
    if (prev !== null && prev > 0) slopeLong = ((vwma - prev) / prev) * 100;
  }

  const slopeShortPctile = rankPercentile(slopeShort, historicalSlopeShort);
  const slopeLongPctile = rankPercentile(slopeLong, historicalSlopeLong);
  const rsi = calcRSI(ohlcv.slice(-100));

  const pctInto6sUpper = Math.max(0, ((currentPrice - upper3s) / spread3to6) * 100);
  const pctInto6sLower = Math.max(0, ((lower3s - currentPrice) / spread3to6) * 100);

  return {
    vwma, sigma, currentPrice,
    priceZScore: Math.round(priceZScore * 100) / 100,
    rsi,
    upper3s, lower3s, upper6s, lower6s,
    spread3to6,
    spreadPct: Math.round(spreadPct * 100) / 100,
    spreadPercentile,
    slopeShort: slopeShort !== null ? Math.round(slopeShort * 1000) / 1000 : null,
    slopeLong: slopeLong !== null ? Math.round(slopeLong * 1000) / 1000 : null,
    slopeShortPctile,
    slopeLongPctile,
    pctInto6sUpper: Math.round(pctInto6sUpper * 10) / 10,
    pctInto6sLower: Math.round(pctInto6sLower * 10) / 10,
  };
}

// ── Alert detection ───────────────────────────────────────────────────

function shouldAlert(bandData) {
  if (!bandData) return null;
  if (bandData.currentPrice >= bandData.upper3s) {
    return { direction: 'UPPER', side: 'SHORT', zScore: bandData.priceZScore };
  }
  if (bandData.currentPrice <= bandData.lower3s) {
    return { direction: 'LOWER', side: 'LONG', zScore: bandData.priceZScore };
  }
  return null;
}

// ── Telegram message ──────────────────────────────────────────────────

function formatAlert(symbol, timeframe, data, trigger) {
  const fmt = (v) => v > 100 ? `$${v.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}` : `$${v.toFixed(4)}`;
  const isUpper = trigger.direction === 'UPPER';
  const bandPrice = isUpper ? data.upper3s : data.lower3s;
  const sixSigmaPrice = isUpper ? data.upper6s : data.lower6s;
  const line = '\u2501'.repeat(26);

  // Price/VWMA delta
  const vwmaDelta = ((data.currentPrice - data.vwma) / data.vwma) * 100;
  const vwmaDeltaStr = `${vwmaDelta >= 0 ? '+' : ''}${vwmaDelta.toFixed(2)}%`;

  // Distance from price to 6σ band
  const distTo6s = ((sixSigmaPrice - data.currentPrice) / data.currentPrice) * 100;
  const distTo6sStr = `${distTo6s >= 0 ? '+' : ''}${distTo6s.toFixed(2)}%`;

  // Ordinal percentile helper
  const ordinal = (n) => {
    if (n == null) return '\u2014';
    const s = ['th', 'st', 'nd', 'rd'];
    const v = n % 100;
    return `${n}${s[(v - 20) % 10] || s[v] || s[0]}`;
  };

  // Slope formatting
  const fmtSlope = (v) => {
    if (v === null) return '\u2014';
    return `${v >= 0 ? '+' : ''}${v.toFixed(2)}%`;
  };

  let msg = `${line}\n`;
  msg += `\u26A0\uFE0F <b>VWMA 3\u03C3 ALERT</b> \u2014 ${symbol} ${timeframe}\n`;
  msg += `${line}\n\n`;

  msg += `<code>PRICE      ${fmt(data.currentPrice)}</code>\n`;
  msg += `<code>VWMA       ${fmt(data.vwma)}</code>    (${vwmaDeltaStr})\n`;
  msg += `<code>Z-SCORE    ${data.priceZScore}\u03C3</code>\n`;
  if (data.rsi != null) {
    msg += `<code>RSI(14)    ${data.rsi}</code>\n`;
  }
  msg += `\n`;

  msg += `<code>\u25B8 3\u03C3 BAND    ${fmt(bandPrice)}</code>\n`;
  msg += `<code>\u25B8 6\u03C3 BAND    ${fmt(sixSigmaPrice)}</code>    (${distTo6sStr} away)\n`;
  msg += `\n`;

  msg += `<code>SPREAD 3\u21926\u03C3   ${data.spreadPct}%</code>  \u2503 ${ordinal(data.spreadPercentile)} pctile\n`;
  msg += `<code>SLOPE SHORT   ${fmtSlope(data.slopeShort)}</code> \u2503 ${ordinal(data.slopeShortPctile)} pctile\n`;
  msg += `<code>SLOPE LONG    ${fmtSlope(data.slopeLong)}</code> \u2503 ${ordinal(data.slopeLongPctile)} pctile\n`;
  msg += `\n`;

  msg += `<b>SIGNAL \u25B8 ${trigger.side}</b>\n`;
  msg += line;

  return msg;
}

// ── Forecast cache (read-only, written by forecast-job.js) ────────────

const FORECAST_CACHE_FILE = path.join(__dirname, 'forecast-cache.json');
const STATE_FILE = path.join(__dirname, 'state.json');

function loadForecastCache() {
  try {
    if (fs.existsSync(FORECAST_CACHE_FILE)) {
      const data = JSON.parse(fs.readFileSync(FORECAST_CACHE_FILE, 'utf8'));
      // Only use if less than 36 hours old
      if (Date.now() - data.timestamp < 36 * 60 * 60 * 1000) {
        return data.forecasts || {};
      }
    }
  } catch (e) {}
  return {};
}

function loadState() {
  try {
    if (fs.existsSync(STATE_FILE)) return JSON.parse(fs.readFileSync(STATE_FILE, 'utf8'));
  } catch (e) {}
  return { lastUpdateId: 0, lastScan: null };
}

function saveState(state) {
  fs.writeFileSync(STATE_FILE, JSON.stringify(state));
}

// ── Forecast context for alerts ───────────────────────────────────────

function getForecastContext(symbol, forecastCache) {
  const data = forecastCache[symbol];
  if (!data) return '';

  const h = data['7d'];
  if (!h) return '';

  const q50 = h.q50Raw || h.ridgeReturnRaw || 0;
  const q10 = h.q10Raw || 0;
  const q90 = h.q90Raw || 0;
  const dir = q50 > 0.005 ? 'LONG' : q50 < -0.005 ? 'SHORT' : 'FLAT';
  const q50Str = `${q50 >= 0 ? '+' : ''}${(q50 * 100).toFixed(1)}%`;
  const rangeStr = `${(q10 * 100).toFixed(1)}% / ${(q90 * 100).toFixed(1)}%`;

  let ctx = `\n<code>FORECAST 7D:</code>\n`;
  ctx += `<code>  Q50: ${q50Str}  \u2503  Q10/Q90: ${rangeStr}</code>\n`;
  return ctx;
}

// ── Command handling (checks for pending Telegram commands) ───────────

async function handleCommands(state, forecastCache) {
  try {
    const updates = await bot.getUpdates({ offset: state.lastUpdateId + 1, timeout: 0, limit: 10 });
    if (!updates || updates.length === 0) return;

    for (const update of updates) {
      state.lastUpdateId = update.update_id;
      const msg = update.message;
      if (!msg || !msg.text || String(msg.chat.id) !== TELEGRAM_CHAT_ID) continue;

      const cmd = msg.text.split(' ')[0].toLowerCase();
      const line = '\u2501'.repeat(26);

      if (cmd === '/help') {
        await bot.sendMessage(TELEGRAM_CHAT_ID,
          `${line}\n<b>VWMA CLOUD SCANNER</b>\n${line}\n\n` +
          `<code>/scan</code>      \u25B8 Force scan now\n` +
          `<code>/forecast</code>  \u25B8 Daily forecast summary\n` +
          `<code>/status</code>    \u25B8 Scanner health\n` +
          `<code>/help</code>      \u25B8 This message\n\n` +
          `Scanner runs every 5 min via GitHub Actions.\n` +
          `Forecasts refresh daily at 8 PM ET.`,
          { parse_mode: 'HTML' });

      } else if (cmd === '/status') {
        const lastScan = state.lastScan ? new Date(state.lastScan).toLocaleString('en-US', { timeZone: 'America/New_York' }) : 'Never';
        const hasForecast = Object.keys(forecastCache).length > 0;
        await bot.sendMessage(TELEGRAM_CHAT_ID,
          `${line}\n<b>SCANNER STATUS</b>\n${line}\n\n` +
          `<code>Last scan:    ${lastScan} ET</code>\n` +
          `<code>Symbols:      ${CONFIG.symbols.length}</code>\n` +
          `<code>Timeframes:   ${CONFIG.timeframes.join(', ')}</code>\n` +
          `<code>Cooldown:     ${CONFIG.cooldownMinutes} min</code>\n` +
          `<code>Forecast:     ${hasForecast ? 'Cached' : 'No data'}</code>`,
          { parse_mode: 'HTML' });

      } else if (cmd === '/forecast') {
        if (Object.keys(forecastCache).length === 0) {
          await bot.sendMessage(TELEGRAM_CHAT_ID, 'No forecast data cached. Forecasts refresh daily at 8 PM ET.');
          continue;
        }

        const rows = [];
        for (const [sym, data] of Object.entries(forecastCache)) {
          const h = data?.['7d'];
          if (!h) continue;
          const q50 = h.q50Raw || h.ridgeReturnRaw || 0;
          const q10 = h.q10Raw || 0;
          const q90 = h.q90Raw || 0;
          const dir = q50 > 0.005 ? 'LONG' : q50 < -0.005 ? 'SHORT' : 'FLAT';
          rows.push({ sym, dir, q50, q10, q90, absQ50: Math.abs(q50) });
        }
        rows.sort((a, b) => b.absQ50 - a.absQ50);

        let reply = `${line}\n<b>7D FORECAST (Ridge + GBM)</b>\n${line}\n\n`;
        reply += `<code>SYMBOL   DIR      Q50     Q10 / Q90</code>\n`;
        reply += `<code>${'\u2500'.repeat(40)}</code>\n`;
        for (const r of rows) {
          const q50Str = `${r.q50 >= 0 ? '+' : ''}${(r.q50 * 100).toFixed(1)}%`;
          const q10Str = `${(r.q10 * 100).toFixed(1)}%`;
          const q90Str = `${(r.q90 * 100).toFixed(1)}%`;
          reply += `<code>${r.sym.padEnd(8)} ${r.dir.padEnd(6)}  ${q50Str.padStart(6)}   ${q10Str.padStart(6)} / ${q90Str.padStart(5)}</code>\n`;
        }
        reply += `\n${line}`;
        await bot.sendMessage(TELEGRAM_CHAT_ID, reply, { parse_mode: 'HTML' });

      } else if (cmd === '/scan') {
        await bot.sendMessage(TELEGRAM_CHAT_ID, 'Scanning now...');
        // The main scan will run right after this function returns
      }
    }
  } catch (err) {
    console.warn(`[Commands] ${err.message}`);
  }
}

// ── Main scan pass ────────────────────────────────────────────────────

async function main() {
  const state = loadState();
  const forecastCache = loadForecastCache();

  // Check for pending Telegram commands first
  await handleCommands(state, forecastCache);

  console.log(`[Scanner] Starting scan — ${CONFIG.symbols.length} symbols, timeframes: ${CONFIG.timeframes.join(', ')}`);

  const cooldowns = loadCooldowns();
  let totalAlerts = 0;

  for (const timeframe of CONFIG.timeframes) {
    for (const symbol of CONFIG.symbols) {
      try {
        const ccxtSymbol = `${symbol}/${CONFIG.quote}:${CONFIG.quote}`;
        const ohlcv = await exchange.fetchOHLCV(ccxtSymbol, timeframe, undefined, BARS_TO_FETCH);

        const bandData = analyze(ohlcv);
        if (!bandData) continue;

        const trigger = shouldAlert(bandData);
        if (trigger && !isOnCooldown(cooldowns, symbol, timeframe, trigger.direction)) {
          let msg = formatAlert(symbol, timeframe, bandData, trigger);
          // Append forecast context if available
          msg += getForecastContext(symbol, forecastCache);
          try {
            await bot.sendMessage(TELEGRAM_CHAT_ID, msg, { parse_mode: 'HTML' });
            console.log(`[ALERT] ${symbol} ${timeframe} ${trigger.side} @ ${bandData.currentPrice} (Z: ${bandData.priceZScore}, RSI: ${bandData.rsi})`);
            totalAlerts++;
          } catch (err) {
            console.error(`[Telegram] Failed: ${err.message}`);
          }
          setCooldown(cooldowns, symbol, timeframe, trigger.direction);
        }
      } catch (err) {
        console.warn(`[Skip] ${symbol} ${timeframe}: ${err.message}`);
      }

      // Rate limit
      await new Promise(r => setTimeout(r, 100));
    }
  }

  saveCooldowns(cooldowns);
  state.lastScan = Date.now();
  saveState(state);
  console.log(`[Scanner] Done — ${totalAlerts} alert(s) sent`);
}

main().catch(err => {
  console.error('[Fatal]', err.message);
  process.exit(1);
});
