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

const exchange = new ccxt.binance({ enableRateLimit: true });
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
  const d = (v, dp) => v > 100 ? v.toFixed(dp || 2) : v.toFixed(dp || 4);
  const isUpper = trigger.direction === 'UPPER';
  const emoji = isUpper ? '\u{1F534}' : '\u{1F7E2}';
  const arrow = isUpper ? '\u25B2' : '\u25BC';
  const bandLabel = isUpper ? 'Upper' : 'Lower';
  const bandPrice = isUpper ? data.upper3s : data.lower3s;
  const outerPrice = isUpper ? data.upper6s : data.lower6s;
  const pctInto6s = isUpper ? data.pctInto6sUpper : data.pctInto6sLower;

  const slopeArrow = (v) => {
    if (v === null) return '\u2014';
    if (v > 0.5) return `\u2191 +${v.toFixed(2)}%`;
    if (v < -0.5) return `\u2193 ${v.toFixed(2)}%`;
    return `\u2192 ${v.toFixed(2)}%`;
  };
  const pctile = (v) => v != null ? `${v}%` : '\u2014';

  let msg = `${emoji} <b>VWMA BAND ALERT</b> \u2014 ${symbol} ${timeframe}\n`;
  msg += `${arrow} Price at <b>${bandLabel} 3\u03C3</b>\n`;
  msg += `\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\n`;
  msg += `\u{1F4B0} Price:     <code>${d(data.currentPrice)}</code>\n`;
  msg += `\u{1F4CA} 100-VWMA:  <code>${d(data.vwma)}</code>\n`;
  msg += `\u{1F4CF} 3\u03C3 Band:   <code>${d(bandPrice)}</code>\n`;
  msg += `\u{1F4CF} 6\u03C3 Band:   <code>${d(outerPrice)}</code>\n`;
  msg += `\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\n`;
  msg += `\u{1F3AF} Z-Score:        <code>${data.priceZScore}\u03C3</code>\n`;
  if (data.rsi != null) {
    msg += `\u{1F4C8} RSI(14):        <code>${data.rsi}</code>\n`;
  }
  msg += `\u{1F4D0} 3\u03C3\u21926\u03C3 Spread:   <code>${data.spreadPct}%</code>  (${pctile(data.spreadPercentile)} pctile)\n`;
  if (pctInto6s > 0) {
    msg += `\u26A1 Into 6\u03C3 zone:   <code>${pctInto6s}%</code>\n`;
  }
  msg += `\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\n`;
  msg += `\u{1F4C8} VWMA Slope:\n`;
  msg += `   Short (${CONFIG.slopeShort}): <code>${slopeArrow(data.slopeShort)}</code>  (${pctile(data.slopeShortPctile)} pctile)\n`;
  msg += `   Long  (${CONFIG.slopeLong}):  <code>${slopeArrow(data.slopeLong)}</code>  (${pctile(data.slopeLongPctile)} pctile)\n`;
  msg += `\n\u{1F4A1} Bias: <b>${trigger.side}</b>`;

  return msg;
}

// ── Main scan pass ────────────────────────────────────────────────────

async function main() {
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
          const msg = formatAlert(symbol, timeframe, bandData, trigger);
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
  console.log(`[Scanner] Done — ${totalAlerts} alert(s) sent`);
}

main().catch(err => {
  console.error('[Fatal]', err.message);
  process.exit(1);
});
