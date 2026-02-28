// ============================================================================
// DAILY FORECAST JOB — GitHub Actions Edition
//
// Runs once daily (8 PM ET / 1 AM UTC). Computes Ridge + GBM forecasts
// for all 22 assets, sends Telegram summary, caches results for scanner.
//
// Env vars (GitHub Secrets):
//   TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
// ============================================================================

const path = require('path');
const fs = require('fs');
const TelegramBot = require('node-telegram-bot-api');

// ── Config ────────────────────────────────────────────────────────────

const TELEGRAM_BOT_TOKEN = process.env.TELEGRAM_BOT_TOKEN;
const TELEGRAM_CHAT_ID = process.env.TELEGRAM_CHAT_ID;

if (!TELEGRAM_BOT_TOKEN || !TELEGRAM_CHAT_ID) {
  console.error('ERROR: TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set');
  process.exit(1);
}

const bot = new TelegramBot(TELEGRAM_BOT_TOKEN, { polling: false });
const FORECAST_CACHE_FILE = path.join(__dirname, 'forecast-cache.json');

// ── Main ──────────────────────────────────────────────────────────────

async function main() {
  console.log('[Forecast] Starting daily forecast generation...');
  const startTime = Date.now();

  // Import forecast engine (uses relative paths to coefficients/)
  const { generateAllForecasts } = require('./backend/forecast-engine');

  let result;
  try {
    result = await generateAllForecasts(true); // force refresh
  } catch (err) {
    console.error('[Forecast] Generation failed:', err.message);
    await bot.sendMessage(TELEGRAM_CHAT_ID,
      `\u274C <b>Forecast generation failed</b>\n<code>${err.message}</code>`,
      { parse_mode: 'HTML' });
    process.exit(1);
  }

  const forecasts = result.forecasts || {};
  const symbols = Object.keys(forecasts);
  const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
  console.log(`[Forecast] Generated forecasts for ${symbols.length} assets in ${elapsed}s`);

  // Save cache for scanner to use
  fs.writeFileSync(FORECAST_CACHE_FILE, JSON.stringify({
    timestamp: Date.now(),
    date: new Date().toISOString().split('T')[0],
    forecasts
  }));
  console.log(`[Forecast] Cache saved to ${FORECAST_CACHE_FILE}`);

  // Build Telegram summary
  const line = '\u2501'.repeat(30);
  const today = new Date().toLocaleDateString('en-US', {
    timeZone: 'America/New_York',
    month: 'short', day: 'numeric', year: 'numeric'
  });

  let msg = `${line}\n`;
  msg += `<b>DAILY FORECAST</b> \u2014 ${today}\n`;
  msg += `${line}\n\n`;
  msg += `<b>7D OUTLOOK (Ridge + GBM)</b>\n\n`;
  msg += `<code>SYMBOL   DIR      Q50     Q10 / Q90</code>\n`;
  msg += `<code>${'\u2500'.repeat(40)}</code>\n`;

  // Build rows sorted by absolute Q50 (highest conviction first)
  const rows = [];
  for (const [sym, data] of Object.entries(forecasts)) {
    const h = data?.['7d'];
    if (!h) continue;

    const q50 = h.q50Raw || h.ridgeReturnRaw || 0;
    const q10 = h.q10Raw || 0;
    const q90 = h.q90Raw || 0;
    const dir = q50 > 0.005 ? 'LONG' : q50 < -0.005 ? 'SHORT' : 'FLAT';
    const confidence = h.confidence || (Math.abs(q50) > 0.03 ? 'HIGH' : Math.abs(q50) > 0.01 ? 'MID' : 'LOW');

    rows.push({ sym, dir, q50, q10, q90, confidence, absQ50: Math.abs(q50) });
  }

  rows.sort((a, b) => b.absQ50 - a.absQ50);

  for (const r of rows) {
    const dirPad = r.dir.padEnd(6);
    const q50Str = `${r.q50 >= 0 ? '+' : ''}${(r.q50 * 100).toFixed(1)}%`;
    const q10Str = `${(r.q10 * 100).toFixed(1)}%`;
    const q90Str = `${(r.q90 * 100).toFixed(1)}%`;
    msg += `<code>${r.sym.padEnd(8)} ${dirPad}  ${q50Str.padStart(6)}   ${q10Str.padStart(6)} / ${q90Str.padStart(5)}</code>\n`;
  }

  // Top conviction picks
  const topPicks = rows.filter(r => r.dir !== 'FLAT').slice(0, 5);
  if (topPicks.length > 0) {
    msg += `\n<b>TOP CONVICTION:</b>\n`;
    for (const r of topPicks) {
      const q50Str = `${r.q50 >= 0 ? '+' : ''}${(r.q50 * 100).toFixed(1)}%`;
      msg += `\u25B8 ${r.sym.padEnd(6)} ${r.dir.padEnd(6)} ${r.confidence.padEnd(4)}  (${q50Str})\n`;
    }
  }

  msg += `\n${line}`;

  // Send to Telegram
  try {
    await bot.sendMessage(TELEGRAM_CHAT_ID, msg, { parse_mode: 'HTML' });
    console.log('[Forecast] Summary sent to Telegram');
  } catch (err) {
    console.error('[Forecast] Telegram send failed:', err.message);
  }

  console.log(`[Forecast] Done in ${elapsed}s`);
}

main().catch(err => {
  console.error('[Fatal]', err.message);
  process.exit(1);
});
