// ============================================================================
// ONE-TIME BOT SETUP — Registers commands with Telegram
//
// Run once: TELEGRAM_BOT_TOKEN=xxx node setup-bot.js
// After this, commands appear as a clickable menu in Telegram chat.
// ============================================================================

const TelegramBot = require('node-telegram-bot-api');

const TELEGRAM_BOT_TOKEN = process.env.TELEGRAM_BOT_TOKEN;
if (!TELEGRAM_BOT_TOKEN) {
  console.error('Set TELEGRAM_BOT_TOKEN env var');
  process.exit(1);
}

const bot = new TelegramBot(TELEGRAM_BOT_TOKEN, { polling: false });

async function setup() {
  const commands = [
    { command: 'vwma', description: 'Full VWMA band scan (select timeframe)' },
    { command: 'forecast', description: 'Show latest daily forecast' },
    { command: 'status', description: 'Scanner health + last scan time' },
    { command: 'help', description: 'List available commands' },
  ];

  await bot.setMyCommands(commands);
  console.log('Commands registered:');
  commands.forEach(c => console.log(`  /${c.command} — ${c.description}`));
  console.log('\nOpen Telegram → your bot → click the menu button next to text input');
}

setup().catch(err => {
  console.error('Failed:', err.message);
  process.exit(1);
});
