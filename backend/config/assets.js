// ============================================================================
// CANONICAL ASSET CONFIGURATION
// Single source of truth for all asset lists across the application
// ============================================================================

// CORE_ASSETS: 14 assets with precomputed regression data
// Used for: Portfolio backtesting, Regime detection, Conviction scoring,
//           HRP allocation, Signal alerts, VWMA analysis
const CORE_ASSETS = [
  'BTC', 'ETH', 'BNB', 'XRP', 'SOL',
  'DOGE', 'ADA', 'LINK', 'AVAX', 'HBAR',
  'LTC', 'DOT', 'ETC', 'ATOM'
];

// EXTENDED_ASSETS: 28 assets for broader screening
// Used for: Momentum screening, Pairs trading exploration
const EXTENDED_ASSETS = [
  ...CORE_ASSETS,
  'TRX', 'XLM', 'BCH', 'SUI', 'SHIB',
  'UNI', 'MATIC', 'ALGO', 'FIL', 'NEAR',
  'APT', 'OP', 'ARB', 'INJ'
];

// Binance trading pair suffix
const PAIR_SUFFIX = 'USDT';

// Helper to get full trading pair symbol
const getTradingPair = (symbol) => `${symbol}${PAIR_SUFFIX}`;

// Helper to get all trading pairs
const getCoreTradingPairs = () => CORE_ASSETS.map(getTradingPair);
const getExtendedTradingPairs = () => EXTENDED_ASSETS.map(getTradingPair);

module.exports = {
  CORE_ASSETS,
  EXTENDED_ASSETS,
  PAIR_SUFFIX,
  getTradingPair,
  getCoreTradingPairs,
  getExtendedTradingPairs
};
