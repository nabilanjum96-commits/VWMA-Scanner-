// ============================================================================
// 97-FACTOR COMPUTATION ENGINE FOR NEW RIDGE MODEL
// Computes all factors required by ridge_model_export.json
// Each factor is z-scored using a rolling 180-day expanding window
// ============================================================================

// Module-level helpers
const _lastW = (arr, w) => arr.slice(Math.max(0, arr.length - w));

const _mean = (arr) => {
  if (!arr.length) return 0;
  return arr.reduce((a, b) => a + b, 0) / arr.length;
};

const _std = (arr) => {
  if (arr.length < 2) return 0;
  const m = _mean(arr);
  const v = arr.reduce((a, x) => a + (x - m) ** 2, 0) / (arr.length - 1);
  return Math.sqrt(v);
};

/**
 * Compute all 97 RAW (un-z-scored) factors given truncated arrays ending at a specific time index.
 * All array parameters should already be sliced to the desired end point.
 *
 * @param {number[]} closes  - Daily close prices
 * @param {number[]} highs   - Daily high prices
 * @param {number[]} lows    - Daily low prices
 * @param {number[]} opens   - Daily open prices
 * @param {number[]} volumes - Daily volumes
 * @param {number[]} returns - Daily log returns (length = closes.length - 1)
 * @param {Object}   macroData - { spy, dxy, gold, tnx, m2, vix, ... }
 * @param {Object}   crossSectional - { allReturns: { symbol: returns[] } }
 * @returns {Object} { FACTOR_NAME: raw_value, ... }
 */
function computeRawFactors(closes, highs, lows, opens, volumes, returns, macroData, crossSectional) {
  const N = returns.length;
  if (N < 30) return null;

  const lastW = _lastW;
  const mean = _mean;
  const std = _std;

  const factors = {};

  // =========================================================================
  // VOLATILITY FACTORS (19)
  // =========================================================================
  const rvolRaw = (w) => std(lastW(returns, w)) * Math.sqrt(365);
  factors.RVOL_7D = rvolRaw(7);
  factors.RVOL_30D = rvolRaw(30);

  const parkinson = (w) => {
    const recent = lastW(closes, w + 1);
    const recentH = lastW(highs, w + 1);
    const recentL = lastW(lows, w + 1);
    let sum = 0, count = 0;
    for (let i = 1; i < recent.length; i++) {
      if (recentH[i] > 0 && recentL[i] > 0 && recentH[i] >= recentL[i]) {
        const logHL = Math.log(recentH[i] / recentL[i]);
        sum += logHL * logHL;
        count++;
      }
    }
    if (count === 0) return 0;
    return Math.sqrt(sum / (4 * count * Math.log(2))) * Math.sqrt(365);
  };
  factors.PARKINSON_7D = parkinson(7);
  factors.PARKINSON_30D = parkinson(30);

  const overnightVol = (w) => {
    const rets = [];
    const recentC = lastW(closes, w + 1);
    const recentO = lastW(opens, w + 1);
    for (let i = 1; i < recentC.length; i++) {
      if (recentC[i - 1] > 0 && recentO[i] > 0) {
        rets.push(Math.log(recentO[i] / recentC[i - 1]));
      }
    }
    return std(rets) * Math.sqrt(365);
  };
  factors.OVERNIGHT_VOL_7D = overnightVol(7);

  const rollingVols = [];
  for (let i = 7; i <= N; i++) {
    rollingVols.push(std(returns.slice(i - 7, i)) * Math.sqrt(365));
  }
  factors.VOLVOL_14D = std(lastW(rollingVols, 14));

  const downsideVol = (w) => {
    const neg = lastW(returns, w).filter(r => r < 0);
    return neg.length >= 2 ? std(neg) * Math.sqrt(365) : 0;
  };
  factors.DOWNVOL_7D = downsideVol(7);
  factors.DOWNVOL_30D = downsideVol(30);

  const skewness = (w) => {
    const r = lastW(returns, w);
    if (r.length < 3) return 0;
    const m = mean(r);
    const s = std(r);
    if (s === 0) return 0;
    const n = r.length;
    return (n / ((n - 1) * (n - 2))) * r.reduce((a, x) => a + ((x - m) / s) ** 3, 0);
  };
  factors.VOL_SKEW_14D = skewness(14);

  const kurtosis = (w) => {
    const r = lastW(returns, w);
    if (r.length < 4) return 0;
    const m = mean(r);
    const s = std(r);
    if (s === 0) return 0;
    return mean(r.map(x => ((x - m) / s) ** 4)) - 3;
  };
  factors.VOL_KURT_14D = kurtosis(14);

  factors.VOL_RATIO_7_30 = factors.RVOL_30D > 0 ? factors.RVOL_7D / factors.RVOL_30D : 1;
  const rvol90 = rvolRaw(90);
  factors.VOL_RATIO_7_90 = rvol90 > 0 ? factors.RVOL_7D / rvol90 : 1;

  const volHistory = [];
  for (let i = 7; i <= Math.min(N, 90); i++) {
    volHistory.push(std(returns.slice(Math.max(0, i - 7), i)) * Math.sqrt(365));
  }
  if (volHistory.length > 5) {
    const sorted = [...volHistory].sort((a, b) => a - b);
    const rank = sorted.filter(v => v <= factors.RVOL_7D).length;
    factors.VOL_CLIMAX = rank / sorted.length;
  } else {
    factors.VOL_CLIMAX = 0.5;
  }

  const rvol7_prev7 = N >= 14 ? std(returns.slice(N - 14, N - 7)) * Math.sqrt(365) : factors.RVOL_7D;
  const rvol30_prev30 = N >= 60 ? std(returns.slice(N - 60, N - 30)) * Math.sqrt(365) : factors.RVOL_30D;
  factors.VOL_MOM_7D = factors.RVOL_7D - rvol7_prev7;
  factors.VOL_MOM_30D = factors.RVOL_30D - rvol30_prev30;

  const rvol3 = rvolRaw(3);
  factors.VOL_DECAY_3D = factors.RVOL_7D > 0 ? rvol3 / factors.RVOL_7D : 1;

  const garchForecast = (() => {
    const omega = 0.00001, alpha = 0.1, beta = 0.85;
    let sigma2 = (factors.RVOL_30D / Math.sqrt(365)) ** 2 / 365;
    const lastRet = returns[N - 1] || 0;
    return Math.sqrt(omega + alpha * lastRet * lastRet + beta * sigma2) * Math.sqrt(365);
  })();
  factors.GARCH_FORECAST_1D = garchForecast;
  factors.GARCH_SURPRISE = factors.RVOL_7D > 0 ? (factors.RVOL_7D - garchForecast) / factors.RVOL_7D : 0;

  const bb20 = lastW(closes, 20);
  const bb20Mean = mean(bb20);
  const bb20Std = std(bb20);
  factors.BB_WIDTH_20D = bb20Mean > 0 ? (2 * bb20Std) / bb20Mean : 0;

  // =========================================================================
  // MOMENTUM / TREND FACTORS (18)
  // =========================================================================
  const tsmom = (w) => {
    if (closes.length < w + 1) return 0;
    const cur = closes[closes.length - 1];
    const prev = closes[closes.length - 1 - w];
    return prev > 0 ? Math.log(cur / prev) : 0;
  };
  factors.TSMOM_1D = tsmom(1);
  factors.TSMOM_3D = tsmom(3);
  factors.TSMOM_7D = tsmom(7);
  factors.TSMOM_14D = tsmom(14);
  factors.TSMOM_30D = tsmom(30);

  const xsmom = (w) => {
    if (!crossSectional || !crossSectional.allReturns) return 0;
    const myRet = tsmom(w);
    const allRets = Object.values(crossSectional.allReturns).map(rets => {
      const r = rets.slice(-w);
      return r.reduce((a, b) => a + b, 0);
    });
    if (allRets.length < 3) return 0;
    return myRet - mean(allRets);
  };
  factors.XSMOM_7D = xsmom(7);
  factors.XSMOM_30D = xsmom(30);

  const skipMom = (w1, w2) => {
    if (closes.length < w1 + 1) return 0;
    const from = closes[closes.length - 1 - w1];
    const to = closes[closes.length - 1 - w2];
    return from > 0 && to > 0 ? Math.log(to / from) : 0;
  };
  factors.SKIP_MOM_7_1 = skipMom(7, 1);
  factors.SKIP_MOM_30_7 = skipMom(30, 7);

  factors.MOM_ACCEL_7D = tsmom(7) - skipMom(14, 7);
  factors.MOM_ACCEL_30D = tsmom(30) - skipMom(60, 30);

  const trendStrength = (w) => {
    const r = lastW(returns, w);
    const s = std(r);
    return s > 0 ? Math.abs(mean(r)) / s * Math.sqrt(r.length) : 0;
  };
  factors.TREND_STRENGTH_14D = trendStrength(14);
  factors.TREND_STRENGTH_30D = trendStrength(30);

  const linregSlope = (w) => {
    const r = lastW(returns, w);
    if (r.length < 5) return 0;
    const n = r.length;
    const xMean = (n - 1) / 2;
    let num = 0, den = 0;
    for (let i = 0; i < n; i++) {
      num += (i - xMean) * r[i];
      den += (i - xMean) ** 2;
    }
    const slope = den > 0 ? num / den : 0;
    const s = std(r);
    return s > 0 ? slope / s : 0;
  };
  factors.LINREG_SLOPE_14D = linregSlope(14);
  factors.LINREG_SLOPE_30D = linregSlope(30);

  const linregR2 = (w) => {
    const r = lastW(returns, w);
    if (r.length < 5) return 0;
    const n = r.length;
    const xMean = (n - 1) / 2;
    const yMean = mean(r);
    let ssRes = 0, ssTot = 0;
    let num = 0, den = 0;
    for (let i = 0; i < n; i++) {
      num += (i - xMean) * (r[i] - yMean);
      den += (i - xMean) ** 2;
    }
    const slope = den > 0 ? num / den : 0;
    const intercept = yMean - slope * xMean;
    for (let i = 0; i < n; i++) {
      const pred = intercept + slope * i;
      ssRes += (r[i] - pred) ** 2;
      ssTot += (r[i] - yMean) ** 2;
    }
    return ssTot > 0 ? 1 - ssRes / ssTot : 0;
  };
  factors.LINREG_R2_14D = linregR2(14);

  const retPersist = (w) => {
    const r = lastW(returns, w);
    if (r.length < 5) return 0;
    const m = mean(r);
    let num = 0, den = 0;
    for (let i = 1; i < r.length; i++) {
      num += (r[i] - m) * (r[i - 1] - m);
      den += (r[i] - m) ** 2;
    }
    return den > 0 ? num / den : 0;
  };
  factors.RET_PERSIST_3D = retPersist(3);
  factors.RET_PERSIST_7D = retPersist(7);

  // =========================================================================
  // MEAN REVERSION / RELATIVE VALUE (10)
  // =========================================================================
  const priceZscore = (w) => {
    const c = lastW(closes, w);
    if (c.length < w) return 0;
    const m = mean(c);
    const s = std(c);
    return s > 0 ? (closes[closes.length - 1] - m) / s : 0;
  };
  factors.ZSCORE_7D = priceZscore(7);
  factors.ZSCORE_21D = priceZscore(21);
  factors.ZSCORE_60D = priceZscore(60);

  const xsZscore = (w) => {
    if (!crossSectional || !crossSectional.allReturns) return 0;
    const myRet = mean(lastW(returns, w));
    const allMeans = Object.values(crossSectional.allReturns).map(r => mean(lastW(r, w)));
    if (allMeans.length < 3) return 0;
    const m = mean(allMeans);
    const s = std(allMeans);
    return s > 0 ? (myRet - m) / s : 0;
  };
  factors.XS_ZSCORE_7D = xsZscore(7);
  factors.XS_ZSCORE_30D = xsZscore(30);

  factors.XS_DISPERSION_7D = (() => {
    if (!crossSectional || !crossSectional.allReturns) return 0;
    const allRets = Object.values(crossSectional.allReturns).map(r => {
      const recent = lastW(r, 7);
      return recent.reduce((a, b) => a + b, 0);
    });
    return std(allRets);
  })();

  factors.CORR_TO_MEAN_30D = (() => {
    if (!crossSectional || !crossSectional.allReturns) return 0;
    const myRets = lastW(returns, 30);
    const symbols = Object.keys(crossSectional.allReturns);
    if (symbols.length < 3 || myRets.length < 10) return 0;
    const meanRets = [];
    for (let i = 0; i < myRets.length; i++) {
      const dailyVals = symbols.map(s => {
        const r = crossSectional.allReturns[s];
        return r[r.length - myRets.length + i] || 0;
      });
      meanRets.push(mean(dailyVals));
    }
    const mx = mean(myRets), my = mean(meanRets);
    let num = 0, dx = 0, dy = 0;
    for (let i = 0; i < myRets.length; i++) {
      num += (myRets[i] - mx) * (meanRets[i] - my);
      dx += (myRets[i] - mx) ** 2;
      dy += (meanRets[i] - my) ** 2;
    }
    return dx > 0 && dy > 0 ? num / Math.sqrt(dx * dy) : 0;
  })();

  factors.BB_PCTB_20D = bb20Std > 0 ? (closes[closes.length - 1] - (bb20Mean - 2 * bb20Std)) / (4 * bb20Std) : 0.5;

  const vwapDev = (w) => {
    if (volumes.length < w) return 0;
    const recentC = lastW(closes, w);
    const recentV = lastW(volumes, w);
    let vwap = 0, totalVol = 0;
    for (let i = 0; i < recentC.length; i++) {
      vwap += recentC[i] * (recentV[i] || 1);
      totalVol += (recentV[i] || 1);
    }
    vwap = totalVol > 0 ? vwap / totalVol : mean(recentC);
    return vwap > 0 ? (closes[closes.length - 1] - vwap) / vwap : 0;
  };
  factors.VWAP_DEV_7D = vwapDev(7);
  factors.VWAP_DEV_30D = vwapDev(30);

  // =========================================================================
  // TECHNICAL INDICATORS (9)
  // =========================================================================
  const computeRSI = (w) => {
    const r = lastW(returns, w);
    if (r.length < w) return 50;
    let gains = 0, losses = 0;
    for (const ret of r) {
      if (ret > 0) gains += ret;
      else losses += Math.abs(ret);
    }
    if (losses === 0) return 100;
    const rs = gains / losses;
    return 100 - 100 / (1 + rs);
  };
  factors.RSI_14D = computeRSI(14);

  factors.RSI_DIVERGENCE_14D = (() => {
    const priceHigh = Math.max(...lastW(closes, 14));
    const curPrice = closes[closes.length - 1];
    const rsi = factors.RSI_14D;
    if (curPrice >= priceHigh * 0.99 && rsi < 70) return 1;
    const priceLow = Math.min(...lastW(closes, 14));
    if (curPrice <= priceLow * 1.01 && rsi > 30) return -1;
    return 0;
  })();

  factors.STOCH_K_14D = (() => {
    const h = lastW(highs, 14);
    const l = lastW(lows, 14);
    const highest = Math.max(...h);
    const lowest = Math.min(...l);
    const range = highest - lowest;
    return range > 0 ? (closes[closes.length - 1] - lowest) / range * 100 : 50;
  })();

  const sma = (w) => mean(lastW(closes, w));
  factors.MA_CROSS_7_21 = sma(21) > 0 ? sma(7) / sma(21) - 1 : 0;
  factors.MA_CROSS_21_50 = sma(50) > 0 ? sma(21) / sma(50) - 1 : 0;
  factors.MA_CROSS_50_200 = closes.length >= 200 ? sma(50) / sma(200) - 1 : 0;

  const hhhlCount = (w) => {
    const recentH = lastW(highs, w);
    const recentL = lastW(lows, w);
    let hh = 0, hl = 0;
    for (let i = 1; i < recentH.length; i++) {
      if (recentH[i] > recentH[i - 1]) hh++;
      if (recentL[i] > recentL[i - 1]) hl++;
    }
    return (hh + hl) / Math.max(1, 2 * (recentH.length - 1));
  };
  factors.HH_HL_COUNT_14D = hhhlCount(14);
  factors.HH_HL_COUNT_30D = hhhlCount(30);

  factors.DONCHIAN_BREAK_20D = (() => {
    const h20 = Math.max(...lastW(highs, 20));
    const l20 = Math.min(...lastW(lows, 20));
    const cur = closes[closes.length - 1];
    if (cur >= h20) return 1;
    if (cur <= l20) return -1;
    return 0;
  })();

  // =========================================================================
  // PRICE STRUCTURE (6)
  // =========================================================================
  const hlRatio = (w) => {
    const h = lastW(highs, w);
    const l = lastW(lows, w);
    const avgHL = mean(h.map((hi, i) => l[i] > 0 ? hi / l[i] : 1));
    return avgHL - 1;
  };
  factors.HL_RATIO_7D = hlRatio(7);

  const computeATR = (w) => {
    const recentH = lastW(highs, w + 1);
    const recentL = lastW(lows, w + 1);
    const recentC = lastW(closes, w + 1);
    let sum = 0;
    for (let i = 1; i < recentH.length; i++) {
      const tr = Math.max(
        recentH[i] - recentL[i],
        Math.abs(recentH[i] - recentC[i - 1]),
        Math.abs(recentL[i] - recentC[i - 1])
      );
      sum += tr;
    }
    return sum / Math.max(1, recentH.length - 1);
  };
  const atr14 = computeATR(14);
  factors.ATR_14D = closes[closes.length - 1] > 0 ? atr14 / closes[closes.length - 1] : 0;

  const atr14_prev = N >= 28 ? (() => {
    const h = highs.slice(-29, -15);
    const l = lows.slice(-29, -15);
    const c = closes.slice(-29, -15);
    let sum = 0;
    for (let i = 1; i < h.length; i++) {
      sum += Math.max(h[i] - l[i], Math.abs(h[i] - c[i - 1]), Math.abs(l[i] - c[i - 1]));
    }
    const prevPrice = closes[closes.length - 15] || closes[closes.length - 1];
    return prevPrice > 0 ? (sum / Math.max(1, h.length - 1)) / prevPrice : 0;
  })() : factors.ATR_14D;
  factors.ATR_CHANGE_7D = factors.ATR_14D - atr14_prev;

  const todayRange = highs.length > 0 ? (highs[highs.length - 1] - lows[lows.length - 1]) / closes[closes.length - 1] : 0;
  factors.ATR_BREAK_14D = factors.ATR_14D > 0 ? todayRange / factors.ATR_14D : 0;

  const rangeW = (w) => {
    const h = Math.max(...lastW(highs, w));
    const l = Math.min(...lastW(lows, w));
    return closes[closes.length - 1] > 0 ? (h - l) / closes[closes.length - 1] : 0;
  };
  const range30 = rangeW(30);
  factors.RANGE_CONTRACT_7D = range30 > 0 ? rangeW(7) / range30 : 1;

  const upDown = (w) => {
    const r = lastW(returns, w);
    const up = r.filter(x => x > 0).length;
    const down = r.filter(x => x < 0).length;
    return down > 0 ? up / down : up > 0 ? 2 : 1;
  };
  factors.UP_DOWN_RATIO_7D = upDown(7);

  // =========================================================================
  // POSITION / LEVEL (3)
  // =========================================================================
  const pricePos = (w) => {
    const c = lastW(closes, w);
    if (c.length < 10) return 0.5;
    const cur = closes[closes.length - 1];
    const rank = c.filter(x => x <= cur).length;
    return rank / c.length;
  };
  factors.PRICE_POS_50D = pricePos(50);
  factors.PRICE_POS_200D = closes.length >= 200 ? pricePos(200) : pricePos(closes.length);

  factors.WICK_RATIO_7D = (() => {
    const recentH = lastW(highs, 7);
    const recentL = lastW(lows, 7);
    const recentO = lastW(opens, 7);
    const recentC = lastW(closes, 7);
    let sumWick = 0, sumRange = 0;
    for (let i = 0; i < recentH.length; i++) {
      const range = recentH[i] - recentL[i];
      if (range > 0) {
        const body = Math.abs(recentC[i] - recentO[i]);
        sumWick += range - body;
        sumRange += range;
      }
    }
    return sumRange > 0 ? sumWick / sumRange : 0.5;
  })();

  // =========================================================================
  // CROSS-ASSET BETAS (16)
  // =========================================================================
  const computeBeta = (assetKey, w) => {
    const macroAsset = macroData?.[assetKey];
    if (!macroAsset || macroAsset.length < w + 5) return 0;
    const macroCloses = macroAsset.map(d => d.close || d);
    const macroRets = [];
    for (let i = 1; i < macroCloses.length; i++) {
      macroRets.push(Math.log(macroCloses[i] / macroCloses[i - 1]));
    }
    const myRets = lastW(returns, w);
    const macroR = lastW(macroRets, w);
    const len = Math.min(myRets.length, macroR.length);
    if (len < 10) return 0;
    const mx = mean(myRets.slice(-len));
    const my = mean(macroR.slice(-len));
    let cov = 0, varM = 0;
    for (let i = 0; i < len; i++) {
      const dx = myRets[myRets.length - len + i] - mx;
      const dy = macroR[macroR.length - len + i] - my;
      cov += dx * dy;
      varM += dy * dy;
    }
    return varM > 0 ? cov / varM : 0;
  };

  factors.BETA_SPY_30D = computeBeta('spy', 30);
  factors.BETA_SPY_60D = computeBeta('spy', 60);
  factors.BETA_SPY_90D = computeBeta('spy', 90);
  factors.BETA_DXY_30D = computeBeta('dxy', 30);
  factors.BETA_DXY_60D = computeBeta('dxy', 60);
  factors.BETA_DXY_90D = computeBeta('dxy', 90);
  factors.BETA_TNX_30D = computeBeta('tnx', 30);
  factors.BETA_TNX_60D = computeBeta('tnx', 60);
  factors.BETA_TNX_90D = computeBeta('tnx', 90);
  factors.BETA_GOLD_30D = computeBeta('gold', 30);
  factors.BETA_GOLD_60D = computeBeta('gold', 60);
  factors.BETA_GOLD_90D = computeBeta('gold', 90);
  factors.BETA_M2_30D = computeBeta('m2', 30);
  factors.BETA_M2_60D = computeBeta('m2', 60);
  factors.BETA_M2_90D = computeBeta('m2', 90);
  factors.BETA_VOL_30D = computeBeta('vix', 30);

  // =========================================================================
  // MICROSTRUCTURE / DERIVATIVES (5)
  // =========================================================================
  factors.BASIS_PERP_SPOT = 0;
  factors.BASIS_ZSCORE_7D = 0;
  factors.FUND_ZSCORE_7D = 0;
  factors.FUND_ZSCORE_30D = 0;
  factors.OPTIONS_EXPIRY_DIST = (() => {
    const now = new Date();
    const dayOfWeek = now.getUTCDay();
    const daysToFri = (5 - dayOfWeek + 7) % 7 || 7;
    return daysToFri / 7;
  })();

  // =========================================================================
  // STATISTICAL (6)
  // =========================================================================
  factors.AUTOCORR_7D = retPersist(7);
  factors.AUTOCORR_30D = retPersist(30);

  factors.HURST_30D = (() => {
    const r = lastW(returns, 30);
    if (r.length < 20) return 0.5;
    const sizes = [5, 10, 15];
    const logRS = [], logN = [];
    for (const w of sizes) {
      const nBlocks = Math.floor(r.length / w);
      if (nBlocks < 2) continue;
      let totalRS = 0;
      for (let b = 0; b < nBlocks; b++) {
        const block = r.slice(b * w, (b + 1) * w);
        const m = mean(block);
        const cumDev = block.map((x, i) => block.slice(0, i + 1).reduce((a, v) => a + (v - m), 0));
        const R = Math.max(...cumDev) - Math.min(...cumDev);
        const S = std(block);
        totalRS += S > 0 ? R / S : 0;
      }
      if (totalRS > 0) {
        logRS.push(Math.log(totalRS / nBlocks));
        logN.push(Math.log(w));
      }
    }
    if (logRS.length < 2) return 0.5;
    const mx = mean(logN), my = mean(logRS);
    let num = 0, den = 0;
    for (let i = 0; i < logRS.length; i++) {
      num += (logN[i] - mx) * (logRS[i] - my);
      den += (logN[i] - mx) ** 2;
    }
    return den > 0 ? Math.max(0, Math.min(1, num / den)) : 0.5;
  })();

  factors.PERM_ENTROPY_21D = (() => {
    const r = lastW(returns, 21);
    if (r.length < 10) return 1;
    const counts = {};
    for (let i = 0; i < r.length - 2; i++) {
      const triple = [r[i], r[i + 1], r[i + 2]];
      const order = triple.map((v, j) => [v, j]).sort((a, b) => a[0] - b[0]).map(x => x[1]).join('');
      counts[order] = (counts[order] || 0) + 1;
    }
    const total = r.length - 2;
    let entropy = 0;
    for (const c of Object.values(counts)) {
      const p = c / total;
      if (p > 0) entropy -= p * Math.log(p);
    }
    return entropy / Math.log(6);
  })();

  const maxDD = (w) => {
    const c = lastW(closes, w);
    let peak = c[0], maxDd = 0;
    for (const price of c) {
      if (price > peak) peak = price;
      const dd = (peak - price) / peak;
      if (dd > maxDd) maxDd = dd;
    }
    return maxDd;
  };
  factors.MAX_DD_14D = maxDD(14);
  factors.MAX_DD_30D = maxDD(30);

  // =========================================================================
  // SEASONALITY / MACRO (5)
  // =========================================================================
  factors.BTC_HALVING_PHASE = (() => {
    const lastHalving = new Date('2024-04-20').getTime();
    const cycleDays = 4 * 365.25;
    const daysSince = (Date.now() - lastHalving) / (24 * 60 * 60 * 1000);
    return (daysSince % cycleDays) / cycleDays;
  })();

  factors.MOY_RETURN_AVG = (() => {
    const currentMonth = new Date().getMonth();
    const monthReturns = {};
    for (let i = Math.max(0, returns.length - 365); i < returns.length; i++) {
      const approxMonth = Math.floor(i / 30) % 12;
      if (!monthReturns[approxMonth]) monthReturns[approxMonth] = [];
      monthReturns[approxMonth].push(returns[i]);
    }
    const monthRets = monthReturns[currentMonth] || [];
    return monthRets.length > 0 ? mean(monthRets) * 30 : 0;
  })();

  factors.DOW_RETURN_AVG = (() => {
    const currentDow = new Date().getDay();
    const dowReturns = {};
    for (let i = Math.max(0, returns.length - 90); i < returns.length; i++) {
      const dow = i % 7;
      if (!dowReturns[dow]) dowReturns[dow] = [];
      dowReturns[dow].push(returns[i]);
    }
    const dowRets = dowReturns[currentDow] || [];
    return dowRets.length > 0 ? mean(dowRets) : 0;
  })();

  const sharpe = (w) => {
    const r = lastW(returns, w);
    const s = std(r);
    return s > 0 ? (mean(r) / s) * Math.sqrt(365) : 0;
  };
  factors.SHARPE_7D = sharpe(7);
  factors.SHARPE_30D = sharpe(30);

  return factors;
}


/**
 * Compute all 97 factors for a single asset, z-scored against expanding 180-day history.
 * Matches the Python training pipeline: computeRawFactors at each historical time step,
 * then z-score = (current - mean(history)) / std(history).
 *
 * @param {Object} cryptoData - { closes, opens, highs, lows, volumes }
 * @param {Object} macroData  - { spy, dxy, gold, tnx, m2, vix, ... }
 * @param {Object} crossSectional - { allReturns: { symbol: returns[] } }
 * @returns {Object} factors - { FACTOR_NAME: z-scored value, ... } or null
 */
function computeFactors97(cryptoData, macroData, crossSectional = {}) {
  const closes = cryptoData.closes || [];
  const highs = cryptoData.highs || closes;
  const lows = cryptoData.lows || closes;
  const opens = cryptoData.opens || closes;
  const volumes = cryptoData.volumes || [];

  if (closes.length < 60) return null;

  // Pre-compute full returns array
  const allReturns = [];
  for (let i = 1; i < closes.length; i++) {
    allReturns.push(Math.log(closes[i] / closes[i - 1]));
  }

  const NORM_WINDOW = 180;
  const MIN_HISTORY = 30;
  const totalBars = closes.length;

  // Compute raw factors at each of the last NORM_WINDOW time steps
  // endIdx is the number of bars to include (1-indexed: closes[0..endIdx-1])
  const factorHistories = {};
  const startOffset = Math.min(NORM_WINDOW - 1, totalBars - 60);

  for (let offset = startOffset; offset >= 0; offset--) {
    const endIdx = totalBars - offset;
    if (endIdx < 60) continue;

    const tCloses = closes.slice(0, endIdx);
    const tHighs = highs.slice(0, endIdx);
    const tLows = lows.slice(0, endIdx);
    const tOpens = opens.slice(0, endIdx);
    const tVolumes = volumes.slice(0, endIdx);
    const tReturns = allReturns.slice(0, endIdx - 1);

    const rawFactors = computeRawFactors(
      tCloses, tHighs, tLows, tOpens, tVolumes, tReturns,
      macroData, crossSectional
    );
    if (!rawFactors) continue;

    for (const [name, val] of Object.entries(rawFactors)) {
      if (!factorHistories[name]) factorHistories[name] = [];
      factorHistories[name].push(isFinite(val) ? val : NaN);
    }
  }

  // The current (most recent) raw factors are the last entry in each history
  const zScoredFactors = {};

  for (const [name, history] of Object.entries(factorHistories)) {
    const current = history[history.length - 1];
    if (!isFinite(current)) {
      zScoredFactors[name] = 0;
      continue;
    }

    // Filter valid (non-NaN) values for statistics
    const valid = history.filter(v => isFinite(v));
    if (valid.length < MIN_HISTORY) {
      // Not enough history — return 0 (neutral) to avoid noisy z-scores
      zScoredFactors[name] = 0;
      continue;
    }

    const m = _mean(valid);
    const s = _std(valid);
    if (s === 0 || !isFinite(s)) {
      zScoredFactors[name] = 0;
      continue;
    }

    zScoredFactors[name] = (current - m) / s;
  }

  return zScoredFactors;
}

module.exports = {
  computeFactors97,
  computeRawFactors
};
