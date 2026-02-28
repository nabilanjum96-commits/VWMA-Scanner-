// ============================================================================
// REAL ML FORECAST ENGINE
// Trained coefficients from Ridge + Quantile Regression (2020-2023)
// Runs daily at 4:00 PM ET using closing prices
// Supports walk-forward coefficient loading for backtesting
// ============================================================================

const axios = require('axios');
const fs = require('fs');
const path = require('path');
const { computeFactors97 } = require('./ridge-factor-engine');

// ============================================================================
// WALK-FORWARD COEFFICIENT LOADING
// ============================================================================

// Cache for loaded coefficients
const coefficientCache = new Map();

/**
 * Get the quarter string BEFORE a given date
 * For predictions made on date D, we use coefficients trained on data ending before D
 * @param {Date|string} date - The date to find coefficients for
 * @returns {string} Quarter string like "2023_Q4"
 */
function getQuarterBeforeDate(date) {
  const d = new Date(date);
  const year = d.getFullYear();
  const month = d.getMonth(); // 0-indexed

  // Current quarter: Q1=Jan-Mar (0-2), Q2=Apr-Jun (3-5), Q3=Jul-Sep (6-8), Q4=Oct-Dec (9-11)
  let quarterNum = Math.floor(month / 3) + 1;
  let quarterYear = year;

  // Use PREVIOUS quarter's coefficients (trained before this quarter started)
  quarterNum -= 1;
  if (quarterNum === 0) {
    quarterNum = 4;
    quarterYear -= 1;
  }

  return `${quarterYear}_Q${quarterNum}`;
}

/**
 * Load coefficients for a specific date
 * Uses the most recent quarter BEFORE the given date
 * @param {Date|string} date - The date to load coefficients for
 * @returns {Object} Coefficient object matching TRAINED_COEFFICIENTS structure
 */
function loadCoefficientsForDate(date) {
  const quarter = getQuarterBeforeDate(date);
  const cacheKey = quarter;

  // Check cache first
  if (coefficientCache.has(cacheKey)) {
    return coefficientCache.get(cacheKey);
  }

  const coeffDir = path.join(__dirname, 'coefficients');
  const filePath = path.join(coeffDir, `${quarter}.json`);

  // Check if file exists
  if (!fs.existsSync(filePath)) {
    console.warn(`[Forecast] No coefficients for ${quarter}, using default`);
    coefficientCache.set(cacheKey, TRAINED_COEFFICIENTS);
    return TRAINED_COEFFICIENTS; // Fallback to hardcoded
  }

  try {
    const content = fs.readFileSync(filePath, 'utf-8');
    const coeffs = JSON.parse(content);

    // Validate structure
    if (!coeffs.horizons || !coeffs.horizons['3d']) {
      console.warn(`[Forecast] Invalid coefficient structure for ${quarter}, using default`);
      return TRAINED_COEFFICIENTS;
    }

    // Cache and return
    coefficientCache.set(cacheKey, coeffs);
    console.log(`[Forecast] Loaded coefficients for ${quarter}`);
    return coeffs;

  } catch (error) {
    console.error(`[Forecast] Error loading coefficients for ${quarter}:`, error.message);
    return TRAINED_COEFFICIENTS;
  }
}

/**
 * List all available coefficient quarters
 * @returns {Array<string>} Array of quarter strings
 */
function listAvailableCoefficients() {
  const coeffDir = path.join(__dirname, 'coefficients');

  if (!fs.existsSync(coeffDir)) {
    return [];
  }

  return fs.readdirSync(coeffDir)
    .filter(f => f.endsWith('.json'))
    .map(f => f.replace('.json', ''))
    .sort();
}

/**
 * Clear coefficient cache (useful for reloading after retraining)
 */
function clearCoefficientCache() {
  coefficientCache.clear();
  console.log('[Forecast] Coefficient cache cleared');
}

// ============================================================================
// RIDGE MODEL EXPORT (per-asset, per-horizon 97-factor model)
// ============================================================================

let ridgeModelExport = null;

/**
 * Load the new Ridge model export with per-asset coefficients.
 * 14 assets × 3 horizons, 97 factors each, walk-forward fitted.
 */
function loadRidgeModelExport() {
  if (ridgeModelExport) return ridgeModelExport;

  const exportPath = path.join(__dirname, 'ridge_model_export.json');
  try {
    if (fs.existsSync(exportPath)) {
      const raw = fs.readFileSync(exportPath, 'utf8');
      ridgeModelExport = JSON.parse(raw);
      console.log(`[Forecast] Loaded Ridge model export: ${ridgeModelExport.universe?.length} assets, ` +
        `horizons [${ridgeModelExport.horizons?.join(',')}]`);
      return ridgeModelExport;
    }
  } catch (err) {
    console.warn(`[Forecast] Error loading ridge_model_export.json:`, err.message);
  }
  return null;
}

function clearRidgeModelExportCache() {
  ridgeModelExport = null;
}

// ============================================================================
// TRAINED COEFFICIENTS (Exported from Jupyter - DO NOT MODIFY)
// ============================================================================

const TRAINED_COEFFICIENTS = {
  "metadata": {
    "trainedOn": "2025-12-10",
    "trainingPeriod": "2020-01-01 to 2023-12-31",
    "nextRetrain": "2026-03-10",
    "retrainIntervalDays": 90
  },
  "horizons": {
    "3d": {
      "factors": [
        "ar1_z", "mr_speed_z", "var_ratio_z", "hurst_z", "trend_slope_z",
        "mom_rank_z", "mom_sharpe_z", "vol_skew_z", "gold_mom_z",
        "mstr_mom_z", "m2_mom_z", "m2_beta_z"
      ],
      "factorParams": {
        "ar1_z": { "factor_window": 40, "norm_window": 180 },
        "mr_speed_z": { "factor_window": 40, "norm_window": 180 },
        "var_ratio_z": { "factor_window": 40, "norm_window": 504 },
        "hurst_z": { "factor_window": 20, "norm_window": 504 },
        "trend_slope_z": { "factor_window": 120, "norm_window": 180 },
        "mom_rank_z": { "factor_window": 90, "norm_window": 180 },
        "mom_sharpe_z": { "factor_window": 120, "norm_window": 180 },
        "vol_skew_z": { "factor_window": 20, "norm_window": 180 },
        "gold_mom_z": { "factor_window": 65, "norm_window": 180 },
        "mstr_mom_z": { "factor_window": 65, "norm_window": 180 },
        "m2_mom_z": { "factor_window": 65, "norm_window": 180 },
        "m2_beta_z": { "factor_window": 65, "norm_window": 180 }
      },
      "ridge": {
        "alpha": 1000.0,
        "intercept": 0.0008690218869060334,
        "coefs": {
          "ar1_z": 0.0020758047392829978,
          "mr_speed_z": -0.001986357989501474,
          "var_ratio_z": 0.0016601411061351653,
          "hurst_z": 0.0018147074056350113,
          "trend_slope_z": -0.0002708806334963058,
          "mom_rank_z": 0.00037097533091047687,
          "mom_sharpe_z": 0.0005324002417733859,
          "vol_skew_z": -0.0005633480973967946,
          "gold_mom_z": 0.00014329892434201825,
          "mstr_mom_z": -0.0010369073529375278,
          "m2_mom_z": 0.0011124421407796604,
          "m2_beta_z": -0.00012695904367484014
        }
      },
      "q10": {
        "intercept": -0.05527529788181582,
        "coefs": {
          "ar1_z": 0.0031912876581685077,
          "mr_speed_z": -0.001877813781562669,
          "var_ratio_z": 0.0075931817951438605,
          "hurst_z": 0.0030351860498639793,
          "trend_slope_z": -0.012513673255050511,
          "mom_rank_z": -0.012002120715023792,
          "mom_sharpe_z": 0.0198101774140258,
          "vol_skew_z": -0.002122579473661604,
          "gold_mom_z": -0.0015290177386653562,
          "mstr_mom_z": 0.0013699204310585321,
          "m2_mom_z": 0.0052333550294826605,
          "m2_beta_z": 0.002266572604563405
        }
      },
      "q50": {
        "intercept": 0.0011491606727575695,
        "coefs": {
          "ar1_z": 0.007924104697185179,
          "mr_speed_z": 0.0049798066816792354,
          "var_ratio_z": 0.0012517727081245052,
          "hurst_z": 0.0011536873385660507,
          "trend_slope_z": 0.0007770848427311383,
          "mom_rank_z": 0.0029718405342824316,
          "mom_sharpe_z": -0.0008468965670180329,
          "vol_skew_z": 0.0003829209648405856,
          "gold_mom_z": 0.0006594439452914047,
          "mstr_mom_z": -0.005283071329871545,
          "m2_mom_z": 0.0018259068723724644,
          "m2_beta_z": 0.0012383641968557893
        }
      },
      "q90": {
        "intercept": 0.05713770855131722,
        "coefs": {
          "ar1_z": 0.006187369545496616,
          "mr_speed_z": -0.0008736216966185517,
          "var_ratio_z": -0.003196625535382559,
          "hurst_z": 0.0026168355135496118,
          "trend_slope_z": 0.014776621508191874,
          "mom_rank_z": 0.008774749321626977,
          "mom_sharpe_z": -0.018536684943482273,
          "vol_skew_z": 0.0035492404829803544,
          "gold_mom_z": 0.0006824379490303877,
          "mstr_mom_z": -0.0032747678310829495,
          "m2_mom_z": -0.003300347762301441,
          "m2_beta_z": -0.00125022824062109
        }
      },
      "spreadBaseline": {
        "mean": 0.10397397747729885,
        "std": 0.02794593009905555
      }
    },
    "7d": {
      "factors": [
        "ar1_z", "mr_speed_z", "var_ratio_z", "hurst_z", "trend_slope_z",
        "mom_rank_z", "mom_sharpe_z", "vol_downside_z", "vol_skew_z",
        "nasdaq_beta_z", "spy_beta_z", "gold_mom_z", "gold_beta_z",
        "dxy_mom_z", "mstr_mom_z", "m2_beta_z", "vix_level_z"
      ],
      "factorParams": {
        "ar1_z": { "factor_window": 40, "norm_window": 180 },
        "mr_speed_z": { "factor_window": 40, "norm_window": 180 },
        "var_ratio_z": { "factor_window": 20, "norm_window": 180 },
        "hurst_z": { "factor_window": 20, "norm_window": 360 },
        "trend_slope_z": { "factor_window": 120, "norm_window": 180 },
        "mom_rank_z": { "factor_window": 60, "norm_window": 180 },
        "mom_sharpe_z": { "factor_window": 120, "norm_window": 180 },
        "vol_downside_z": { "factor_window": 60, "norm_window": 504 },
        "vol_skew_z": { "factor_window": 20, "norm_window": 180 },
        "nasdaq_beta_z": { "factor_window": 50, "norm_window": 180 },
        "spy_beta_z": { "factor_window": 50, "norm_window": 180 },
        "gold_mom_z": { "factor_window": 50, "norm_window": 180 },
        "gold_beta_z": { "factor_window": 50, "norm_window": 180 },
        "dxy_mom_z": { "factor_window": 50, "norm_window": 180 },
        "mstr_mom_z": { "factor_window": 50, "norm_window": 180 },
        "m2_beta_z": { "factor_window": 50, "norm_window": 180 },
        "vix_level_z": { "factor_window": 50, "norm_window": 180 }
      },
      "ridge": {
        "alpha": 1000.0,
        "intercept": 0.0016803466557956267,
        "coefs": {
          "ar1_z": 0.005355828206298302,
          "mr_speed_z": -0.0051907235904514355,
          "var_ratio_z": 0.003014165753300156,
          "hurst_z": 0.003067381554070358,
          "trend_slope_z": -0.0006087652367768444,
          "mom_rank_z": -0.0014274898503874888,
          "mom_sharpe_z": 0.0008177002950744159,
          "vol_downside_z": 0.0017680727103265052,
          "vol_skew_z": -0.0012422858206827068,
          "nasdaq_beta_z": 0.0009992665786712474,
          "spy_beta_z": -0.0012099114881542862,
          "gold_mom_z": 0.002542534065607116,
          "gold_beta_z": -0.0042796356117418425,
          "dxy_mom_z": 0.003895800552961862,
          "mstr_mom_z": 0.00013138661151199137,
          "m2_beta_z": 0.0005849190469364432,
          "vix_level_z": -0.0005925440394148915
        }
      },
      "q10": {
        "intercept": -0.07594992032741685,
        "coefs": {
          "ar1_z": -0.04952594153377965,
          "mr_speed_z": -0.07928252393221435,
          "var_ratio_z": -0.004417684519101517,
          "hurst_z": 0.015865889231792968,
          "trend_slope_z": -0.006112735687258508,
          "mom_rank_z": -0.028571959595769775,
          "mom_sharpe_z": 0.012862762849754503,
          "vol_downside_z": -0.0029364487415011707,
          "vol_skew_z": -0.0017813346958452936,
          "nasdaq_beta_z": 0.04088976986420212,
          "spy_beta_z": -0.024029565388104038,
          "gold_mom_z": 0.002551265890068939,
          "gold_beta_z": -0.021862412048570284,
          "dxy_mom_z": 0.007893440250924977,
          "mstr_mom_z": 0.012366825015291683,
          "m2_beta_z": 1.0909315985296875e-06,
          "vix_level_z": -0.0105743462887683
        }
      },
      "q50": {
        "intercept": 0.003363171835488574,
        "coefs": {
          "ar1_z": 0.05155730103362966,
          "mr_speed_z": 0.038230402744652225,
          "var_ratio_z": 0.0032908824212952714,
          "hurst_z": 0.0009592293787192814,
          "trend_slope_z": -0.011782640097973515,
          "mom_rank_z": -0.010831314254554913,
          "mom_sharpe_z": 0.015949481279202385,
          "vol_downside_z": 0.010676958478246179,
          "vol_skew_z": -0.0029627491696872926,
          "nasdaq_beta_z": 0.01491716415785227,
          "spy_beta_z": -0.016154637886631157,
          "gold_mom_z": 0.0038353811849206887,
          "gold_beta_z": -0.00782011941325833,
          "dxy_mom_z": 0.010704667223221664,
          "mstr_mom_z": 0.006882065413275851,
          "m2_beta_z": -0.0006229909770221476,
          "vix_level_z": -0.0021423974752793062
        }
      },
      "q90": {
        "intercept": 0.09610371977016469,
        "coefs": {
          "ar1_z": 0.013605165015715848,
          "mr_speed_z": -0.00488500088955357,
          "var_ratio_z": 0.010176722970202992,
          "hurst_z": -0.01729317218705452,
          "trend_slope_z": 0.016264632422464294,
          "mom_rank_z": -0.005251939359314159,
          "mom_sharpe_z": -0.017791736221516877,
          "vol_downside_z": 0.0005459890515468516,
          "vol_skew_z": 0.008392724292112456,
          "nasdaq_beta_z": 0.03251982361954027,
          "spy_beta_z": -0.04450457182172301,
          "gold_mom_z": 0.0060094724338143335,
          "gold_beta_z": -0.008458293014378393,
          "dxy_mom_z": 0.00990022426853758,
          "mstr_mom_z": -0.007305861597644025,
          "m2_beta_z": 0.004292786030685315,
          "vix_level_z": 0.01695387121792333
        }
      },
      "spreadBaseline": {
        "mean": 0.1705261773508928,
        "std": 0.07476520668349781
      }
    },
    "15d": {
      "factors": [
        "ar1_z", "mr_speed_z", "var_ratio_z", "hurst_z", "trend_slope_z",
        "mom_rank_z", "mom_sharpe_z", "vol_downside_z", "vol_skew_z",
        "gold_mom_z", "gold_beta_z", "dxy_mom_z", "mstr_mom_z",
        "m2_mom_z", "m2_beta_z", "vix_level_z", "vix_chg_z"
      ],
      "factorParams": {
        "ar1_z": { "factor_window": 40, "norm_window": 180 },
        "mr_speed_z": { "factor_window": 40, "norm_window": 180 },
        "var_ratio_z": { "factor_window": 20, "norm_window": 180 },
        "hurst_z": { "factor_window": 20, "norm_window": 504 },
        "trend_slope_z": { "factor_window": 120, "norm_window": 180 },
        "mom_rank_z": { "factor_window": 60, "norm_window": 180 },
        "mom_sharpe_z": { "factor_window": 90, "norm_window": 180 },
        "vol_downside_z": { "factor_window": 60, "norm_window": 180 },
        "vol_skew_z": { "factor_window": 20, "norm_window": 504 },
        "gold_mom_z": { "factor_window": 40, "norm_window": 180 },
        "gold_beta_z": { "factor_window": 40, "norm_window": 180 },
        "dxy_mom_z": { "factor_window": 40, "norm_window": 180 },
        "mstr_mom_z": { "factor_window": 40, "norm_window": 180 },
        "m2_mom_z": { "factor_window": 40, "norm_window": 180 },
        "m2_beta_z": { "factor_window": 40, "norm_window": 180 },
        "vix_level_z": { "factor_window": 40, "norm_window": 180 },
        "vix_chg_z": { "factor_window": 40, "norm_window": 180 }
      },
      "ridge": {
        "alpha": 1000.0,
        "intercept": 0.002815743436130034,
        "coefs": {
          "ar1_z": 0.00859535979059885,
          "mr_speed_z": -0.008467611867994647,
          "var_ratio_z": 0.0036693767978969657,
          "hurst_z": 0.006188319671240928,
          "trend_slope_z": 0.00039680763227323165,
          "mom_rank_z": -0.0033347945645803433,
          "mom_sharpe_z": 0.0029354591535159704,
          "vol_downside_z": 0.006601801499400619,
          "vol_skew_z": 0.00485404694229452,
          "gold_mom_z": 0.002966346181761619,
          "gold_beta_z": -0.0075996731850171174,
          "dxy_mom_z": 0.009507638863411655,
          "mstr_mom_z": -0.0031702507359397626,
          "m2_mom_z": 0.003672236870413534,
          "m2_beta_z": 0.005525105673427335,
          "vix_level_z": 0.00038015407110357925,
          "vix_chg_z": -9.830209468828459e-05
        }
      },
      "q10": {
        "intercept": -0.11960427004180563,
        "coefs": {
          "ar1_z": 0.06173548806306073,
          "mr_speed_z": 0.04130747964347137,
          "var_ratio_z": -0.004174012644037939,
          "hurst_z": 0.05123376288164616,
          "trend_slope_z": -0.02187113587479317,
          "mom_rank_z": -0.03883959220720001,
          "mom_sharpe_z": 0.04519612688487129,
          "vol_downside_z": 0.029879776252275336,
          "vol_skew_z": 0.0034303119908143387,
          "gold_mom_z": 0.025919151214029434,
          "gold_beta_z": -0.006641353940269212,
          "dxy_mom_z": 0.022318100603988977,
          "mstr_mom_z": 0.01894037478689725,
          "m2_mom_z": 0.022021415749676976,
          "m2_beta_z": 0.0017752246710741604,
          "vix_level_z": 0.0029315222880375025,
          "vix_chg_z": -0.012774351601720468
        }
      },
      "q50": {
        "intercept": 0.0016983398970417257,
        "coefs": {
          "ar1_z": 0.061897202067319856,
          "mr_speed_z": 0.03538049695745138,
          "var_ratio_z": -0.0015962667078525117,
          "hurst_z": 0.007025830645195713,
          "trend_slope_z": -0.005831780756789584,
          "mom_rank_z": -0.010375253480208069,
          "mom_sharpe_z": 0.02123675756997287,
          "vol_downside_z": 0.0209108570811415,
          "vol_skew_z": 0.00968985552173636,
          "gold_mom_z": 0.004244683059548365,
          "gold_beta_z": -0.015435558916358824,
          "dxy_mom_z": 0.013407306790424262,
          "mstr_mom_z": -0.0034940275570345264,
          "m2_mom_z": 0.007337534597583822,
          "m2_beta_z": 0.0015236078814299323,
          "vix_level_z": 0.007462164095055124,
          "vix_chg_z": -0.0039517519708132665
        }
      },
      "q90": {
        "intercept": 0.1383487103922011,
        "coefs": {
          "ar1_z": -0.018861234169907704,
          "mr_speed_z": -0.05405811515715797,
          "var_ratio_z": 0.014205399149839204,
          "hurst_z": -0.018115583577366756,
          "trend_slope_z": -0.005328539419056533,
          "mom_rank_z": -0.01319845429474209,
          "mom_sharpe_z": 0.014763159727844144,
          "vol_downside_z": 0.003676504902883515,
          "vol_skew_z": 0.015219596942657176,
          "gold_mom_z": -0.021143887825826102,
          "gold_beta_z": -0.03261384337839168,
          "dxy_mom_z": 0.013128371415678686,
          "mstr_mom_z": -0.03448180799211942,
          "m2_mom_z": 0.0006435080996203268,
          "m2_beta_z": 0.012988418450089245,
          "vix_level_z": 0.011973342705228485,
          "vix_chg_z": 0.009438737251907803
        }
      },
      "spreadBaseline": {
        "mean": 0.23984364736811797,
        "std": 0.1167732990583865
      }
    }
  }
};

// ============================================================================
// STATISTICAL HELPER FUNCTIONS
// ============================================================================

function mean(arr) {
  if (!arr || arr.length === 0) return NaN;
  const valid = arr.filter(x => !isNaN(x) && x !== null);
  if (valid.length === 0) return NaN;
  return valid.reduce((a, b) => a + b, 0) / valid.length;
}

function std(arr) {
  if (!arr || arr.length < 2) return NaN;
  const valid = arr.filter(x => !isNaN(x) && x !== null);
  if (valid.length < 2) return NaN;
  const m = mean(valid);
  const variance = valid.reduce((sum, x) => sum + Math.pow(x - m, 2), 0) / (valid.length - 1);
  return Math.sqrt(variance);
}

function correlation(arr1, arr2) {
  if (arr1.length !== arr2.length || arr1.length < 2) return NaN;
  const n = arr1.length;
  const m1 = mean(arr1);
  const m2 = mean(arr2);
  const s1 = std(arr1);
  const s2 = std(arr2);
  if (s1 === 0 || s2 === 0) return NaN;
  
  let sum = 0;
  for (let i = 0; i < n; i++) {
    sum += (arr1[i] - m1) * (arr2[i] - m2);
  }
  return sum / ((n - 1) * s1 * s2);
}

function covariance(arr1, arr2) {
  if (arr1.length !== arr2.length || arr1.length < 2) return NaN;
  const n = arr1.length;
  const m1 = mean(arr1);
  const m2 = mean(arr2);
  let sum = 0;
  for (let i = 0; i < n; i++) {
    sum += (arr1[i] - m1) * (arr2[i] - m2);
  }
  return sum / (n - 1);
}

function variance(arr) {
  if (!arr || arr.length < 2) return NaN;
  const m = mean(arr);
  return arr.reduce((sum, x) => sum + Math.pow(x - m, 2), 0) / (arr.length - 1);
}

function skewness(arr) {
  if (!arr || arr.length < 3) return NaN;
  const n = arr.length;
  const m = mean(arr);
  const s = std(arr);
  if (s === 0) return NaN;
  
  let sum = 0;
  for (let i = 0; i < n; i++) {
    sum += Math.pow((arr[i] - m) / s, 3);
  }
  return (n / ((n - 1) * (n - 2))) * sum;
}

function percentileRank(arr, value) {
  if (!arr || arr.length === 0) return NaN;
  const sorted = [...arr].sort((a, b) => a - b);
  let count = 0;
  for (const v of sorted) {
    if (v < value) count++;
  }
  return count / sorted.length;
}

// ============================================================================
// FACTOR COMPUTATION FUNCTIONS
// ============================================================================

// AR(1) - Autocorrelation of returns
function computeAR1(returns, window) {
  if (returns.length < window + 1) return NaN;
  const slice = returns.slice(-window);
  const x = slice.slice(0, -1);
  const y = slice.slice(1);
  return correlation(x, y);
}

// Mean Reversion Speed
function computeMRSpeed(returns, window) {
  if (returns.length < window + 1) return NaN;
  const slice = returns.slice(-window);
  const x = slice.slice(0, -1);
  const y = slice.slice(1);
  
  const varX = variance(x);
  if (varX === 0) return NaN;
  
  const cov = covariance(y, x);
  const beta = cov / varX;
  
  if (1 + beta <= 0) return NaN;
  return -Math.log(1 + beta);
}

// Variance Ratio
function computeVarianceRatio(returns, window, q = 5) {
  if (returns.length < window) return NaN;
  const slice = returns.slice(-window);
  
  const var1 = variance(slice);
  if (var1 === 0) return NaN;
  
  // q-period returns
  const qReturns = [];
  for (let i = 0; i <= slice.length - q; i += q) {
    let sum = 0;
    for (let j = 0; j < q; j++) {
      sum += slice[i + j];
    }
    qReturns.push(sum);
  }
  
  if (qReturns.length < 2) return NaN;
  const varQ = variance(qReturns);
  
  return varQ / (q * var1);
}

// Hurst Exponent (R/S method)
function computeHurst(returns, window) {
  if (returns.length < window || window < 20) return NaN;
  const slice = returns.slice(-window);
  
  const m = mean(slice);
  const cumDev = [];
  let cumSum = 0;
  for (let i = 0; i < slice.length; i++) {
    cumSum += slice[i] - m;
    cumDev.push(cumSum);
  }
  
  const r = Math.max(...cumDev) - Math.min(...cumDev);
  const s = std(slice);
  
  if (s === 0 || r === 0) return NaN;
  return Math.log(r / s) / Math.log(slice.length);
}

// Trend Slope (normalized by volatility)
function computeTrendSlope(closes, window) {
  if (closes.length < window) return NaN;
  const slice = closes.slice(-window);
  
  // Simple linear regression
  const n = slice.length;
  const x = Array.from({ length: n }, (_, i) => i);
  const xMean = mean(x);
  const yMean = mean(slice);
  
  let num = 0, den = 0;
  for (let i = 0; i < n; i++) {
    num += (x[i] - xMean) * (slice[i] - yMean);
    den += Math.pow(x[i] - xMean, 2);
  }
  
  const slope = num / den;
  const s = std(slice);
  
  if (s === 0) return NaN;
  return slope / s;
}

// Momentum Rank (percentile of current price in window)
function computeMomRank(closes, window) {
  if (closes.length < window) return NaN;
  const slice = closes.slice(-window);
  const current = slice[slice.length - 1];
  const history = slice.slice(0, -1);
  return percentileRank(history, current);
}

// Momentum Sharpe
function computeMomSharpe(returns, window) {
  if (returns.length < window) return NaN;
  const slice = returns.slice(-window);
  const m = mean(slice);
  const s = std(slice);
  if (s === 0) return NaN;
  return m / s;
}

// Downside Volatility
function computeDownsideVol(returns, window) {
  if (returns.length < window) return NaN;
  const slice = returns.slice(-window);
  const negative = slice.filter(r => r < 0);
  if (negative.length < 2) return NaN;
  return std(negative);
}

// Volatility Skewness
function computeVolSkew(returns, window) {
  if (returns.length < window) return NaN;
  const slice = returns.slice(-window);
  return skewness(slice);
}

// Rolling Beta
function computeRollingBeta(coinReturns, assetReturns, window) {
  if (coinReturns.length < window || assetReturns.length < window) return NaN;
  const coinSlice = coinReturns.slice(-window);
  const assetSlice = assetReturns.slice(-window);
  
  const varAsset = variance(assetSlice);
  if (varAsset === 0) return NaN;
  
  const cov = covariance(coinSlice, assetSlice);
  return cov / varAsset;
}

// Momentum (log return)
function computeMomentum(prices, window) {
  if (prices.length < window + 1) return NaN;
  const current = prices[prices.length - 1];
  const past = prices[prices.length - 1 - window];
  if (past <= 0 || current <= 0) return NaN;
  return Math.log(current / past);
}

// Z-Score - use available history, minimum 30 observations
function computeZScore(value, history, window) {
  // Use available history, but at least 30 observations
  const actualWindow = Math.min(window, history.length);
  if (actualWindow < 30) return NaN;
  
  const slice = history.slice(-actualWindow);
  const m = mean(slice);
  const s = std(slice);
  if (s === 0 || isNaN(s)) return NaN;
  return (value - m) / s;
}

// ============================================================================
// DATA FETCHING
// ============================================================================

// Helper: Sleep for retry delays
const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

// Fetch crypto daily data from Binance (with retry logic)
async function fetchCryptoDaily(symbol, days = 600, maxRetries = 3) {
  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      const url = `https://api.binance.com/api/v3/klines?symbol=${symbol}USDT&interval=1d&limit=${days}`;
      const response = await axios.get(url, { timeout: 30000 });

      return response.data.map(candle => ({
        timestamp: candle[0],
        open: parseFloat(candle[1]),
        high: parseFloat(candle[2]),
        low: parseFloat(candle[3]),
        close: parseFloat(candle[4]),
        volume: parseFloat(candle[5])
      }));
    } catch (error) {
      console.error(`[Fetch] ${symbol} attempt ${attempt}/${maxRetries} failed:`, error.message);
      if (attempt < maxRetries) {
        const delay = 1000 * attempt; // 1s, 2s, 3s
        console.log(`[Fetch] Retrying ${symbol} in ${delay}ms...`);
        await sleep(delay);
      }
    }
  }
  console.error(`[Fetch] ${symbol} failed after ${maxRetries} attempts`);
  return null;
}

// Fetch macro data from Yahoo Finance (with retry logic)
async function fetchYahooDaily(ticker, days = 600, maxRetries = 3) {
  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      const period2 = Math.floor(Date.now() / 1000);
      const period1 = period2 - (days * 24 * 60 * 60);

      const url = `https://query1.finance.yahoo.com/v8/finance/chart/${ticker}?period1=${period1}&period2=${period2}&interval=1d`;

      const response = await axios.get(url, {
        timeout: 30000,
        headers: {
          'User-Agent': 'Mozilla/5.0'
        }
      });

      const result = response.data.chart.result[0];
      const timestamps = result.timestamp;
      const closes = result.indicators.quote[0].close;

      return timestamps.map((ts, i) => ({
        timestamp: ts * 1000,
        close: closes[i]
      })).filter(d => d.close !== null);
    } catch (error) {
      console.error(`[Fetch] ${ticker} attempt ${attempt}/${maxRetries} failed:`, error.message);
      if (attempt < maxRetries) {
        const delay = 1000 * attempt;
        console.log(`[Fetch] Retrying ${ticker} in ${delay}ms...`);
        await sleep(delay);
      }
    }
  }
  console.error(`[Fetch] ${ticker} failed after ${maxRetries} attempts`);
  return null;
}

// Fetch M2 from FRED
async function fetchM2(apiKey = '8e7470ef009ffd473e465770729a7620') {
  try {
    const url = `https://api.stlouisfed.org/fred/series/observations?series_id=M2SL&api_key=${apiKey}&file_type=json&observation_start=2020-01-01`;
    const response = await axios.get(url);
    
    const observations = response.data.observations;
    return observations.map(obs => ({
      date: obs.date,
      value: parseFloat(obs.value)
    })).filter(d => !isNaN(d.value));
  } catch (error) {
    console.error('Error fetching M2:', error.message);
    return null;
  }
}

// ============================================================================
// MAIN FORECAST FUNCTION
// ============================================================================

// Cache for storing daily forecasts
let forecastCache = {
  timestamp: null,
  data: null
};

// Check if cache is valid (refreshes once daily at 8 PM ET)
function isCacheValid() {
  if (!forecastCache.timestamp) return false;

  const now = new Date();
  const cached = new Date(forecastCache.timestamp);

  // Get current hour in ET
  const currentETHour = parseInt(new Date().toLocaleString('en-US', {
    timeZone: 'America/New_York',
    hour: 'numeric',
    hour12: false
  }));

  // Get cached date in ET for comparison
  const cachedETDate = cached.toLocaleDateString('en-US', { timeZone: 'America/New_York' });
  const nowETDate = now.toLocaleDateString('en-US', { timeZone: 'America/New_York' });

  // If it's before 8 PM ET today, cache is valid if fetched since last 8 PM ET
  if (currentETHour < 20) {
    // Before 8 PM ET - cache valid if fetched today OR yesterday (still within 24hr window)
    const yesterday = new Date(now);
    yesterday.setDate(yesterday.getDate() - 1);
    const yesterdayETDate = yesterday.toLocaleDateString('en-US', { timeZone: 'America/New_York' });

    if (cachedETDate === nowETDate) return true;
    if (cachedETDate === yesterdayETDate) return true;
    return false;
  }

  // After 8 PM ET - cache only valid if fetched today after 8 PM ET
  if (cachedETDate === nowETDate) {
    const cachedETHour = parseInt(cached.toLocaleString('en-US', {
      timeZone: 'America/New_York',
      hour: 'numeric',
      hour12: false
    }));
    return cachedETHour >= 20; // Must have been cached after 8 PM ET today
  }

  return false;
}

// Compute all factors for a coin
// coefficients param is optional - if not provided, uses TRAINED_COEFFICIENTS
function computeFactors(cryptoData, macroData, horizon, coefficients = null) {
  const coeffs = coefficients || TRAINED_COEFFICIENTS;
  const horizonConfig = coeffs.horizons?.[`${horizon}d`];
  if (!horizonConfig) return null;
  
  const factors = {};
  const closes = cryptoData.map(d => d.close);
  const returns = [];
  for (let i = 1; i < closes.length; i++) {
    returns.push(Math.log(closes[i] / closes[i - 1]));
  }
  
  // Compute each factor
  for (const factorName of horizonConfig.factors) {
    const params = horizonConfig.factorParams[factorName];
    const fw = params.factor_window;
    const nw = params.norm_window;
    
    let rawValue = NaN;
    let rawHistory = [];
    
    // Statistical factors
    if (factorName === 'ar1_z') {
      rawValue = computeAR1(returns, fw);
      for (let i = nw; i < returns.length; i++) {
        rawHistory.push(computeAR1(returns.slice(0, i), fw));
      }
    } else if (factorName === 'mr_speed_z') {
      rawValue = computeMRSpeed(returns, fw);
      for (let i = nw; i < returns.length; i++) {
        rawHistory.push(computeMRSpeed(returns.slice(0, i), fw));
      }
    } else if (factorName === 'var_ratio_z') {
      rawValue = computeVarianceRatio(returns, fw);
      for (let i = nw; i < returns.length; i++) {
        rawHistory.push(computeVarianceRatio(returns.slice(0, i), fw));
      }
    } else if (factorName === 'hurst_z') {
      rawValue = computeHurst(returns, fw);
      for (let i = nw; i < returns.length; i++) {
        rawHistory.push(computeHurst(returns.slice(0, i), fw));
      }
    } else if (factorName === 'trend_slope_z') {
      rawValue = computeTrendSlope(closes, fw);
      for (let i = nw; i < closes.length; i++) {
        rawHistory.push(computeTrendSlope(closes.slice(0, i), fw));
      }
    } else if (factorName === 'mom_rank_z') {
      rawValue = computeMomRank(closes, fw);
      for (let i = nw; i < closes.length; i++) {
        rawHistory.push(computeMomRank(closes.slice(0, i), fw));
      }
    } else if (factorName === 'mom_sharpe_z') {
      rawValue = computeMomSharpe(returns, fw);
      for (let i = nw; i < returns.length; i++) {
        rawHistory.push(computeMomSharpe(returns.slice(0, i), fw));
      }
    } else if (factorName === 'vol_downside_z') {
      rawValue = computeDownsideVol(returns, fw);
      for (let i = nw; i < returns.length; i++) {
        rawHistory.push(computeDownsideVol(returns.slice(0, i), fw));
      }
    } else if (factorName === 'vol_skew_z') {
      rawValue = computeVolSkew(returns, fw);
      for (let i = nw; i < returns.length; i++) {
        rawHistory.push(computeVolSkew(returns.slice(0, i), fw));
      }
    }
    // Macro momentum factors
    else if (factorName.endsWith('_mom_z')) {
      const macroName = factorName.replace('_mom_z', '');
      const macroKey = {
        'gold': 'gold',
        'dxy': 'dxy',
        'mstr': 'mstr',
        'm2': 'm2',
        'nasdaq': 'nasdaq',
        'spy': 'spy',
        'vix': 'vix',
        'tlt': 'tlt'
      }[macroName];
      
      if (macroKey && macroData[macroKey]) {
        const macroPrices = macroData[macroKey].map(d => d.close);
        rawValue = computeMomentum(macroPrices, fw);
        for (let i = nw + fw; i < macroPrices.length; i++) {
          rawHistory.push(computeMomentum(macroPrices.slice(0, i), fw));
        }
      }
    }
    // Macro beta factors
    else if (factorName.endsWith('_beta_z')) {
      const macroName = factorName.replace('_beta_z', '');
      const macroKey = {
        'gold': 'gold',
        'dxy': 'dxy',
        'mstr': 'mstr',
        'm2': 'm2',
        'nasdaq': 'nasdaq',
        'spy': 'spy',
        'vix': 'vix',
        'tlt': 'tlt'
      }[macroName];
      
      if (macroKey && macroData[macroKey]) {
        const macroPrices = macroData[macroKey].map(d => d.close);
        const macroReturns = [];
        for (let i = 1; i < macroPrices.length; i++) {
          macroReturns.push(Math.log(macroPrices[i] / macroPrices[i - 1]));
        }
        
        // Align lengths
        const minLen = Math.min(returns.length, macroReturns.length);
        const alignedCrypto = returns.slice(-minLen);
        const alignedMacro = macroReturns.slice(-minLen);
        
        rawValue = computeRollingBeta(alignedCrypto, alignedMacro, fw);
        for (let i = nw + fw; i < minLen; i++) {
          rawHistory.push(computeRollingBeta(alignedCrypto.slice(0, i), alignedMacro.slice(0, i), fw));
        }
      }
    }
    // VIX level
    else if (factorName === 'vix_level_z') {
      if (macroData.vix) {
        const vixPrices = macroData.vix.map(d => d.close);
        rawValue = vixPrices[vixPrices.length - 1];
        rawHistory = vixPrices.slice(-nw);
      }
    }
    // VIX change
    else if (factorName === 'vix_chg_z') {
      if (macroData.vix) {
        const vixPrices = macroData.vix.map(d => d.close);
        if (vixPrices.length > 5) {
          rawValue = vixPrices[vixPrices.length - 1] - vixPrices[vixPrices.length - 6];
          for (let i = nw + 5; i < vixPrices.length; i++) {
            rawHistory.push(vixPrices[i] - vixPrices[i - 5]);
          }
        }
      }
    }
    
    // Compute z-score
    if (!isNaN(rawValue) && rawHistory.length >= 30) {
      const validHistory = rawHistory.filter(v => !isNaN(v) && v !== null);
      if (validHistory.length >= 30) {
        const zScore = computeZScore(rawValue, validHistory, nw);
        factors[factorName] = isNaN(zScore) ? 0 : zScore;
      } else {
        factors[factorName] = 0;
      }
    } else {
      factors[factorName] = 0; // Default to 0 if cannot compute
    }
  }
  
  return factors;
}

// Apply coefficients to get prediction - skip NaN factors
function applyCoefficients(factors, coefs, intercept) {
  let prediction = intercept;
  for (const [factor, value] of Object.entries(factors)) {
    if (coefs[factor] !== undefined && !isNaN(value) && value !== null) {
      prediction += coefs[factor] * value;
    }
  }
  return prediction;
}

// Traverse a single sklearn GBR decision tree (XGBoost-compatible nested format)
function traverseForecastTree(factors, node) {
  let current = node;
  let depth = 0;
  while (current.leaf === undefined && depth < 20) {
    const raw = factors[current.split];
    const val = (raw !== undefined && !isNaN(raw)) ? raw : 0;
    const targetId = val < current.split_condition ? current.yes : current.no;
    const next = current.children?.find(c => c.nodeid === targetId);
    if (!next) break;
    current = next;
    depth++;
  }
  return current?.leaf ?? 0;
}

// Predict using sklearn GBR ensemble: base_score + lr * sum(tree_outputs)
function predictGBMQuantile(factors, gbmModel) {
  const trees = gbmModel.trees || [];
  const baseScore = gbmModel.base_score || 0;
  const lr = gbmModel.learning_rate || 0.05;
  let sum = 0;
  for (const tree of trees) {
    sum += traverseForecastTree(factors, tree);
  }
  return baseScore + lr * sum;
}

// Generate forecast for a single coin
// Optional asOfDate parameter for walk-forward backtesting
// crossSectional: { allReturns: { symbol: returns[] } } for cross-asset factors
async function generateCoinForecast(symbol, cryptoData, macroData, asOfDate = null, crossSectional = null) {
  // Load appropriate coefficients based on date (for GBM Q10/Q90 + legacy Ridge fallback)
  const coefficients = asOfDate
    ? loadCoefficientsForDate(asOfDate)
    : TRAINED_COEFFICIENTS;

  // Check if this symbol has per-asset Ridge coefficients from the new export
  const ridgeExport = loadRidgeModelExport();
  const hasNewRidge = !asOfDate && ridgeExport?.assets?.[symbol];

  // Pre-compute 97-factor vector for new Ridge model (once, reused across horizons)
  let factors97 = null;
  if (hasNewRidge && cryptoData && cryptoData.length > 60) {
    const cryptoDataObj = {
      closes: cryptoData.map(d => d.close),
      opens: cryptoData.map(d => d.open || d.close),
      highs: cryptoData.map(d => d.high || d.close),
      lows: cryptoData.map(d => d.low || d.close),
      volumes: cryptoData.map(d => d.volume || 0),
    };
    factors97 = computeFactors97(cryptoDataObj, macroData, crossSectional || {});
  }

  const forecasts = {};

  for (const horizon of [3, 7, 15]) {
    const horizonConfig = coefficients.horizons[`${horizon}d`];
    if (!horizonConfig) continue;

    // Compute OLD factors (needed for GBM Q10/Q90 and legacy Ridge fallback)
    const factors = computeFactors(cryptoData, macroData, horizon);
    if (!factors) continue;

    // --- Ridge Q50 Point Forecast ---
    let ridgeReturn;
    let modelSource = 'legacy';

    if (hasNewRidge && factors97) {
      // Use new per-asset Ridge model (97 factors, walk-forward fitted)
      const horizonKey = `${horizon}D`;
      const assetModel = ridgeExport.assets[symbol][horizonKey];
      if (assetModel?.coefficients) {
        ridgeReturn = assetModel.intercept || 0;
        const factorOrder = assetModel.factor_order || [];
        for (const factorName of factorOrder) {
          const coef = assetModel.coefficients[factorName];
          const val = factors97[factorName];
          if (coef !== undefined && val !== undefined && isFinite(val)) {
            ridgeReturn += coef * val;
          }
        }
        modelSource = 'ridge_v2';
      } else {
        // Fallback to old Ridge if horizon missing in export
        ridgeReturn = applyCoefficients(factors, horizonConfig.ridge.coefs, horizonConfig.ridge.intercept);
      }
    } else {
      // Use old universal Ridge model
      ridgeReturn = applyCoefficients(factors, horizonConfig.ridge.coefs, horizonConfig.ridge.intercept);
    }

    // --- GBM Q10/Q90 (uses old factors — separate model) ---
    let q10, q50, q90;

    if (horizonConfig.q10?.gbm?.trees && horizonConfig.q90?.gbm?.trees) {
      // GBM tree structures available — use actual quantile predictions
      q10 = predictGBMQuantile(factors, horizonConfig.q10.gbm);
      q90 = predictGBMQuantile(factors, horizonConfig.q90.gbm);
      q50 = ridgeReturn;  // Ridge as stable point forecast

      // Guard quantile crossings: if GBM produces q10 > q50 or q50 > q90, re-center
      if (q10 > q50 || q50 > q90) {
        const gbmSpread = Math.abs(q90 - q10);
        q10 = ridgeReturn - gbmSpread / 2;
        q90 = ridgeReturn + gbmSpread / 2;
      }
    } else {
      // Legacy: fixed intercept-based spreads (backward compatible with old coefficient files)
      const q10Intercept = horizonConfig.q10?.intercept || -0.05;
      const q90Intercept = horizonConfig.q90?.intercept || 0.05;
      const spreadHalf = (q90Intercept - q10Intercept) / 2;
      q10 = ridgeReturn - spreadHalf;
      q50 = ridgeReturn;
      q90 = ridgeReturn + spreadHalf;
    }

    // Calculate spread and confidence
    const spread = q90 - q10;
    const spreadMean = horizonConfig.spreadBaseline.mean;
    const spreadStd = horizonConfig.spreadBaseline.std;
    const spreadZScore = isNaN(spread) ? 0 : (spread - spreadMean) / spreadStd;
    const confidence = spreadZScore < 0 ? 'HIGH' : 'LOW';

    // Determine direction - handle NaN
    let direction = 'FLAT';
    if (!isNaN(ridgeReturn)) {
      if (ridgeReturn > 0.005) direction = 'LONG';
      else if (ridgeReturn < -0.005) direction = 'SHORT';
    }

    // Safe formatting function
    const safeFormat = (val) => isNaN(val) ? '0.00%' : (val * 100).toFixed(2) + '%';
    const safeRaw = (val) => isNaN(val) ? 0 : val;

    forecasts[`${horizon}d`] = {
      ridgeReturn: safeFormat(ridgeReturn),
      ridgeReturnRaw: safeRaw(ridgeReturn),
      q10: safeFormat(q10),
      q10Raw: safeRaw(q10),
      q50: safeFormat(q50),
      q50Raw: safeRaw(q50),
      q90: safeFormat(q90),
      q90Raw: safeRaw(q90),
      spread: safeFormat(spread),
      spreadZScore: isNaN(spreadZScore) ? '0.00' : spreadZScore.toFixed(2),
      confidence,
      direction,
      modelSource,
      factors: Object.fromEntries(
        Object.entries(factors).map(([k, v]) => [k, isNaN(v) ? '0.000' : v.toFixed(3)])
      )
    };
  }

  return forecasts;
}

// Main forecast function
async function generateAllForecasts(forceRefresh = false) {
  // Check cache
  if (!forceRefresh && isCacheValid() && forecastCache.data) {
    console.log('Returning cached forecasts');
    return forecastCache.data;
  }

  console.log('Generating fresh forecasts...');

  // Fetch macro data
  console.log('Fetching macro data...');
  const macroData = {};

  const macroTickers = {
    nasdaq: '^IXIC',
    spy: 'SPY',
    gold: 'GC=F',
    dxy: 'DX-Y.NYB',
    vix: '^VIX',
    tlt: 'TLT',
    tnx: '^TNX',   // 10-Year Treasury yield (needed for BETA_TNX_* factors)
    mstr: 'MSTR',
    mara: 'MARA'
  };

  for (const [name, ticker] of Object.entries(macroTickers)) {
    const data = await fetchYahooDaily(ticker);
    if (data) {
      macroData[name] = data;
      console.log(`  ✓ ${name}: ${data.length} days`);
    }
  }

  // Fetch M2
  const m2Data = await fetchM2();
  if (m2Data) {
    // Convert to daily (forward fill)
    const m2Daily = [];
    for (const obs of m2Data) {
      const date = new Date(obs.date);
      m2Daily.push({ timestamp: date.getTime(), close: obs.value });
    }
    macroData.m2 = m2Daily;
    console.log(`  ✓ m2: ${m2Daily.length} observations`);
  }

  // Fetch crypto data and generate forecasts
  // Extended to match CONFIG.symbols (22 coins total)
  const coins = [
    'BTC', 'ETH', 'BNB', 'XRP', 'SOL', 'TRX', 'DOGE', 'ADA', 'LINK', 'XLM',
    'BCH', 'SUI', 'AVAX', 'HBAR', 'LTC', 'SHIB', 'DOT',
    'ETC', 'XMR', 'EOS', 'NEO', 'ATOM'
  ];

  // Pass 1: Fetch all crypto OHLCV data
  console.log('Fetching crypto data...');
  const allCryptoData = {};
  for (let i = 0; i < coins.length; i++) {
    const coin = coins[i];
    if (i > 0) await sleep(150);
    const cryptoData = await fetchCryptoDaily(coin);
    if (cryptoData) {
      allCryptoData[coin] = cryptoData;
      console.log(`  ✓ ${coin}: ${cryptoData.length} bars`);
    } else {
      console.warn(`  ✗ ${coin}: no data returned`);
    }
  }

  // Pass 2: Build cross-sectional returns for XS factors (XSMOM, XS_ZSCORE, etc.)
  const crossSectional = { allReturns: {} };
  for (const [coin, data] of Object.entries(allCryptoData)) {
    const rets = [];
    for (let i = 1; i < data.length; i++) {
      if (data[i].close > 0 && data[i - 1].close > 0) {
        rets.push(Math.log(data[i].close / data[i - 1].close));
      }
    }
    if (rets.length > 30) {
      crossSectional.allReturns[coin] = rets;
    }
  }
  console.log(`  Cross-sectional: ${Object.keys(crossSectional.allReturns).length} assets`);

  // Pass 3: Generate forecasts with cross-sectional context
  const forecasts = {};
  console.log('Generating coin forecasts...');
  for (const coin of coins) {
    if (!allCryptoData[coin]) continue;
    forecasts[coin] = await generateCoinForecast(coin, allCryptoData[coin], macroData, null, crossSectional);
    const src = forecasts[coin]?.['7d']?.modelSource || 'unknown';
    console.log(`  ✓ ${coin}: forecast generated (${src})`);
  }

  // Build result
  const ridgeExport = loadRidgeModelExport();
  const result = {
    metadata: {
      ...TRAINED_COEFFICIENTS.metadata,
      generatedAt: new Date().toISOString(),
      generatedAtET: new Date().toLocaleString('en-US', { timeZone: 'America/New_York' }),
      ridgeModelVersion: ridgeExport ? 'v2_97factor' : 'legacy',
      ridgeModelDate: ridgeExport?.export_date || null,
      ridgeUniverse: ridgeExport?.universe || [],
    },
    forecasts
  };

  // Update cache
  forecastCache = {
    timestamp: Date.now(),
    data: result
  };

  return result;
}

// Get forecast for a single coin
async function getCoinForecast(symbol) {
  const all = await generateAllForecasts();
  return all.forecasts[symbol.toUpperCase()] || null;
}

// Get model metadata
function getModelMetadata() {
  return TRAINED_COEFFICIENTS.metadata;
}

// ============================================================================
// FACTOR SNAPSHOT FOR GRADIENT BOOSTING
// ============================================================================

// Cache for macro data (reuse within same session)
let macroDataCache = {
  timestamp: null,
  data: null
};

// Check if macro cache is fresh (within 1 hour)
function isMacroCacheFresh() {
  if (!macroDataCache.timestamp) return false;
  const age = Date.now() - macroDataCache.timestamp;
  return age < 60 * 60 * 1000; // 1 hour
}

/**
 * Get a comprehensive factor snapshot for all symbols
 * Used by trade-logger for gradient boosting training data
 * @param {Array<string>} symbols - Array of crypto symbols (e.g., ['BTC', 'ETH'])
 * @returns {Object} Aggregated factors across all symbols + cross-asset factors
 */
async function getFactorSnapshot(symbols) {
  // Fetch or use cached macro data
  let macroData = macroDataCache.data;

  if (!isMacroCacheFresh() || !macroData) {
    console.log('[FactorSnapshot] Fetching macro data...');
    macroData = {};

    const macroTickers = {
      nasdaq: '^IXIC',
      spy: 'SPY',
      gold: 'GC=F',
      dxy: 'DX-Y.NYB',
      vix: '^VIX',
      tlt: 'TLT',
      mstr: 'MSTR'
    };

    for (const [name, ticker] of Object.entries(macroTickers)) {
      const data = await fetchYahooDaily(ticker);
      if (data) {
        macroData[name] = data;
      }
    }

    // Fetch M2
    const m2Data = await fetchM2();
    if (m2Data) {
      const m2Daily = m2Data.map(obs => ({
        timestamp: new Date(obs.date).getTime(),
        close: obs.value
      }));
      macroData.m2 = m2Daily;
    }

    macroDataCache = {
      timestamp: Date.now(),
      data: macroData
    };
  }

  // Collect per-symbol factors
  const symbolFactors = {};

  for (const symbol of symbols) {
    const cryptoData = await fetchCryptoDaily(symbol.toUpperCase().replace('USDT', ''));
    if (!cryptoData) continue;

    // Compute factors for 7d horizon (most comprehensive factor set)
    const factors = computeFactors(cryptoData, macroData, 7);
    if (factors) {
      symbolFactors[symbol] = factors;
    }
  }

  // Calculate aggregated factors (mean across symbols)
  const allSymbolKeys = Object.keys(symbolFactors);
  const numSymbols = allSymbolKeys.length;

  if (numSymbols === 0) {
    return getEmptyFactorSnapshot();
  }

  // Core statistical factors - aggregate
  const coreFactors = ['ar1_z', 'mr_speed_z', 'var_ratio_z', 'hurst_z', 'trend_slope_z',
                       'mom_rank_z', 'mom_sharpe_z', 'vol_downside_z', 'vol_skew_z'];

  const aggregated = {};

  // Calculate averages for core factors
  for (const factor of coreFactors) {
    const values = allSymbolKeys
      .map(s => symbolFactors[s][factor])
      .filter(v => !isNaN(v) && v !== null);

    aggregated[`avg_${factor}`] = values.length > 0
      ? values.reduce((a, b) => a + b, 0) / values.length
      : 0;
  }

  // Cross-asset factors (same for all symbols, take from first)
  const firstSymbol = allSymbolKeys[0];
  const crossAssetFactors = [
    'gold_mom_z', 'nasdaq_mom_z', 'spy_mom_z', 'dxy_mom_z', 'mstr_mom_z', 'm2_mom_z', 'vix_mom_z',
    'gold_beta_z', 'nasdaq_beta_z', 'spy_beta_z', 'm2_beta_z',
    'vix_level_z', 'vix_chg_z'
  ];

  for (const factor of crossAssetFactors) {
    // Cross-asset factors from first symbol (they're macro, same for all)
    const value = symbolFactors[firstSymbol][factor];
    aggregated[factor] = (value !== undefined && !isNaN(value)) ? value : 0;
  }

  // Also compute cross-symbol dispersion (useful for regime detection)
  for (const factor of coreFactors) {
    const values = allSymbolKeys
      .map(s => symbolFactors[s][factor])
      .filter(v => !isNaN(v) && v !== null);

    if (values.length > 1) {
      const m = values.reduce((a, b) => a + b, 0) / values.length;
      const variance = values.reduce((sum, v) => sum + Math.pow(v - m, 2), 0) / (values.length - 1);
      aggregated[`std_${factor}`] = Math.sqrt(variance);
    } else {
      aggregated[`std_${factor}`] = 0;
    }
  }

  return {
    // Raw aggregated factors
    ...aggregated,

    // Metadata
    _numSymbols: numSymbols,
    _symbols: allSymbolKeys,
    _timestamp: new Date().toISOString(),

    // Per-symbol factors for detailed analysis
    _perSymbol: symbolFactors
  };
}

/**
 * Get an empty factor snapshot (when no data available)
 */
function getEmptyFactorSnapshot() {
  return {
    avg_ar1_z: 0,
    avg_mr_speed_z: 0,
    avg_var_ratio_z: 0,
    avg_hurst_z: 0,
    avg_trend_slope_z: 0,
    avg_mom_rank_z: 0,
    avg_mom_sharpe_z: 0,
    avg_vol_downside_z: 0,
    avg_vol_skew_z: 0,
    gold_mom_z: 0,
    nasdaq_mom_z: 0,
    spy_mom_z: 0,
    dxy_mom_z: 0,
    mstr_mom_z: 0,
    m2_mom_z: 0,
    vix_mom_z: 0,
    gold_beta_z: 0,
    nasdaq_beta_z: 0,
    spy_beta_z: 0,
    m2_beta_z: 0,
    vix_level_z: 0,
    vix_chg_z: 0,
    std_ar1_z: 0,
    std_mr_speed_z: 0,
    std_var_ratio_z: 0,
    std_hurst_z: 0,
    std_trend_slope_z: 0,
    std_mom_rank_z: 0,
    std_mom_sharpe_z: 0,
    std_vol_downside_z: 0,
    std_vol_skew_z: 0,
    _numSymbols: 0,
    _symbols: [],
    _timestamp: new Date().toISOString(),
    _perSymbol: {}
  };
}

// ============================================================================
// MACRO BETAS & PRICES FOR SCENARIO ANALYSIS
// ============================================================================

// Cache for macro data
let macroBetasCache = {
  timestamp: null,
  data: null
};

let macroPricesCache = {
  timestamp: null,
  data: null,
  days: null
};

const MACRO_BETA_TTL = 60 * 60 * 1000; // 1 hour cache
const MACRO_PRICES_TTL = 60 * 60 * 1000; // 1 hour cache

// Get current macro betas for BTC
async function getMacroBetas(forceRefresh = false) {
  // Check cache
  if (!forceRefresh && macroBetasCache.data &&
      Date.now() - macroBetasCache.timestamp < MACRO_BETA_TTL) {
    return macroBetasCache.data;
  }

  console.log('[Macro] Computing macro betas for BTC...');

  const macroTickers = {
    gold: 'GC=F',
    spy: 'SPY',
    dxy: 'DX-Y.NYB',
    vix: '^VIX',
    mstr: 'MSTR'
  };

  const betaWindow = 60; // 60-day rolling beta

  try {
    // Fetch BTC data
    const btcData = await fetchCryptoDaily('BTC', 120);
    if (!btcData || btcData.length < betaWindow) {
      console.error('[Macro] Insufficient BTC data');
      return null;
    }

    const btcReturns = [];
    for (let i = 1; i < btcData.length; i++) {
      if (btcData[i].close > 0 && btcData[i-1].close > 0) {
        btcReturns.push(Math.log(btcData[i].close / btcData[i-1].close));
      }
    }

    const results = {};

    // Fetch all macros in parallel for speed
    const betaPromises = Object.entries(macroTickers).map(async ([name, ticker]) => {
      try {
        const macroData = await fetchYahooDaily(ticker, 120);
        if (!macroData || macroData.length < betaWindow) {
          console.log(`[Macro] Insufficient data for ${name}`);
          return { name, result: { beta: null, momentum7d: null, currentPrice: null } };
        }

        // Calculate returns
        const macroReturns = [];
        for (let i = 1; i < macroData.length; i++) {
          if (macroData[i].close > 0 && macroData[i-1].close > 0) {
            macroReturns.push(Math.log(macroData[i].close / macroData[i-1].close));
          }
        }

        // Align lengths
        const minLen = Math.min(btcReturns.length, macroReturns.length, betaWindow);
        const btcSlice = btcReturns.slice(-minLen);
        const macroSlice = macroReturns.slice(-minLen);

        // Compute beta
        const varMacro = variance(macroSlice);
        let beta = null;
        if (varMacro > 0) {
          const cov = covariance(btcSlice, macroSlice);
          beta = cov / varMacro;
        }

        // 7-day momentum
        const prices = macroData.map(d => d.close);
        let momentum7d = null;
        if (prices.length >= 8) {
          const current = prices[prices.length - 1];
          const past = prices[prices.length - 8];
          if (past > 0 && current > 0) {
            momentum7d = (current - past) / past;
          }
        }

        return {
          name,
          result: {
            beta: beta !== null ? parseFloat(beta.toFixed(4)) : null,
            momentum7d: momentum7d !== null ? parseFloat((momentum7d * 100).toFixed(2)) : null,
            currentPrice: macroData[macroData.length - 1]?.close || null
          }
        };

      } catch (err) {
        console.error(`[Macro] Error fetching ${name}:`, err.message);
        return { name, result: { beta: null, momentum7d: null, currentPrice: null } };
      }
    });

    const betaResults = await Promise.all(betaPromises);
    betaResults.forEach(({ name, result }) => {
      results[name] = result;
    });

    // Cache results
    macroBetasCache.data = results;
    macroBetasCache.timestamp = Date.now();

    console.log('[Macro] Betas computed:', results);
    return results;

  } catch (error) {
    console.error('[Macro] Error computing betas:', error);
    return null;
  }
}

// Get historical macro prices for overlay
async function getMacroPrices(days = 90, forceRefresh = false) {
  // Check cache - reuse if same days requested and not expired
  if (!forceRefresh && macroPricesCache.data &&
      macroPricesCache.days === days &&
      Date.now() - macroPricesCache.timestamp < MACRO_PRICES_TTL) {
    console.log('[Macro] Returning cached macro prices');
    return macroPricesCache.data;
  }

  console.log(`[Macro] Fetching ${days} days of macro prices...`);

  const macroTickers = {
    gold: 'GC=F',
    spy: 'SPY',
    dxy: 'DX-Y.NYB',
    vix: '^VIX',
    mstr: 'MSTR'
  };

  const results = {};

  // Fetch all in parallel for speed
  const fetchPromises = Object.entries(macroTickers).map(async ([name, ticker]) => {
    try {
      const data = await fetchYahooDaily(ticker, days + 10);
      if (data && data.length > 0) {
        const startPrice = data[0].close;
        return {
          name,
          data: data.slice(-days).map(d => ({
            timestamp: d.timestamp,
            price: d.close,
            normalized: (d.close / startPrice) * 100
          }))
        };
      }
      return { name, data: [] };
    } catch (err) {
      console.error(`[Macro] Error fetching ${name} prices:`, err.message);
      return { name, data: [] };
    }
  });

  const fetchResults = await Promise.all(fetchPromises);
  fetchResults.forEach(({ name, data }) => {
    results[name] = data;
  });

  // Cache results
  macroPricesCache.data = results;
  macroPricesCache.days = days;
  macroPricesCache.timestamp = Date.now();

  console.log('[Macro] Prices cached');
  return results;
}

module.exports = {
  generateAllForecasts,
  getCoinForecast,
  getModelMetadata,
  getFactorSnapshot,
  getEmptyFactorSnapshot,
  TRAINED_COEFFICIENTS,
  // Export helper functions for external use
  mean,
  std,
  computeFactors,
  // Macro scenario functions
  getMacroBetas,
  getMacroPrices,
  // Data fetching functions for return attribution
  fetchYahooDaily,
  fetchCryptoDaily,
  // Walk-forward coefficient functions
  loadCoefficientsForDate,
  getQuarterBeforeDate,
  listAvailableCoefficients,
  clearCoefficientCache,
  // For walk-forward backtesting
  generateCoinForecast,
  // GBM quantile prediction
  predictGBMQuantile,
  // Ridge model export (97-factor per-asset model)
  loadRidgeModelExport,
  clearRidgeModelExportCache,
};