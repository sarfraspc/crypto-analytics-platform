/**
 * API service module for crypto analytics platform.
 *
 * Provides functions for fetching price forecasts, sentiment analysis,
 * on-chain metrics, technical patterns, and agent chat interactions.
 * @module services/api
 */

const rawApiBase =
  import.meta.env.VITE_API_URL ||
  import.meta.env.REACT_APP_API_URL ||
  (import.meta.env.MODE === 'production' ? '/api' : 'http://localhost:8000')

/** @constant {string} API_BASE_URL - Base URL for API requests */
const API_BASE_URL = rawApiBase.replace(/\/$/, '')

/** @constant {string} GENERIC_MARKET_SYMBOL - Default symbol for market queries */
const GENERIC_MARKET_SYMBOL = 'BTC'

/**
 * Handle API response and parse JSON or text.
 * @param {Response} response - Fetch response object
 * @returns {Promise<Object|string>} Parsed response body
 * @throws {Error} If response is not ok
 */
const handleResponse = async (response) => {
  const contentType = response.headers.get('content-type')
  const isJSON = contentType && contentType.includes('application/json')
  const body = isJSON ? await response.json() : await response.text()
  if (!response.ok) {
    const message = typeof body === 'string' ? body : body?.error || 'Request failed'
    throw new Error(message)
  }
  return body
}

/**
 * Make an API request with automatic response handling.
 * @param {string} path - API endpoint path
 * @param {Object} [options] - Fetch options
 * @returns {Promise<Object>} Parsed response
 */
const request = (path, options) => fetch(`${API_BASE_URL}${path}`, options).then(handleResponse)

/**
 * Build URL query string from parameters object.
 * @param {Object} [params={}] - Query parameters
 * @returns {string} Query string with leading '?' or empty string
 */
const buildQuery = (params = {}) => {
  const query = new URLSearchParams()
  Object.entries(params).forEach(([key, value]) => {
    if (value === undefined || value === null || value === '') return
    query.append(key, value)
  })
  const queryString = query.toString()
  return queryString ? `?${queryString}` : ''
}

/**
 * Parse raw forecast text into structured data points.
 * @param {string} [rawText=''] - Raw forecast text with timestamp and price
 * @returns {Array<Object>} Array of {timestamp, predicted_close} objects
 */
const parseRawForecastTable = (rawText = '') => {
  const rows = []
  const lines = rawText.split('\n')

  for (const line of lines) {
    const match = line.match(/(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2})\s+(-?\d+(?:\.\d+)?)/)
    if (match) {
      const timestamp = match[1].replace(' ', 'T')
      const price = Number(match[2])

      rows.push({
        timestamp,
        predicted_close: price,
      })
    }
  }

  return rows
}

/**
 * Normalize forecast API response into consistent format.
 * @param {Object} [payload={}] - Raw forecast payload
 * @returns {Object} Normalized forecast with model, points, lastPoint, rawText
 */
const normalizeForecastPayload = (payload = {}) => {
  let points = []

  if (Array.isArray(payload.forecast_points) && payload.forecast_points.length) {
    points = payload.forecast_points
      .map((point) => {
        if (!point || typeof point !== 'object') return null

        const price = Number(point.predicted_close ?? point.price ?? point.value)
        let ts = point.timestamp || point.ts

        if (ts && typeof ts === 'string' && ts.includes(' ')) {
          ts = ts.replace(' ', 'T')
        }

        if (!Number.isFinite(price) || price <= 0) {
          return null
        }

        return {
          timestamp: ts,
          predicted_close: price,
        }
      })
      .filter(Boolean)
  }

  if (points.length === 0) {
    const raw = payload.raw_text || payload.forecast?.raw_text
    if (typeof raw === 'string') {
      const parsedFromRaw = parseRawForecastTable(raw)
      const realisticPoints = parsedFromRaw.filter((entry) => Number(entry.predicted_close) > 1000)
      if (realisticPoints.length > 0) {
        points = realisticPoints
      }
    }
  }

  if (points.length > 0 && (!points[0].timestamp || points[0].timestamp === 'null')) {
    const baseDate = payload.generated_at
      ? new Date(payload.generated_at)
      : payload.timestamp
        ? new Date(payload.timestamp)
        : new Date()
    points = points.map((point, index) => ({
      ...point,
      timestamp: new Date(baseDate.getTime() + (index + 1) * 60 * 60 * 1000).toISOString(),
    }))
  }

  const lastPoint =
    payload.last_point ||
    payload.forecast?.last_point ||
    (points.length ? points[points.length - 1] : null)

  return {
    model: payload.model_used || payload.forecast?.model_used || 'sarimax_v3',
    generatedAt: payload.generated_at || payload.timestamp || payload.forecast?.generated_at || null,
    points,
    lastPoint,
    rawText: payload.raw_text || payload.forecast?.raw_text || null,
  }
}

/**
 * Normalize patterns data into array format.
 * @param {Array|Object} input - Patterns as array or object keyed by symbol
 * @returns {Array<Object>} Array of pattern objects with symbol property
 */
const normalizePatternsList = (input) => {
  if (Array.isArray(input)) {
    return input
  }
  if (input && typeof input === 'object') {
    return Object.entries(input).map(([symbol, details]) => ({
      symbol,
      ...(typeof details === 'object' ? details : { pattern: String(details) }),
    }))
  }
  return []
}

/**
 * Normalize aggregated sentiment data into consistent format.
 * @param {Object} [raw={}] - Raw sentiment data
 * @returns {Object} Normalized sentiment with top_sentiment, scores
 */
const normalizeAggregatedSentiment = (raw = {}) => {
  if (!raw || typeof raw !== 'object') return {}

  // Support both dashboard-style { top_sentiment, top_confidence, scores: { bullish, ... } }
  // and asset-style { top_sentiment, top_confidence, bullish_score, ... }.
  const scores = raw.scores || {}

  const bullish = raw.bullish_score ?? scores.bullish ?? scores.BULLISH ?? 0
  const bearish = raw.bearish_score ?? scores.bearish ?? scores.BEARISH ?? 0
  const neutral = raw.neutral_score ?? scores.neutral ?? scores.NEUTRAL ?? 0

  let topSentiment = raw.top_sentiment || raw.sentiment || null
  let topConfidence = raw.top_confidence ?? raw.confidence

  const scoreMap = {
    BULLISH: typeof bullish === 'number' ? bullish : 0,
    BEARISH: typeof bearish === 'number' ? bearish : 0,
    NEUTRAL: typeof neutral === 'number' ? neutral : 0,
  }

  if (!topSentiment) {
    const sorted = Object.entries(scoreMap).sort((a, b) => b[1] - a[1])
    if (sorted[0] && sorted[0][1] > 0) {
      const [label, value] = sorted[0]
      topSentiment = label
      topConfidence = value
    }
  }

  const normalized = {
    top_sentiment: topSentiment || 'UNKNOWN',
    top_confidence:
      typeof topConfidence === 'number'
        ? topConfidence
        : scoreMap[(topSentiment || '').toUpperCase()] || 0.5,
    bullish_score: scoreMap.BULLISH,
    bearish_score: scoreMap.BEARISH,
    neutral_score: scoreMap.NEUTRAL,
  }

  return normalized
}

/**
 * Fetch price forecast for a symbol.
 * @param {string} symbol - Crypto symbol (e.g., 'BTC')
 * @param {Object} [options] - Forecast options
 * @param {number} [options.horizonDays=3] - Forecast horizon in days
 * @param {string} [options.startDate] - Start date for forecast
 * @returns {Promise<Object>} Forecast data with points and metadata
 */
export const getPriceForecast = async (symbol, { horizonDays = 3, startDate } = {}) => {
  const response = await request(
    `/price/forecast/${symbol}${buildQuery({ horizon_days: horizonDays, start_date: startDate })}`,
  )
  const normalized = normalizeForecastPayload(response)
  return {
    ...response,
    model_used: normalized.model,
    forecast_points: normalized.points,
    last_point: normalized.lastPoint,
    raw_text: normalized.rawText,
    generated_at: normalized.generatedAt,
  }
}

/**
 * Fetch sentiment analysis for a symbol.
 * @param {string} [symbol='BTC'] - Crypto symbol
 * @param {Object} [options] - Analysis options
 * @param {number} [options.k=5] - Number of sources to retrieve
 * @param {boolean} [options.refresh=false] - Force refresh cache
 * @param {number} [options.daysBack=7] - Days of history to analyze
 * @returns {Promise<Object>} Sentiment data with aggregated scores and sources
 */
export const getSentimentAnalysis = (symbol = GENERIC_MARKET_SYMBOL, { k = 5, refresh = false, daysBack = 7 } = {}) =>
  request(`/sentiment/asset/${symbol}${buildQuery({ k, refresh, days_back: daysBack })}`).then((result) => {
    const rawAggregated =
      result.aggregated ||
      result.sentiment?.aggregated ||
      result.sentiment ||
      result.rag?.aggregated ||
      {}

    const aggregated = normalizeAggregatedSentiment(rawAggregated)

    const sources =
      (Array.isArray(result.sources) && result.sources) ||
      (Array.isArray(result.sentiment?.sources) && result.sentiment.sources) ||
      (Array.isArray(result.rag?.sources) && result.rag.sources) ||
      []

    return {
      ...result,
      aggregated,
      sources,
    }
  })

/**
 * Fetch recent sentiment sources across all assets.
 * @param {Object} [options] - Query options
 * @param {number} [options.k=5] - Number of sources
 * @param {boolean} [options.refresh=true] - Force refresh
 * @returns {Promise<Object>} Recent sources with aggregated sentiment
 */
export const getRecentSentimentSources = ({ k = 5, refresh = true } = {}) =>
  request(`/sentiment/sources/recent${buildQuery({ k, refresh })}`).then((result) => ({
    ...result,
    aggregated: result.aggregated || {},
    sources: Array.isArray(result.sources) ? result.sources : [],
  }))

/**
 * Fetch current Fear & Greed Index.
 * @returns {Promise<Object>} FNG data with current value and classification
 */
export const getFngCurrent = () =>
  request('/sentiment/fng/current').then((result) => ({
    ...result,
    fng: result.fng || {},
  }))

/**
 * Fetch on-chain metrics for a symbol.
 * @param {string} [symbol='BTC'] - Crypto symbol
 * @param {string} [window='24h'] - Time window for metrics
 * @returns {Promise<Object>} On-chain metrics data
 */
export const getOnChainMetrics = (symbol = GENERIC_MARKET_SYMBOL, window = '24h') =>
  request(`/onchain/metrics${buildQuery({ window, symbol })}`).then((result) => ({
    ...result,
    metrics: result.metrics || {},
  }))

/**
 * Fetch available trading symbols.
 * @param {Object} [options] - Query options
 * @param {string} [options.exchange] - Exchange filter
 * @param {string} [options.interval] - Time interval
 * @param {number} [options.limit] - Max symbols to return
 * @returns {Promise<Array>} List of available symbols
 */
export const getAvailableSymbols = ({ exchange, interval, limit } = {}) =>
  request(`/price/symbols${buildQuery({ exchange, interval, limit })}`)

/**
 * Fetch technical analysis patterns.
 * @param {Object} [options] - Query options
 * @param {string} [options.exchange] - Exchange filter
 * @param {string} [options.interval='1d'] - Candle interval
 * @param {number} [options.limit=20] - Max patterns to return
 * @returns {Promise<Object>} Technical patterns data
 */
export const getTechnicalPatterns = ({ exchange, interval = '1d', limit = 20 } = {}) =>
  request(`/onchain/patterns${buildQuery({ exchange, interval, limit })}`).then((result) => ({
    ...result,
    patterns: normalizePatternsList(result.patterns),
  }))

/**
 * Fetch symbols with detected patterns.
 * @param {Object} [options] - Query options
 * @param {string} [options.exchange] - Exchange filter
 * @param {string} [options.interval='1d'] - Candle interval
 * @param {number} [options.limit=200] - Max symbols to return
 * @returns {Promise<Array>} Symbols with pattern data
 */
export const getPatternSymbols = ({ exchange, interval = '1d', limit = 200 } = {}) =>
  request(`/onchain/pattern-symbols${buildQuery({ exchange, interval, limit })}`)

/**
 * Fetch combined insight summary for dashboard.
 * @param {string} symbol - Crypto symbol
 * @param {Object} [options] - Query options
 * @param {number} [options.horizonDays=3] - Forecast horizon
 * @param {string} [options.window='24h'] - Metrics time window
 * @param {number} [options.kDocs=5] - Number of sentiment docs
 * @returns {Promise<Object>} Combined forecast, sentiment, and metrics
 */
export const getInsightSummary = async (symbol, { horizonDays = 3, window = '24h', kDocs = 5 } = {}) => {
  const overview = await request(
    `/dashboard/overview/${symbol}${buildQuery({ horizon_days: horizonDays, window, k_docs: kDocs })}`,
  )
  const normalized = overview.forecast ? normalizeForecastPayload(overview.forecast) : null
  return {
    ...overview,
    forecast: normalized
      ? {
          ...overview.forecast,
          model_used: normalized.model,
          forecast_points: normalized.points,
          last_point: normalized.lastPoint,
          raw_text: normalized.rawText,
          generated_at: normalized.generatedAt,
        }
      : overview.forecast,
  }
}

/**
 * Send chat message to AI agent.
 * @param {string} symbol - Crypto symbol context
 * @param {string} question - User question
 * @param {Array} [history=[]] - Conversation history
 * @param {Object} [options={}] - Agent options (horizon, etc.)
 * @returns {Promise<Object>} Agent response with insights
 */
export const sendChatMessage = (symbol, question, history = [], options = {}) => {
  const payload = {
    question,
    options: {
      horizon: 7,
      ...options,
    },
    history,
  }

  return request(`/agent/insight/${symbol}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
}
