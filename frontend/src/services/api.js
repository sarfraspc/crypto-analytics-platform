const API_BASE_URL = (import.meta.env.VITE_API_URL || import.meta.env.REACT_APP_API_URL || 'http://localhost:8000').replace(/\/$/, '')
const GENERIC_MARKET_SYMBOL = 'BTC'

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

const request = (path, options) => fetch(`${API_BASE_URL}${path}`, options).then(handleResponse)

const buildQuery = (params = {}) => {
  const query = new URLSearchParams()
  Object.entries(params).forEach(([key, value]) => {
    if (value === undefined || value === null || value === '') return
    query.append(key, value)
  })
  const queryString = query.toString()
  return queryString ? `?${queryString}` : ''
}

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

export const getSentimentAnalysis = (symbol = GENERIC_MARKET_SYMBOL, { k = 5, refresh = false, daysBack = 7 } = {}) =>
  request(`/sentiment/asset/${symbol}${buildQuery({ k, refresh, days_back: daysBack })}`).then((result) => ({
    ...result,
    aggregated: result.aggregated || {},
    sources: Array.isArray(result.sources) ? result.sources : [],
  }))

export const getRecentSentimentSources = ({ k = 5, refresh = true } = {}) =>
  request(`/sentiment/sources/recent${buildQuery({ k, refresh })}`).then((result) => ({
    ...result,
    aggregated: result.aggregated || {},
    sources: Array.isArray(result.sources) ? result.sources : [],
  }))

export const getFngCurrent = () =>
  request('/sentiment/fng/current').then((result) => ({
    ...result,
    fng: result.fng || {},
  }))

export const getOnChainMetrics = (symbol = GENERIC_MARKET_SYMBOL, window = '24h') =>
  request(`/onchain/metrics${buildQuery({ window })}`).then((result) => ({
    ...result,
    metrics: result.metrics || {},
  }))

export const getAvailableSymbols = ({ exchange, interval, limit } = {}) =>
  request(`/price/symbols${buildQuery({ exchange, interval, limit })}`)

export const getTechnicalPatterns = ({ exchange = 'binance', interval = '1d', limit = 20 } = {}) =>
  request(`/onchain/patterns${buildQuery({ exchange, interval, limit })}`).then((result) => ({
    ...result,
    patterns: normalizePatternsList(result.patterns),
  }))

export const getPatternSymbols = ({ exchange = 'binance', interval = '1d', limit = 200 } = {}) =>
  request(`/onchain/pattern-symbols${buildQuery({ exchange, interval, limit })}`)

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
