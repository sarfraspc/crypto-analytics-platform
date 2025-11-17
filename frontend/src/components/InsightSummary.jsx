import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Brain } from 'lucide-react'
import { useTheme } from '../hooks/useTheme'
import { useSymbol } from '../hooks/useSymbol'
import { getInsightSummary } from '../services/api'
import Loader from './Loader'
import ErrorBox from './ErrorBox'

const InsightSummary = () => {
  const { isDark } = useTheme()
  const { symbol } = useSymbol()
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const isMountedRef = useRef(true)
  const [lastUpdated, setLastUpdated] = useState(null)

  const fetchSummary = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const result = await getInsightSummary(symbol)
      if (isMountedRef.current) {
        setData(result)
        setLastUpdated(result.generated_at || result.timestamp || new Date().toISOString())
      }
    } catch (err) {
      if (isMountedRef.current) {
        setError(err.message ?? 'Unable to load insight summary')
      }
    } finally {
      if (isMountedRef.current) {
        setLoading(false)
      }
    }
  }, [symbol])

  useEffect(() => {
    isMountedRef.current = true
    fetchSummary()
    return () => {
      isMountedRef.current = false
    }
  }, [fetchSummary])

  const handleRetry = () => {
    if (!loading) {
      fetchSummary()
    }
  }

  const summary = useMemo(() => {
    if (!data) return null
    const sentimentLabel = data.sentiment?.top_sentiment ?? 'mixed'
    const pressure = data.onchain?.market_pressure_index
    const whaleTx = data.onchain?.whale_transactions
    const flow = data.onchain?.dominant_flow
    const bias = data.onchain?.market_bias

    const pressureText =
      typeof pressure === 'number' ? pressure.toFixed(2) : pressure

    const computed = `Market mood for ${symbol} is ${sentimentLabel.toLowerCase()} with ${
      whaleTx ? `${whaleTx} whale transactions` : 'limited whale data'
    }. ${flow ? `Dominant flow is ${flow}.` : ''} ${
      pressure != null ? `Market pressure index stands at ${pressureText}.` : ''
    } ${
      bias ? `TA bias is ${bias}.` : ''
    }`.trim()

    if (computed.trim().length < 50 || !computed.toLowerCase().includes(symbol.toLowerCase())) {
      return data.response || `Limited dashboard data for ${symbol}. Sentiment: ${sentimentLabel}.`
    }
    return computed
  }, [data, symbol])

  const badges = useMemo(() => {
    if (!data) return []
    const result = []
    if (data.sentiment?.top_sentiment) {
      result.push({ label: data.sentiment.top_sentiment, tone: 'green' })
    }
    if (data.onchain?.dominant_flow) {
      result.push({ label: `${data.onchain.dominant_flow} flow`, tone: 'blue' })
    }
    if (data.forecast?.model_used) {
      result.push({ label: data.forecast.model_used, tone: 'purple' })
    }
    return result
  }, [data])

  const formattedUpdated =
    lastUpdated && !Number.isNaN(new Date(lastUpdated).getTime())
      ? new Date(lastUpdated).toLocaleString()
      : null

  return (
    <div
      className="relative rounded-xl p-6 shadow-lg bg-gray-900 border border-indigo-700 overflow-hidden"
    >
      {/* Gradient and blur overlay */}
      <div className="absolute inset-0 rounded-xl bg-gradient-to-br from-indigo-900/50 to-purple-900/50 backdrop-blur-xl"></div>

      {/* Content */}
      <div className="relative z-10">
        <div className="flex items-center gap-2 mb-4">
          <Brain className="text-indigo-500" size={24} />
          <h3 className="text-lg font-semibold text-white">
            AI Insight Summary
          </h3>
        </div>

        {loading && <Loader label="Gathering latest dashboard data..." />}
        {!loading && error && <ErrorBox message={error} onRetry={handleRetry} />}
        {!loading && !error && summary && (
          <p className="text-sm leading-relaxed text-gray-300">
            {summary}
          </p>
        )}
        {!loading && !error && !summary && data?.response && (
          <p className="text-sm leading-relaxed text-gray-300">{data.response}</p>
        )}

        {!loading && !error && badges.length > 0 && (
          <div className="flex flex-wrap gap-2 mt-4">
            {badges.map((badge) => (
              <span
                key={badge.label}
                className={`px-3 py-1 rounded-full text-xs font-medium ${
                  badge.tone === 'green'
                    ? 'bg-green-500/20 text-green-300'
                    : badge.tone === 'blue'
                      ? 'bg-blue-500/20 text-blue-300'
                      : 'bg-purple-500/20 text-purple-300'
                }`}
              >
                {badge.label}
              </span>
            ))}
          </div>
        )}

        {formattedUpdated && (
          <p className="mt-4 text-xs text-gray-400">
            Last updated {formattedUpdated}
          </p>
        )}
      </div>
    </div>
  )
}

export default InsightSummary
