import { useEffect, useMemo, useState } from 'react'
import { Activity, ChevronDown, ChevronUp } from 'lucide-react'
import { useTheme } from '../hooks/useTheme'
import { useSymbol } from '../hooks/useSymbol'
import { getSentimentAnalysis } from '../services/api'
import Loader from './Loader'
import ErrorBox from './ErrorBox'

const SentimentGauge = () => {
  const { isDark } = useTheme()
  const { symbol } = useSymbol()
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [expanded, setExpanded] = useState(false)
  const [lastUpdated, setLastUpdated] = useState(null)

  useEffect(() => {
    let isMounted = true
    const fetchData = async () => {
      setLoading(true)
      setError(null)
      try {
        const result = await getSentimentAnalysis(symbol)
        if (isMounted) {
          setData(result)
          setLastUpdated(result.generated_at || result.timestamp || new Date().toISOString())
        }
      } catch (err) {
        if (isMounted) setError(err.message ?? 'Unable to load sentiment data')
      } finally {
        if (isMounted) setLoading(false)
      }
    }

    fetchData()
    return () => {
      isMounted = false
    }
  }, [symbol])

  const aggregated = data?.aggregated || {}
  const sentimentScores = {
    bullish: aggregated.bullish_score ?? aggregated.BULLISH ?? 0,
    bearish: aggregated.bearish_score ?? aggregated.BEARISH ?? 0,
    neutral: aggregated.neutral_score ?? aggregated.NEUTRAL ?? 0,
  }
  const label = aggregated.top_sentiment ?? aggregated.sentiment ?? 'UNKNOWN'
  const topScore = sentimentScores[label.toLowerCase()] ?? aggregated.top_confidence ?? 0.5
  const score = topScore
  const confidence = aggregated.top_confidence ?? score ?? 0.5
  const sources = Array.isArray(data?.sources) ? data.sources : []
  const formattedUpdated =
    lastUpdated && !Number.isNaN(new Date(lastUpdated).getTime())
      ? new Date(lastUpdated).toLocaleString()
      : null

  const gaugeColor = useMemo(() => {
    if (score > 0.6) return 'text-green-500'
    if (score < 0.4) return 'text-red-500'
    return 'text-yellow-500'
  }, [score])

  return (
    <div
      className={`rounded-xl p-6 shadow-lg ${
        isDark
          ? 'bg-slate-800/50 backdrop-blur-xl border border-slate-700'
          : 'bg-white border border-gray-200'
      }`}
    >
      <div className="flex items-center justify-between mb-6 gap-3">
        <div className="flex items-center gap-2">
          <Activity className="text-indigo-500" size={24} />
          <h3 className={`text-lg font-semibold ${isDark ? 'text-white' : 'text-gray-900'}`}>
            Market Sentiment
          </h3>
        </div>
        <div className="flex items-center gap-3">
          {formattedUpdated && (
            <span className={`text-xs ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
              Last updated {formattedUpdated}
            </span>
          )}
          <button
            type="button"
            onClick={() => setExpanded((prev) => !prev)}
            className={`p-2 rounded-lg transition-all ${
              isDark ? 'hover:bg-slate-700' : 'hover:bg-gray-100'
            }`}
          >
            {expanded ? <ChevronUp size={20} /> : <ChevronDown size={20} />}
          </button>
        </div>
      </div>

      {loading && <Loader label="Analyzing sentiment" />}
      {!loading && error && <ErrorBox message={error} />}

      {!loading && !error && (
        <>
          <div className="flex items-center justify-center mb-6">
            <div className="relative w-48 h-48">
              <svg className="transform -rotate-90 w-48 h-48">
                <circle cx="96" cy="96" r="80" stroke={isDark ? '#334155' : '#e5e7eb'} strokeWidth="12" fill="none" />
                <circle
                  cx="96"
                  cy="96"
                  r="80"
                  stroke={score > 0.6 ? '#10b981' : score < 0.4 ? '#ef4444' : '#eab308'}
                  strokeWidth="12"
                  fill="none"
                  strokeDasharray={`${Math.min(score, 1) * 502.4} 502.4`}
                  className="transition-all duration-1000"
                />
              </svg>
              <div className="absolute inset-0 flex flex-col items-center justify-center">
                <span className={`text-4xl font-bold ${gaugeColor}`}>{Math.round(Math.min(score, 1) * 100)}%</span>
                <span className={`text-sm font-medium mt-1 ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
                  {label}
                </span>
                <span className={`text-xs mt-1 ${isDark ? 'text-gray-500' : 'text-gray-500'}`}>
                  {Math.round(Math.min(confidence, 1) * 100)}% confidence
                </span>
              </div>
            </div>
          </div>

          {expanded && (
            <div className="space-y-3 mt-6 pt-6 border-t border-slate-700 dark:border-slate-700/70">
              <h4 className={`text-sm font-semibold mb-3 ${isDark ? 'text-gray-300' : 'text-gray-700'}`}>
                Top Sources
              </h4>
              {sources.length > 0 ? (
                sources.map((source, index) => (
                  <div
                    key={source.metadata?.doc_id || index}
                    className={`p-3 rounded-lg ${isDark ? 'bg-slate-700/50' : 'bg-gray-50'}`}
                  >
                    <div className="flex items-center gap-2 mb-1">
                      <span
                        className={`text-xs px-2 py-0.5 rounded-full ${
                          isDark ? 'bg-slate-600 text-gray-300' : 'bg-gray-200 text-gray-700'
                        }`}
                      >
                        {source.metadata?.source || 'source'}
                      </span>
                      {typeof source.score === 'number' && (
                        <span
                          className={`text-xs font-medium ${
                            source.score > 0.7
                              ? 'text-green-400'
                              : source.score < 0.4
                                ? 'text-red-400'
                                : 'text-yellow-400'
                          }`}
                        >
                          {Math.round(Math.min(source.score, 1) * 100)}%
                        </span>
                      )}
                    </div>
                    <p className={`text-xs ${isDark ? 'text-gray-300' : 'text-gray-700'}`}>
                      {(source.content || source.excerpt || 'No excerpt available.').slice(0, 100)}
                      {source.content && source.content.length > 100 ? '…' : ''}
                    </p>
                  </div>
                ))
              ) : (
                <p className={`text-xs italic ${isDark ? 'text-gray-500' : 'text-gray-400'}`}>
                  No sources available right now—try refreshing the page or switching symbols.
                </p>
              )}
            </div>
          )}
        </>
      )}
    </div>
  )
}

export default SentimentGauge
