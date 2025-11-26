import { useMemo } from 'react'
import { Activity } from 'lucide-react'
import { useTheme } from '../hooks/useTheme'

const SentimentGauge = ({ aggregated }) => { // Removed lastUpdated prop
  const { isDark } = useTheme()

  const sentimentScores = {
    bullish: aggregated.bullish_score ?? aggregated.BULLISH ?? 0,
    bearish: aggregated.bearish_score ?? aggregated.BEARISH ?? 0,
    neutral: aggregated.neutral_score ?? aggregated.NEUTRAL ?? 0,
  }
  const label = aggregated.top_sentiment ?? aggregated.sentiment ?? 'UNKNOWN'
  const topScore = sentimentScores[label.toLowerCase()] ?? aggregated.top_confidence ?? 0.5
  const score = topScore
  const confidence = aggregated.top_confidence ?? score ?? 0.5
  
  const gaugeColor = useMemo(() => {
    if (score > 0.6) return 'text-success-400'
    if (score < 0.4) return 'text-red-400'
    return 'text-yellow-400'
  }, [score])

  return (
    <div>
      <div className="flex items-center justify-between mb-6 gap-3">
        <div className="flex items-center gap-2">
          <Activity className="text-accent-500" size={24} />
          <h3 className={`text-lg font-semibold ${isDark ? 'text-white' : 'text-gray-900'}`}>
            Market Sentiment
          </h3>
        </div>
      </div>

      <div className="flex items-center justify-center">
        <div className="relative w-48 h-48">
          <svg className="transform -rotate-90 w-48 h-48">
            <circle cx="96" cy="96" r="80" stroke={isDark ? '#1e293b' : '#e5e7eb'} strokeWidth="12" fill="none" />
            <circle
              cx="96"
              cy="96"
              r="80"
              stroke={score > 0.6 ? '#22c55e' : score < 0.4 ? '#ef4444' : '#eab308'}
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
    </div>
  )
}

export default SentimentGauge