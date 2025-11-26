import { useMemo } from 'react'
import { Gauge } from 'lucide-react'
import { useTheme } from '../hooks/useTheme'

const clamp = (val, min, max) => Math.min(Math.max(val, min), max)

const FngGauge = ({ fng }) => {
  const { isDark } = useTheme()
  const value = clamp(Number(fng?.value ?? fng?.current_value ?? 50), 0, 100)
  const sentiment = (fng?.sentiment || fng?.classification || 'UNKNOWN').toString().toUpperCase()
  const bias = (fng?.market_bias || 'NEUTRAL').toString().toUpperCase()

  const score = value / 100
  const gaugeColor = useMemo(() => {
    if (bias === 'BULLISH' || bias === 'OPPORTUNITY') return 'text-success-400'
    if (bias === 'BEARISH') return 'text-red-400'
    if (bias === 'CAUTION') return 'text-yellow-400'
    return score > 0.6 ? 'text-success-400' : score < 0.4 ? 'text-red-400' : 'text-yellow-400'
  }, [bias, score])

  return (
    <div className="flex flex-col items-center">
      <div className="flex items-center gap-2 mb-6">
        <Gauge className="text-accent-500" size={24} />
        <h3 className={`text-lg font-semibold ${isDark ? 'text-white' : 'text-gray-900'}`}>Fear &amp; Greed</h3>
      </div>

      <div className="flex items-center justify-center">
        <div className="relative w-32 h-32 sm:w-48 sm:h-48">
          <svg className="transform -rotate-90 w-full h-full" viewBox="0 0 192 192">
            <circle
              cx="96"
              cy="96"
              r="80"
              stroke={isDark ? '#1e293b' : '#e5e7eb'}
              strokeWidth="12"
              fill="none"
            />
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
            <span className={`text-2xl sm:text-4xl font-bold ${gaugeColor}`}>{Math.round(score * 100)}</span>
            <span className={`text-xs sm:text-sm font-medium mt-1 ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
              {sentiment}
            </span>
            <span className={`text-xs mt-1 ${isDark ? 'text-gray-500' : 'text-gray-500'}`}>{bias} bias</span>
          </div>
        </div>
      </div>
    </div>
  )
}

export default FngGauge
