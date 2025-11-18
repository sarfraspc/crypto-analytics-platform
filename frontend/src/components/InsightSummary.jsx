import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Brain } from 'lucide-react'

// Mock hooks and services for demo
const useTheme = () => ({ isDark: true })
const useSymbol = () => ({ symbol: 'BTC' })
const getInsightSummary = async (symbol) => {
  await new Promise(resolve => setTimeout(resolve, 1000))
  return {
    sentiment: { top_sentiment: 'NEUTRAL' },
    onchain: { 
      market_pressure_index: 0.33,
      whale_transactions: 'limited',
      dominant_flow: 'balanced',
      market_bias: 'neutral'
    },
    forecast: { model_used: 'sarimax_v3' },
    generated_at: new Date().toISOString()
  }
}

const Loader = ({ label }) => (
  <div className="flex items-center gap-2">
    <div className="animate-spin h-4 w-4 border-2 border-indigo-500 border-t-transparent rounded-full"></div>
    <span className="text-gray-400 text-sm">{label}</span>
  </div>
)

const ErrorBox = ({ message, onRetry }) => (
  <div className="bg-red-500/10 border border-red-500/30 rounded p-3">
    <p className="text-red-300 text-sm">{message}</p>
    <button onClick={onRetry} className="mt-2 text-xs text-red-400 underline">Retry</button>
  </div>
)

// Animated Crypto Symbol Component - 3D Neural Network Brain
const AnimatedCryptoSymbol = () => {
  return (
    <div className="relative w-20 h-20" style={{ perspective: '1000px' }}>
      {/* 3D Container with tilt animation */}
      <div className="absolute inset-0 animate-tilt-3d" style={{ transformStyle: 'preserve-3d' }}>
        
        {/* Outer energy rings with depth */}
        <div className="absolute inset-0 rounded-full border-2 border-indigo-500/20 animate-ping-slow" 
             style={{ transform: 'translateZ(20px)' }}></div>
        <div className="absolute inset-2 rounded-full border-2 border-purple-500/30 animate-ping-slower"
             style={{ transform: 'translateZ(10px)' }}></div>
        
        {/* Rotating orbital ring with gradient - 3D effect */}
        <div className="absolute inset-0 animate-spin-3d" style={{ transformStyle: 'preserve-3d' }}>
          <div className="absolute inset-0 rounded-full" style={{
            background: 'conic-gradient(from 0deg, transparent 0%, rgba(99, 102, 241, 0.8) 10%, transparent 20%, transparent 80%, rgba(168, 85, 247, 0.8) 90%, transparent 100%)',
            transform: 'translateZ(15px)',
            boxShadow: '0 0 30px rgba(99, 102, 241, 0.5)',
          }}></div>
        </div>
        
        {/* Back orbital ring for depth */}
        <div className="absolute inset-0 animate-spin-3d-reverse" style={{ transformStyle: 'preserve-3d' }}>
          <div className="absolute inset-0 rounded-full opacity-40" style={{
            background: 'conic-gradient(from 180deg, transparent 0%, rgba(236, 72, 153, 0.6) 10%, transparent 20%, transparent 80%, rgba(168, 85, 247, 0.6) 90%, transparent 100%)',
            transform: 'translateZ(-10px) rotateX(60deg)',
            boxShadow: '0 0 20px rgba(236, 72, 153, 0.3)',
          }}></div>
        </div>
        
        {/* Neural network nodes - central brain with 3D depth */}
        <div className="absolute inset-0 flex items-center justify-center" style={{ transformStyle: 'preserve-3d' }}>
          <div className="relative w-12 h-12" style={{ transformStyle: 'preserve-3d' }}>
            {/* Center core - AI brain with 3D pop */}
            <div className="absolute inset-0 flex items-center justify-center" style={{ transformStyle: 'preserve-3d' }}>
              <div className="w-10 h-10 rounded-full animate-pulse-glow-3d" 
                   style={{
                     background: 'radial-gradient(circle at 30% 30%, rgba(255, 255, 255, 0.8), rgba(99, 102, 241, 0.9) 20%, rgba(168, 85, 247, 0.9) 60%, rgba(236, 72, 153, 0.9))',
                     transform: 'translateZ(30px)',
                     boxShadow: '0 0 40px rgba(99, 102, 241, 0.8), 0 0 80px rgba(168, 85, 247, 0.6), inset -5px -5px 20px rgba(0, 0, 0, 0.3), inset 5px 5px 20px rgba(255, 255, 255, 0.2)',
                   }}>
                {/* Shine effect */}
                <div className="absolute top-2 left-2 w-4 h-4 bg-white/40 rounded-full blur-sm"></div>
              </div>
            </div>
            
            {/* Neural nodes around the center with 3D positioning */}
            <div className="absolute top-0 left-1/2 w-3 h-3 bg-gradient-to-br from-indigo-300 to-indigo-500 rounded-full -translate-x-1/2 animate-float-3d shadow-lg shadow-indigo-500/50"
                 style={{ transform: 'translateX(-50%) translateZ(20px)' }}></div>
            <div className="absolute bottom-0 left-1/2 w-3 h-3 bg-gradient-to-br from-purple-300 to-purple-500 rounded-full -translate-x-1/2 animate-float-3d animation-delay-300 shadow-lg shadow-purple-500/50"
                 style={{ transform: 'translateX(-50%) translateZ(15px)' }}></div>
            <div className="absolute left-0 top-1/2 w-3 h-3 bg-gradient-to-br from-pink-300 to-pink-500 rounded-full -translate-y-1/2 animate-float-3d animation-delay-600 shadow-lg shadow-pink-500/50"
                 style={{ transform: 'translateY(-50%) translateZ(18px)' }}></div>
            <div className="absolute right-0 top-1/2 w-3 h-3 bg-gradient-to-br from-cyan-300 to-cyan-500 rounded-full -translate-y-1/2 animate-float-3d animation-delay-900 shadow-lg shadow-cyan-500/50"
                 style={{ transform: 'translateY(-50%) translateZ(22px)' }}></div>
            
            {/* Connecting lines - neural connections with glow */}
            <svg className="absolute inset-0 w-full h-full animate-pulse-slow" viewBox="0 0 48 48" style={{ transform: 'translateZ(5px)' }}>
              <defs>
                <linearGradient id="lineGrad1" x1="0%" y1="0%" x2="0%" y2="100%">
                  <stop offset="0%" style={{ stopColor: 'rgb(99, 102, 241)', stopOpacity: 0.6 }} />
                  <stop offset="100%" style={{ stopColor: 'rgb(168, 85, 247)', stopOpacity: 0.3 }} />
                </linearGradient>
              </defs>
              <line x1="24" y1="8" x2="24" y2="19" stroke="url(#lineGrad1)" strokeWidth="2" filter="url(#glow)" />
              <line x1="24" y1="29" x2="24" y2="40" stroke="url(#lineGrad1)" strokeWidth="2" filter="url(#glow)" />
              <line x1="8" y1="24" x2="19" y2="24" stroke="url(#lineGrad1)" strokeWidth="2" filter="url(#glow)" />
              <line x1="29" y1="24" x2="40" y2="24" stroke="url(#lineGrad1)" strokeWidth="2" filter="url(#glow)" />
              <filter id="glow">
                <feGaussianBlur stdDeviation="2" result="coloredBlur"/>
                <feMerge>
                  <feMergeNode in="coloredBlur"/>
                  <feMergeNode in="SourceGraphic"/>
                </feMerge>
              </filter>
            </svg>
          </div>
        </div>
        
        {/* Data particles flowing in 3D orbit */}
        <div className="absolute inset-0 animate-orbit-3d" style={{ transformStyle: 'preserve-3d' }}>
          <div className="absolute top-1 left-1/2 w-2 h-2 bg-cyan-400 rounded-full shadow-lg shadow-cyan-400/70"
               style={{ transform: 'translateZ(25px)', boxShadow: '0 0 15px rgba(34, 211, 238, 0.8)' }}></div>
        </div>
        <div className="absolute inset-0 animate-orbit-3d-delayed" style={{ transformStyle: 'preserve-3d' }}>
          <div className="absolute top-1 left-1/2 w-2 h-2 bg-pink-400 rounded-full shadow-lg shadow-pink-400/70"
               style={{ transform: 'translateZ(20px)', boxShadow: '0 0 15px rgba(236, 72, 153, 0.8)' }}></div>
        </div>
        <div className="absolute inset-0 animate-orbit-3d-more-delayed" style={{ transformStyle: 'preserve-3d' }}>
          <div className="absolute top-1 left-1/2 w-2 h-2 bg-indigo-400 rounded-full shadow-lg shadow-indigo-400/70"
               style={{ transform: 'translateZ(28px)', boxShadow: '0 0 15px rgba(99, 102, 241, 0.8)' }}></div>
        </div>
      </div>
    </div>
  )
}

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
    <div className="relative rounded-xl p-6 shadow-lg bg-gray-900 border border-indigo-700 overflow-hidden">
      {/* Gradient and blur overlay */}
      <div className="absolute inset-0 rounded-xl bg-gradient-to-br from-indigo-900/50 to-purple-900/50 backdrop-blur-xl"></div>

      {/* Content */}
      <div className="relative z-10">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <Brain className="text-indigo-500" size={24} />
            <h3 className="text-lg font-semibold text-white">
              AI Insight Summary
            </h3>
          </div>
          
          {/* Animated Symbol on the right */}
          <AnimatedCryptoSymbol />
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