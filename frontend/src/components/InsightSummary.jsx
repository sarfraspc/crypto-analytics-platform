import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Brain } from 'lucide-react'
import { useTheme } from '../hooks/useTheme'
import { useSymbol } from '../hooks/useSymbol'
import { getInsightSummary, getPriceForecast } from '../services/api'

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

// Animated Crypto Symbol Component - Sci-Fi Blockchain Hologram
const AnimatedCryptoSymbol = () => {
  return (
    <div className="relative w-24 h-24" style={{ perspective: '1200px' }}>
      {/* 3D Container with sci-fi rotation */}
      <div className="absolute inset-0 animate-scifi-rotate" style={{ transformStyle: 'preserve-3d' }}>
        
        {/* Holographic scanning lines */}
        <div className="absolute inset-0 overflow-hidden rounded-full opacity-30">
          <div className="absolute w-full h-1 bg-gradient-to-r from-transparent via-cyan-400 to-transparent animate-scan"></div>
        </div>
        
        {/* Outer hexagonal ring - blockchain structure */}
        <div className="absolute inset-0 animate-hex-spin" style={{ transformStyle: 'preserve-3d' }}>
          <svg className="absolute inset-0 w-full h-full" viewBox="0 0 100 100" style={{ transform: 'translateZ(25px)' }}>
            <polygon points="50,5 90,27.5 90,72.5 50,95 10,72.5 10,27.5" 
                     fill="none" 
                     stroke="url(#hexGrad)" 
                     strokeWidth="2"
                     className="animate-pulse-glow-hex" />
            <defs>
              <linearGradient id="hexGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" style={{ stopColor: 'rgb(6, 182, 212)', stopOpacity: 0.8 }}>
                  <animate attributeName="stop-color" values="rgb(6, 182, 212); rgb(168, 85, 247); rgb(6, 182, 212)" dur="3s" repeatCount="indefinite" />
                </stop>
                <stop offset="100%" style={{ stopColor: 'rgb(168, 85, 247)', stopOpacity: 0.8 }}>
                  <animate attributeName="stop-color" values="rgb(168, 85, 247); rgb(6, 182, 212); rgb(168, 85, 247)" dur="3s" repeatCount="indefinite" />
                </stop>
              </linearGradient>
            </defs>
          </svg>
        </div>
        
        {/* Middle hexagonal ring rotating opposite */}
        <div className="absolute inset-3 animate-hex-spin-reverse" style={{ transformStyle: 'preserve-3d' }}>
          <svg className="absolute inset-0 w-full h-full" viewBox="0 0 100 100" style={{ transform: 'translateZ(15px)' }}>
            <polygon points="50,10 85,30 85,70 50,90 15,70 15,30" 
                     fill="none" 
                     stroke="rgba(236, 72, 153, 0.6)" 
                     strokeWidth="1.5"
                     strokeDasharray="10,5"
                     className="animate-dash" />
          </svg>
        </div>
        
        {/* Blockchain core - cube with connected nodes */}
        <div className="absolute inset-0 flex items-center justify-center" style={{ transformStyle: 'preserve-3d' }}>
          <div className="relative w-14 h-14 animate-cube-float" style={{ transformStyle: 'preserve-3d' }}>
            {/* Central blockchain cube */}
            <div className="absolute inset-0 flex items-center justify-center" style={{ transformStyle: 'preserve-3d' }}>
              <div className="w-12 h-12" style={{
                transformStyle: 'preserve-3d',
                transform: 'translateZ(20px)',
              }}>
                {/* Front face */}
                <div className="absolute inset-0 border-2 border-cyan-400/80 bg-gradient-to-br from-cyan-500/30 to-purple-600/30"
                     style={{ 
                       transform: 'translateZ(6px)',
                       boxShadow: 'inset 0 0 20px rgba(6, 182, 212, 0.5), 0 0 30px rgba(6, 182, 212, 0.3)',
                     }}>
                  {/* Block symbol */}
                  <div className="absolute inset-0 flex items-center justify-center">
                    <svg className="w-6 h-6 text-cyan-300 animate-pulse-slow" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <rect x="4" y="4" width="6" height="6" />
                      <rect x="14" y="4" width="6" height="6" />
                      <rect x="4" y="14" width="6" height="6" />
                      <rect x="14" y="14" width="6" height="6" />
                      <line x1="10" y1="7" x2="14" y2="7" />
                      <line x1="10" y1="17" x2="14" y2="17" />
                      <line x1="7" y1="10" x2="7" y2="14" />
                      <line x1="17" y1="10" x2="17" y2="14" />
                    </svg>
                  </div>
                </div>
                {/* Top face */}
                <div className="absolute inset-0 border-2 border-purple-400/60 bg-gradient-to-br from-purple-500/20 to-pink-600/20"
                     style={{ 
                       transform: 'rotateX(90deg) translateZ(6px)',
                       boxShadow: 'inset 0 0 15px rgba(168, 85, 247, 0.4)',
                     }}></div>
                {/* Right face */}
                <div className="absolute inset-0 border-2 border-indigo-400/60 bg-gradient-to-br from-indigo-500/20 to-cyan-600/20"
                     style={{ 
                       transform: 'rotateY(90deg) translateZ(6px)',
                       boxShadow: 'inset 0 0 15px rgba(99, 102, 241, 0.4)',
                     }}></div>
              </div>
            </div>
            
            {/* Corner nodes - blockchain connections */}
            <div className="absolute -top-1 -left-1 w-2 h-2 bg-cyan-400 rounded-full animate-pulse-node shadow-lg shadow-cyan-400"
                 style={{ transform: 'translateZ(30px)', boxShadow: '0 0 10px rgba(6, 182, 212, 1)' }}></div>
            <div className="absolute -top-1 -right-1 w-2 h-2 bg-purple-400 rounded-full animate-pulse-node animation-delay-300 shadow-lg shadow-purple-400"
                 style={{ transform: 'translateZ(28px)', boxShadow: '0 0 10px rgba(168, 85, 247, 1)' }}></div>
            <div className="absolute -bottom-1 -left-1 w-2 h-2 bg-pink-400 rounded-full animate-pulse-node animation-delay-600 shadow-lg shadow-pink-400"
                 style={{ transform: 'translateZ(26px)', boxShadow: '0 0 10px rgba(236, 72, 153, 1)' }}></div>
            <div className="absolute -bottom-1 -right-1 w-2 h-2 bg-indigo-400 rounded-full animate-pulse-node animation-delay-900 shadow-lg shadow-indigo-400"
                 style={{ transform: 'translateZ(32px)', boxShadow: '0 0 10px rgba(99, 102, 241, 1)' }}></div>
          </div>
        </div>
        
        {/* Orbiting data packets */}
        <div className="absolute inset-0 animate-data-orbit" style={{ transformStyle: 'preserve-3d' }}>
          <div className="absolute top-0 left-1/2 w-1.5 h-1.5 bg-cyan-400 rounded-sm shadow-glow-cyan"
               style={{ transform: 'translateX(-50%) translateZ(35px)', boxShadow: '0 0 12px rgba(6, 182, 212, 1)' }}></div>
        </div>
        <div className="absolute inset-0 animate-data-orbit-delayed" style={{ transformStyle: 'preserve-3d' }}>
          <div className="absolute top-0 left-1/2 w-1.5 h-1.5 bg-purple-400 rounded-sm shadow-glow-purple"
               style={{ transform: 'translateX(-50%) translateZ(32px)', boxShadow: '0 0 12px rgba(168, 85, 247, 1)' }}></div>
        </div>
        <div className="absolute inset-0 animate-data-orbit-more-delayed" style={{ transformStyle: 'preserve-3d' }}>
          <div className="absolute top-0 left-1/2 w-1.5 h-1.5 bg-pink-400 rounded-sm shadow-glow-pink"
               style={{ transform: 'translateX(-50%) translateZ(38px)', boxShadow: '0 0 12px rgba(236, 72, 153, 1)' }}></div>
        </div>
        
        {/* Energy field rings */}
        <div className="absolute inset-0 rounded-full border border-cyan-500/20 animate-energy-pulse"></div>
        <div className="absolute inset-2 rounded-full border border-purple-500/20 animate-energy-pulse animation-delay-300"></div>
        
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
      const [summaryResult, forecastResult] = await Promise.all([
        getInsightSummary(symbol),
        getPriceForecast(symbol, { horizonDays: 3 })
      ])
      
      if (isMountedRef.current) {
        // Override forecast data with fresh forecast to get current model and last point
        const mergedData = {
          ...summaryResult,
          forecast: {
            ...summaryResult.forecast,
            model_used: forecastResult.model_used,
            last_point: forecastResult.last_point
          }
        }
        setData(mergedData)
        setLastUpdated(summaryResult.generated_at || summaryResult.timestamp || new Date().toISOString())
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
    
    // Forecast information
    const forecastPrice = data.forecast?.last_point?.predicted_close
    const modelUsed = data.forecast?.model_used
    const modelName = modelUsed ? modelUsed.match(/^([a-zA-Z]+)/)?.[1] || modelUsed : null

    const pressureText = typeof pressure === 'number' ? pressure.toFixed(2) : pressure
    
    // Build summary in logical order: Forecast -> Sentiment -> On-chain activity -> Technical analysis
    const parts = []
    
    // 1. Price Forecast (most important for traders)
    if (forecastPrice && modelName) {
      const modelDisplay = modelName.charAt(0).toUpperCase() + modelName.slice(1)
      const priceDisplay = forecastPrice.toLocaleString(undefined, { 
        style: 'currency', 
        currency: 'USD', 
        maximumFractionDigits: 2 
      })
      parts.push(`${modelDisplay} forecasts ${symbol} reaching ${priceDisplay} in 10 days.`)
    }
    
    // 2. Market Sentiment
    parts.push(`Market sentiment is ${sentimentLabel.toLowerCase()}.`)
    
    // 3. On-chain Activity
    if (whaleTx) {
      parts.push(`Detected ${whaleTx} whale transactions.`)
    }
    
    if (flow) {
      const flowText = flow.includes('dominated') ? flow : `${flow}-dominated flow`
      parts.push(`On-chain activity shows ${flowText}.`)
    }
    
    if (pressure != null) {
      parts.push(`Market pressure index is at ${pressureText}.`)
    }
    
    // 4. Technical Analysis
    if (bias) {
      parts.push(`Technical indicators show ${bias} bias.`)
    }

    const computed = parts.join(' ')

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
      // Extract first part of model name before underscore or version number
      // e.g., "prophet_v1_stochastic" -> "prophet", "sarimax_v3" -> "sarimax"
      const modelName = data.forecast.model_used.match(/^([a-zA-Z]+)/)?.[1] || data.forecast.model_used
      result.push({ label: modelName, tone: 'purple' })
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
        <div className="flex items-start justify-between mb-4">
          <div className="flex-1">
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
          </div>
          
          {/* Animated Symbol on the right, vertically centered */}
          <div className="flex-shrink-0 ml-6">
            <AnimatedCryptoSymbol />
          </div>
        </div>




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
      
      <style jsx>{`
        @keyframes scifi-rotate {
          0% {
            transform: rotateX(20deg) rotateY(0deg) rotateZ(0deg);
          }
          25% {
            transform: rotateX(20deg) rotateY(90deg) rotateZ(90deg);
          }
          50% {
            transform: rotateX(20deg) rotateY(180deg) rotateZ(180deg);
          }
          75% {
            transform: rotateX(20deg) rotateY(270deg) rotateZ(270deg);
          }
          100% {
            transform: rotateX(20deg) rotateY(360deg) rotateZ(360deg);
          }
        }
        
        @keyframes hex-spin {
          from { 
            transform: translateZ(25px) rotateZ(0deg);
          }
          to { 
            transform: translateZ(25px) rotateZ(360deg);
          }
        }
        
        @keyframes hex-spin-reverse {
          from { 
            transform: translateZ(15px) rotateZ(360deg);
          }
          to { 
            transform: translateZ(15px) rotateZ(0deg);
          }
        }
        
        @keyframes cube-float {
          0%, 100% {
            transform: translateY(0px) rotateX(0deg) rotateY(0deg);
          }
          50% {
            transform: translateY(-5px) rotateX(10deg) rotateY(10deg);
          }
        }
        
        @keyframes pulse-glow-hex {
          0%, 100% {
            filter: drop-shadow(0 0 5px currentColor);
            opacity: 0.8;
          }
          50% {
            filter: drop-shadow(0 0 15px currentColor);
            opacity: 1;
          }
        }
        
        @keyframes pulse-node {
          0%, 100% {
            transform: translateZ(30px) scale(1);
            opacity: 0.8;
          }
          50% {
            transform: translateZ(35px) scale(1.5);
            opacity: 1;
          }
        }
        
        @keyframes dash {
          to {
            stroke-dashoffset: -100;
          }
        }
        
        @keyframes scan {
          0% {
            top: 0%;
            opacity: 0;
          }
          50% {
            opacity: 1;
          }
          100% {
            top: 100%;
            opacity: 0;
          }
        }
        
        @keyframes data-orbit {
          from { 
            transform: rotateZ(0deg);
          }
          to { 
            transform: rotateZ(360deg);
          }
        }
        
        @keyframes energy-pulse {
          0%, 100% {
            transform: scale(1);
            opacity: 0.2;
          }
          50% {
            transform: scale(1.15);
            opacity: 0.5;
          }
        }
        
        @keyframes pulse-slow {
          0%, 100% { 
            opacity: 0.6;
          }
          50% { 
            opacity: 1;
          }
        }
        
        .animate-scifi-rotate {
          animation: scifi-rotate 10s linear infinite;
        }
        
        .animate-hex-spin {
          animation: hex-spin 8s linear infinite;
        }
        
        .animate-hex-spin-reverse {
          animation: hex-spin-reverse 6s linear infinite;
        }
        
        .animate-cube-float {
          animation: cube-float 4s ease-in-out infinite;
        }
        
        .animate-pulse-glow-hex {
          animation: pulse-glow-hex 2s ease-in-out infinite;
        }
        
        .animate-pulse-node {
          animation: pulse-node 2s ease-in-out infinite;
        }
        
        .animate-dash {
          animation: dash 2s linear infinite;
        }
        
        .animate-scan {
          animation: scan 3s ease-in-out infinite;
        }
        
        .animate-data-orbit {
          animation: data-orbit 3s linear infinite;
        }
        
        .animate-data-orbit-delayed {
          animation: data-orbit 3s linear infinite;
          animation-delay: 1s;
        }
        
        .animate-data-orbit-more-delayed {
          animation: data-orbit 3s linear infinite;
          animation-delay: 2s;
        }
        
        .animate-energy-pulse {
          animation: energy-pulse 3s ease-in-out infinite;
        }
        
        .animate-pulse-slow {
          animation: pulse-slow 2s ease-in-out infinite;
        }
        
        .animation-delay-300 {
          animation-delay: 0.3s;
        }
        
        .animation-delay-600 {
          animation-delay: 0.6s;
        }
        
        .animation-delay-900 {
          animation-delay: 0.9s;
        }
      `}</style>
    </div>
  )
}

export default InsightSummary
