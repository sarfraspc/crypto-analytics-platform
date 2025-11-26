import { useCallback, useEffect, useMemo, useState } from 'react'
import { ArrowDownRight, ArrowUpRight, BarChart3, Minus } from 'lucide-react'
import { useTheme } from '../hooks/useTheme'
import { useSymbol } from '../hooks/useSymbol'
import { getOnChainMetrics } from '../services/api'
import Loader from './Loader'
import ErrorBox from './ErrorBox'

const formatNumber = (
  value,
  { style = 'decimal', maximumFractionDigits = 2, currency = 'USD' } = {},
) => {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '--'
  
  const num = Number(value)
  const absNum = Math.abs(num)
  
  // For currency, format with M/B suffixes for large numbers
  if (style === 'currency') {
    let formattedValue
    let suffix = ''
    
    if (absNum >= 1_000_000_000) {
      formattedValue = num / 1_000_000_000
      suffix = 'B'
    } else if (absNum >= 1_000_000) {
      formattedValue = num / 1_000_000
      suffix = 'M'
    } else {
      formattedValue = num
    }
    
    const formatted = formattedValue.toLocaleString(undefined, {
      style: 'currency',
      currency,
      maximumFractionDigits: suffix ? 2 : maximumFractionDigits,
      minimumFractionDigits: 0,
    })
    
    return suffix ? `${formatted}${suffix}` : formatted
  }
  
  // For regular numbers
  const options = { style, maximumFractionDigits }
  return num.toLocaleString(undefined, options)
}

const MetricCard = ({ title, value, trend, isDark }) => {
  const TrendIcon = trend === 'up' ? ArrowUpRight : trend === 'down' ? ArrowDownRight : Minus
  const trendColor = trend === 'up' ? 'text-success-400' : trend === 'down' ? 'text-red-400' : 'text-gray-500'
  const isNA = value === 'N/A'
  return (
    <div
      className={`rounded-xl p-4 shadow-lg flex flex-col justify-between min-h-[110px] transition-all hover:scale-[1.02] ${
        isDark 
          ? 'bg-dark-900/60 border border-accent-500/10 hover:border-accent-500/30' 
          : 'bg-white border border-slate-200 hover:border-accent-500/30'
      }`}
    >
      {/* Title at top */}
      <p className={`text-xs font-medium ${isDark ? 'text-gray-400' : 'text-gray-500'}`}>{title}</p>
      
      {/* Value in middle with more space */}
      <div className="flex items-center my-2">
        <p className={`text-xl font-bold whitespace-nowrap ${isDark ? 'text-white' : 'text-gray-900'}`}>
          {isNA ? <span className="text-sm text-gray-400 dark:text-gray-500">N/A</span> : value}
        </p>
      </div>
      
      {/* Trend at bottom */}
      <div className={`flex items-center justify-end gap-1 ${trendColor}`}>
        <TrendIcon size={14} />
        <span className="text-xs font-medium whitespace-nowrap">{trend === 'flat' ? '—' : trend}</span>
      </div>
    </div>
  )
}

const MetricsGrid = () => {
  const { isDark } = useTheme()
  const { symbol } = useSymbol()
  const [metrics, setMetrics] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [lastUpdated, setLastUpdated] = useState(null)

  const fetchMetrics = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const result = await getOnChainMetrics(symbol)
      setMetrics(result.metrics || {})
      setLastUpdated(result.generated_at || result.timestamp || new Date().toISOString())
    } catch (err) {
      setError(err.message ?? 'Unable to load metrics')
    } finally {
      setLoading(false)
    }
  }, [symbol])

  useEffect(() => {
    fetchMetrics()
  }, [fetchMetrics])

  const normalizedMetrics = useMemo(
    () => ({
      whale_transactions: metrics?.whale_transactions ?? metrics?.whale_count,
      net_flow_usd: metrics?.net_flow_usd,
      exchange_inflow_usd: metrics?.exchange_inflow_usd ?? metrics?.inflow_usd,
      exchange_outflow_usd: metrics?.exchange_outflow_usd ?? metrics?.outflow_usd,
      market_pressure_index: metrics?.market_pressure_index,
      price_change_pct: metrics?.price_change_pct,
      flow_trend_24h: metrics?.flow_trend_24h,
    }),
    [metrics],
  )
  const safeTrend = (value, threshold = 0) => {
    if (typeof value !== 'number') return 'flat'
    if (value > threshold) return 'up'
    if (value < -threshold) return 'down'
    return 'flat'
  }

  const displayValue = (value, opts) => {
    if (value === null || value === undefined) {
      return 'N/A'
    }
    return formatNumber(value, opts)
  }

  const cards = [
    {
      title: 'Whale Transactions',
      value: displayValue(normalizedMetrics.whale_transactions),
      trend: safeTrend(normalizedMetrics.flow_trend_24h),
    },
    {
      title: 'Net Flow USD',
      value: displayValue(normalizedMetrics.net_flow_usd, { style: 'currency' }),
      trend: safeTrend(normalizedMetrics.net_flow_usd),
    },
    {
      title: 'Exchange Inflow',
      value: displayValue(normalizedMetrics.exchange_inflow_usd, { style: 'currency' }),
      trend: 'flat',
    },
    {
      title: 'Exchange Outflow',
      value: displayValue(normalizedMetrics.exchange_outflow_usd, { style: 'currency' }),
      trend: 'flat',
    },
    {
      title: 'Market Pressure Index',
      value: displayValue(normalizedMetrics.market_pressure_index),
      trend: safeTrend(normalizedMetrics.market_pressure_index, 0.05),
    },
    {
      title: 'Price Change % (24h)',
      value:
        normalizedMetrics.price_change_pct === null || normalizedMetrics.price_change_pct === undefined
          ? 'N/A'
          : `${formatNumber(normalizedMetrics.price_change_pct)}%`,
      trend: safeTrend(normalizedMetrics.price_change_pct),
    },
  ]

  const formattedUpdated =
    lastUpdated && !Number.isNaN(new Date(lastUpdated).getTime())
      ? new Date(lastUpdated).toLocaleString()
      : null

  const handleRetry = () => {
    if (!loading) {
      fetchMetrics()
    }
  }

  return (
    <div
      className={`rounded-xl p-6 shadow-lg flex flex-col flex-grow ${
        isDark
          ? 'bg-dark-800/50 backdrop-blur-xl border border-accent-500/10'
          : 'bg-white/80 backdrop-blur-xl border border-slate-200'
      }`}
    >
      <div className="flex items-center justify-between gap-3 mb-6">
        <div className="flex items-center gap-2">
          <BarChart3 className="text-accent-500" size={24} />
          <h3 className={`text-lg font-semibold ${isDark ? 'text-white' : 'text-gray-900'}`}>
            On-Chain Metrics
          </h3>
        </div>
        {formattedUpdated && (
          <span className={`text-xs ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
            Last updated {formattedUpdated}
          </span>
        )}
      </div>

      {loading && <Loader label="Fetching on-chain data" />}
      {!loading && error && <ErrorBox message={error} onRetry={handleRetry} />}
      {!loading && !error && (
        <div className="grid grid-cols-2 gap-3 sm:gap-4 flex-grow">
          {cards.map((card) => (
            <MetricCard key={card.title} {...card} isDark={isDark} />
          ))}
        </div>
      )}
    </div>
  )
}

export default MetricsGrid
