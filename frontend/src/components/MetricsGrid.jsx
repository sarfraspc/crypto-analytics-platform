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
  const options =
    style === 'currency'
      ? { style, currency, maximumFractionDigits }
      : { style, maximumFractionDigits }
  return Number(value).toLocaleString(undefined, options)
}

const MetricCard = ({ title, value, trend, isDark }) => {
  const TrendIcon = trend === 'up' ? ArrowUpRight : trend === 'down' ? ArrowDownRight : Minus
  const trendColor = trend === 'up' ? 'text-green-500' : trend === 'down' ? 'text-red-500' : 'text-gray-500'
  const isNA = value === 'N/A'
  return (
    <div
      className={`rounded-xl p-4 shadow-lg ${
        isDark ? 'bg-slate-900/40 border border-slate-700' : 'bg-white border border-gray-200'
      }`}
    >
      <p className={`text-xs font-medium mb-2 ${isDark ? 'text-gray-400' : 'text-gray-500'}`}>{title}</p>
      <div className="flex items-end justify-between">
        <p className={`text-2xl font-bold ${isDark ? 'text-white' : 'text-gray-900'}`}>
          {isNA ? <span className="text-sm text-gray-400 dark:text-gray-500">N/A</span> : value}
        </p>
        <div className={`flex items-center gap-1 ${trendColor}`}>
          <TrendIcon size={16} />
          <span className="text-sm font-medium">{trend === 'flat' ? '—' : trend}</span>
        </div>
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

  const fetchMetrics = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const result = await getOnChainMetrics(symbol)
      setMetrics(result.metrics || {})
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
  console.log('MetricsGrid normalized metrics:', normalizedMetrics)
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

  const handleRetry = () => {
    if (!loading) {
      fetchMetrics()
    }
  }

  return (
    <div
      className={`rounded-xl p-6 shadow-lg ${
        isDark
          ? 'bg-slate-800/50 backdrop-blur-xl border border-slate-700'
          : 'bg-white border border-gray-200'
      }`}
    >
      <div className="flex items-center gap-2 mb-6">
        <BarChart3 className="text-indigo-500" size={24} />
        <h3 className={`text-lg font-semibold ${isDark ? 'text-white' : 'text-gray-900'}`}>
          On-Chain Metrics
        </h3>
      </div>

      {loading && <Loader label="Fetching on-chain data" />}
      {!loading && error && <ErrorBox message={error} onRetry={handleRetry} />}
      {!loading && !error && (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {cards.map((card) => (
            <MetricCard key={card.title} {...card} isDark={isDark} />
          ))}
        </div>
      )}
    </div>
  )
}

export default MetricsGrid
