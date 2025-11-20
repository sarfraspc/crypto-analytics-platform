import { useEffect, useMemo, useState } from 'react'
import { ComposedChart, Bar, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts'
import { TrendingUp } from 'lucide-react'
import { useTheme } from '../hooks/useTheme'
import { useSymbol } from '../hooks/useSymbol'
import { getPriceForecast } from '../services/api'
import Loader from './Loader'
import ErrorBox from './ErrorBox'

const ForecastChart = () => {
  const { isDark } = useTheme()
  const { symbol } = useSymbol()
  const HORIZON_DAYS = 10
  const [points, setPoints] = useState([])
  const [meta, setMeta] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    let isMounted = true
    const fetchData = async () => {
      setLoading(true)
      setError(null)
      try {
        const result = await getPriceForecast(symbol, { horizonDays: HORIZON_DAYS })
        if (!isMounted) return
        const normalizedPoints = (result.forecast_points || []).filter((point) => {
          const value = Number(point.predicted_close)
          const isHorizonFlag = result.horizon_hours && value === result.horizon_hours
          return Number.isFinite(value) && !isHorizonFlag
        })
        setPoints(normalizedPoints)
        setMeta({
          model: result.model_used,
          generatedAt: result.generated_at,
          lastPoint: result.last_point,
          rawText: result.raw_text,
        })
      } catch (err) {
        if (isMounted) {
          setError(err.message ?? 'Unable to load forecast data')
        }
      } finally {
        if (isMounted) setLoading(false)
      }
    }

    fetchData()
    return () => {
      isMounted = false
    }
  }, [symbol])

  const chartData = useMemo(() => {
    if (points.length === 0) return []

    // Group hourly points into 6-hour candles for better visualization
    const candleInterval = 6 // hours per candle
    const candles = []
    
    for (let i = 0; i < points.length; i += candleInterval) {
      const candlePoints = points.slice(i, i + candleInterval)
      if (candlePoints.length === 0) continue

      const prices = candlePoints.map(p => Number(p.predicted_close))
      const open = prices[0]
      const close = prices[prices.length - 1]
      const high = Math.max(...prices)
      const low = Math.min(...prices)
      
      const timestamp = new Date(candlePoints[0].timestamp)
      const timePart = timestamp.toLocaleTimeString(undefined, {
        hour: '2-digit',
        minute: '2-digit',
        hour12: false
      })
      const datePart = timestamp.toLocaleDateString(undefined, {
        month: 'short',
        day: 'numeric'
      })
      const label = `${datePart} ${timePart}`

      candles.push({
        timestamp: candlePoints[0].timestamp,
        label,
        open,
        close,
        high,
        low,
        isPositive: close >= open,
        lowHigh: [low, high]
      })
    }

    return candles
  }, [points])

  // Calculate dynamic Y-axis domain to zoom into price range
  const yAxisDomain = useMemo(() => {
    if (chartData.length === 0) return ['auto', 'auto']
    
    const allPrices = chartData.flatMap(d => [d.high, d.low])
    const min = Math.min(...allPrices)
    const max = Math.max(...allPrices)
    const range = max - min
    
    // Zoom in based on the price range
    let paddingPercent
    if (range < 1) {
      paddingPercent = 5 // Very small movements: 5x zoom
    } else if (range < 10) {
      paddingPercent = 2 // Small movements: 2x zoom
    } else if (range < 100) {
      paddingPercent = 0.5 // Medium movements
    } else {
      paddingPercent = 0.2 // Large movements
    }
    
    const padding = Math.max(range * paddingPercent, max * 0.0001)
    
    return [
      Number((min - padding).toFixed(2)),
      Number((max + padding).toFixed(2))
    ]
  }, [chartData])

  const headline = useMemo(() => {
    if (!meta?.lastPoint) return null
    const value = Number(meta.lastPoint.predicted_close)
    if (Number.isNaN(value)) return null
    return value.toLocaleString(undefined, { 
      style: 'currency', 
      currency: 'USD', 
      minimumFractionDigits: 2,
      maximumFractionDigits: 2 
    })
  }, [meta])

  // Custom candlestick renderer
  const Candlestick = (props) => {
    const { x, y, width, height, payload } = props
    if (!payload) return null

    const { high, low, open, close } = payload
    const isPositive = close >= open
    const color = isPositive ? '#10b981' : '#ef4444'
    
    // Calculate Y positions
    const priceRange = high - low || 0.01
    const yHigh = y
    const yLow = y + height
    const yOpen = y + ((high - open) / priceRange) * height
    const yClose = y + ((high - close) / priceRange) * height
    
    const wickX = x + width / 2
    const bodyTop = Math.min(yOpen, yClose)
    const bodyBottom = Math.max(yOpen, yClose)
    const bodyHeight = Math.max(bodyBottom - bodyTop, 1)

    return (
      <g>
        {/* Wick line */}
        <line
          x1={wickX}
          y1={yHigh}
          x2={wickX}
          y2={yLow}
          stroke={color}
          strokeWidth={1}
        />
        {/* Candle body */}
        <rect
          x={x + width * 0.2}
          y={bodyTop}
          width={width * 0.6}
          height={bodyHeight}
          fill={color}
          stroke={color}
        />
      </g>
    )
  }

  const CustomTooltip = ({ active, payload }) => {
    if (!active || !payload?.[0]) return null
    
    const data = payload[0].payload
    const change = data.close - data.open
    const changePercent = ((change / data.open) * 100).toFixed(2)
    
    return (
      <div
        className={`rounded-lg border px-3 py-2 shadow-xl ${
          isDark ? 'bg-slate-900 border-slate-700' : 'bg-white border-gray-200'
        }`}
      >
        <p className={`text-xs font-semibold mb-2 ${isDark ? 'text-gray-200' : 'text-gray-900'}`}>
          {data.label}
        </p>
        <div className="space-y-1 text-xs">
          <div className="flex justify-between gap-4">
            <span className={isDark ? 'text-gray-400' : 'text-gray-600'}>Open:</span>
            <span className={`font-mono ${isDark ? 'text-gray-200' : 'text-gray-900'}`}>
              ${data.open.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
            </span>
          </div>
          <div className="flex justify-between gap-4">
            <span className={isDark ? 'text-gray-400' : 'text-gray-600'}>High:</span>
            <span className={`font-mono ${isDark ? 'text-gray-200' : 'text-gray-900'}`}>
              ${data.high.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
            </span>
          </div>
          <div className="flex justify-between gap-4">
            <span className={isDark ? 'text-gray-400' : 'text-gray-600'}>Low:</span>
            <span className={`font-mono ${isDark ? 'text-gray-200' : 'text-gray-900'}`}>
              ${data.low.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
            </span>
          </div>
          <div className="flex justify-between gap-4">
            <span className={isDark ? 'text-gray-400' : 'text-gray-600'}>Close:</span>
            <span className={`font-mono ${isDark ? 'text-gray-200' : 'text-gray-900'}`}>
              ${data.close.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
            </span>
          </div>
          <div className={`flex justify-between gap-4 pt-1 mt-1 border-t ${isDark ? 'border-slate-700' : 'border-gray-200'}`}>
            <span className={isDark ? 'text-gray-400' : 'text-gray-600'}>Change:</span>
            <span className={`font-mono font-semibold ${data.isPositive ? 'text-green-500' : 'text-red-500'}`}>
              {data.isPositive ? '+' : ''}{change.toFixed(2)} ({changePercent}%)
            </span>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div
      className={`rounded-xl p-6 shadow-lg ${
        isDark
          ? 'bg-slate-800/50 backdrop-blur-xl border border-slate-700'
          : 'bg-white border border-gray-200'
      }`}
    >
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-2">
          <TrendingUp className="text-indigo-500" size={24} />
          <h3 className={`text-lg font-semibold ${isDark ? 'text-white' : 'text-gray-900'}`}>
            Price Forecast
          </h3>
        </div>
        {meta && (
          <div className="text-right">
            <p className={`text-xs ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
              Model: {meta.model || 'SARIMAX'}
            </p>
            {headline && (
              <p className={`text-lg font-bold ${isDark ? 'text-green-400' : 'text-green-600'}`}>
                {headline}
              </p>
            )}
            {meta.generatedAt && (
              <p className={`text-xs ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
                Last updated {new Date(meta.generatedAt).toLocaleString()}
              </p>
            )}
          </div>
        )}
      </div>

      {loading && <Loader label="Loading forecast" />}
      {!loading && error && <ErrorBox message={error} />}
      {!loading && !error && chartData.length > 0 && (
        <ResponsiveContainer width="100%" height={400}>
          <ComposedChart data={chartData} margin={{ top: 10, right: 10, bottom: 40, left: 10 }}>
            <defs>
              <linearGradient id="volumeGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.2} />
                <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid 
              strokeDasharray="3 3" 
              stroke={isDark ? '#334155' : '#e5e7eb'} 
              vertical={false}
            />
            <XAxis 
              dataKey="label"
              stroke={isDark ? '#94a3b8' : '#6b7280'}
              style={{ fontSize: '11px' }}
              angle={-45}
              textAnchor="end"
              height={80}
              interval="preserveStartEnd"
            />
            <YAxis 
              domain={yAxisDomain}
              stroke={isDark ? '#94a3b8' : '#6b7280'}
              style={{ fontSize: '11px' }}
              width={85}
              tickFormatter={(value) => 
                `$${Number(value).toLocaleString(undefined, { 
                  minimumFractionDigits: 2, 
                  maximumFractionDigits: 2 
                })}`
              }
            />
            <Tooltip content={<CustomTooltip />} />
            <Bar 
              dataKey="close" 
              shape={<Candlestick />}
            />
            <Line 
              type="monotone" 
              dataKey="close"
              stroke="#8b5cf6"
              strokeWidth={1.5}
              dot={false}
              strokeDasharray="3 3"
              strokeOpacity={0.4}
            />
          </ComposedChart>
        </ResponsiveContainer>
      )}
      {!loading && !error && chartData.length === 0 && (
        <div className={`rounded-lg border px-4 py-3 text-sm ${
          isDark ? 'border-slate-700 text-gray-300' : 'border-gray-200 text-gray-600'
        }`}>
          <p className="font-medium mb-2">No valid forecast data available.</p>
          <p className="text-xs">
            The forecast service is currently returning data that doesn't meet display criteria.
            This might be due to model training or data source issues.
          </p>
        </div>
      )}
    </div>
  )
}

export default ForecastChart
