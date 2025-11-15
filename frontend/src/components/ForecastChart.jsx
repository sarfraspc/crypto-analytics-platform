import { useEffect, useMemo, useState } from 'react'
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts'
import { TrendingUp } from 'lucide-react'
import { useTheme } from '../hooks/useTheme'
import { useSymbol } from '../hooks/useSymbol'
import { getPriceForecast } from '../services/api'
import Loader from './Loader'
import ErrorBox from './ErrorBox'

const formatDate = (value) => {
  if (!value) return ''
  const date = new Date(value)
  return date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
}

const ForecastChart = () => {
  const { isDark } = useTheme()
  const { symbol } = useSymbol()
  const HORIZON_DAYS = 30
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
        const normalizedPoints = (result.forecast_points || []).filter((point, idx) => {
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

  const chartData = useMemo(
    () =>
      points.map((point, index) => ({
        id: index,
        date: formatDate(point.timestamp) || `T+${index + 1}h`,
        predicted: Number(point.predicted_close) || null,
      })),
    [points],
  )

  const headline = useMemo(() => {
    if (!meta?.lastPoint) return null
    const value = Number(meta.lastPoint.predicted_close)
    if (Number.isNaN(value)) return null
    return value.toLocaleString(undefined, { style: 'currency', currency: 'USD', maximumFractionDigits: 0 })
  }, [meta])

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
              <p className={`text-lg font-bold ${isDark ? 'text-green-400' : 'text-green-600'}`}>{headline}</p>
            )}
            {meta.generatedAt && (
              <p className={`text-xs ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
                Updated {new Date(meta.generatedAt).toLocaleString()}
              </p>
            )}
          </div>
        )}
      </div>

      {loading && <Loader label="Loading forecast" />}
      {!loading && error && <ErrorBox message={error} />}
      {!loading && !error && chartData.length > 0 && (
        <ResponsiveContainer width="100%" height={300}>
          <AreaChart data={chartData}>
            <defs>
              <linearGradient id="colorPredicted" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke={isDark ? '#334155' : '#e5e7eb'} />
            <XAxis dataKey="date" stroke={isDark ? '#94a3b8' : '#6b7280'} />
            <YAxis stroke={isDark ? '#94a3b8' : '#6b7280'} />
            <Tooltip
              contentStyle={{
                backgroundColor: isDark ? '#1e293b' : '#ffffff',
                border: `1px solid ${isDark ? '#334155' : '#e5e7eb'}`,
                borderRadius: '8px',
              }}
            />
            <Area type="monotone" dataKey="predicted" stroke="#8b5cf6" fill="url(#colorPredicted)" strokeWidth={2} />
          </AreaChart>
        </ResponsiveContainer>
      )}
      {!loading && !error && chartData.length === 0 && (
        <div className={`rounded-lg border px-4 py-3 text-sm ${isDark ? 'border-slate-700 text-gray-300' : 'border-gray-200 text-gray-600'}`}>
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
