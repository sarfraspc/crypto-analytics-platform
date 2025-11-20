import { useEffect, useMemo, useState } from 'react'
import { Info, Target } from 'lucide-react'
import { useTheme } from '../hooks/useTheme'
import { getPatternSymbols, getTechnicalPatterns } from '../services/api'
import Loader from './Loader'
import ErrorBox from './ErrorBox'

const confidenceWidth = (value) => {
  if (value === null || value === undefined) return 0
  return Math.min(Math.max(value, 0), 1) * 100
}

const buildPatternExplanation = (pattern) => {
  if (!pattern) return 'No strong technical signals detected.'

  const parts = []
  const rsi = typeof pattern.rsi === 'number' ? pattern.rsi : Number(pattern.rsi)
  const macd = typeof pattern.macd_hist === 'number' ? pattern.macd_hist : Number(pattern.macd_hist)
  const patternName = pattern.pattern || pattern.name
  const signal = (pattern.signal || '').toString().toLowerCase()

  if (!Number.isNaN(rsi)) {
    if (rsi < 30) {
      parts.push('RSI below 30 indicates oversold conditions')
    } else if (rsi > 70) {
      parts.push('RSI above 70 indicates overbought conditions')
    } else {
      parts.push(`RSI around ${Math.round(rsi)} suggests neutral momentum`)
    }
  }

  if (!Number.isNaN(macd)) {
    if (macd > 0) {
      parts.push('Positive MACD histogram suggests bullish momentum')
    } else if (macd < 0) {
      parts.push('Negative MACD histogram suggests bearish momentum')
    }
  }

  if (patternName && patternName !== 'none') {
    const prettyName = patternName.replace(/_/g, ' ')
    parts.push(`Candlestick pattern ${prettyName} supports the ${signal || 'current'} bias`)
  }

  if (!parts.length) return 'No strong technical signals detected.'
  return parts.join('. ') + '.'
}

const PATTERN_QUERY = {
  exchange: 'binance',
  interval: '1d',
  limit: 100,
}

const MAIN_SYMBOLS = ['BTC', 'ETH', 'SOL', 'BNB', 'XRP', 'ADA', 'AVAX', 'DOGE', 'MATIC', 'LTC', 'DOT']

const sortByPriority = (items) => {
  const priorityIndex = new Map(MAIN_SYMBOLS.map((symbol, index) => [symbol, index]))
  return [...items].sort((a, b) => {
    const aSymbol = (a.symbol || '').toUpperCase()
    const bSymbol = (b.symbol || '').toUpperCase()
    const aPriority = priorityIndex.has(aSymbol) ? priorityIndex.get(aSymbol) : MAIN_SYMBOLS.length
    const bPriority = priorityIndex.has(bSymbol) ? priorityIndex.get(bSymbol) : MAIN_SYMBOLS.length
    if (aPriority !== bPriority) return aPriority - bPriority
    return aSymbol.localeCompare(bSymbol)
  })
}

const PatternsTable = () => {
  const { isDark } = useTheme()
  const [patterns, setPatterns] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [lastUpdated, setLastUpdated] = useState(null)
  const [taSymbols, setTaSymbols] = useState([])
  const [symbolsLoading, setSymbolsLoading] = useState(false)
  const [symbolsError, setSymbolsError] = useState(null)

  useEffect(() => {
    let isMounted = true
    const fetchData = async () => {
      setLoading(true)
      setError(null)
      try {
        const result = await getTechnicalPatterns(PATTERN_QUERY)
        if (isMounted) {
          const rawPatterns = result.patterns
          let safePatterns = []
          if (Array.isArray(rawPatterns)) {
            safePatterns = rawPatterns
          } else if (rawPatterns && typeof rawPatterns === 'object') {
            safePatterns = Object.entries(rawPatterns).map(([symbol, details]) => ({
              symbol,
              ...(typeof details === 'object' ? details : { pattern: String(details) }),
            }))
          }
          setPatterns(safePatterns)
          setLastUpdated(result.generated_at || result.timestamp || new Date().toISOString())
        }
      } catch (err) {
        if (isMounted) setError(err.message ?? 'Unable to load patterns')
      } finally {
        if (isMounted) setLoading(false)
      }
    }

    const fetchSymbols = async () => {
      setSymbolsLoading(true)
      setSymbolsError(null)
      try {
        const response = await getPatternSymbols(PATTERN_QUERY)
        if (isMounted) {
          const fetched = Array.isArray(response?.symbols)
            ? response.symbols.filter((item) => typeof item === 'string')
            : []
          setTaSymbols(fetched)
        }
      } catch (err) {
        if (isMounted) {
          setSymbolsError(err.message ?? 'Unable to load TA symbol list')
          setTaSymbols([])
        }
      } finally {
        if (isMounted) setSymbolsLoading(false)
      }
    }

    fetchData()
    fetchSymbols()
    return () => {
      isMounted = false
    }
  }, [])

  const formattedUpdated =
    lastUpdated && !Number.isNaN(new Date(lastUpdated).getTime())
      ? new Date(lastUpdated).toLocaleString()
      : null

  const filteredPatterns = useMemo(() => {
    if (!patterns.length) return []
    if (!taSymbols.length) return sortByPriority(patterns)
    const allowSet = new Set(
      taSymbols.map((item) => (item?.toUpperCase?.() || '').trim()).filter(Boolean),
    )
    const filtered = patterns.filter((pattern) =>
      allowSet.has((pattern.symbol || '').toUpperCase()),
    )
    return sortByPriority(filtered)
  }, [patterns, taSymbols])

  return (
    <div
      className={`rounded-xl p-6 shadow-lg ${
        isDark
          ? 'bg-slate-800/50 backdrop-blur-xl border border-slate-700'
          : 'bg-white border border-gray-200'
      }`}
    >
      <div className="flex items-center justify-between gap-3 mb-6">
        <div className="flex items-center gap-2">
          <Target className="text-indigo-500" size={24} />
          <h3 className={`text-lg font-semibold ${isDark ? 'text-white' : 'text-gray-900'}`}>
            Technical Patterns
          </h3>
        </div>
        {formattedUpdated && (
          <span className={`text-xs ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
            Last updated {formattedUpdated}
          </span>
        )}
      </div>

      {symbolsLoading && (
        <p className={`text-xs mb-3 ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
          Syncing available TA symbols…
        </p>
      )}
      {!symbolsLoading && symbolsError && (
        <p className={`text-xs mb-3 ${isDark ? 'text-rose-300' : 'text-rose-500'}`}>{symbolsError}</p>
      )}

      {loading && <Loader label="Scanning market patterns" />}
      {!loading && error && <ErrorBox message={error} />}
      {!loading && !error && (
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className={`border-b ${isDark ? 'border-slate-700' : 'border-gray-200'}`}>
                {['Symbol', 'Pattern', 'Signal', 'Confidence', 'Details'].map((heading) => (
                  <th
                    key={heading}
                    className={`text-left py-3 px-4 text-xs font-semibold ${
                      isDark ? 'text-gray-400' : 'text-gray-600'
                    }`}
                  >
                    {heading}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {filteredPatterns.map((pattern, index) => (
                <tr
                  key={`${pattern.symbol}-${pattern.pattern}-${index}`}
                  className={`border-b ${isDark ? 'border-slate-700' : 'border-gray-200'} ${
                    isDark ? 'hover:bg-slate-700/30' : 'hover:bg-gray-50'
                  } transition-colors`}
                >
                  <td className={`py-3 px-4 ${isDark ? 'text-white' : 'text-gray-900'}`}>
                    <span className="font-medium">{pattern.symbol}</span>
                  </td>
                  <td className={`py-3 px-4 ${isDark ? 'text-gray-300' : 'text-gray-700'}`}>
                    {pattern.pattern || pattern.name}
                  </td>
                  <td className={`py-3 px-4 ${isDark ? 'text-gray-300' : 'text-gray-700'}`}>
                    {(pattern.signal || '').toString().toLowerCase() || 'neutral'}
                  </td>
                  <td className="py-3 px-4">
                  <div className="flex items-center gap-2">
                    <div className="flex-1 h-2 bg-slate-700 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-gradient-to-r from-indigo-500 to-purple-500 rounded-full"
                        style={{
                          width: `${confidenceWidth(
                            pattern.confidence ?? (pattern.rsi ? pattern.rsi / 100 : 0.5),
                          )}%`,
                        }}
                      />
                    </div>
                    <span className={`text-sm font-medium ${isDark ? 'text-gray-300' : 'text-gray-700'}`}>
                      {Math.round(
                        (pattern.confidence ?? (pattern.rsi ? pattern.rsi / 100 : 0.5)) * 100,
                      )}
                      %
                    </span>
                  </div>
                  </td>
                  <td className="py-3 px-4">
                    <div className="group relative">
                      <Info size={16} className={`${isDark ? 'text-gray-400' : 'text-gray-500'} cursor-help`} />
                      <div
                        className={`absolute bottom-full right-0 mb-2 w-64 p-3 rounded-lg shadow-xl opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all z-10 ${
                          isDark
                            ? 'bg-slate-900 border border-slate-700 text-gray-300'
                            : 'bg-white border border-gray-200 text-gray-700'
                        }`}
                      >
                        <p className="text-xs">
                          {pattern.explanation || pattern.details || buildPatternExplanation(pattern)}
                        </p>
                      </div>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          {filteredPatterns.length === 0 && (
            <p className="text-sm text-center text-gray-500 py-6">
              {taSymbols.length === 0
                ? 'No TA symbols found in the database.'
                : 'No patterns detected at the moment.'}
            </p>
          )}
        </div>
      )}
    </div>
  )
}

export default PatternsTable
