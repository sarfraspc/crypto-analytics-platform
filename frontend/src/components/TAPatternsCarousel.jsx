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

const PATTERN_QUERY = {
  exchange: 'binance',
  interval: '1d',
  limit: 100,
}

const TAPatternsCarousel = () => {
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
    if (!taSymbols.length) return patterns
    const allowSet = new Set(
      taSymbols.map((item) => (item?.toUpperCase?.() || '').trim()).filter(Boolean),
    )
    return patterns.filter((pattern) => allowSet.has((pattern.symbol || '').toUpperCase()))
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
        <div className="flex space-x-4 overflow-x-auto pb-4">
          {filteredPatterns.map((pattern, index) => (
            <div
              key={`${pattern.symbol}-${pattern.pattern}-${index}`}
              className={`flex-shrink-0 w-44 h-36 p-4 rounded-lg border flex flex-col ${
                isDark
                  ? 'bg-slate-800 border-slate-700'
                  : 'bg-white border-gray-200'
              }`}
            >
              <div>
                <div className="flex justify-between items-center">
                  <span className={`font-medium ${isDark ? 'text-white' : 'text-gray-900'}`}>{pattern.symbol}</span>
                  <div className="group relative">
                    <Info size={16} className={`${isDark ? 'text-gray-400' : 'text-gray-500'} cursor-help`} />
                    <div
                      className={`absolute top-full left-0 mt-2 w-64 p-3 rounded-lg shadow-xl opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all z-10 ${
                        isDark
                          ? 'bg-slate-900 border border-slate-700 text-gray-300'
                          : 'bg-white border border-gray-200 text-gray-700'
                      }`}
                    >
                      <p className="text-xs">{pattern.explanation || pattern.details || 'Pattern detected'}</p>
                    </div>
                  </div>
                </div>
                <p className={`text-sm mt-2 ${isDark ? 'text-gray-300' : 'text-gray-700'}`}>
                  {pattern.pattern || pattern.name}
                  {pattern.signal ? ` (${String(pattern.signal).toLowerCase()})` : ''}
                </p>
              </div>
              <div className="flex-grow" />
              <div className="mt-3">
                <div className="flex items-center gap-2">
                  <div className={`flex-1 h-2 ${isDark ? 'bg-slate-700' : 'bg-gray-200'} rounded-full overflow-hidden`}>
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
              </div>
            </div>
          ))}
          {filteredPatterns.length === 0 && (
            <div className="w-full text-center">
              <p className="text-sm text-gray-500 py-6">
                {taSymbols.length === 0
                  ? 'No TA symbols found in the database.'
                  : 'No patterns detected at the moment.'}
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default TAPatternsCarousel
