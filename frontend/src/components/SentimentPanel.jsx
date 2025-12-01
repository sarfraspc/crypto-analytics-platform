import { useEffect, useState } from 'react'
import SentimentGauge from './SentimentGauge'
import SentimentSourcesCarousel from './SentimentSourcesCarousel'
import FngGauge from './FngGauge'
import { useTheme } from '../hooks/useTheme'
import { useSymbol } from '../hooks/useSymbol'
import { getSentimentAnalysis, getFngCurrent } from '../services/api'
import Loader from './Loader'
import ErrorBox from './ErrorBox'

const SentimentPanel = () => {
  const { isDark } = useTheme()
  const { symbol } = useSymbol()
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [lastUpdated, setLastUpdated] = useState(null)
  const [fng, setFng] = useState(null)

  useEffect(() => {
    let isMounted = true
    const fetchData = async () => {
      setLoading(true)
      setError(null)
      try {
        // Use cached sentiment for fast initial load
        const [result, fngResult] = await Promise.all([
          getSentimentAnalysis(symbol, { useCached: true, maxAgeHours: 4 }),
          getFngCurrent().catch(() => null),
        ])
        if (isMounted) {
          setData(result)
          setFng(fngResult)
          setLastUpdated(
            result.generated_at ||
              result.timestamp ||
              fngResult?.last_updated ||
              fngResult?.fng?.last_updated ||
              new Date().toISOString(),
          )
        }
      } catch (err) {
        if (isMounted) setError(err.message ?? 'Unable to load sentiment data')
      } finally {
        if (isMounted) setLoading(false)
      }
    }

    fetchData()
    return () => {
      isMounted = false
    }
  }, [symbol])

  const aggregated = data?.aggregated || {}
  const sources = Array.isArray(data?.sources) ? data.sources : []
  const fngData = fng?.fng || fng || {}
  const showFng = fngData && Object.keys(fngData).length > 0
  const formattedUpdated =
    lastUpdated && !Number.isNaN(new Date(lastUpdated).getTime())
      ? new Date(lastUpdated).toLocaleString()
      : null

  return (
    <div
      className={`rounded-xl p-6 shadow-lg ${
        isDark
          ? 'bg-dark-800/50 backdrop-blur-xl border border-accent-500/10'
          : 'bg-white/80 backdrop-blur-xl border border-slate-200'
      }`}
    >
      {loading && <Loader label="Analyzing sentiment" />}
      {!loading && error && <ErrorBox message={error} />}
      {!loading && !error && data && (
        <div className="flex flex-col lg:flex-row gap-6 items-center lg:items-start">
          <div className="flex flex-col items-center">
            <SentimentGauge aggregated={aggregated} />
          </div>
          {showFng && (
            <div className="flex flex-col items-center">
              <FngGauge fng={fngData} size={116} />
            </div>
          )}
          <div className="flex-1 w-full flex flex-col min-w-0">
            <div className="flex flex-wrap items-center justify-between w-full mb-4 gap-2 sm:gap-3">
              <h3 className={`text-lg font-semibold ${isDark ? 'text-white' : 'text-gray-900'}`}>
                Sentiment Sources
              </h3>
              {formattedUpdated && (
                <span className={`text-xs ${isDark ? 'text-gray-400' : 'text-gray-600'} truncate`}>
                  Last updated {formattedUpdated}
                </span>
              )}
            </div>
            <div className="flex-grow w-full">
              <SentimentSourcesCarousel sources={sources} />
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default SentimentPanel
