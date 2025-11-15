import { useEffect, useState } from 'react'
import SentimentGauge from './SentimentGauge'
import SentimentSourcesCarousel from './SentimentSourcesCarousel'
import { useTheme } from '../hooks/useTheme'
import { getSentimentAnalysis } from '../services/api'
import Loader from './Loader'
import ErrorBox from './ErrorBox'

const SentimentPanel = () => {
  const { isDark } = useTheme()
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [lastUpdated, setLastUpdated] = useState(null)

  useEffect(() => {
    let isMounted = true
    const fetchData = async () => {
      setLoading(true)
      setError(null)
      try {
        const result = await getSentimentAnalysis()
        if (isMounted) {
          setData(result)
          setLastUpdated(result.generated_at || result.timestamp || new Date().toISOString())
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
  }, [])

  const aggregated = data?.aggregated || {}
  const sources = Array.isArray(data?.sources) ? data.sources : []

  const formattedUpdated =
    lastUpdated && !Number.isNaN(new Date(lastUpdated).getTime())
      ? new Date(lastUpdated).toLocaleString()
      : null

  return (
    <div
      className={`rounded-xl p-6 shadow-lg ${
        isDark
          ? 'bg-slate-800/50 backdrop-blur-xl border border-slate-700'
          : 'bg-white border border-gray-200'
      }`}
    >
      {loading && <Loader label="Analyzing sentiment" />}
      {!loading && error && <ErrorBox message={error} />}
      {!loading && !error && data && (
        <div className="flex flex-col lg:flex-row gap-6">
          <div className="lg:w-1/3">
            <SentimentGauge aggregated={aggregated} />
          </div>
          <div className="lg:w-2/3 flex flex-col">
            <div className="flex items-center justify-between mb-6 gap-3">
              <h3 className={`text-lg font-semibold ${isDark ? 'text-white' : 'text-gray-900'}`}>
                Sentiment Sources
              </h3>
              {formattedUpdated && (
                <span className={`text-xs ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
                  Last updated {formattedUpdated}
                </span>
              )}
            </div>
            <div className="flex-grow">
              <SentimentSourcesCarousel sources={sources} />
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default SentimentPanel
