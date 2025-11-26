import { useEffect, useRef, useState } from 'react'
import { useTheme } from '../hooks/useTheme'

const SentimentSourcesCarousel = ({ sources }) => {
  const { isDark } = useTheme()
  const scrollRef = useRef(null)
  const [isHovered, setIsHovered] = useState(false)

  useEffect(() => {
    const scrollInterval = setInterval(() => {
      if (scrollRef.current && !isHovered) {
        const { scrollLeft, scrollWidth, clientWidth } = scrollRef.current
        const cardWidth = 304 // w-72 (288px) + space-x-4 (16px)

        // Check if we are near the end of the scroll
        if (scrollLeft + clientWidth >= scrollWidth - 1) {
          scrollRef.current.scrollTo({ left: 0, behavior: 'smooth' })
        } else {
          scrollRef.current.scrollBy({ left: cardWidth, behavior: 'smooth' })
        }
      }
    }, 4000) // Scroll every 4 seconds

    return () => clearInterval(scrollInterval)
  }, [isHovered])

  if (!sources || sources.length === 0) {
    return (
      <div className="flex items-center justify-center h-full">
        <p className={`text-xs italic ${isDark ? 'text-gray-500' : 'text-gray-400'}`}>
          No sources available right now.
        </p>
      </div>
    )
  }

  return (
    <div 
      className="w-full h-full flex items-center"
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      <div ref={scrollRef} className="flex space-x-4 overflow-x-auto pb-4 scroll-smooth">
        {sources.map((source, index) => (
          <div
            key={source.metadata?.doc_id || index}
            className={`flex-shrink-0 w-72 h-40 p-4 rounded-lg border flex flex-col transition-all hover:scale-[1.02] ${
              isDark
                ? 'bg-dark-900/60 border-accent-500/10 hover:border-accent-500/30'
                : 'bg-white border-slate-200 hover:border-accent-500/30'
            }`}
          >
            <div className="flex-shrink-0">
              <div className="flex items-center gap-2 mb-2">
                <span
                  className={`text-xs px-2 py-0.5 rounded-full ${
                    isDark ? 'bg-dark-700 text-gray-300' : 'bg-slate-100 text-gray-700'
                  }`}
                >
                  {source.metadata?.source || 'source'}
                </span>
                {typeof source.score === 'number' && (
                  <span
                    className={`text-xs font-medium ${
                      source.score > 0.7
                        ? 'text-success-400'
                        : source.score < 0.4
                          ? 'text-red-400'
                          : 'text-yellow-400'
                    }`}
                  >
                    {Math.round(Math.min(source.score, 1) * 100)}%
                  </span>
                )}
              </div>
            </div>
            <div className="flex-grow overflow-y-auto mt-1 pr-2">
              <p className={`text-sm ${isDark ? 'text-gray-300' : 'text-gray-700'}`}>
                {(source.content || source.excerpt || 'No excerpt available.').slice(0, 140)}
                {source.content && source.content.length > 140 ? '…' : ''}
              </p>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

export default SentimentSourcesCarousel