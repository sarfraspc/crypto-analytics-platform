import InsightSummary from '../components/InsightSummary'
import ForecastChart from '../components/ForecastChart'
import MetricsGrid from '../components/MetricsGrid'
import SentimentPanel from '../components/SentimentPanel'
import TAPatternsCarousel from '../components/TAPatternsCarousel'

const Dashboard = () => (
  <div className="space-y-6">
    {/* Row 1: InsightSummary (full width) */}
    <div className="grid grid-cols-1">
      <InsightSummary />
    </div>

    {/* Row 2: Sentiment Panel (full width) */}
    <div className="grid grid-cols-1">
      <SentimentPanel />
    </div>

    {/* Row 3: ForecastChart and MetricsGrid (side-by-side) */}
    <div className="grid gap-6 lg:grid-cols-3 lg:items-stretch">
      <div className="lg:col-span-2">
        <ForecastChart />
      </div>
      <div className="lg:col-span-1 flex">
        <MetricsGrid />
      </div>
    </div>

    {/* Row 4: TA Patterns Carousel (full width) */}
    <div className="grid grid-cols-1">
      <TAPatternsCarousel />
    </div>
  </div>
)

export default Dashboard