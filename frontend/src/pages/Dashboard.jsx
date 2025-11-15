import InsightSummary from '../components/InsightSummary'
import ForecastChart from '../components/ForecastChart'
import SentimentGauge from '../components/SentimentGauge'
import MetricsGrid from '../components/MetricsGrid'
import PatternsTable from '../components/PatternsTable'

const Dashboard = () => (
  <div className="space-y-6">
    <div className="grid gap-6 lg:grid-cols-3">
      <div className="lg:col-span-3">
        <InsightSummary />
      </div>
    </div>

    <div className="grid gap-6 lg:grid-cols-3">
      <div className="lg:col-span-2">
        <ForecastChart />
      </div>
      <div className="lg:col-span-1">
        <SentimentGauge />
      </div>
    </div>

    <div className="grid gap-6 lg:grid-cols-3">
      <div className="lg:col-span-2">
        <MetricsGrid />
      </div>
      <div className="lg:col-span-1">
        <PatternsTable />
      </div>
    </div>
  </div>
)

export default Dashboard
