import { Activity, Brain, Globe2 } from 'lucide-react'
import ChatBox from '../components/ChatBox'
import { useTheme } from '../hooks/useTheme'

const PlaceholderCard = ({ icon: Icon, title, description, isDark }) => (
  <div
    className={`rounded-xl p-6 border border-dashed transition-all hover:scale-[1.01] ${
      isDark ? 'border-accent-500/40 bg-accent-950/20 text-accent-100' : 'border-accent-200/40 bg-accent-50/20 text-accent-800'
    }`}
  >
    <div className="flex items-center gap-3 mb-3">
      <Icon className="text-accent-400" size={20} />
      <h4 className="text-base font-semibold">{title}</h4>
    </div>
    <p className="text-sm opacity-80">{description}</p>
  </div>
)

const AdvancedAnalytics = () => {
  const { isDark } = useTheme()

  return (
    <div className="space-y-6">
      <div className="grid gap-6 lg:grid-cols-2">
        <ChatBox />
        <div
          className={`rounded-xl p-6 shadow-lg ${
            isDark
              ? 'bg-dark-800/50 backdrop-blur-xl border border-accent-500/10'
              : 'bg-white/80 backdrop-blur-xl border border-slate-200'
          }`}
        >
          <h3 className={`text-lg font-semibold mb-4 ${isDark ? 'text-white' : 'text-gray-900'}`}>
            Advanced Playbooks
          </h3>
          <p className={`text-sm mb-4 ${isDark ? 'text-gray-300' : 'text-gray-600'}`}>
            Design custom strategies that blend forecast confidence, whale flow triggers, and narrative
            sentiment thresholds. Configure alerts, backtests, and automated trade checklists directly from the
            agent.
          </p>
          <div className="space-y-4">
            <PlaceholderCard
              icon={Brain}
              title="LLM Strategy Composer"
              description="Chain multi-step reasoning with agent memory to produce actionable playbooks for any asset."
              isDark={isDark}
            />
            <PlaceholderCard
              icon={Activity}
              title="Hybrid Backtesting"
              description="Stress-test strategies using historical SARIMAX signals, on-chain anomalies, and sentiment shocks."
              isDark={isDark}
            />
            <PlaceholderCard
              icon={Globe2}
              title="Global Alerts"
              description="Subscribe to macro, regulatory, and funding signals that can impact your current watchlist."
              isDark={isDark}
            />
          </div>
        </div>
      </div>
    </div>
  )
}

export default AdvancedAnalytics
