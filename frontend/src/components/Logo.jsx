import { useTheme } from '../hooks/useTheme'
import logo from '../assets/logo.png'

const Logo = () => {
  const { isDark } = useTheme()
  return (
    <div className="flex items-center gap-3">
      <img src={logo} alt="Novum Crypto Logo" className="w-10 h-10 rounded-full" />
      <div>
        <span className={`text-xl font-bold ${isDark ? 'text-white' : 'text-gray-900'}`}>
          Novum Crypto
        </span>
        <span className="block text-xs text-gray-500 dark:text-gray-400">
          AI Powered Crypto Analysis Platform
        </span>
      </div>
    </div>
  )
}

export default Logo
