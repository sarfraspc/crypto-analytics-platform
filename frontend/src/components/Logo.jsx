import { useTheme } from '../hooks/useTheme'
import logo from '../assets/logo.png'

const Logo = () => {
  const { isDark } = useTheme()
  return (
    <div className="flex items-center gap-2 sm:gap-3">
      <img src={logo} alt="Novum Crypto Logo" className="w-8 h-8 sm:w-10 sm:h-10 rounded-full flex-shrink-0" />
      <div className="hidden sm:block">
        <span className={`text-lg sm:text-xl font-bold ${isDark ? 'text-white' : 'text-gray-900'}`}>
          Novum Crypto
        </span>
        <span className="block text-xs text-gray-500 dark:text-gray-400">
          AI Powered Crypto Analysis Platform
        </span>
      </div>
      {/* Mobile: Show abbreviated version */}
      <div className="block sm:hidden">
        <span className={`text-base font-bold ${isDark ? 'text-white' : 'text-gray-900'}`}>
          Novum
        </span>
      </div>
    </div>
  )
}

export default Logo