import { Moon, Sun } from 'lucide-react'
import Logo from './Logo'
import { useTheme } from '../hooks/useTheme'
import { useSymbol } from '../hooks/useSymbol'

const NavBar = ({ currentPage, onPageChange }) => {
  const { isDark, toggleTheme } = useTheme()
  const { symbol, setSymbol } = useSymbol()
  const symbols = ['BTC', 'ETH', 'SOL', 'AVAX', 'MATIC']

  return (
    <nav
      className={`sticky top-0 z-50 backdrop-blur-xl border-b ${
        isDark ? 'bg-slate-900/80 border-slate-800' : 'bg-white/80 border-gray-200'
      }`}
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <Logo />
          <div className="flex items-center gap-4">
            <select
              value={symbol}
              onChange={(event) => setSymbol(event.target.value)}
              className={`px-4 py-2 rounded-lg font-medium border focus:outline-none focus:ring-2 focus:ring-indigo-500 ${
                isDark
                  ? 'bg-slate-800 text-white border-slate-700'
                  : 'bg-gray-100 text-gray-900 border-gray-300'
              }`}
            >
              {symbols.map((item) => (
                <option key={item} value={item}>
                  {item}
                </option>
              ))}
            </select>

            <div
              className={`hidden sm:flex gap-2 px-2 py-1 rounded-lg ${
                isDark ? 'bg-slate-800' : 'bg-gray-100'
              }`}
            >
              {[
                { id: 'dashboard', label: 'Dashboard' },
                { id: 'advanced', label: 'Advanced Analytics' },
              ].map((item) => (
                <button
                  key={item.id}
                  type="button"
                  onClick={() => onPageChange(item.id)}
                  className={`px-4 py-2 rounded-md font-medium transition-all ${
                    currentPage === item.id
                      ? 'bg-indigo-600 text-white shadow-lg'
                      : isDark
                        ? 'text-gray-400 hover:text-white'
                        : 'text-gray-600 hover:text-gray-900'
                  }`}
                >
                  {item.label}
                </button>
              ))}
            </div>

            <div className="sm:hidden">
              <select
                value={currentPage}
                onChange={(event) => onPageChange(event.target.value)}
                className={`px-3 py-2 rounded-lg font-medium border focus:outline-none focus:ring-2 focus:ring-indigo-500 ${
                  isDark
                    ? 'bg-slate-800 text-white border-slate-700'
                    : 'bg-gray-100 text-gray-900 border-gray-300'
                }`}
              >
                <option value="dashboard">Dashboard</option>
                <option value="advanced">Advanced Analytics</option>
              </select>
            </div>

            <button
              type="button"
              onClick={toggleTheme}
              className={`p-2 rounded-lg transition-all ${
                isDark
                  ? 'bg-slate-800 text-yellow-400 hover:bg-slate-700'
                  : 'bg-gray-100 text-indigo-600 hover:bg-gray-200'
              }`}
              aria-label="Toggle color theme"
            >
              {isDark ? <Sun size={20} /> : <Moon size={20} />}
            </button>
          </div>
        </div>
      </div>
    </nav>
  )
}

export default NavBar
