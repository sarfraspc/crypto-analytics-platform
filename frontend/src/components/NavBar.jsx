import { useEffect, useRef, useState } from 'react'
import { Moon, Sun, ChevronDown, Loader2, Check, Menu, X } from 'lucide-react'
import Logo from './Logo'
import { useTheme } from '../hooks/useTheme'
import { useSymbol } from '../hooks/useSymbol'

const NavBar = ({ currentPage, onPageChange }) => {
  const { isDark, toggleTheme } = useTheme()
  const { symbol, setSymbol, availableSymbols, loadingSymbols, symbolsError } = useSymbol()
  const [dropdownOpen, setDropdownOpen] = useState(false)
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)
  const desktopDropdownRef = useRef(null)
  const mobileDropdownRef = useRef(null)
  
  const symbolOptions =
    Array.isArray(availableSymbols) && availableSymbols.length > 0
      ? availableSymbols
      : symbol
        ? [symbol]
        : []
  const dropdownDisabled = loadingSymbols || symbolOptions.length === 0

  useEffect(() => {
    const handleClickOutside = (event) => {
      const target = event.target
      const desktopEl = desktopDropdownRef.current
      const mobileEl = mobileDropdownRef.current
      const insideDesktop = desktopEl && desktopEl.contains(target)
      const insideMobile = mobileEl && mobileEl.contains(target)
      if (!insideDesktop && !insideMobile) {
        setDropdownOpen(false)
      }
    }
    const handleEscape = (event) => {
      if (event.key === 'Escape') {
        setDropdownOpen(false)
        setMobileMenuOpen(false)
      }
    }
    document.addEventListener('mousedown', handleClickOutside)
    document.addEventListener('keydown', handleEscape)
    return () => {
      document.removeEventListener('mousedown', handleClickOutside)
      document.removeEventListener('keydown', handleEscape)
    }
  }, [])

  const handleSymbolSelect = (value) => {
    setSymbol(value)
    setDropdownOpen(false)
  }

  const handlePageChange = (page) => {
    onPageChange(page)
    setMobileMenuOpen(false)
  }

  return (
    <nav
      className={`sticky top-0 z-50 backdrop-blur-xl border-b ${
        isDark ? 'bg-slate-900/80 border-slate-800' : 'bg-white/80 border-gray-200'
      }`}
    >
      <div className="max-w-7xl mx-auto px-3 sm:px-4 lg:px-8">
        <div className="flex items-center justify-between h-14 sm:h-16">
          {/* Logo - scaled down on mobile */}
          <div className="flex-shrink-0">
            <Logo />
          </div>

          {/* Desktop Navigation - hidden on mobile/tablet */}
          <div className="hidden lg:flex items-center gap-3 xl:gap-4">
            {/* Page Toggle Buttons */}
            <div
              className={`flex gap-2 px-2 py-1 rounded-lg ${
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
                  className={`px-3 xl:px-4 py-2 rounded-md font-medium text-sm transition-all ${
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

            {/* Symbol Dropdown (Desktop) */}
            <div className="relative" ref={desktopDropdownRef}>
              <button
                type="button"
                onClick={() => !dropdownDisabled && setDropdownOpen((prev) => !prev)}
                className={`w-36 xl:w-40 px-3 xl:px-4 py-2 text-sm rounded-lg font-medium border focus:outline-none focus:ring-2 focus:ring-indigo-500 flex items-center justify-between ${
                  isDark
                    ? 'bg-slate-800 text-white border-slate-700 disabled:bg-slate-800/60'
                    : 'bg-white text-gray-900 border-gray-200 disabled:bg-gray-50'
                } ${dropdownDisabled ? 'opacity-70 cursor-not-allowed' : 'cursor-pointer'}`}
                aria-haspopup="listbox"
                aria-expanded={dropdownOpen}
                disabled={dropdownDisabled}
              >
                <span className="truncate">
                  {loadingSymbols ? 'Loading...' : symbol || 'Select symbol'}
                </span>
                {loadingSymbols ? <Loader2 size={16} className="animate-spin" /> : <ChevronDown size={18} />}
              </button>
              {dropdownOpen && (
                <div
                  className={`absolute right-0 mt-2 w-48 rounded-lg border shadow-lg z-50 ${
                    isDark ? 'bg-slate-900 border-slate-700' : 'bg-white border-gray-200'
                  }`}
                >
                  <div className="max-h-64 overflow-y-auto py-1">
                    {symbolOptions.map((option) => (
                      <button
                        key={option}
                        type="button"
                        onClick={() => handleSymbolSelect(option)}
                        className={`w-full flex items-center justify-between px-4 py-2 text-sm ${
                          isDark
                            ? 'text-gray-100 hover:bg-slate-800'
                            : 'text-gray-800 hover:bg-gray-100'
                        } ${option === symbol ? 'font-semibold' : ''}`}
                      >
                        <span>{option}</span>
                        {option === symbol && <Check size={14} className="text-indigo-500" />}
                      </button>
                    ))}
                    {symbolOptions.length === 0 && (
                      <p className={`px-4 py-3 text-sm ${isDark ? 'text-gray-400' : 'text-gray-500'}`}>
                        No symbols available
                      </p>
                    )}
                  </div>
                </div>
              )}
              {symbolsError && (
                <p className={`mt-1 text-xs ${isDark ? 'text-rose-400' : 'text-rose-500'}`}>{symbolsError}</p>
              )}
            </div>

            {/* Theme Toggle */}
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

          {/* Mobile/Tablet Controls - visible only on mobile/tablet */}
          <div className="flex lg:hidden items-center gap-2">
            {/* Symbol Dropdown - Mobile */}
            <div className="relative" ref={mobileDropdownRef}>
              <button
                type="button"
                onClick={() => !dropdownDisabled && setDropdownOpen((prev) => !prev)}
                className={`w-20 sm:w-24 px-2 sm:px-3 py-2 text-xs sm:text-sm rounded-lg font-medium border focus:outline-none focus:ring-2 focus:ring-indigo-500 flex items-center justify-between ${
                  isDark
                    ? 'bg-slate-800 text-white border-slate-700 disabled:bg-slate-800/60'
                    : 'bg-white text-gray-900 border-gray-200 disabled:bg-gray-50'
                } ${dropdownDisabled ? 'opacity-70 cursor-not-allowed' : 'cursor-pointer'}`}
                aria-haspopup="listbox"
                aria-expanded={dropdownOpen}
                disabled={dropdownDisabled}
              >
                <span className="truncate text-xs sm:text-sm">
                  {loadingSymbols ? '...' : symbol || 'Symbol'}
                </span>
                {loadingSymbols ? (
                  <Loader2 size={14} className="animate-spin flex-shrink-0" />
                ) : (
                  <ChevronDown size={14} className="flex-shrink-0" />
                )}
              </button>
              {dropdownOpen && (
                <div
                  className={`absolute right-0 mt-2 w-40 sm:w-48 rounded-lg border shadow-lg z-50 ${
                    isDark ? 'bg-slate-900 border-slate-700' : 'bg-white border-gray-200'
                  }`}
                >
                  <div className="max-h-64 overflow-y-auto py-1">
                    {symbolOptions.map((option) => (
                      <button
                        key={option}
                        type="button"
                        onClick={() => handleSymbolSelect(option)}
                        className={`w-full flex items-center justify-between px-3 sm:px-4 py-2 text-sm ${
                          isDark
                            ? 'text-gray-100 hover:bg-slate-800'
                            : 'text-gray-800 hover:bg-gray-100'
                        } ${option === symbol ? 'font-semibold' : ''}`}
                      >
                        <span>{option}</span>
                        {option === symbol && <Check size={14} className="text-indigo-500" />}
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {/* Theme Toggle - Mobile */}
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
              {isDark ? <Sun size={18} /> : <Moon size={18} />}
            </button>

            {/* Mobile Menu Button */}
            <button
              type="button"
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className={`p-2 rounded-lg transition-all ${
                isDark
                  ? 'bg-slate-800 text-white hover:bg-slate-700'
                  : 'bg-gray-100 text-gray-900 hover:bg-gray-200'
              }`}
              aria-label="Toggle menu"
            >
              {mobileMenuOpen ? <X size={20} /> : <Menu size={20} />}
            </button>
          </div>
        </div>

        {/* Mobile Menu Dropdown */}
        {mobileMenuOpen && (
          <div className={`lg:hidden py-4 border-t ${isDark ? 'border-slate-800' : 'border-gray-200'}`}>
            <div className="flex flex-col space-y-2">
              {[
                { id: 'dashboard', label: 'Dashboard' },
                { id: 'advanced', label: 'Advanced Analytics' },
              ].map((item) => (
                <button
                  key={item.id}
                  type="button"
                  onClick={() => handlePageChange(item.id)}
                  className={`w-full px-4 py-3 rounded-lg font-medium text-sm text-left transition-all ${
                    currentPage === item.id
                      ? 'bg-indigo-600 text-white shadow-lg'
                      : isDark
                        ? 'text-gray-300 hover:bg-slate-800'
                        : 'text-gray-700 hover:bg-gray-100'
                  }`}
                >
                  {item.label}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>
    </nav>
  )
}

export default NavBar
