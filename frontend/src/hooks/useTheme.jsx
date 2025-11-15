import { createContext, useContext, useEffect, useMemo, useState } from 'react'

const ThemeContext = createContext(null)

const getInitialTheme = () => {
  if (typeof window === 'undefined') {
    return true
  }
  const stored = window.localStorage.getItem('nc-theme')
  if (stored) {
    return stored === 'dark'
  }
  return window.matchMedia('(prefers-color-scheme: dark)').matches
}

export const ThemeProvider = ({ children }) => {
  const [isDark, setIsDark] = useState(getInitialTheme)

  useEffect(() => {
    if (typeof document === 'undefined') return
    const root = document.documentElement
    if (isDark) {
      root.classList.add('dark')
    } else {
      root.classList.remove('dark')
    }
    window.localStorage.setItem('nc-theme', isDark ? 'dark' : 'light')
  }, [isDark])

  const value = useMemo(
    () => ({
      isDark,
      toggleTheme: () => setIsDark((prev) => !prev),
    }),
    [isDark],
  )

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>
}

export const useTheme = () => {
  const context = useContext(ThemeContext)
  if (!context) {
    throw new Error('useTheme must be used within a ThemeProvider')
  }
  return context
}
