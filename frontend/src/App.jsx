import { useState } from 'react'
import NavBar from './components/NavBar'
import Footer from './components/Footer'
import Dashboard from './pages/Dashboard'
import AdvancedAnalytics from './pages/AdvancedAnalytics'
import { ThemeProvider, useTheme } from './hooks/useTheme'
import { SymbolProvider } from './hooks/useSymbol'

const AppContent = () => {
  const { isDark } = useTheme()
  const [currentPage, setCurrentPage] = useState('dashboard')

  return (
    <div className={`${isDark ? 'bg-slate-950 text-white' : 'bg-gray-50 text-gray-900'} min-h-screen`}> 
      <NavBar currentPage={currentPage} onPageChange={setCurrentPage} />
      <main className="max-w-7xl mx-auto px-3 sm:px-4 md:px-6 lg:px-8 py-4 sm:py-8">
        {currentPage === 'dashboard' ? <Dashboard /> : <AdvancedAnalytics />}
      </main>
      <Footer />
    </div>
  )
}

const App = () => (
  <ThemeProvider>
    <SymbolProvider>
      <AppContent />
    </SymbolProvider>
  </ThemeProvider>
)

export default App
