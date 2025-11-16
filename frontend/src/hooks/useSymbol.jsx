import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react'
import { getAvailableSymbols } from '../services/api'

const SymbolContext = createContext(null)
const FALLBACK_SYMBOLS = ['BTC', 'ETH', 'SOL', 'AVAX', 'MATIC']

export const SymbolProvider = ({ children, defaultSymbol = 'BTC' }) => {
  const [symbol, setSymbol] = useState(defaultSymbol)
  const [availableSymbols, setAvailableSymbols] = useState(
    defaultSymbol ? [defaultSymbol, ...FALLBACK_SYMBOLS.filter((item) => item !== defaultSymbol)] : FALLBACK_SYMBOLS,
  )
  const [loadingSymbols, setLoadingSymbols] = useState(false)
  const [symbolsError, setSymbolsError] = useState(null)

  const refreshSymbols = useCallback(async () => {
    setLoadingSymbols(true)
    setSymbolsError(null)
    try {
      const response = await getAvailableSymbols()
      const fetched = Array.isArray(response?.symbols) ? response.symbols.filter((item) => typeof item === 'string') : []
      if (fetched.length === 0) {
        throw new Error('No symbols returned')
      }
      setAvailableSymbols(fetched)
      setSymbol((prev) => (prev && fetched.includes(prev) ? prev : fetched[0]))
    } catch (err) {
      console.error('Failed to load available symbols', err)
      setSymbolsError(err?.message ?? 'Unable to load available symbols')
      setAvailableSymbols((prev) => (prev.length ? prev : FALLBACK_SYMBOLS))
      setSymbol((prev) => prev || defaultSymbol || FALLBACK_SYMBOLS[0])
    } finally {
      setLoadingSymbols(false)
    }
  }, [defaultSymbol])

  useEffect(() => {
    refreshSymbols()
  }, [refreshSymbols])

  const contextValue = useMemo(
    () => ({
      symbol,
      setSymbol,
      availableSymbols,
      loadingSymbols,
      symbolsError,
      refreshSymbols,
    }),
    [symbol, availableSymbols, loadingSymbols, symbolsError, refreshSymbols],
  )

  return (
    <SymbolContext.Provider value={contextValue}>
      {children}
    </SymbolContext.Provider>
  )
}

export const useSymbol = () => {
  const context = useContext(SymbolContext)
  if (!context) {
    throw new Error('useSymbol must be used within a SymbolProvider')
  }
  return context
}
