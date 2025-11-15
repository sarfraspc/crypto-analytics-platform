import { createContext, useContext, useState } from 'react'

const SymbolContext = createContext(null)

export const SymbolProvider = ({ children, defaultSymbol = 'BTC' }) => {
  const [symbol, setSymbol] = useState(defaultSymbol)
  return (
    <SymbolContext.Provider value={{ symbol, setSymbol }}>
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
