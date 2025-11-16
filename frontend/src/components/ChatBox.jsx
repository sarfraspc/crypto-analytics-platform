import { useEffect, useMemo, useState, useRef } from 'react'
import ReactMarkdown from 'react-markdown'
import { MessageSquare, Send, Loader2, RefreshCw } from 'lucide-react'
import { useTheme } from '../hooks/useTheme'
import { useSymbol } from '../hooks/useSymbol'
import { sendChatMessage } from '../services/api'
import ErrorBox from './ErrorBox'

const QUERY_CATEGORIES = ['forecast', 'onchain', 'sentiment', 'combined', 'backtest', 'patterns']

const DataCards = ({ title, data, isDark }) => {
  if (!data || typeof data !== 'object') return null
  const entriesSource = typeof data.metrics === 'object' && data.metrics ? data.metrics : data
  const entries = Object.entries(entriesSource).filter(
    ([, value]) => ['string', 'number', 'boolean'].includes(typeof value),
  )
  if (entries.length === 0) return null

  return (
    <div className={`mt-2 rounded-lg border ${isDark ? 'border-slate-700 bg-slate-800/60' : 'border-gray-200 bg-gray-50'}`}>
      <p className={`px-3 pt-3 text-xs font-semibold uppercase ${isDark ? 'text-gray-300' : 'text-gray-700'}`}>{title}</p>
      <div className="grid grid-cols-2 gap-2 p-3">
        {entries.map(([key, value]) => (
          <div
            key={key}
            className={`rounded-md px-2 py-2 ${isDark ? 'bg-slate-700/60 text-gray-100' : 'bg-white text-gray-800'}`}
          >
            <p className="text-[10px] font-semibold uppercase text-gray-400">{key.replace(/_/g, ' ')}</p>
            <p className="text-sm">
              {typeof value === 'number' ? value.toLocaleString(undefined, { maximumFractionDigits: 3 }) : String(value)}
            </p>
          </div>
        ))}
      </div>
    </div>
  )
}

const AssistantBubble = ({ message, isDark, onChipClick }) => {
  const isUser = message.role === 'user'
  const bubbleClasses = isUser
    ? 'bg-indigo-600 text-white'
    : isDark
      ? 'bg-slate-700 text-gray-100'
      : 'bg-gray-100 text-gray-900'

  const dataSections =
    message.data && typeof message.data === 'object' && Object.keys(message.data).length > 0
      ? Object.entries(message.data)
      : []

  return (
    <div className={`max-w-[80%] rounded-xl px-4 py-3 ${bubbleClasses}`}>
      {isUser ? (
        <p className="text-sm leading-relaxed whitespace-pre-line">{message.content}</p>
      ) : (
        <div className="markdown-body">
          <ReactMarkdown className="prose prose-invert prose-sm max-w-none">{message.content}</ReactMarkdown>
        </div>
      )}

      {!isUser && message.chips?.length > 0 && (
        <div className="flex flex-wrap gap-2 mt-3">
          {message.chips.map((chip) => (
            <button
              type="button"
              key={chip}
              onClick={() => onChipClick?.(chip)}
              className={`text-xs px-2 py-1 rounded-full ${
                isDark ? 'bg-slate-600 text-gray-200 hover:bg-slate-500' : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
              }`}
            >
              {chip}
            </button>
          ))}
        </div>
      )}

      {!isUser &&
        dataSections.map(([key, value]) => (
          <DataCards key={key} title={`${key} insights`} data={value} isDark={isDark} />
        ))}
    </div>
  )
}

const ChatBox = () => {
  const { isDark } = useTheme()
  const { symbol } = useSymbol()
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const placeholder = 'Ask about forecast, onchain, or sentiment...'
  const [isTyping, setIsTyping] = useState(false)
  const [error, setError] = useState(null)
  const [regenerating, setRegenerating] = useState(false)
  const messagesEndRef = useRef(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages, isTyping])

  useEffect(() => {
    setMessages((prev) => {
      const intro = {
        role: 'assistant',
        content: `Hi! I'm your AI crypto analyst.\nAsk about **price**, **on-chain**, **sentiment**, **reports**, **backtests**, or **patterns** for any crypto.`,
        chips: QUERY_CATEGORIES,
      }
      if (prev.length === 0 || prev[0].role !== 'assistant') {
        return [intro, ...prev]
      }
      if (prev[0].content === intro.content) {
        return prev
      }
      const [, ...rest] = prev
      return [intro, ...rest]
    })
  }, [symbol])

  const lastUserQuestion = useMemo(
    () => [...messages].reverse().find((message) => message.role === 'user')?.content || null,
    [messages],
  )

  const buildQuickQuestion = (category) => {
    switch (category) {
      case 'forecast':
        return `Run a forecast for ${symbol}`
      case 'onchain':
        return `Onchain flows for ${symbol}`
      case 'sentiment':
        return `Market sentiment for ${symbol}`
      case 'combined':
        return `Combined view (forecast/onchain/sentiment)`
      case 'backtest':
        return `Backtest a simple strategy`
      case 'patterns':
        return `Chart patterns and TA signals`
      default:
        return `${category} insights`
    }
  }

  const handleSend = async (overrideQuestion, options = {}) => {
    const question = (overrideQuestion ?? input).trim()
    if (!question || isTyping || regenerating) return

    const nextMessages = [...messages, { role: 'user', content: question }]
    setMessages(nextMessages)
    setInput('')
    setIsTyping(true)
    setError(null)

    try {
      const response = await sendChatMessage(symbol, question, nextMessages, options)
      const content = response.final_answer || response.response || 'No response received.'
      const chips = response.categories || Object.keys(response.data_insights || {}).filter((k) => response.data_insights[k])
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content,
          chips: chips?.filter(Boolean),
          data: response.data || response.data_insights,
          queryType: response.query_type,
        },
      ])
    } catch (err) {
      const fallback = err?.message || 'Assistant unavailable. Please try later.'
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: fallback,
        },
      ])
      setError(fallback)
    } finally {
      setIsTyping(false)
    }
  }

  const handleRegenerate = async () => {
    if (!lastUserQuestion || isTyping || regenerating) return
    setRegenerating(true)
    setError(null)
    const history = messages.filter((_, idx) => idx !== messages.length - 1 || messages[idx].role !== 'assistant')
    try {
      const response = await sendChatMessage(symbol, lastUserQuestion, history)
      const content = response.final_answer || response.response || 'No response received.'
      const chips = response.categories || Object.keys(response.data_insights || {}).filter((k) => response.data_insights[k])
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content,
          chips: chips?.filter(Boolean),
          data: response.data || response.data_insights,
          queryType: response.query_type,
        },
      ])
    } catch (err) {
      const fallback = err?.message || 'Assistant unavailable. Please try later.'
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: fallback,
        },
      ])
      setError(fallback)
    } finally {
      setRegenerating(false)
    }
  }

  const handleChipClick = (chip) => {
    const prompt = buildQuickQuestion(chip)
    setInput(prompt)
    const forceType = QUERY_CATEGORIES.includes(chip) ? chip : chip.toLowerCase()
    handleSend(prompt, { force_query_type: forceType })
  }

  return (
    <div
      className={`rounded-xl shadow-lg flex flex-col ${
        isDark ? 'bg-slate-800/50 backdrop-blur-xl border border-slate-700' : 'bg-white border border-gray-200'
      }`}
      style={{ height: '600px' }}
    >
      <div className={`px-6 py-4 border-b flex-shrink-0 ${isDark ? 'border-slate-700 bg-slate-800/80' : 'border-gray-200 bg-gray-50'}`}>
        <div className="flex items-center gap-2">
          <MessageSquare className="text-indigo-500" size={20} />
          <h3 className={`font-semibold ${isDark ? 'text-white' : 'text-gray-900'}`}>AI Assistant</h3>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-6 space-y-4 min-h-0">
        {messages.map((message, index) => (
          <div
            key={`${message.role}-${index}`}
            className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <AssistantBubble message={message} isDark={isDark} onChipClick={handleChipClick} />
          </div>
        ))}
        {isTyping && (
          <div className="flex justify-start">
            <div className={`rounded-xl px-4 py-3 ${isDark ? 'bg-slate-700' : 'bg-gray-100'}`}>
              <Loader2 className="animate-spin text-indigo-500" size={20} />
            </div>
          </div>
        )}
        {!isTyping && !regenerating && messages.some((m) => m.role === 'assistant') && (
          <div className="sticky bottom-2 flex justify-end">
            <button
              type="button"
              onClick={handleRegenerate}
              disabled={!lastUserQuestion}
              className={`inline-flex items-center gap-2 px-3 py-2 rounded-lg text-xs font-medium shadow ${
                isDark
                  ? 'bg-slate-700 text-gray-100 hover:bg-slate-600 disabled:bg-slate-800 disabled:text-gray-500'
                  : 'bg-white border border-gray-200 text-gray-700 hover:bg-gray-50 disabled:bg-gray-100 disabled:text-gray-400'
              }`}
            >
              <RefreshCw size={16} className="shrink-0" />
              Regenerate
            </button>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <div className={`p-4 border-t space-y-3 flex-shrink-0 ${isDark ? 'border-slate-700 bg-slate-800/80' : 'border-gray-200 bg-gray-50'}`}>
        {error && <ErrorBox message={error} />}
        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === 'Enter') {
                handleSend()
              }
            }}
            placeholder={placeholder}
            className={`flex-1 px-4 py-2 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 ${
              isDark ? 'bg-slate-700 text-white placeholder-gray-400' : 'bg-white text-gray-900 placeholder-gray-500'
            }`}
          />
          <button
            type="button"
            onClick={() => handleSend()}
            className="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-all"
          >
            <Send size={20} />
          </button>
        </div>
        <div className="flex flex-wrap gap-2">
          {QUERY_CATEGORIES.map((category) => (
            <button
              type="button"
              key={category}
              onClick={() => handleChipClick(category)}
              className={`text-xs px-3 py-1 rounded-full border ${
                isDark ? 'border-slate-600 text-gray-200 hover:bg-slate-700' : 'border-gray-300 text-gray-700 hover:bg-gray-100'
              }`}
            >
              {buildQuickQuestion(category)}
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}

export default ChatBox