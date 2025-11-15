import { useEffect, useState } from 'react'
import { MessageSquare, Send, Loader2 } from 'lucide-react'
import { useTheme } from '../hooks/useTheme'
import { useSymbol } from '../hooks/useSymbol'
import { sendChatMessage } from '../services/api'
import ErrorBox from './ErrorBox'

const AssistantBubble = ({ message, isDark }) => (
  <div
    className={`max-w-[80%] rounded-xl px-4 py-3 ${
      message.role === 'user'
        ? 'bg-indigo-600 text-white'
        : isDark
          ? 'bg-slate-700 text-gray-100'
          : 'bg-gray-100 text-gray-900'
    }`}
  >
    <p className="text-sm leading-relaxed whitespace-pre-line">{message.content}</p>
    {message.chips && message.chips.length > 0 && (
      <div className="flex flex-wrap gap-2 mt-3">
        {message.chips.map((chip) => (
          <span
            key={chip}
            className={`text-xs px-2 py-1 rounded-full ${
              isDark ? 'bg-slate-600 text-gray-300' : 'bg-gray-200 text-gray-700'
            }`}
          >
            {chip}
          </span>
        ))}
      </div>
    )}
  </div>
)

const ChatBox = () => {
  const { isDark } = useTheme()
  const { symbol } = useSymbol()
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [isTyping, setIsTyping] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    setMessages((prev) => {
      const intro = {
        role: 'assistant',
        content: `Hi! I'm your AI crypto analyst. Ask me anything about ${symbol} or the crypto markets.`,
        chips: ['forecast', 'sentiment', 'on-chain'],
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

  const handleSend = async () => {
    const question = input.trim()
    if (!question || isTyping) return

    const nextMessages = [...messages, { role: 'user', content: question }]
    setMessages(nextMessages)
    setInput('')
    setIsTyping(true)
    setError(null)

    try {
      const response = await sendChatMessage(symbol, question, nextMessages)
      console.log('Chat response:', response)
      const content = response.final_answer || response.response || 'No response received.'
      const chips = response.categories || Object.keys(response.data_insights || {})
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content,
          chips: chips?.filter(Boolean),
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

  return (
    <div
      className={`rounded-xl shadow-lg overflow-hidden h-full flex flex-col ${
        isDark
          ? 'bg-slate-800/50 backdrop-blur-xl border border-slate-700'
          : 'bg-white border border-gray-200'
      }`}
    >
      <div className={`px-6 py-4 border-b ${isDark ? 'border-slate-700 bg-slate-800/80' : 'border-gray-200 bg-gray-50'}`}>
        <div className="flex items-center gap-2">
          <MessageSquare className="text-indigo-500" size={20} />
          <h3 className={`font-semibold ${isDark ? 'text-white' : 'text-gray-900'}`}>AI Assistant</h3>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-6 space-y-4">
        {messages.map((message, index) => (
          <div
            key={`${message.role}-${index}`}
            className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <AssistantBubble message={message} isDark={isDark} />
          </div>
        ))}
        {isTyping && (
          <div className="flex justify-start">
            <div className={`rounded-xl px-4 py-3 ${isDark ? 'bg-slate-700' : 'bg-gray-100'}`}>
              <Loader2 className="animate-spin text-indigo-500" size={20} />
            </div>
          </div>
        )}
      </div>

      <div className={`p-4 border-t ${isDark ? 'border-slate-700 bg-slate-800/80' : 'border-gray-200 bg-gray-50'}`}>
        {error && <ErrorBox message={error} />}
        <div className="flex gap-2 mt-2">
          <input
            type="text"
            value={input}
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === 'Enter') {
                handleSend()
              }
            }}
            placeholder="Ask about price trends, sentiment, or market analysis..."
            className={`flex-1 px-4 py-2 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 ${
              isDark
                ? 'bg-slate-700 text-white placeholder-gray-400'
                : 'bg-white text-gray-900 placeholder-gray-500'
            }`}
          />
          <button
            type="button"
            onClick={handleSend}
            className="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-all"
          >
            <Send size={20} />
          </button>
        </div>
      </div>
    </div>
  )
}

export default ChatBox
