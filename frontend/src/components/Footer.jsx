import { useTheme } from '../hooks/useTheme'

const Footer = () => {
  const { isDark } = useTheme()
  return (
    <footer className={`mt-10 border-t ${isDark ? 'border-slate-800' : 'border-gray-200'}`}>
      <div className="max-w-7xl mx-auto px-4 py-6 flex flex-col sm:flex-row items-center justify-between text-sm text-gray-500 dark:text-gray-400 gap-2">
        <p>© {new Date().getFullYear()} Novum Crypto - AI Powered Crypto Analysis Platform. All rights reserved.</p>
        <div className="flex gap-4">
          <a href="https://github.com/sarfraspc" className="hover:text-indigo-500" target="_blank" rel="noopener noreferrer">
            GitHub
          </a>
          <a href="https://www.linkedin.com/in/muhammedsarfras" className="hover:text-indigo-500" target="_blank" rel="noopener noreferrer">
            LinkedIn
          </a>
          <a href="mailto:sarfrasspc@gmail.com" className="hover:text-indigo-500" target="_blank" rel="noopener noreferrer">
            Contact
          </a>
        </div>
      </div>
    </footer>
  )
}

export default Footer
