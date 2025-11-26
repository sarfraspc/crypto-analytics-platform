import { useTheme } from '../hooks/useTheme'

const Footer = () => {
  const { isDark } = useTheme()
  return (
    <footer className={`mt-10 border-t ${isDark ? 'border-accent-500/20' : 'border-accent-500/20'}`}>
      <div className={`max-w-7xl mx-auto px-4 py-6 flex flex-col sm:flex-row items-center justify-between text-sm gap-2 ${isDark ? 'text-gray-400' : 'text-gray-500'}`}>
        <p>© {new Date().getFullYear()} Novum Crypto - AI Powered Crypto Analysis Platform. All rights reserved.</p>
        <div className="flex gap-4">
          <a href="https://github.com/sarfraspc" className="hover:text-accent-400 transition-colors" target="_blank" rel="noopener noreferrer">
            GitHub
          </a>
          <a href="https://www.linkedin.com/in/muhammedsarfras" className="hover:text-accent-400 transition-colors" target="_blank" rel="noopener noreferrer">
            LinkedIn
          </a>
          <a href="mailto:sarfrasspc@gmail.com" className="hover:text-accent-400 transition-colors" target="_blank" rel="noopener noreferrer">
            Contact
          </a>
        </div>
      </div>
    </footer>
  )
}

export default Footer
