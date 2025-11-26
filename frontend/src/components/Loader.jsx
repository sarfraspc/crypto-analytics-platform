import { Loader2 } from 'lucide-react'

const Loader = ({ label = 'Loading data...' }) => (
  <div className="flex flex-col items-center justify-center gap-2 py-10 text-accent-500">
    <Loader2 className="animate-spin" size={32} />
    <p className="text-sm font-medium text-center text-accent-400">{label}</p>
  </div>
)

export default Loader
