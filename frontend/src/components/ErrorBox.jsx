const ErrorBox = ({ message = 'Unable to load data. Please try again.', onRetry }) => (
  <div className="rounded-xl border border-red-500/30 bg-red-500/10 text-red-200 px-4 py-3 text-sm flex items-center justify-between gap-4">
    <span>{message}</span>
    {onRetry && (
      <button
        type="button"
        onClick={onRetry}
        className="px-3 py-1 rounded-md border border-red-400/40 text-red-100 hover:bg-red-500/20 text-xs font-semibold"
      >
        Retry
      </button>
    )}
  </div>
)

export default ErrorBox
