export async function onRequest(context) {
  const { request, env } = context
  const incomingUrl = new URL(request.url)

  // Lightweight debug helper to confirm what env keys are visible to the Function
  if (incomingUrl.pathname === '/api/debug-env') {
    const keys = env ? Object.keys(env) : []
    return new Response(JSON.stringify({ hasEnv: !!env, keys }), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    })
  }

  // Prefer BACKEND_ORIGIN, but also allow a fallback name just in case
  const backendOrigin = env.BACKEND_ORIGIN || env.VITE_BACKEND_ORIGIN
  if (!backendOrigin) {
    return new Response('Misconfigured: BACKEND_ORIGIN not set', { status: 500 })
  }

  const backendUrl = new URL(backendOrigin)
  backendUrl.pathname = incomingUrl.pathname.replace(/^\/api/, '')
  backendUrl.search = incomingUrl.search

  const backendRequest = new Request(backendUrl.toString(), request)

  try {
    return await fetch(backendRequest)
  } catch (error) {
    return new Response(`Backend Error: ${error.message || 'Unknown error'}`, {
      status: 502,
    })
  }
}
