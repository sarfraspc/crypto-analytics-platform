export async function onRequest(context) {
  const { request, env } = context
  const incomingUrl = new URL(request.url)

  const backendOrigin = env.BACKEND_ORIGIN
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
