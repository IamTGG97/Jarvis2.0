export type SessionSummary = {
  id: string
  created_at: string
  updated_at: string
  turn_count: number
  preview: string | null
}

export type SessionTurn = {
  id: number
  transcript: string
  response: string
  created_at: string
}

export const SESSION_STORAGE_KEY = 'jarvis-conversation-session'

async function readJson<T>(response: Response, fallbackMessage: string): Promise<T> {
  const body = await response.json().catch(() => ({})) as T & { detail?: string }
  if (!response.ok) throw new Error(body.detail || fallbackMessage)
  return body
}

export async function listSessions(): Promise<SessionSummary[]> {
  const response = await fetch('/api/sessions')
  const body = await readJson<{ sessions: SessionSummary[] }>(response, 'Could not load conversations.')
  return body.sessions
}

export async function createSession(): Promise<SessionSummary> {
  const response = await fetch('/api/sessions', { method: 'POST' })
  return readJson<SessionSummary>(response, 'Could not start a new conversation.')
}

export async function deleteSession(sessionId: string): Promise<void> {
  const response = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}`, { method: 'DELETE' })
  if (!response.ok && response.status !== 404) {
    await readJson(response, 'Could not delete the conversation.')
  }
}

/** Returns the session's turns oldest first, or an empty list if it has none yet. */
export async function fetchSessionTurns(sessionId: string): Promise<SessionTurn[]> {
  const response = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}/turns`)
  if (response.status === 404) return []
  const body = await readJson<{ turns: SessionTurn[] }>(response, 'Could not load the conversation.')
  return body.turns
}

export function formatRelativeTime(isoTime: string, now = Date.now()): string {
  const seconds = Math.round((now - new Date(isoTime).getTime()) / 1000)
  if (seconds < 60) return 'just now'
  const minutes = Math.round(seconds / 60)
  if (minutes < 60) return `${minutes} min ago`
  const hours = Math.round(minutes / 60)
  if (hours < 24) return `${hours} h ago`
  const days = Math.round(hours / 24)
  if (days < 7) return `${days} d ago`
  return new Date(isoTime).toLocaleDateString()
}
