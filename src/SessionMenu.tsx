import { useCallback, useEffect, useRef, useState } from 'react'
import {
  createSession,
  deleteSession,
  formatRelativeTime,
  listSessions,
  type SessionSummary,
} from './sessionsApi'

type SessionMenuProps = {
  activeSessionId: string
  disabled: boolean
  onSelectSession: (sessionId: string) => void
}

function SessionMenu({ activeSessionId, disabled, onSelectSession }: SessionMenuProps) {
  const [isOpen, setIsOpen] = useState(false)
  const [sessions, setSessions] = useState<SessionSummary[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [isWorking, setIsWorking] = useState(false)
  const [error, setError] = useState('')
  const [pendingDeleteId, setPendingDeleteId] = useState<string | null>(null)
  const panelRef = useRef<HTMLDivElement>(null)

  const refresh = useCallback(async () => {
    setIsLoading(true)
    setError('')
    try {
      setSessions(await listSessions())
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : 'Could not load conversations.')
    } finally {
      setIsLoading(false)
    }
  }, [])

  useEffect(() => {
    if (!isOpen) return
    void refresh()
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setIsOpen(false)
    }
    const closeOnOutsideClick = (event: PointerEvent) => {
      if (panelRef.current && !panelRef.current.contains(event.target as Node)) setIsOpen(false)
    }
    window.addEventListener('keydown', closeOnEscape)
    window.addEventListener('pointerdown', closeOnOutsideClick)
    return () => {
      window.removeEventListener('keydown', closeOnEscape)
      window.removeEventListener('pointerdown', closeOnOutsideClick)
    }
  }, [isOpen, refresh])

  useEffect(() => {
    if (disabled) setIsOpen(false)
  }, [disabled])

  async function startNewConversation() {
    setIsWorking(true)
    setError('')
    try {
      const session = await createSession()
      onSelectSession(session.id)
      setIsOpen(false)
    } catch (createError) {
      setError(createError instanceof Error ? createError.message : 'Could not start a new conversation.')
    } finally {
      setIsWorking(false)
    }
  }

  async function confirmDelete(sessionId: string) {
    setIsWorking(true)
    setError('')
    try {
      await deleteSession(sessionId)
      setPendingDeleteId(null)
      if (sessionId === activeSessionId) {
        const replacement = await createSession()
        onSelectSession(replacement.id)
      }
      await refresh()
    } catch (deleteError) {
      setError(deleteError instanceof Error ? deleteError.message : 'Could not delete the conversation.')
    } finally {
      setIsWorking(false)
    }
  }

  const activeIsListed = sessions.some((session) => session.id === activeSessionId)

  return (
    <div className="session-menu" ref={panelRef}>
      <button
        type="button"
        className="session-toggle"
        onClick={() => {
          setPendingDeleteId(null)
          setIsOpen((open) => !open)
        }}
        disabled={disabled}
        aria-expanded={isOpen}
        aria-controls="session-panel"
        title="Conversations"
      >
        <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 5.5h16M4 12h16M4 18.5h10" /></svg>
        <span className="sr-only">Conversations</span>
      </button>

      {isOpen && (
        <div className="session-panel" id="session-panel" role="dialog" aria-label="Conversations">
          <button
            type="button"
            className="session-new"
            onClick={() => void startNewConversation()}
            disabled={isWorking}
          >
            + New conversation
          </button>

          {error && <p className="session-error" role="alert">{error}</p>}
          {isLoading && !sessions.length && <p className="session-empty">Loading…</p>}

          <ul className="session-list">
            {!activeIsListed && !isLoading && (
              <li className="session-row is-active">
                <span className="session-select" aria-current="true">
                  <span className="session-preview">Current conversation</span>
                  <span className="session-meta">No messages yet</span>
                </span>
              </li>
            )}
            {sessions.map((session) => {
              const isActive = session.id === activeSessionId
              const preview = session.preview || 'No messages yet'
              if (pendingDeleteId === session.id) {
                return (
                  <li key={session.id} className="session-row is-confirming">
                    <span className="session-confirm-text">Delete “{preview}”? This erases its memory.</span>
                    <span className="session-confirm-actions">
                      <button type="button" className="session-danger" onClick={() => void confirmDelete(session.id)} disabled={isWorking}>
                        Delete
                      </button>
                      <button type="button" onClick={() => setPendingDeleteId(null)} disabled={isWorking}>
                        Cancel
                      </button>
                    </span>
                  </li>
                )
              }
              return (
                <li key={session.id} className={`session-row${isActive ? ' is-active' : ''}`}>
                  <button
                    type="button"
                    className="session-select"
                    onClick={() => {
                      onSelectSession(session.id)
                      setIsOpen(false)
                    }}
                    aria-current={isActive ? 'true' : undefined}
                    disabled={isWorking}
                  >
                    <span className="session-preview">{preview}</span>
                    <span className="session-meta">
                      {session.turn_count} {session.turn_count === 1 ? 'turn' : 'turns'} · {formatRelativeTime(session.updated_at)}
                    </span>
                  </button>
                  <button
                    type="button"
                    className="session-delete"
                    onClick={() => setPendingDeleteId(session.id)}
                    disabled={isWorking}
                    aria-label={`Delete conversation: ${preview}`}
                    title="Delete conversation"
                  >
                    <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 7h14M10 7V5h4v2M7 7l1 12h8l1-12" /></svg>
                  </button>
                </li>
              )
            })}
          </ul>
        </div>
      )}
    </div>
  )
}

export default SessionMenu
