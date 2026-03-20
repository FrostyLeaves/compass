import { createContext, useContext, useEffect, useMemo, useState } from 'react'
import { askQuestion, type SearchResultItem } from '@/lib/api'
import { useLang } from '@/lib/lang'

const STORAGE_KEY = 'compass-qa-chats'

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  sources?: SearchResultItem[]
  createdAt: string
  status?: 'pending' | 'done' | 'error'
}

export interface ChatSession {
  id: string
  title: string
  createdAt: string
  updatedAt: string
  messages: Message[]
}

interface ChatState {
  chats: ChatSession[]
  activeChatId: string
}

interface QAContextValue {
  chats: ChatSession[]
  activeChat: ChatSession | undefined
  activeChatId: string
  loadingChatId: string | null
  isLoading: boolean
  newChat: () => void
  selectChat: (chatId: string) => void
  deleteChat: (chatId: string) => void
  sendQuestion: (question: string) => Promise<void>
}

const QAContext = createContext<QAContextValue>({
  chats: [],
  activeChat: undefined,
  activeChatId: '',
  loadingChatId: null,
  isLoading: false,
  newChat: () => {},
  selectChat: () => {},
  deleteChat: () => {},
  sendQuestion: async () => {},
})

function createId() {
  if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) return crypto.randomUUID()
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`
}

function createChat(): ChatSession {
  const now = new Date().toISOString()
  return {
    id: createId(),
    title: 'New Chat',
    createdAt: now,
    updatedAt: now,
    messages: [],
  }
}

function createMessage(
  role: Message['role'],
  content: string,
  sources?: SearchResultItem[],
  status: Message['status'] = 'done',
): Message {
  return {
    id: createId(),
    role,
    content,
    sources,
    createdAt: new Date().toISOString(),
    status,
  }
}

function summarizeTitle(content: string) {
  const compact = content.replace(/\s+/g, ' ').trim()
  if (!compact) return 'New Chat'
  return compact.length > 40 ? `${compact.slice(0, 40)}...` : compact
}

function sortChats(chats: ChatSession[]) {
  return [...chats].sort((a, b) => Date.parse(b.updatedAt) - Date.parse(a.updatedAt))
}

function loadChatState(): ChatState {
  const fallback = createChat()
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return { chats: [fallback], activeChatId: fallback.id }

    const parsed = JSON.parse(raw) as Partial<ChatState>
    if (!parsed || !Array.isArray(parsed.chats) || parsed.chats.length === 0) {
      return { chats: [fallback], activeChatId: fallback.id }
    }

    const chats = parsed.chats.flatMap((chat): ChatSession[] => {
      if (!chat || typeof chat.id !== 'string' || !Array.isArray(chat.messages)) return []
      const messages = chat.messages.flatMap((message): Message[] => {
        if (!message || (message.role !== 'user' && message.role !== 'assistant') || typeof message.content !== 'string') {
          return []
        }
        return [{
          id: typeof message.id === 'string' ? message.id : createId(),
          role: message.role,
          content:
            message.status === 'pending'
              ? (typeof message.content === 'string' && message.content.trim() ? message.content : 'Request interrupted. Please retry.')
              : message.content,
          sources: Array.isArray(message.sources) ? message.sources : undefined,
          createdAt: typeof message.createdAt === 'string' ? message.createdAt : new Date().toISOString(),
          status:
            message.status === 'pending'
              ? 'error'
              : (message.status === 'done' || message.status === 'error' ? message.status : 'done'),
        }]
      })
      return [{
        id: chat.id,
        title: typeof chat.title === 'string' && chat.title.trim() ? chat.title : 'New Chat',
        createdAt: typeof chat.createdAt === 'string' ? chat.createdAt : new Date().toISOString(),
        updatedAt: typeof chat.updatedAt === 'string' ? chat.updatedAt : new Date().toISOString(),
        messages,
      }]
    })

    if (chats.length === 0) return { chats: [fallback], activeChatId: fallback.id }

    const sorted = sortChats(chats)
    const activeChatId = sorted.some(chat => chat.id === parsed.activeChatId) ? parsed.activeChatId! : sorted[0].id
    return { chats: sorted, activeChatId }
  } catch {
    return { chats: [fallback], activeChatId: fallback.id }
  }
}

export function QAProvider({ children }: { children: React.ReactNode }) {
  const { activeLang } = useLang()
  const [chatState, setChatState] = useState<ChatState>(loadChatState)
  const [loadingChatId, setLoadingChatId] = useState<string | null>(null)

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(chatState))
    } catch {
      /* storage unavailable */
    }
  }, [chatState])

  const chats = chatState.chats
  const activeChat = chats.find(chat => chat.id === chatState.activeChatId) ?? chats[0]

  const updateChat = (chatId: string, updater: (chat: ChatSession) => ChatSession) => {
    setChatState(prev => ({
      ...prev,
      chats: sortChats(prev.chats.map(chat => (chat.id === chatId ? updater(chat) : chat))),
    }))
  }

  const appendMessage = (chatId: string, message: Message) => {
    updateChat(chatId, chat => {
      const nextMessages = [...chat.messages, message]
      const nextTitle = chat.title === 'New Chat' && message.role === 'user'
        ? summarizeTitle(message.content)
        : chat.title
      return {
        ...chat,
        title: nextTitle,
        updatedAt: message.createdAt,
        messages: nextMessages,
      }
    })
  }

  const updateMessage = (chatId: string, messageId: string, updater: (message: Message) => Message) => {
    updateChat(chatId, chat => ({
      ...chat,
      messages: chat.messages.map(message => (message.id === messageId ? updater(message) : message)),
    }))
  }

  const newChat = () => {
    const chat = createChat()
    setChatState(prev => ({
      chats: [chat, ...prev.chats],
      activeChatId: chat.id,
    }))
  }

  const selectChat = (chatId: string) => {
    setChatState(prev => ({ ...prev, activeChatId: chatId }))
  }

  const deleteChat = (chatId: string) => {
    if (loadingChatId === chatId) return
    setChatState(prev => {
      const remaining = prev.chats.filter(chat => chat.id !== chatId)
      if (remaining.length === 0) {
        const chat = createChat()
        return { chats: [chat], activeChatId: chat.id }
      }
      return {
        chats: remaining,
        activeChatId: prev.activeChatId === chatId ? remaining[0].id : prev.activeChatId,
      }
    })
  }

  const sendQuestion = async (question: string) => {
    const q = question.trim()
    if (!q || loadingChatId !== null || !activeChat) return

    const chatId = activeChat.id
    const userMessage = createMessage('user', q)
    const pendingReply = createMessage('assistant', '', undefined, 'pending')

    appendMessage(chatId, userMessage)
    appendMessage(chatId, pendingReply)
    setLoadingChatId(chatId)

    try {
      const res = await askQuestion(q, 50, activeLang)
      updateMessage(chatId, pendingReply.id, message => ({
        ...message,
        content: res.answer,
        sources: res.sources,
        status: 'done',
      }))
    } catch (err) {
      const detail = err instanceof Error ? err.message : 'Unknown error'
      updateMessage(chatId, pendingReply.id, message => ({
        ...message,
        content: `Error: ${detail}`,
        status: 'error',
      }))
    } finally {
      setLoadingChatId(current => (current === chatId ? null : current))
    }
  }

  const value = useMemo(() => ({
    chats,
    activeChat,
    activeChatId: chatState.activeChatId,
    loadingChatId,
    isLoading: loadingChatId !== null,
    newChat,
    selectChat,
    deleteChat,
    sendQuestion,
  }), [activeChat, chatState.activeChatId, chats, loadingChatId])

  return (
    <QAContext.Provider value={value}>
      {children}
    </QAContext.Provider>
  )
}

export function useQA() {
  return useContext(QAContext)
}
