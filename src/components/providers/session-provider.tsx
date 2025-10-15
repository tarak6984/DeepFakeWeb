'use client'

import { SessionProvider } from 'next-auth/react'
import { ReactNode } from 'react'

interface Props {
  children: ReactNode
  session?: any
}

export function NextAuthProvider({ children, session }: Props) {
  return <SessionProvider session={session}>{children}</SessionProvider>
}