'use client'

import { useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { AlertTriangle, RefreshCw, Home } from 'lucide-react'
import Link from 'next/link'

export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string }
  reset: () => void
}) {
  useEffect(() => {
    // Log the error to an error reporting service
    console.error('Application error:', error)
  }, [error])

  return (
    <div className="container mx-auto px-4 py-16 flex flex-col items-center justify-center min-h-[60vh] text-center">
      <div className="mb-8">
        <AlertTriangle className="w-24 h-24 text-red-500 mx-auto mb-4" />
        <h1 className="text-4xl font-bold mb-2">Something went wrong</h1>
        <p className="text-xl text-muted-foreground mb-4">
          An unexpected error occurred while processing your request.
        </p>
        {process.env.NODE_ENV === 'development' && (
          <details className="mt-4 p-4 bg-muted rounded-lg text-left max-w-2xl">
            <summary className="cursor-pointer font-semibold mb-2">
              Error Details (Development Only)
            </summary>
            <pre className="text-sm text-red-600 whitespace-pre-wrap">
              {error.message}
            </pre>
            {error.digest && (
              <p className="text-xs text-muted-foreground mt-2">
                Error ID: {error.digest}
              </p>
            )}
          </details>
        )}
      </div>

      <div className="flex flex-col sm:flex-row gap-4">
        <Button onClick={reset} variant="default">
          <RefreshCw className="w-4 h-4 mr-2" />
          Try Again
        </Button>
        
        <Button asChild variant="outline">
          <Link href="/">
            <Home className="w-4 h-4 mr-2" />
            Go Home
          </Link>
        </Button>
      </div>

      <div className="mt-12 text-sm text-muted-foreground">
        <p>
          If this error persists, please contact support with error ID:{' '}
          {error.digest || 'N/A'}
        </p>
      </div>
    </div>
  )
}