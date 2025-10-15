import Link from 'next/link'
import { Button } from '@/components/ui/button'
import { FileX, Home, ArrowLeft } from 'lucide-react'

export default function NotFound() {
  return (
    <div className="container mx-auto px-4 py-16 flex flex-col items-center justify-center min-h-[60vh] text-center">
      <div className="mb-8">
        <FileX className="w-24 h-24 text-muted-foreground mx-auto mb-4" />
        <h1 className="text-4xl font-bold mb-2">Page Not Found</h1>
        <p className="text-xl text-muted-foreground mb-8">
          The page you're looking for doesn't exist or has been moved.
        </p>
      </div>

      <div className="flex flex-col sm:flex-row gap-4">
        <Button asChild variant="default">
          <Link href="/">
            <Home className="w-4 h-4 mr-2" />
            Go Home
          </Link>
        </Button>
        
        <Button asChild variant="outline">
          <Link href="/dashboard">
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back to Dashboard
          </Link>
        </Button>
      </div>

      <div className="mt-12 text-sm text-muted-foreground">
        <p>If you think this is a mistake, please contact support.</p>
      </div>
    </div>
  )
}