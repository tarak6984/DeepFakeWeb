import { withAuth } from "next-auth/middleware"

export default withAuth(
  function middleware(req) {
    // Add any additional middleware logic here
  },
  {
    callbacks: {
      authorized: ({ token, req }) => {
        // Check if user is authenticated
        if (!token) return false

        // Admin routes require admin role (temporarily disabled for testing)
        if (req.nextUrl.pathname.startsWith('/admin')) {
          return true // Temporarily allow all authenticated users
          // return token.role === 'ADMIN' || token.role === 'SUPER_ADMIN'
        }

        // Dashboard and protected routes require authentication
        if (req.nextUrl.pathname.startsWith('/dashboard') ||
            req.nextUrl.pathname.startsWith('/profile') ||
            req.nextUrl.pathname.startsWith('/settings')) {
          return !!token
        }

        return true
      },
    },
  }
)

export const config = {
  matcher: [
    '/dashboard/:path*',
    '/profile/:path*',
    '/settings/:path*',
    '/admin/:path*',
  ]
}