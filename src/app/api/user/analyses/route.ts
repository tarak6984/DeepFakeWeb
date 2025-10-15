import { NextRequest, NextResponse } from 'next/server'
import { getServerSession } from 'next-auth'
import { authOptions } from '@/lib/auth'
import { databaseService } from '@/lib/database-service'

export async function GET(request: NextRequest) {
  try {
    const session = await getServerSession(authOptions)

    if (!session?.user?.id) {
      return NextResponse.json(
        { message: 'Unauthorized' },
        { status: 401 }
      )
    }

    const { searchParams } = new URL(request.url)
    const limit = parseInt(searchParams.get('limit') || '50')
    const offset = parseInt(searchParams.get('offset') || '0')

    const analyses = await databaseService.getUserAnalyses(session.user.id, limit, offset)

    return NextResponse.json({
      analyses,
      pagination: {
        limit,
        offset,
        total: analyses.length // In a real app, you'd get the total count separately
      }
    })
  } catch (error) {
    console.error('Error fetching user analyses:', error)
    return NextResponse.json(
      { message: 'Internal server error' },
      { status: 500 }
    )
  }
}