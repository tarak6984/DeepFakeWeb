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

    const [usage, limits] = await Promise.all([
      databaseService.getUserUsage(session.user.id),
      databaseService.checkUsageLimit(session.user.id)
    ])

    return NextResponse.json({
      monthlyScans: usage?.monthlyScans || 0,
      totalScans: usage?.totalScans || 0,
      monthlyLimit: limits.limit,
      remaining: limits.remaining,
      allowed: limits.allowed
    })
  } catch (error) {
    console.error('Error fetching user usage:', error)
    return NextResponse.json(
      { message: 'Internal server error' },
      { status: 500 }
    )
  }
}