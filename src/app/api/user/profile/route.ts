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

    const profile = await databaseService.getUserProfile(session.user.id)
    
    if (!profile) {
      return NextResponse.json(
        { message: 'Profile not found' },
        { status: 404 }
      )
    }

    // Format the response
    const profileData = {
      id: profile.id,
      name: profile.name,
      email: profile.email,
      image: profile.image,
      role: profile.role,
      plan: profile.plan,
      createdAt: profile.createdAt.toISOString(),
      preferences: profile.preferences ? {
        theme: profile.preferences.theme,
        notifications: profile.preferences.enableNotifications,
        dataRetention: profile.preferences.dataRetentionDays,
        shareAnalytics: profile.preferences.shareAnalytics
      } : null,
      usage: profile.usage ? {
        totalScans: profile.usage.totalScans,
        monthlyScans: profile.usage.monthlyScans,
        monthlyLimit: profile.usage.monthlyLimit
      } : null,
      recentAnalyses: profile.analyses
    }

    return NextResponse.json(profileData)
  } catch (error) {
    console.error('Error fetching profile:', error)
    return NextResponse.json(
      { message: 'Internal server error' },
      { status: 500 }
    )
  }
}

export async function PATCH(request: NextRequest) {
  try {
    const session = await getServerSession(authOptions)

    if (!session?.user?.id) {
      return NextResponse.json(
        { message: 'Unauthorized' },
        { status: 401 }
      )
    }

    const { name, image, preferences } = await request.json()

    // Update user profile
    if (name !== undefined || image !== undefined) {
      await databaseService.updateUserProfile(session.user.id, {
        ...(name !== undefined && { name }),
        ...(image !== undefined && { image })
      })
    }

    // Update user preferences
    if (preferences) {
      await databaseService.updateUserPreferences(session.user.id, {
        ...(preferences.theme && { theme: preferences.theme.toUpperCase() }),
        ...(preferences.notifications !== undefined && { enableNotifications: preferences.notifications }),
        ...(preferences.dataRetention !== undefined && { dataRetentionDays: preferences.dataRetention }),
        ...(preferences.shareAnalytics !== undefined && { shareAnalytics: preferences.shareAnalytics })
      })
    }

    return NextResponse.json({ message: 'Profile updated successfully' })
  } catch (error) {
    console.error('Error updating profile:', error)
    return NextResponse.json(
      { message: 'Internal server error' },
      { status: 500 }
    )
  }
}