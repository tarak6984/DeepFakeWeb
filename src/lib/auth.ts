import { NextAuthOptions } from "next-auth"
import { PrismaAdapter } from "@next-auth/prisma-adapter"
import CredentialsProvider from "next-auth/providers/credentials"
import GoogleProvider from "next-auth/providers/google"
import GitHubProvider from "next-auth/providers/github"
import bcrypt from "bcryptjs"
import { prisma } from "@/lib/prisma"

export const authOptions: NextAuthOptions = {
  adapter: PrismaAdapter(prisma),
  providers: [
    // Email/Password Authentication
    CredentialsProvider({
      name: "credentials",
      credentials: {
        email: { label: "Email", type: "email" },
        password: { label: "Password", type: "password" }
      },
      async authorize(credentials) {
        if (!credentials?.email || !credentials?.password) {
          throw new Error("Email and password required")
        }

        const user = await prisma.user.findUnique({
          where: { email: credentials.email }
        })

        if (!user || !user.password) {
          throw new Error("Invalid credentials")
        }

        const isPasswordValid = await bcrypt.compare(
          credentials.password,
          user.password
        )

        if (!isPasswordValid) {
          throw new Error("Invalid credentials")
        }

        return {
          id: user.id,
          email: user.email,
          name: user.name,
          image: user.image,
          role: user.role,
          plan: user.plan,
        }
      }
    }),

    // Google OAuth (optional)
    ...(process.env.GOOGLE_CLIENT_ID && process.env.GOOGLE_CLIENT_SECRET
      ? [GoogleProvider({
          clientId: process.env.GOOGLE_CLIENT_ID,
          clientSecret: process.env.GOOGLE_CLIENT_SECRET,
        })]
      : []
    ),

    // GitHub OAuth (optional)
    ...(process.env.GITHUB_CLIENT_ID && process.env.GITHUB_CLIENT_SECRET
      ? [GitHubProvider({
          clientId: process.env.GITHUB_CLIENT_ID,
          clientSecret: process.env.GITHUB_CLIENT_SECRET,
        })]
      : []
    ),
  ],

  session: {
    strategy: "jwt",
  },

  callbacks: {
    async jwt({ token, user, trigger, session }) {
      // Initial sign in
      if (user) {
        token.role = user.role
        token.plan = user.plan
        token.userId = user.id
      }

      // Update session
      if (trigger === "update" && session) {
        token.name = session.name
        token.email = session.email
      }

      return token
    },

    async session({ session, token }) {
      if (token) {
        session.user.id = token.userId as string
        session.user.role = token.role as string
        session.user.plan = token.plan as string
      }

      return session
    },

    async signIn({ user, account, profile }) {
      // For OAuth providers, create user preferences and usage tracking
      if (account?.provider !== "credentials") {
        try {
          const existingUser = await prisma.user.findUnique({
            where: { email: user.email! },
            include: { preferences: true, usage: true }
          })

          if (existingUser && !existingUser.preferences) {
            // Create default preferences
            await prisma.userPreferences.create({
              data: { userId: existingUser.id }
            })
          }

          if (existingUser && !existingUser.usage) {
            // Create usage tracking
            await prisma.usageTracking.create({
              data: { 
                userId: existingUser.id,
                currentMonth: new Date().getMonth() + 1,
                currentYear: new Date().getFullYear()
              }
            })
          }
        } catch (error) {
          console.error("Error creating user defaults:", error)
        }
      }

      return true
    },
  },

  pages: {
    signIn: "/auth/signin",
    error: "/auth/error",
  },

  events: {
    async createUser({ user }) {
      try {
        // Create default preferences for new users
        await prisma.userPreferences.create({
          data: { userId: user.id }
        })

        // Create usage tracking for new users
        await prisma.usageTracking.create({
          data: { 
            userId: user.id,
            currentMonth: new Date().getMonth() + 1,
            currentYear: new Date().getFullYear()
          }
        })
      } catch (error) {
        console.error("Error creating user defaults:", error)
      }
    },
  },
}

// Helper functions for user management
export async function createUser(email: string, password: string, name?: string) {
  const hashedPassword = await bcrypt.hash(password, 12)
  
  const user = await prisma.user.create({
    data: {
      email,
      password: hashedPassword,
      name,
    },
  })

  return user
}

export async function getUserById(id: string) {
  return await prisma.user.findUnique({
    where: { id },
    include: {
      preferences: true,
      usage: true,
      analyses: {
        orderBy: { createdAt: 'desc' },
        take: 10
      }
    }
  })
}

export async function updateUserProfile(id: string, data: { name?: string; image?: string }) {
  return await prisma.user.update({
    where: { id },
    data,
  })
}