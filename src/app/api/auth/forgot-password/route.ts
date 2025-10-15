import { NextResponse } from "next/server";
import { prisma } from "@/lib/prisma";
import { emailService } from "@/lib/email-service";
import crypto from 'crypto';

export async function POST(req: Request) {
  try {
    const { email } = await req.json();

    if (!email || typeof email !== 'string') {
      return NextResponse.json(
        { error: "Email is required" },
        { status: 400 }
      );
    }

    // Validate email format
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email)) {
      return NextResponse.json(
        { error: "Invalid email format" },
        { status: 400 }
      );
    }

    // Check if user exists
    const user = await prisma.user.findUnique({
      where: { email: email.toLowerCase() },
    });

    // For security, always return success even if user doesn't exist
    // This prevents email enumeration attacks
    if (!user) {
      return NextResponse.json({
        success: true,
        message: "If an account with that email exists, we've sent a password reset link.",
      });
    }

    // Check if user has a password (not OAuth-only account)
    if (!user.password) {
      return NextResponse.json({
        success: true,
        message: "If an account with that email exists, we've sent a password reset link.",
      });
    }

    // Generate secure reset token
    const resetToken = crypto.randomBytes(32).toString('hex');
    const expiresAt = new Date(Date.now() + 60 * 60 * 1000); // 1 hour from now

    // Clean up any existing reset tokens for this user
    await prisma.passwordResetToken.deleteMany({
      where: { userId: user.id },
    });

    // Create new reset token
    await prisma.passwordResetToken.create({
      data: {
        token: resetToken,
        userId: user.id,
        expires: expiresAt,
      },
    });

    // Generate reset link
    const baseUrl = process.env.NEXTAUTH_URL || 'http://localhost:3000';
    const resetLink = `${baseUrl}/auth/reset-password?token=${resetToken}`;

    // Send email if service is configured
    if (emailService.isConfigured()) {
      const emailSent = await emailService.sendPasswordResetEmail({
        to: user.email,
        userName: user.name || user.email.split('@')[0],
        resetLink,
      });

      if (!emailSent) {
        console.error('Failed to send password reset email to:', user.email);
        return NextResponse.json(
          { error: "Failed to send reset email. Please try again later." },
          { status: 500 }
        );
      }
    } else {
      // For development/testing when email is not configured
      console.log('📧 Password reset email would be sent to:', user.email);
      console.log('🔗 Reset link:', resetLink);
      
      return NextResponse.json({
        success: true,
        message: "Password reset link generated successfully.",
        // Include reset link in development mode for testing
        ...(process.env.NODE_ENV === 'development' && { 
          resetLink,
          devNote: "Email service not configured. Use the resetLink above for testing." 
        }),
      });
    }

    return NextResponse.json({
      success: true,
      message: "If an account with that email exists, we've sent a password reset link.",
    });

  } catch (error) {
    console.error('Forgot password error:', error);
    return NextResponse.json(
      { error: "An error occurred while processing your request. Please try again later." },
      { status: 500 }
    );
  }
}