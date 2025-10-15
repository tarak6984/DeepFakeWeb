import { NextResponse } from "next/server";
import { prisma } from "@/lib/prisma";
import bcrypt from "bcryptjs";

export async function GET(req: Request) {
  try {
    const { searchParams } = new URL(req.url);
    const token = searchParams.get('token');

    if (!token) {
      return NextResponse.json(
        { error: "Reset token is required" },
        { status: 400 }
      );
    }

    // Find and validate reset token
    const resetToken = await prisma.passwordResetToken.findUnique({
      where: { token },
      include: { user: { select: { email: true } } },
    });

    if (!resetToken) {
      return NextResponse.json(
        { error: "Invalid or expired reset token" },
        { status: 400 }
      );
    }

    // Check if token has expired
    if (new Date() > resetToken.expires) {
      // Clean up expired token
      await prisma.passwordResetToken.delete({
        where: { id: resetToken.id },
      });

      return NextResponse.json(
        { error: "Reset token has expired" },
        { status: 400 }
      );
    }

    // Check if token has already been used
    if (resetToken.used) {
      return NextResponse.json(
        { error: "Reset token has already been used" },
        { status: 400 }
      );
    }

    return NextResponse.json({
      success: true,
      message: "Valid reset token",
      userEmail: resetToken.user.email,
    });

  } catch (error) {
    console.error('Reset token validation error:', error);
    return NextResponse.json(
      { error: "An error occurred while validating the reset token" },
      { status: 500 }
    );
  }
}

export async function POST(req: Request) {
  try {
    const { token, password, confirmPassword } = await req.json();

    if (!token || !password || !confirmPassword) {
      return NextResponse.json(
        { error: "Token, password, and confirm password are required" },
        { status: 400 }
      );
    }

    if (password !== confirmPassword) {
      return NextResponse.json(
        { error: "Passwords do not match" },
        { status: 400 }
      );
    }

    // Validate password strength
    if (password.length < 8) {
      return NextResponse.json(
        { error: "Password must be at least 8 characters long" },
        { status: 400 }
      );
    }

    // Password strength validation
    const passwordRegex = /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]/;
    if (!passwordRegex.test(password)) {
      return NextResponse.json(
        { 
          error: "Password must contain at least one uppercase letter, one lowercase letter, one number, and one special character" 
        },
        { status: 400 }
      );
    }

    // Find and validate reset token
    const resetToken = await prisma.passwordResetToken.findUnique({
      where: { token },
      include: { user: true },
    });

    if (!resetToken) {
      return NextResponse.json(
        { error: "Invalid or expired reset token" },
        { status: 400 }
      );
    }

    // Check if token has expired
    if (new Date() > resetToken.expires) {
      // Clean up expired token
      await prisma.passwordResetToken.delete({
        where: { id: resetToken.id },
      });

      return NextResponse.json(
        { error: "Reset token has expired" },
        { status: 400 }
      );
    }

    // Check if token has already been used
    if (resetToken.used) {
      return NextResponse.json(
        { error: "Reset token has already been used" },
        { status: 400 }
      );
    }

    // Hash the new password
    const hashedPassword = await bcrypt.hash(password, 12);

    // Update user's password and mark token as used
    await prisma.$transaction([
      prisma.user.update({
        where: { id: resetToken.userId },
        data: { password: hashedPassword },
      }),
      prisma.passwordResetToken.update({
        where: { id: resetToken.id },
        data: { used: true },
      }),
    ]);

    return NextResponse.json({
      success: true,
      message: "Password has been reset successfully. You can now sign in with your new password.",
    });

  } catch (error) {
    console.error('Reset password error:', error);
    return NextResponse.json(
      { error: "An error occurred while resetting your password. Please try again." },
      { status: 500 }
    );
  }
}