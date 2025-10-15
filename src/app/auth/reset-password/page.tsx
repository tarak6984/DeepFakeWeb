'use client';

import { useState, useEffect, Suspense } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import Link from 'next/link';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Progress } from '@/components/ui/progress';
import { 
  Loader2, 
  Key, 
  Eye, 
  EyeOff, 
  CheckCircle, 
  XCircle,
  AlertTriangle,
  Shield,
  ArrowLeft
} from 'lucide-react';
import { motion } from 'framer-motion';

interface PasswordValidation {
  minLength: boolean;
  hasUppercase: boolean;
  hasLowercase: boolean;
  hasNumber: boolean;
  hasSpecialChar: boolean;
}

function ResetPasswordContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const token = searchParams?.get('token');

  const [isValidating, setIsValidating] = useState(true);
  const [tokenValid, setTokenValid] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState(false);
  
  // Form state
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);

  // Password validation
  const [validation, setValidation] = useState<PasswordValidation>({
    minLength: false,
    hasUppercase: false,
    hasLowercase: false,
    hasNumber: false,
    hasSpecialChar: false,
  });

  // Validate token on mount
  useEffect(() => {
    const validateToken = async () => {
      if (!token) {
        setError('Invalid or missing reset token');
        setIsValidating(false);
        return;
      }

      try {
        const response = await fetch(`/api/auth/reset-password?token=${encodeURIComponent(token)}`);
        const data = await response.json();

        if (response.ok) {
          setTokenValid(true);
        } else {
          setError(data.error || 'Invalid or expired reset token');
        }
      } catch (error) {
        setError('Network error. Please check your connection and try again.');
      } finally {
        setIsValidating(false);
      }
    };

    validateToken();
  }, [token]);

  // Validate password as user types
  useEffect(() => {
    setValidation({
      minLength: password.length >= 8,
      hasUppercase: /[A-Z]/.test(password),
      hasLowercase: /[a-z]/.test(password),
      hasNumber: /[0-9]/.test(password),
      hasSpecialChar: /[^A-Za-z0-9]/.test(password),
    });
  }, [password]);

  const getPasswordStrength = () => {
    const score = Object.values(validation).filter(Boolean).length;
    return Math.round((score / 5) * 100);
  };

  const getStrengthColor = () => {
    const strength = getPasswordStrength();
    if (strength <= 40) return 'bg-red-500';
    if (strength <= 60) return 'bg-yellow-500';
    if (strength <= 80) return 'bg-orange-500';
    return 'bg-green-500';
  };

  const getStrengthText = () => {
    const strength = getPasswordStrength();
    if (strength <= 40) return 'Weak';
    if (strength <= 60) return 'Fair';
    if (strength <= 80) return 'Good';
    return 'Strong';
  };

  const isPasswordValid = Object.values(validation).every(Boolean);
  const passwordsMatch = password === confirmPassword && confirmPassword.length > 0;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!isPasswordValid) {
      setError('Please ensure your password meets all requirements');
      return;
    }

    if (!passwordsMatch) {
      setError('Passwords do not match');
      return;
    }

    setIsSubmitting(true);
    setError('');

    try {
      const response = await fetch('/api/auth/reset-password', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          token,
          password,
          confirmPassword,
        }),
      });

      const data = await response.json();

      if (response.ok) {
        setSuccess(true);
      } else {
        setError(data.error || 'An error occurred. Please try again.');
      }
    } catch (error) {
      setError('Network error. Please check your connection and try again.');
    } finally {
      setIsSubmitting(false);
    }
  };

  // Loading state
  if (isValidating) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-background via-background to-muted/20 p-4">
        <Card className="w-full max-w-md shadow-2xl border-0 bg-card/80 backdrop-blur-sm">
          <CardContent className="flex flex-col items-center justify-center py-12">
            <Loader2 className="w-8 h-8 animate-spin text-primary mb-4" />
            <p className="text-muted-foreground">Validating reset token...</p>
          </CardContent>
        </Card>
      </div>
    );
  }

  // Success state
  if (success) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-background via-background to-muted/20 p-4">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="w-full max-w-md"
        >
          <Card className="shadow-2xl border-0 bg-card/80 backdrop-blur-sm">
            <CardHeader className="space-y-4 text-center pb-8">
              <div className="mx-auto w-20 h-20 bg-green-100 dark:bg-green-900/30 rounded-2xl flex items-center justify-center">
                <CheckCircle className="w-10 h-10 text-green-600 dark:text-green-400" />
              </div>
              <div>
                <CardTitle className="text-2xl font-bold">Password Reset Successful</CardTitle>
                <p className="text-muted-foreground mt-2">
                  Your password has been updated successfully.
                </p>
              </div>
            </CardHeader>
            
            <CardContent className="space-y-6">
              <div className="text-center space-y-4">
                <p className="text-sm text-muted-foreground">
                  You can now sign in with your new password.
                </p>
                
                <div className="bg-muted/30 p-4 rounded-lg">
                  <h4 className="font-medium text-sm mb-2">Security Tips:</h4>
                  <ul className="text-xs text-muted-foreground space-y-1 text-left">
                    <li>• Use a unique password for each account</li>
                    <li>• Consider using a password manager</li>
                    <li>• Enable two-factor authentication when available</li>
                  </ul>
                </div>
              </div>

              <Button 
                asChild 
                className="w-full h-12"
                onClick={() => router.push('/auth/signin')}
              >
                <Link href="/auth/signin">
                  Sign In Now
                </Link>
              </Button>
            </CardContent>
          </Card>
        </motion.div>
      </div>
    );
  }

  // Error state or invalid token
  if (!tokenValid) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-background via-background to-muted/20 p-4">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="w-full max-w-md"
        >
          <Card className="shadow-2xl border-0 bg-card/80 backdrop-blur-sm">
            <CardHeader className="space-y-4 text-center pb-8">
              <div className="mx-auto w-20 h-20 bg-red-100 dark:bg-red-900/30 rounded-2xl flex items-center justify-center">
                <XCircle className="w-10 h-10 text-red-600 dark:text-red-400" />
              </div>
              <div>
                <CardTitle className="text-2xl font-bold">Invalid Reset Link</CardTitle>
                <p className="text-muted-foreground mt-2">
                  This password reset link is invalid or has expired.
                </p>
              </div>
            </CardHeader>
            
            <CardContent className="space-y-6">
              {error && (
                <Alert variant="destructive">
                  <AlertTriangle className="h-4 w-4" />
                  <AlertDescription>{error}</AlertDescription>
                </Alert>
              )}

              <div className="text-center space-y-4">
                <p className="text-sm text-muted-foreground">
                  Password reset links expire after 1 hour and can only be used once.
                </p>
              </div>

              <div className="space-y-4">
                <Button asChild className="w-full">
                  <Link href="/auth/forgot-password">
                    Request New Reset Link
                  </Link>
                </Button>
                
                <Button asChild variant="outline" className="w-full">
                  <Link href="/auth/signin">
                    <ArrowLeft className="w-4 h-4 mr-2" />
                    Back to Sign In
                  </Link>
                </Button>
              </div>
            </CardContent>
          </Card>
        </motion.div>
      </div>
    );
  }

  // Main form
  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-background via-background to-muted/20 p-4">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="w-full max-w-md"
      >
        <Card className="shadow-2xl border-0 bg-card/80 backdrop-blur-sm">
          <CardHeader className="space-y-4 text-center pb-8">
            <div className="mx-auto w-20 h-20 bg-primary/10 rounded-2xl flex items-center justify-center">
              <Shield className="w-10 h-10 text-primary" />
            </div>
            <div>
              <CardTitle className="text-2xl font-bold">Reset Your Password</CardTitle>
              <p className="text-muted-foreground mt-2">
                Choose a strong, unique password for your account.
              </p>
            </div>
          </CardHeader>
          
          <CardContent className="space-y-6">
            {error && (
              <Alert variant="destructive">
                <AlertTriangle className="h-4 w-4" />
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}

            <form onSubmit={handleSubmit} className="space-y-6">
              {/* New Password */}
              <div className="space-y-2">
                <Label htmlFor="password" className="flex items-center gap-2">
                  <Key className="w-4 h-4" />
                  New Password
                </Label>
                <div className="relative">
                  <Input
                    id="password"
                    type={showPassword ? 'text' : 'password'}
                    placeholder="Enter your new password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    required
                    className="h-12 pr-12"
                    disabled={isSubmitting}
                  />
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    className="absolute right-2 top-1/2 -translate-y-1/2 h-8 w-8 p-0"
                    onClick={() => setShowPassword(!showPassword)}
                  >
                    {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                  </Button>
                </div>
              </div>

              {/* Password Strength */}
              {password.length > 0 && (
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="text-muted-foreground">Password Strength</span>
                    <span className={`font-medium ${
                      getPasswordStrength() <= 40 ? 'text-red-500' :
                      getPasswordStrength() <= 60 ? 'text-yellow-500' :
                      getPasswordStrength() <= 80 ? 'text-orange-500' :
                      'text-green-500'
                    }`}>
                      {getStrengthText()}
                    </span>
                  </div>
                  <Progress value={getPasswordStrength()} className="h-2">
                    <div 
                      className={`h-full rounded-full transition-all ${getStrengthColor()}`}
                      style={{ width: `${getPasswordStrength()}%` }}
                    />
                  </Progress>
                </div>
              )}

              {/* Password Requirements */}
              {password.length > 0 && (
                <div className="space-y-2">
                  <h4 className="text-sm font-medium">Password Requirements:</h4>
                  <div className="grid grid-cols-1 gap-1">
                    {Object.entries(validation).map(([key, isValid]) => {
                      const labels = {
                        minLength: 'At least 8 characters',
                        hasUppercase: 'One uppercase letter',
                        hasLowercase: 'One lowercase letter',
                        hasNumber: 'One number',
                        hasSpecialChar: 'One special character',
                      };
                      
                      return (
                        <div key={key} className="flex items-center gap-2 text-xs">
                          {isValid ? (
                            <CheckCircle className="w-3 h-3 text-green-500" />
                          ) : (
                            <XCircle className="w-3 h-3 text-red-500" />
                          )}
                          <span className={isValid ? 'text-green-600' : 'text-muted-foreground'}>
                            {labels[key as keyof typeof labels]}
                          </span>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}

              {/* Confirm Password */}
              <div className="space-y-2">
                <Label htmlFor="confirmPassword">Confirm New Password</Label>
                <div className="relative">
                  <Input
                    id="confirmPassword"
                    type={showConfirmPassword ? 'text' : 'password'}
                    placeholder="Confirm your new password"
                    value={confirmPassword}
                    onChange={(e) => setConfirmPassword(e.target.value)}
                    required
                    className="h-12 pr-12"
                    disabled={isSubmitting}
                  />
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    className="absolute right-2 top-1/2 -translate-y-1/2 h-8 w-8 p-0"
                    onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                  >
                    {showConfirmPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                  </Button>
                </div>
                
                {confirmPassword.length > 0 && (
                  <div className="flex items-center gap-2 text-xs">
                    {passwordsMatch ? (
                      <>
                        <CheckCircle className="w-3 h-3 text-green-500" />
                        <span className="text-green-600">Passwords match</span>
                      </>
                    ) : (
                      <>
                        <XCircle className="w-3 h-3 text-red-500" />
                        <span className="text-red-500">Passwords don't match</span>
                      </>
                    )}
                  </div>
                )}
              </div>

              <Button 
                type="submit" 
                className="w-full h-12 text-base font-medium"
                disabled={!isPasswordValid || !passwordsMatch || isSubmitting}
              >
                {isSubmitting ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Updating Password...
                  </>
                ) : (
                  <>
                    <Key className="mr-2 h-4 w-4" />
                    Update Password
                  </>
                )}
              </Button>
            </form>

            <div className="text-center">
              <Link
                href="/auth/signin"
                className="text-sm font-medium text-primary hover:underline flex items-center justify-center gap-2"
              >
                <ArrowLeft className="w-4 h-4" />
                Back to Sign In
              </Link>
            </div>
          </CardContent>
        </Card>
      </motion.div>
    </div>
  );
}

export default function ResetPasswordPage() {
  return (
    <Suspense fallback={
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-background via-background to-muted/20 p-4">
        <Card className="w-full max-w-md shadow-2xl border-0 bg-card/80 backdrop-blur-sm">
          <CardContent className="flex flex-col items-center justify-center py-12">
            <Loader2 className="w-8 h-8 animate-spin text-primary mb-4" />
            <p className="text-muted-foreground">Loading...</p>
          </CardContent>
        </Card>
      </div>
    }>
      <ResetPasswordContent />
    </Suspense>
  );
}
