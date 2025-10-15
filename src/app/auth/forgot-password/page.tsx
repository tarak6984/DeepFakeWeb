'use client';

import { useState } from 'react';
import Link from 'next/link';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Loader2, Mail, ArrowLeft, Shield, CheckCircle, AlertTriangle } from 'lucide-react';
import { motion } from 'framer-motion';

export default function ForgotPasswordPage() {
  const [email, setEmail] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isSubmitted, setIsSubmitted] = useState(false);
  const [error, setError] = useState('');
  const [devResetLink, setDevResetLink] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError('');

    try {
      const response = await fetch('/api/auth/forgot-password', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ email }),
      });

      const data = await response.json();

      if (response.ok) {
        setIsSubmitted(true);
        
        // In development mode, show the reset link for testing
        if (data.resetLink && process.env.NODE_ENV === 'development') {
          setDevResetLink(data.resetLink);
        }
      } else {
        setError(data.error || 'An error occurred. Please try again.');
      }
    } catch (error) {
      setError('Network error. Please check your connection and try again.');
    } finally {
      setIsLoading(false);
    }
  };

  if (isSubmitted) {
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
                <CardTitle className="text-2xl font-bold">Check Your Email</CardTitle>
                <p className="text-muted-foreground mt-2">
                  We've sent password reset instructions to your email address.
                </p>
              </div>
            </CardHeader>
            
            <CardContent className="space-y-6">
              <div className="text-center space-y-4">
                <p className="text-sm text-muted-foreground">
                  If an account with <strong>{email}</strong> exists, you'll receive an email with a link to reset your password.
                </p>
                
                <div className="bg-muted/30 p-4 rounded-lg space-y-2">
                  <h4 className="font-medium text-sm">What's next?</h4>
                  <ul className="text-xs text-muted-foreground space-y-1 text-left">
                    <li>• Check your email inbox (and spam folder)</li>
                    <li>• Click the reset link in the email</li>
                    <li>• The link expires in 1 hour for security</li>
                    <li>• Create a new secure password</li>
                  </ul>
                </div>

                {devResetLink && (
                  <Alert className="border-blue-200 bg-blue-50 dark:bg-blue-900/20">
                    <AlertTriangle className="h-4 w-4 text-blue-600 dark:text-blue-400" />
                    <AlertDescription className="text-left">
                      <strong>Development Mode:</strong><br />
                      Email service not configured. Use this link for testing:<br />
                      <a 
                        href={devResetLink}
                        className="text-blue-600 hover:underline break-all font-mono text-xs"
                        target="_blank"
                        rel="noopener noreferrer"
                      >
                        {devResetLink}
                      </a>
                    </AlertDescription>
                  </Alert>
                )}
              </div>

              <div className="space-y-4">
                <Button asChild className="w-full">
                  <Link href="/auth/signin">
                    <ArrowLeft className="w-4 h-4 mr-2" />
                    Back to Sign In
                  </Link>
                </Button>
                
                <Button 
                  variant="outline" 
                  className="w-full"
                  onClick={() => {
                    setIsSubmitted(false);
                    setEmail('');
                    setError('');
                    setDevResetLink('');
                  }}
                >
                  Try Different Email
                </Button>
              </div>

              <div className="text-center text-xs text-muted-foreground">
                Didn't receive an email? Check your spam folder or try again in a few minutes.
              </div>
            </CardContent>
          </Card>
        </motion.div>
      </div>
    );
  }

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
              <CardTitle className="text-2xl font-bold">Forgot Password?</CardTitle>
              <p className="text-muted-foreground mt-2">
                No worries! Enter your email and we'll send you a reset link.
              </p>
            </div>
          </CardHeader>
          
          <CardContent className="space-y-6">
            {error && (
              <Alert variant="destructive">
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}

            <form onSubmit={handleSubmit} className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="email" className="flex items-center gap-2">
                  <Mail className="w-4 h-4" />
                  Email Address
                </Label>
                <Input
                  id="email"
                  type="email"
                  placeholder="Enter your email address"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  required
                  className="h-12"
                  disabled={isLoading}
                />
              </div>

              <Button 
                type="submit" 
                className="w-full h-12 text-base font-medium"
                disabled={isLoading}
              >
                {isLoading ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Sending Reset Link...
                  </>
                ) : (
                  <>
                    <Mail className="mr-2 h-4 w-4" />
                    Send Reset Link
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

            <div className="pt-4 border-t">
              <div className="text-xs text-muted-foreground space-y-2">
                <p className="font-medium text-center mb-3">Security Information:</p>
                <ul className="space-y-1 text-center">
                  <li>• Reset links expire in 1 hour</li>
                  <li>• Links can only be used once</li>
                  <li>• We'll never ask for your password via email</li>
                </ul>
              </div>
            </div>
          </CardContent>
        </Card>
      </motion.div>
    </div>
  );
}