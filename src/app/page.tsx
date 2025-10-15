'use client';

import { useEffect, useState } from 'react';
import Link from "next/link";
import { useRouter } from 'next/navigation';
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Input } from '@/components/ui/input';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';
import { Label } from '@/components/ui/label';
import { useToast } from '@/hooks/use-toast';
import {
  Shield,
  Zap,
  Eye,
  Upload,
  Brain,
  Lock,
  FileImage,
  FilePlay,
  FileAudio,
  ArrowRight,
  Github,
  Star,
  ShieldCheck,
  EyeOff,
} from "lucide-react";
import Image from 'next/image';
import { motion } from "framer-motion";
import { UsageLimitBanner } from '@/components/usage/usage-limit-banner';

function HomePage() {
  const [adminDialogOpen, setAdminDialogOpen] = useState(false);
  const [adminPassword, setAdminPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [isVerifying, setIsVerifying] = useState(false);
  const [displayedText, setDisplayedText] = useState('');
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isTyping, setIsTyping] = useState(true);
  const [showCursor, setShowCursor] = useState(true);
  const [hasStarted, setHasStarted] = useState(false);
  const { toast } = useToast();
  const router = useRouter();

  const fullText = "Upload Media. Detect Deepfakes.";

  // Initial startup delay
  useEffect(() => {
    const startDelay = setTimeout(() => {
      setHasStarted(true);
    }, 500); // Wait 500ms before starting animation
    
    return () => clearTimeout(startDelay);
  }, []);

  // Looping typing animation effect
  useEffect(() => {
    if (!hasStarted) return;
    
    const typeText = () => {
      if (isTyping) {
        if (currentIndex < fullText.length) {
          // Typing forward
          setDisplayedText(prev => prev + fullText[currentIndex]);
          setCurrentIndex(prev => prev + 1);
        } else {
          // Pause before erasing
          setTimeout(() => {
            setIsTyping(false);
          }, 2000); // Wait 2 seconds before erasing
        }
      } else {
        if (currentIndex > 0) {
          // Erasing backward
          setDisplayedText(prev => prev.slice(0, -1));
          setCurrentIndex(prev => prev - 1);
        } else {
          // Pause before typing again
          setTimeout(() => {
            setIsTyping(true);
          }, 500); // Wait 0.5 seconds before typing again
        }
      }
    };

    const timeout = setTimeout(typeText, isTyping ? 100 : 50); // Typing: 100ms, Erasing: 50ms
    return () => clearTimeout(timeout);
  }, [currentIndex, isTyping, fullText, hasStarted]);

  // Cursor blink animation
  useEffect(() => {
    const cursorBlink = setInterval(() => {
      setShowCursor(prev => !prev);
    }, 500);
    return () => clearInterval(cursorBlink);
  }, []);

  const handleAdminAccess = async () => {
    if (adminPassword !== '101010') {
      toast({
        title: 'Access Denied',
        description: 'Invalid admin password',
        variant: 'destructive',
      });
      return;
    }

    setIsVerifying(true);
    
    setTimeout(() => {
      toast({
        title: 'Admin Access Granted',
        description: 'Redirecting to admin dashboard...',
      });
      
      router.push('/admin/dashboard');
      
      setAdminDialogOpen(false);
      setAdminPassword('');
      setIsVerifying(false);
    }, 1000);
  };

  const currentYear = new Date().getFullYear();

  const features = [
    {
      icon: Brain,
      title: "AI-Powered Detection",
      description: "Advanced machine learning algorithms analyze media for manipulation patterns and deepfake signatures.",
    },
    {
      icon: Zap,
      title: "Lightning Fast",
      description: "Get results in seconds with our optimized processing pipeline and cloud-based analysis.",
    },
    {
      icon: Lock,
      title: "Secure & Private",
      description: "Your files are processed securely and never stored permanently. Complete privacy guaranteed.",
    },
    {
      icon: Eye,
      title: "Detailed Analysis",
      description: "Comprehensive reports with confidence scores, frame-by-frame analysis, and visual breakdowns.",
    },
  ];

  const supportedFormats = [
    { icon: FileImage, name: "Images", formats: "JPG, PNG, GIF, WebP" },
    { icon: FilePlay, name: "Videos", formats: "MP4, WebM, MOV, AVI" },
    { icon: FileAudio, name: "Audio", formats: "MP3, WAV, FLAC, AAC" },
  ];


  return (
    <div className="min-h-screen gradient-bg">
      {/* Hero Section */}
      <section className="relative overflow-hidden py-20 sm:py-32">
        <div className="absolute inset-0 bg-grid-white/[0.02] bg-[size:60px_60px]" />
        <div className="relative container mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center max-w-7xl mx-auto">
            {/* Left Column - Content */}
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.8 }}
              className="text-center lg:text-left"
            >
              <div className="flex items-center justify-center lg:justify-start gap-2 mb-6">
                <Badge variant="outline" className="px-3 py-1">
                  <Star className="w-3 h-3 mr-1 fill-current" />
                  AI-Powered Detection
                </Badge>
              </div>
              
              <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold tracking-tight mb-6 min-h-[1.2em]">
                {hasStarted ? (
                  <span className="inline-block">
                    {displayedText.split(' ').map((word, index) => {
                      if (word === 'Detect' || word === 'Deepfakes.') {
                        return (
                          <span key={index} className="text-transparent bg-clip-text bg-gradient-to-r from-blue-600 to-purple-600">
                            {word}{index < displayedText.split(' ').length - 1 ? ' ' : ''}
                          </span>
                        );
                      }
                      return <span key={index}>{word}{index < displayedText.split(' ').length - 1 ? ' ' : ''}</span>;
                    })}
                    <span className={`inline-block w-1 h-[1em] bg-current ml-1 ${showCursor ? 'opacity-100' : 'opacity-0'} transition-opacity duration-100`}>|</span>
                  </span>
                ) : (
                  <span>
                    Upload Media.{" "}
                    <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-600 to-purple-600">
                      Detect Deepfakes.
                    </span>
                  </span>
                )}
              </h1>
              
              <p className="text-lg sm:text-xl text-muted-foreground mb-8 max-w-2xl mx-auto lg:mx-0">
                Advanced AI-powered deepfake detection for images, videos, and audio files. 
                100% local processing, unlimited usage, and enterprise-grade accuracy in seconds.
              </p>
              
              <div className="flex flex-col sm:flex-row gap-4 justify-center lg:justify-start items-center mb-8">
                <Link href="/fast-upload" prefetch={true}>
                  <Button size="lg" className="font-semibold px-8 py-6 text-base">
                    <Upload className="mr-2 h-5 w-5" />
                    Try Detection
                    <ArrowRight className="ml-2 h-4 w-4" />
                  </Button>
                </Link>
              </div>
            </motion.div>

            {/* Right Column - Hero Image */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.8, delay: 0.2 }}
              className="relative lg:h-[500px] flex items-center justify-center"
            >
              <div className="relative w-full max-w-md mx-auto">
                <div className="absolute inset-0 bg-gradient-to-r from-blue-500/20 to-purple-500/20 rounded-2xl blur-3xl" />
                <div className="relative overflow-hidden rounded-2xl shadow-2xl ring-1 ring-gray-200 dark:ring-gray-800">
                  <Image
                    src="/hero-image.jpg"
                    alt="AI-powered deepfake detection visualization"
                    width={400}
                    height={500}
                    className="w-full h-auto object-cover"
                    priority
                  />
                  <div className="absolute inset-0 bg-gradient-to-t from-background/10 to-transparent" />
                </div>
              </div>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Usage Limit Banner */}
      <section className="py-8">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8">
          <UsageLimitBanner className="max-w-4xl mx-auto" />
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20 bg-muted/30">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
            className="text-center mb-16"
          >
            <h2 className="text-3xl sm:text-4xl font-bold mb-4">
              Why Choose ITL Deepfake Detective?
            </h2>
            <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
              Built with cutting-edge AI technology to provide the most accurate and reliable deepfake detection available.
            </p>
          </motion.div>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
            {features.map((feature, index) => (
              <motion.div
                key={feature.title}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
              >
                <Card className="h-full hover:shadow-lg transition-shadow">
                  <CardHeader>
                    <feature.icon className="h-10 w-10 text-primary mb-4" />
                    <CardTitle className="text-lg">{feature.title}</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <CardDescription className="text-sm leading-relaxed">
                      {feature.description}
                    </CardDescription>
                  </CardContent>
                </Card>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Supported Formats */}
      <section className="py-20">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
            className="text-center mb-16"
          >
            <h2 className="text-3xl sm:text-4xl font-bold mb-4">
              Supports All Media Types
            </h2>
            <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
              Comprehensive analysis for images, videos, and audio files with detailed insights and frame-by-frame detection.
            </p>
          </motion.div>
          
          <div className="grid md:grid-cols-3 gap-8 max-w-4xl mx-auto">
            {supportedFormats.map((format, index) => (
              <motion.div
                key={format.name}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
              >
                <Card className="text-center hover:shadow-lg transition-shadow">
                  <CardHeader>
                    <format.icon className="h-16 w-16 text-primary mx-auto mb-4" />
                    <CardTitle className="text-xl">{format.name}</CardTitle>
                    <CardDescription className="text-sm">
                      {format.formats}
                    </CardDescription>
                  </CardHeader>
                </Card>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 bg-muted/30">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
            className="text-center max-w-3xl mx-auto"
          >
            <Shield className="h-16 w-16 text-primary mx-auto mb-6" />
            <h2 className="text-3xl sm:text-4xl font-bold mb-4">
              Ready to Verify Your Media?
            </h2>
            <p className="text-xl text-muted-foreground mb-8">
              Start detecting deepfakes with our advanced local AI models. 
              Upload your file and get unlimited results in seconds - all processed locally!
            </p>
            <Link href="/fast-upload" prefetch={true}>
              <Button size="lg" className="font-semibold px-8 py-6 text-base">
                <Upload className="mr-2 h-5 w-5" />
                Start Free Analysis
                <ArrowRight className="ml-2 h-4 w-4" />
              </Button>
            </Link>
          </motion.div>
        </div>
      </section>
      
      {/* Simple Footer with Admin Access */}
      <footer className="bg-background border-t">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="flex flex-col md:flex-row justify-between items-center space-y-4 md:space-y-0">
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <Shield className="h-5 w-5 text-primary" />
                <span className="font-semibold">ITL Deepfake Detective</span>
              </div>
              <p className="text-sm text-muted-foreground">
                © {currentYear} All rights reserved.
              </p>
            </div>

            <div className="flex items-center space-x-2">
              <Dialog open={adminDialogOpen} onOpenChange={setAdminDialogOpen}>
                <DialogTrigger asChild>
                  <Button 
                    variant="ghost" 
                    size="sm" 
                    className="text-sm font-bold text-muted-foreground hover:text-foreground transition-colors"
                  >
                    <Lock className="h-4 w-4 mr-1" />
                    System
                  </Button>
                </DialogTrigger>
                <DialogContent className="sm:max-w-md">
                  <DialogHeader>
                    <DialogTitle className="flex items-center">
                      <ShieldCheck className="h-5 w-5 mr-2 text-primary" />
                      Admin Access
                    </DialogTitle>
                    <DialogDescription>
                      Enter the admin password to access the administration panel.
                    </DialogDescription>
                  </DialogHeader>
                  <div className="space-y-4">
                    <div className="space-y-2">
                      <Label htmlFor="admin-password">Admin Password</Label>
                      <div className="relative">
                        <Input
                          id="admin-password"
                          type={showPassword ? "text" : "password"}
                          placeholder="Enter admin password"
                          value={adminPassword}
                          onChange={(e) => setAdminPassword(e.target.value)}
                          onKeyDown={(e) => {
                            if (e.key === 'Enter') {
                              handleAdminAccess();
                            }
                          }}
                          className="pr-10"
                        />
                        <Button
                          type="button"
                          variant="ghost"
                          size="sm"
                          className="absolute right-0 top-0 h-full px-3 py-2 hover:bg-transparent"
                          onClick={() => setShowPassword(!showPassword)}
                        >
                          {showPassword ? (
                            <EyeOff className="h-4 w-4" />
                          ) : (
                            <Eye className="h-4 w-4" />
                          )}
                        </Button>
                      </div>
                    </div>
                    <div className="text-xs text-muted-foreground p-3 bg-muted/50 rounded-md">
                      <p className="font-medium mb-1">Admin Panel Features:</p>
                      <ul className="space-y-1">
                        <li>• System Dashboard & Metrics</li>
                        <li>• User Management</li>
                        <li>• Analysis Statistics</li>
                        <li>• System Maintenance</li>
                      </ul>
                    </div>
                  </div>
                  <DialogFooter>
                    <Button
                      variant="outline"
                      onClick={() => {
                        setAdminDialogOpen(false);
                        setAdminPassword('');
                      }}
                    >
                      Cancel
                    </Button>
                    <Button 
                      onClick={handleAdminAccess}
                      disabled={isVerifying || !adminPassword}
                    >
                      {isVerifying ? (
                        <motion.div
                          animate={{ rotate: 360 }}
                          transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                          className="mr-2"
                        >
                          <ShieldCheck className="h-4 w-4" />
                        </motion.div>
                      ) : (
                        <ShieldCheck className="h-4 w-4 mr-2" />
                      )}
                      {isVerifying ? 'Verifying...' : 'Access Admin'}
                    </Button>
                  </DialogFooter>
                </DialogContent>
              </Dialog>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default HomePage;
