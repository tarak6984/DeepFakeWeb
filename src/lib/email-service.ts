import nodemailer from 'nodemailer';

export interface EmailOptions {
  to: string;
  subject: string;
  html: string;
  text?: string;
}

export interface ResetPasswordEmailOptions {
  to: string;
  userName: string;
  resetLink: string;
}

class EmailService {
  private transporter: nodemailer.Transporter | null = null;

  constructor() {
    this.initializeTransporter();
  }

  private initializeTransporter() {
    // Check if email configuration is available
    const smtpHost = process.env.SMTP_HOST;
    const smtpPort = process.env.SMTP_PORT;
    const smtpUser = process.env.SMTP_USER;
    const smtpPassword = process.env.SMTP_PASSWORD;

    if (!smtpHost || !smtpPort || !smtpUser || !smtpPassword) {
      console.warn('⚠️  Email service not configured. Set SMTP_* environment variables to enable email functionality.');
      return;
    }

    try {
      this.transporter = nodemailer.createTransport({
        host: smtpHost,
        port: parseInt(smtpPort),
        secure: parseInt(smtpPort) === 465, // true for 465, false for other ports
        auth: {
          user: smtpUser,
          pass: smtpPassword,
        },
      });

      console.log('✅ Email service initialized successfully');
    } catch (error) {
      console.error('❌ Failed to initialize email service:', error);
    }
  }

  async sendEmail(options: EmailOptions): Promise<boolean> {
    if (!this.transporter) {
      console.error('❌ Email service not initialized. Cannot send email.');
      return false;
    }

    try {
      const info = await this.transporter.sendMail({
        from: process.env.EMAIL_FROM || process.env.SMTP_USER,
        to: options.to,
        subject: options.subject,
        text: options.text,
        html: options.html,
      });

      console.log('✅ Email sent successfully:', info.messageId);
      return true;
    } catch (error) {
      console.error('❌ Failed to send email:', error);
      return false;
    }
  }

  async sendPasswordResetEmail(options: ResetPasswordEmailOptions): Promise<boolean> {
    const appName = process.env.NEXT_PUBLIC_APP_NAME || 'ITL Deepfake Detective';
    const appUrl = process.env.NEXTAUTH_URL || 'http://localhost:3000';

    const html = `
      <!DOCTYPE html>
      <html lang="en">
      <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Reset Your Password</title>
        <style>
          body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
            line-height: 1.6; 
            color: #333; 
            max-width: 600px; 
            margin: 0 auto; 
            padding: 20px;
            background-color: #f8fafc;
          }
          .container { 
            background: white; 
            padding: 40px; 
            border-radius: 12px; 
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
          }
          .header { 
            text-align: center; 
            margin-bottom: 30px; 
          }
          .logo { 
            font-size: 24px; 
            font-weight: bold; 
            color: #2563eb; 
            margin-bottom: 10px;
          }
          .button { 
            display: inline-block; 
            background: #2563eb; 
            color: white; 
            padding: 14px 28px; 
            text-decoration: none; 
            border-radius: 8px; 
            font-weight: 600;
            margin: 20px 0;
          }
          .button:hover { 
            background: #1d4ed8; 
          }
          .footer { 
            margin-top: 30px; 
            padding-top: 20px; 
            border-top: 1px solid #e5e7eb; 
            font-size: 14px; 
            color: #6b7280; 
            text-align: center;
          }
          .warning {
            background: #fef3c7;
            border: 1px solid #f59e0b;
            padding: 12px;
            border-radius: 6px;
            margin: 20px 0;
            color: #92400e;
          }
        </style>
      </head>
      <body>
        <div class="container">
          <div class="header">
            <div class="logo">${appName}</div>
            <h1>Reset Your Password</h1>
          </div>
          
          <p>Hello ${options.userName || 'there'},</p>
          
          <p>We received a request to reset the password for your ${appName} account. If you made this request, click the button below to reset your password:</p>
          
          <div style="text-align: center; margin: 30px 0;">
            <a href="${options.resetLink}" class="button">Reset Password</a>
          </div>
          
          <div class="warning">
            <strong>Security Notice:</strong> This link will expire in 1 hour for your security. If you didn't request this password reset, you can safely ignore this email.
          </div>
          
          <p>If the button doesn't work, you can copy and paste this link into your browser:</p>
          <p style="word-break: break-all; color: #2563eb;">${options.resetLink}</p>
          
          <div class="footer">
            <p>This email was sent by ${appName}</p>
            <p>If you have questions, contact our support team.</p>
            <p><a href="${appUrl}" style="color: #2563eb;">Visit ${appName}</a></p>
          </div>
        </div>
      </body>
      </html>
    `;

    const text = `
      Reset Your Password - ${appName}
      
      Hello ${options.userName || 'there'},
      
      We received a request to reset the password for your ${appName} account.
      
      Click this link to reset your password:
      ${options.resetLink}
      
      This link will expire in 1 hour for your security.
      
      If you didn't request this password reset, you can safely ignore this email.
      
      Best regards,
      The ${appName} Team
    `;

    return this.sendEmail({
      to: options.to,
      subject: `Reset Your ${appName} Password`,
      html,
      text,
    });
  }

  async testConnection(): Promise<boolean> {
    if (!this.transporter) {
      console.error('❌ Email service not initialized');
      return false;
    }

    try {
      await this.transporter.verify();
      console.log('✅ Email service connection verified');
      return true;
    } catch (error) {
      console.error('❌ Email service connection failed:', error);
      return false;
    }
  }

  // Check if email service is properly configured
  isConfigured(): boolean {
    return this.transporter !== null;
  }
}

// Export singleton instance
export const emailService = new EmailService();
export default EmailService;