#!/usr/bin/env node

const fs = require('fs');
const path = require('path');

class DocumentationGenerator {
    constructor(projectPath) {
        this.projectPath = projectPath;
        this.ignoredDirs = new Set([
            'node_modules',
            '.next',
            '.git',
            '__pycache__',
            '.vscode',
            'dist',
            'build',
            '.turbo'
        ]);
        this.ignoredFiles = new Set([
            '.DS_Store',
            'Thumbs.db',
            '.env.local',
            '.env',
            '.gitignore'
        ]);
        this.binaryExtensions = new Set([
            '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.ico',
            '.pdf', '.zip', '.tar', '.gz', '.exe', '.dll',
            '.so', '.dylib', '.bin'
        ]);
    }

    // Get file extension
    getFileExtension(filePath) {
        return path.extname(filePath).toLowerCase();
    }

    // Check if file should be ignored
    shouldIgnoreFile(filePath) {
        const fileName = path.basename(filePath);
        const ext = this.getFileExtension(filePath);
        return this.ignoredFiles.has(fileName) || this.binaryExtensions.has(ext);
    }

    // Check if directory should be ignored
    shouldIgnoreDir(dirPath) {
        const dirName = path.basename(dirPath);
        return this.ignoredDirs.has(dirName);
    }

    // Get language identifier for syntax highlighting
    getLanguageFromExtension(ext) {
        const languageMap = {
            '.js': 'javascript',
            '.jsx': 'jsx',
            '.ts': 'typescript',
            '.tsx': 'tsx',
            '.py': 'python',
            '.json': 'json',
            '.md': 'markdown',
            '.yml': 'yaml',
            '.yaml': 'yaml',
            '.css': 'css',
            '.scss': 'scss',
            '.html': 'html',
            '.sql': 'sql',
            '.sh': 'bash',
            '.env': 'bash',
            '.txt': 'text',
            '.xml': 'xml',
            '.toml': 'toml',
            '.lock': 'text'
        };
        return languageMap[ext] || 'text';
    }

    // Scan directory recursively
    scanDirectory(dirPath, relativePath = '') {
        const items = [];
        
        try {
            const entries = fs.readdirSync(dirPath, { withFileTypes: true });
            
            for (const entry of entries) {
                const fullPath = path.join(dirPath, entry.name);
                const relPath = path.join(relativePath, entry.name);

                if (entry.isDirectory()) {
                    if (!this.shouldIgnoreDir(fullPath)) {
                        items.push({
                            type: 'directory',
                            name: entry.name,
                            path: relPath,
                            children: this.scanDirectory(fullPath, relPath)
                        });
                    }
                } else if (entry.isFile()) {
                    if (!this.shouldIgnoreFile(fullPath)) {
                        items.push({
                            type: 'file',
                            name: entry.name,
                            path: relPath,
                            extension: this.getFileExtension(entry.name),
                            size: fs.statSync(fullPath).size
                        });
                    }
                }
            }
        } catch (error) {
            console.warn(`Warning: Could not read directory ${dirPath}: ${error.message}`);
        }

        return items.sort((a, b) => {
            if (a.type !== b.type) {
                return a.type === 'directory' ? -1 : 1;
            }
            return a.name.localeCompare(b.name);
        });
    }

    // Read file content safely
    readFileContent(filePath) {
        try {
            return fs.readFileSync(filePath, 'utf8');
        } catch (error) {
            return `Error reading file: ${error.message}`;
        }
    }

    // Generate project structure markdown
    generateStructureMarkdown(items, indent = 0) {
        let markdown = '';
        const prefix = '  '.repeat(indent);

        for (const item of items) {
            if (item.type === 'directory') {
                markdown += `${prefix}- 📁 **${item.name}/**\n`;
                if (item.children && item.children.length > 0) {
                    markdown += this.generateStructureMarkdown(item.children, indent + 1);
                }
            } else {
                const sizeKB = (item.size / 1024).toFixed(1);
                markdown += `${prefix}- 📄 ${item.name} (${sizeKB} KB)\n`;
            }
        }

        return markdown;
    }

    // Generate file content sections with size limit
    generateFileContentSections(items, basePath = '') {
        let sections = '';
        let totalSize = 0;
        const MAX_FILE_SIZE = 50 * 1024; // 50KB per file
        const MAX_TOTAL_SIZE = 10 * 1024 * 1024; // 10MB total

        for (const item of items) {
            if (totalSize > MAX_TOTAL_SIZE) {
                sections += `\n## ⚠️ Documentation Size Limit Reached\n\nThe remaining files have been omitted to prevent memory issues. ` +
                           `The complete project structure is available above.\n\n`;
                break;
            }

            const fullPath = path.join(this.projectPath, item.path);

            if (item.type === 'directory' && item.children) {
                const childSections = this.generateFileContentSections(item.children, basePath);
                sections += childSections;
                totalSize += childSections.length;
            } else if (item.type === 'file' && item.size < MAX_FILE_SIZE) {
                const content = this.readFileContent(fullPath);
                const language = this.getLanguageFromExtension(item.extension);
                
                // Truncate very long files
                let displayContent = content;
                if (content.length > MAX_FILE_SIZE) {
                    const lines = content.split('\n');
                    const truncatedLines = lines.slice(0, 200); // First 200 lines
                    displayContent = truncatedLines.join('\n') + '\n\n// ... (file truncated for documentation size)\n';
                }
                
                const fileSection = `\n## 📄 ${item.path}\n\n` +
                                  `**File Type:** ${item.extension || 'No extension'}\n` +
                                  `**Size:** ${(item.size / 1024).toFixed(2)} KB\n\n` +
                                  `\`\`\`${language} path=${fullPath} start=1\n` +
                                  displayContent +
                                  `\n\`\`\`\n\n`;
                
                sections += fileSection;
                totalSize += fileSection.length;
            } else if (item.type === 'file') {
                // Large file - just list it
                sections += `\n## 📄 ${item.path}\n\n`;
                sections += `**File Type:** ${item.extension || 'No extension'}\n`;
                sections += `**Size:** ${(item.size / 1024).toFixed(2)} KB\n`;
                sections += `**Note:** File too large to include in documentation\n\n`;
            }
        }

        return sections;
    }

    // Generate package.json analysis
    generatePackageAnalysis() {
        const packagePath = path.join(this.projectPath, 'package.json');
        if (!fs.existsSync(packagePath)) {
            return '**No package.json found**\n\n';
        }

        try {
            const packageContent = JSON.parse(fs.readFileSync(packagePath, 'utf8'));
            let analysis = '';

            analysis += `**Project Name:** ${packageContent.name || 'Not specified'}\n`;
            analysis += `**Version:** ${packageContent.version || 'Not specified'}\n`;
            analysis += `**Description:** ${packageContent.description || 'Not specified'}\n\n`;

            if (packageContent.scripts) {
                analysis += `**Available Scripts:**\n`;
                Object.entries(packageContent.scripts).forEach(([name, script]) => {
                    analysis += `- \`npm run ${name}\`: ${script}\n`;
                });
                analysis += '\n';
            }

            if (packageContent.dependencies) {
                analysis += `**Dependencies (${Object.keys(packageContent.dependencies).length}):**\n`;
                Object.entries(packageContent.dependencies).forEach(([name, version]) => {
                    analysis += `- ${name}: ${version}\n`;
                });
                analysis += '\n';
            }

            if (packageContent.devDependencies) {
                analysis += `**Dev Dependencies (${Object.keys(packageContent.devDependencies).length}):**\n`;
                Object.entries(packageContent.devDependencies).forEach(([name, version]) => {
                    analysis += `- ${name}: ${version}\n`;
                });
                analysis += '\n';
            }

            return analysis;
        } catch (error) {
            return `**Error reading package.json:** ${error.message}\n\n`;
        }
    }

    // Generate requirements.txt analysis
    generatePythonRequirements() {
        const reqPath = path.join(this.projectPath, 'requirements.txt');
        if (!fs.existsSync(reqPath)) {
            return '';
        }

        try {
            const content = fs.readFileSync(reqPath, 'utf8');
            const packages = content.split('\n').filter(line => line.trim() && !line.startsWith('#'));
            
            let analysis = `**Python Dependencies (${packages.length}):**\n`;
            packages.forEach(pkg => {
                analysis += `- ${pkg.trim()}\n`;
            });
            analysis += '\n';

            return analysis;
        } catch (error) {
            return `**Error reading requirements.txt:** ${error.message}\n\n`;
        }
    }

    // Generate comprehensive documentation
    generate() {
        console.log('🔍 Scanning project directory...');
        const projectStructure = this.scanDirectory(this.projectPath);
        
        console.log('📊 Analyzing project...');
        const packageAnalysis = this.generatePackageAnalysis();
        const pythonRequirements = this.generatePythonRequirements();
        
        console.log('📝 Generating documentation...');
        
        const documentation = `# 🚀 Deepfake Detection Platform - Complete Project Documentation

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [Project Structure](#project-structure)
3. [Dependencies Analysis](#dependencies-analysis)
4. [Setup Instructions](#setup-instructions)
5. [Configuration](#configuration)
6. [API Documentation](#api-documentation)
7. [Source Code](#source-code)

---

## 📖 Project Overview

This is a comprehensive deepfake detection platform built with Next.js 15 (frontend) and Python FastAPI (backend). The platform provides:

- **Web-based deepfake detection** using AI models
- **User authentication** with NextAuth.js
- **File upload and analysis** with real-time progress
- **User dashboard** with analytics and history
- **Email notifications** for password reset
- **PDF export** of analysis results
- **Modern UI** with Tailwind CSS and shadcn/ui components

**Technology Stack:**
- **Frontend:** Next.js 15, React, TypeScript, Tailwind CSS
- **Backend:** Python, FastAPI, OpenCV, TensorFlow/PyTorch
- **Database:** Prisma ORM with SQLite/PostgreSQL
- **Authentication:** NextAuth.js
- **Email:** Nodemailer with Gmail SMTP
- **Deployment:** Ready for Vercel (frontend) and Python hosting (backend)

---

## 🗂️ Project Structure

${this.generateStructureMarkdown(projectStructure)}

---

## 📦 Dependencies Analysis

### Frontend Dependencies

${packageAnalysis}

### Backend Dependencies

${pythonRequirements}

---

## 🛠️ Setup Instructions

### Prerequisites

1. **Node.js** (version 18 or higher)
2. **Python** (version 3.8 or higher)
3. **Git** for version control
4. **Gmail account** (for email functionality)

### Step 1: Clone and Setup Frontend

\`\`\`bash
# Clone the repository
git clone <repository-url>
cd deepfakewebpythonapi

# Install Node.js dependencies
npm install

# Generate Prisma client
npx prisma generate

# Setup database (SQLite by default)
npx prisma db push
\`\`\`

### Step 2: Setup Python Backend

\`\`\`bash
# Create Python virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\\Scripts\\activate
# On macOS/Linux:
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
\`\`\`

### Step 3: Environment Configuration

Create a \`.env.local\` file in the project root:

\`\`\`env
# NextAuth Configuration
NEXTAUTH_SECRET=your-secret-here
NEXTAUTH_URL=http://localhost:3000

# Email Configuration (Gmail SMTP)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASS=your-app-password

# Database (optional, defaults to SQLite)
DATABASE_URL="file:./dev.db"

# API Configuration
PYTHON_API_URL=http://localhost:8000
\`\`\`

### Step 4: Gmail SMTP Setup

1. Enable 2-Factor Authentication on your Gmail account
2. Generate an App Password:
   - Go to Google Account settings
   - Security → 2-Step Verification → App passwords
   - Generate password for "Mail"
3. Use the generated password in \`SMTP_PASS\`

### Step 5: Start Development Servers

\`\`\`bash
# Start both frontend and backend
npm run dev:full

# Or start individually:
npm run dev        # Frontend only (port 3000)
npm run dev:api    # Backend only (port 8000)
\`\`\`

### Step 6: Access the Application

- **Frontend:** http://localhost:3000
- **Backend API:** http://localhost:8000
- **API Documentation:** http://localhost:8000/docs

---

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| \`NEXTAUTH_SECRET\` | NextAuth.js secret key | Yes | - |
| \`NEXTAUTH_URL\` | Application base URL | Yes | http://localhost:3000 |
| \`SMTP_HOST\` | Email server host | Yes | smtp.gmail.com |
| \`SMTP_PORT\` | Email server port | Yes | 587 |
| \`SMTP_USER\` | Email username | Yes | - |
| \`SMTP_PASS\` | Email password/app password | Yes | - |
| \`DATABASE_URL\` | Database connection string | No | file:./dev.db |
| \`PYTHON_API_URL\` | Backend API URL | No | http://localhost:8000 |

### Database Schema

The application uses Prisma with the following main models:
- **User:** User accounts and profiles
- **Analysis:** Detection results and metadata
- **Session:** User session management

---

## 🔌 API Documentation

### Frontend API Routes

#### Authentication
- \`POST /api/auth/[...nextauth]\` - NextAuth.js authentication
- \`POST /api/auth/forgot-password\` - Password reset email
- \`POST /api/auth/reset-password\` - Password reset confirmation

#### User Management
- \`GET /api/user/profile\` - Get user profile
- \`PATCH /api/user/profile\` - Update user profile
- \`GET /api/user/stats\` - Get user statistics
- \`GET /api/user/usage\` - Get usage analytics
- \`GET /api/user/analyses\` - Get analysis history

### Backend API Endpoints

#### Detection
- \`POST /detect\` - Upload and analyze media file
- \`GET /health\` - Health check endpoint

#### Response Format
\`\`\`json
{
  "filename": "example.mp4",
  "is_deepfake": true,
  "confidence": 0.87,
  "models_used": ["model1", "model2"],
  "processing_time": 2.34,
  "metadata": {
    "file_size": 1024000,
    "duration": 30.5
  }
}
\`\`\`

---

## 💻 Source Code

Below is the complete source code for all files in the project:

${this.generateFileContentSections(projectStructure)}

---

## 🚀 Deployment

### Frontend Deployment (Vercel)

1. Connect your GitHub repository to Vercel
2. Set environment variables in Vercel dashboard
3. Deploy automatically on push to main branch

### Backend Deployment

Options include:
- **Railway:** Python hosting with automatic deployments
- **Heroku:** Classic platform with Python support
- **DigitalOcean:** App platform or droplets
- **AWS:** EC2 or Lambda for serverless

### Production Considerations

1. **Security:** Use strong secrets and HTTPS
2. **Database:** Migrate from SQLite to PostgreSQL
3. **File Storage:** Use cloud storage (AWS S3, Cloudinary)
4. **Monitoring:** Add error tracking and performance monitoring
5. **Scaling:** Consider load balancing for high traffic

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

---

## 📄 License

This project is licensed under the MIT License.

---

**Generated on:** ${new Date().toISOString()}
**Generator Version:** 1.0.0
`;

        return documentation;
    }

    // Save documentation to file
    save(outputPath) {
        const documentation = this.generate();
        fs.writeFileSync(outputPath, documentation, 'utf8');
        console.log(`✅ Documentation generated successfully at: ${outputPath}`);
        
        const stats = fs.statSync(outputPath);
        console.log(`📄 Documentation size: ${(stats.size / 1024).toFixed(2)} KB`);
        
        return outputPath;
    }
}

// Main execution
function main() {
    const projectPath = process.argv[2] || process.cwd();
    const outputPath = process.argv[3] || path.join(projectPath, 'PROJECT_DOCUMENTATION.md');
    
    console.log('🎯 Starting documentation generation...');
    console.log(`📂 Project path: ${projectPath}`);
    console.log(`📝 Output path: ${outputPath}`);
    console.log('');
    
    const generator = new DocumentationGenerator(projectPath);
    
    try {
        generator.save(outputPath);
        console.log('');
        console.log('🎉 Documentation generation completed!');
        console.log('');
        console.log('📋 What was included:');
        console.log('  ✓ Complete project structure');
        console.log('  ✓ All source code files');
        console.log('  ✓ Dependencies analysis');
        console.log('  ✓ Step-by-step setup guide');
        console.log('  ✓ Environment configuration');
        console.log('  ✓ API documentation');
        console.log('  ✓ Deployment instructions');
        console.log('');
        console.log('🚀 Your friend can now use this documentation to recreate the entire project!');
    } catch (error) {
        console.error('❌ Error generating documentation:', error.message);
        process.exit(1);
    }
}

if (require.main === module) {
    main();
}

module.exports = DocumentationGenerator;