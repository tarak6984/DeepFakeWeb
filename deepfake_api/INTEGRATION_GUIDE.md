# 🔄 Reality Defender → Local API Integration Guide

This guide shows you exactly how to replace Reality Defender API with your new local deepfake detection API in your existing Next.js application.

## 📂 Your Current Structure

Your existing app has:
```
src/app/api/rd/
├── signed-url/
│   └── route.ts
└── result/[id]/
    └── route.ts
```

## 🛠️ Step 1: Update API Route Handlers

### Update `src/app/api/rd/signed-url/route.ts`

Replace the entire content with:

```typescript
import { NextResponse } from "next/server";

export async function POST(req: Request) {
  try {
    const { fileName } = await req.json();

    if (!fileName || typeof fileName !== "string") {
      return NextResponse.json({ error: "Invalid fileName" }, { status: 400 });
    }

    // Local API doesn't need signed URLs - return direct upload endpoint
    return NextResponse.json({
      message: "Local API - upload directly to /api/upload",
      upload_url: `${process.env.NEXT_PUBLIC_LOCAL_API_URL || 'http://localhost:8000'}/api/upload`,
      fileName: fileName
    });

  } catch (err: unknown) {
    return NextResponse.json({ 
      error: err instanceof Error ? err.message : "Unexpected error" 
    }, { status: 500 });
  }
}
```

### Update `src/app/api/rd/result/[id]/route.ts`

Replace the entire content with:

```typescript
import { NextResponse } from "next/server";

export async function GET(_req: Request, context: { params: Promise<{ id: string }> }) {
  try {
    const { id } = await context.params || { id: '' };

    if (!id) {
      return NextResponse.json({ error: "Missing id" }, { status: 400 });
    }

    const localApiUrl = process.env.NEXT_PUBLIC_LOCAL_API_URL || 'http://localhost:8000';
    
    // Forward request to local API
    const upstream = await fetch(`${localApiUrl}/api/result/${id}`, {
      method: "GET",
      cache: "no-store",
    });

    const data = await upstream.json().catch(() => ({}));

    if (!upstream.ok) {
      return NextResponse.json(
        { error: "Local API error", status: upstream.status, response: data },
        { status: upstream.status }
      );
    }

    return NextResponse.json(data);
  } catch (err: unknown) {
    return NextResponse.json({ 
      error: err instanceof Error ? err.message : "Unexpected error" 
    }, { status: 500 });
  }
}
```

## 🔧 Step 2: Update Environment Variables

### Update `.env.local`

```bash
# Replace Reality Defender configuration with local API
NEXT_PUBLIC_LOCAL_API_URL=http://localhost:8000

# Keep these for backwards compatibility (but they won't be used)
# NEXT_PUBLIC_RD_API_KEY=not_needed_anymore
# NEXT_PUBLIC_RD_API_URL=http://localhost:8000

# App Configuration (keep existing)
NEXT_PUBLIC_APP_NAME="Deepfake Detective"
NEXT_PUBLIC_APP_DESCRIPTION="Advanced AI-powered deepfake detection for media files"
```

## 📱 Step 3: Update Frontend Code (if needed)

If your frontend code directly calls Reality Defender API, update it:

### Before (Reality Defender):
```javascript
// OLD: Reality Defender API call
const response = await fetch('/api/rd/signed-url', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ fileName: file.name })
});

const { upload_url } = await response.json();

// Upload to Reality Defender
const uploadResponse = await fetch(upload_url, {
  method: 'POST',
  body: formData
});
```

### After (Local API):
```javascript
// NEW: Direct upload to local API
const response = await fetch('http://localhost:8000/api/upload', {
  method: 'POST',
  body: formData  // No additional headers needed!
});

const result = await response.json();
const analysisId = result.analysis_id;
```

## 🚀 Step 4: Start Both APIs

### Terminal 1: Start your local deepfake API
```bash
cd deepfake_api
python main.py
```

### Terminal 2: Start your Next.js app
```bash
npm run dev
```

## 🔍 Step 5: Test the Integration

1. **Open your Next.js app**: http://localhost:3000
2. **Upload a test file** through your existing UI
3. **Verify it works** without any Reality Defender API key!

## 📊 What Changed vs What Stayed the Same

### ✅ What Stayed the Same:
- Your frontend UI and components
- Your existing API route structure (`/api/rd/...`)
- Your file upload workflow
- Your result polling logic

### 🔄 What Changed:
- API calls now go to `localhost:8000` instead of Reality Defender
- No API key authentication needed
- **Unlimited usage** instead of 50/month limit
- **Support for video files** in addition to images and audio
- **Faster processing** (local vs network)
- **Better privacy** (files never leave your machine)

## 🐛 Troubleshooting

### Issue: "Cannot connect to API"
**Solution**: Make sure the local API is running:
```bash
cd deepfake_api
python main.py
```

### Issue: "Module not found" errors in Next.js
**Solution**: The API routes should work as-is. If you have import errors, restart the Next.js dev server:
```bash
npm run dev
```

### Issue: CORS errors
**Solution**: The local API is configured to allow requests from `localhost:3000`. If using a different port, update `deepfake_api/config.yaml`:
```yaml
api:
  cors_origins: ["http://localhost:3000", "http://localhost:3001"]
```

### Issue: "Port already in use"
**Solution**: Stop any existing processes or change the API port:
```bash
# Find process using port 8000
lsof -i :8000
# Kill it
kill -9 <PID>

# Or change the port in config.yaml
api:
  port: 8001
```

## 🎉 Benefits You'll Get

1. **💰 Cost Savings**: No more API fees - unlimited usage!
2. **🚀 Better Performance**: Local processing is faster
3. **📹 Video Support**: Now supports video deepfake detection
4. **🔒 Privacy**: Files never leave your machine
5. **🎛️ Full Control**: Customize models, thresholds, and behavior
6. **📈 Scalability**: No rate limits or quotas

## 🔄 Optional: Frontend Direct Integration

If you want to bypass the Next.js API routes entirely and call the local API directly from your frontend:

```javascript
// Direct integration (optional)
async function analyzeFile(file) {
  const formData = new FormData();
  formData.append('file', file);
  
  // Direct upload to local API
  const response = await fetch('http://localhost:8000/api/upload', {
    method: 'POST',
    body: formData
  });
  
  const { analysis_id } = await response.json();
  
  // Poll for results
  while (true) {
    const resultResponse = await fetch(`http://localhost:8000/api/result/${analysis_id}`);
    const result = await resultResponse.json();
    
    if (result.status === 'completed') {
      return result;
    }
    
    await new Promise(resolve => setTimeout(resolve, 1000));
  }
}
```

---

🎊 **Congratulations!** You've successfully replaced Reality Defender with a powerful local alternative that gives you unlimited deepfake detection capabilities!