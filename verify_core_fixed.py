#!/usr/bin/env python3
"""
Enhanced verification script to check core functionality completeness 
in BUILD_FROM_SCRATCH.md
"""

import os
import re
from pathlib import Path

def get_core_files():
    """Get core essential files for the application."""
    core_files = {
        # Backend Python files
        'deepfake_api/main.py': 'Backend main server',
        'deepfake_api/detectors/audio_detector.py': 'Audio detection',
        'deepfake_api/detectors/video_detector.py': 'Video detection', 
        'deepfake_api/detectors/image_detector.py': 'Image detection',
        'deepfake_api/download_models.py': 'Model downloader',
        
        # Frontend core pages
        'src/app/page.tsx': 'Home page',
        'src/app/layout.tsx': 'Root layout',
        'src/app/upload/page.tsx': 'Upload page',
        'src/app/dashboard/page.tsx': 'Dashboard',
        'src/app/auth/signin/page.tsx': 'Sign in',
        'src/app/auth/signup/page.tsx': 'Sign up',
        
        # API routes
        'src/app/api/upload/route.ts': 'Upload API',
        'src/app/api/auth/signup/route.ts': 'Signup API',
        
        # Core components
        'src/components/ui/button.tsx': 'Button component',
        'src/components/ui/input.tsx': 'Input component',
        'src/components/ui/card.tsx': 'Card component',
        'src/components/layout/navbar.tsx': 'Navigation',
        
        # Configuration
        'package.json': 'Package config',
        'next.config.ts': 'Next.js config',
        'tailwind.config.ts': 'Tailwind config',
        'prisma/schema.prisma': 'Database schema',
        
        # Library files
        'src/lib/auth.ts': 'Authentication config',
        'src/lib/database-service.ts': 'Database service',
        'src/lib/prisma.ts': 'Prisma client'
    }
    
    return core_files

def check_file_in_guide(filename, guide_content):
    """Check if a file is included in the BUILD_FROM_SCRATCH.md guide."""
    patterns = [
        f'cat > {filename}',
        f'cat > ./{filename}',
    ]
    
    for pattern in patterns:
        if pattern in guide_content:
            return True
    
    return False

def extract_code_for_file(filename, guide_content):
    """Extract the code content for a specific file from the guide."""
    patterns = [
        f'cat > {filename} << \'EOF\'',
        f'cat > ./{filename} << \'EOF\'',
    ]
    
    for pattern in patterns:
        start_pos = guide_content.find(pattern)
        if start_pos != -1:
            eof_pos = guide_content.find('EOF', start_pos + len(pattern))
            if eof_pos != -1:
                code_block = guide_content[start_pos + len(pattern):eof_pos]
                return len(code_block.split('\n')) - 2
    
    return 0

def main():
    print("🔍 Verifying BUILD_FROM_SCRATCH.md CORE functionality completeness...")
    print("=" * 70)
    
    try:
        with open('BUILD_FROM_SCRATCH.md', 'r', encoding='utf-8') as f:
            guide_content = f.read()
    except FileNotFoundError:
        print("❌ BUILD_FROM_SCRATCH.md not found!")
        return
    
    core_files = get_core_files()
    print(f"📋 Checking {len(core_files)} core files...\n")
    
    present_files = 0
    missing_files = []
    present_with_code = 0
    total_guide_lines = 0
    
    for file_path, description in core_files.items():
        file_exists = os.path.exists(file_path)
        in_guide = check_file_in_guide(file_path, guide_content)
        guide_lines = extract_code_for_file(file_path, guide_content)
        total_guide_lines += guide_lines
        
        status = "❌"
        if in_guide:
            present_files += 1
            if guide_lines > 10:
                present_with_code += 1
                status = "✅"
            else:
                status = "⚠️"
        else:
            missing_files.append((file_path, description))
        
        original_lines = 0
        if file_exists:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    original_lines = len(f.readlines())
            except:
                pass
        
        print(f"{status} {file_path:<35} {description:<20} " +
              f"({'exists' if file_exists else 'missing'}:{original_lines} -> guide:{guide_lines} lines)")
    
    print(f"\n📊 CORE Functionality Analysis:")
    print(f"   Core files present: {present_files}/{len(core_files)} ({present_files/len(core_files)*100:.1f}%)")
    print(f"   Files with substantial code: {present_with_code}/{len(core_files)} ({present_with_code/len(core_files)*100:.1f}%)")
    print(f"   Total guide code lines for core files: {total_guide_lines:,}")
    
    if missing_files:
        print(f"\n❌ Missing CRITICAL files ({len(missing_files)} files):")
        for file_path, description in missing_files[:10]:
            print(f"   - {file_path:<35} ({description})")
        if len(missing_files) > 10:
            print(f"   ... and {len(missing_files) - 10} more")
    
    print(f"\n🎯 CORE Functionality Assessment:")
    if present_with_code >= len(core_files) * 0.9:
        print("   ✅ BUILD_FROM_SCRATCH.md has EXCELLENT core coverage")
        print("   ✅ All essential functionality is included")
        assessment = "EXCELLENT"
    elif present_with_code >= len(core_files) * 0.7:
        print("   ✅ BUILD_FROM_SCRATCH.md has GOOD core coverage")
        print("   ✅ Most essential functionality is included")
        assessment = "GOOD"
    elif present_files >= len(core_files) * 0.6:
        print("   ⚠️  BUILD_FROM_SCRATCH.md has ADEQUATE core coverage")
        print("   ⚠️  Basic functionality is included, some gaps exist")
        assessment = "ADEQUATE"
    else:
        print("   ❌ BUILD_FROM_SCRATCH.md has INSUFFICIENT core coverage")
        print("   ❌ Critical functionality may be missing")
        assessment = "INSUFFICIENT"
    
    core_completeness = (present_with_code / len(core_files)) * 100
    print(f"   Core completeness: {core_completeness:.1f}%")
    
    print(f"\n📈 Additional Statistics:")
    print(f"   Guide file size: {len(guide_content):,} characters")
    print(f"   Guide file lines: {len(guide_content.split(chr(10))):,} lines")
    print(f"   Code blocks found: {guide_content.count('cat >')}")
    
    return assessment, core_completeness

if __name__ == "__main__":
    main()