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
        'next.config.js': 'Next.js config',
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
    # Look for the file creation pattern
    patterns = [
        f'cat > {filename}',
        f'cat > ./{filename}',
        f'cat > src/{filename}' if not filename.startswith('src/') else f'cat > {filename}',
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
            # Find the end of the code block
            eof_pos = guide_content.find('EOF', start_pos + len(pattern))
            if eof_pos != -1:
                code_block = guide_content[start_pos + len(pattern):eof_pos]
                return len(code_block.split('\n')) - 2  # Subtract 2 for empty lines
    
    return 0

def main():
    print("🔍 Verifying BUILD_FROM_SCRATCH.md CORE functionality completeness...")
    print("=" * 70)
    
    # Read the BUILD_FROM_SCRATCH.md content
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
        # Check if file exists in original project
        file_exists = os.path.exists(file_path)
        
        # Check if file is in guide
        in_guide = check_file_in_guide(file_path, guide_content)
        
        # Count lines in guide for this file
        guide_lines = extract_code_for_file(file_path, guide_content)
        total_guide_lines += guide_lines
        
        status = "❌"
        if in_guide:
            present_files += 1
            if guide_lines > 10:  # Has substantial code
                present_with_code += 1
                status = "✅"
            else:
                status = "⚠️"
        else:
            missing_files.append((file_path, description))
        
        # Get original file line count if it exists
        original_lines = 0
        if file_exists:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    original_lines = len(f.readlines())
            except:
                pass
        
        print(f"{status} {file_path:<35} {description:<20} "
              f"({'exists' if file_exists else 'missing'}:{original_lines} -> guide:{guide_lines} lines)\")\n    
    \n    print(f\"\\n📊 CORE Functionality Analysis:\")\n    print(f\"   Core files present: {present_files}/{len(core_files)} ({present_files/len(core_files)*100:.1f}%)\")\n    print(f\"   Files with substantial code: {present_with_code}/{len(core_files)} ({present_with_code/len(core_files)*100:.1f}%)\")\n    print(f\"   Total guide code lines for core files: {total_guide_lines:,}\")\n    \n    if missing_files:\n        print(f\"\\n❌ Missing CRITICAL files ({len(missing_files)} files):\")\n        for file_path, description in missing_files[:10]:\n            print(f\"   - {file_path:<35} ({description})\")\n        if len(missing_files) > 10:\n            print(f\"   ... and {len(missing_files) - 10} more\")\n    \n    # Overall assessment for CORE functionality\n    print(f\"\\n🎯 CORE Functionality Assessment:\")\n    if present_with_code >= len(core_files) * 0.9:  # 90% of core files with code\n        print(\"   ✅ BUILD_FROM_SCRATCH.md has EXCELLENT core coverage\")\n        print(\"   ✅ All essential functionality is included\")\n        assessment = \"EXCELLENT\"\n    elif present_with_code >= len(core_files) * 0.7:  # 70% of core files with code\n        print(\"   ✅ BUILD_FROM_SCRATCH.md has GOOD core coverage\")\n        print(\"   ✅ Most essential functionality is included\")\n        assessment = \"GOOD\"\n    elif present_files >= len(core_files) * 0.6:  # 60% of core files present\n        print(\"   ⚠️  BUILD_FROM_SCRATCH.md has ADEQUATE core coverage\")\n        print(\"   ⚠️  Basic functionality is included, some gaps exist\")\n        assessment = \"ADEQUATE\"\n    else:\n        print(\"   ❌ BUILD_FROM_SCRATCH.md has INSUFFICIENT core coverage\")\n        print(\"   ❌ Critical functionality may be missing\")\n        assessment = \"INSUFFICIENT\"\n    \n    core_completeness = (present_with_code / len(core_files)) * 100\n    print(f\"   Core completeness: {core_completeness:.1f}%\")\n    \n    # Additional statistics\n    print(f\"\\n📈 Additional Statistics:\")\n    print(f\"   Guide file size: {len(guide_content):,} characters\")\n    print(f\"   Guide file lines: {len(guide_content.split(chr(10))):,} lines\")\n    print(f\"   Code blocks found: {guide_content.count('cat >')}\")\n    \n    return assessment, core_completeness\n\nif __name__ == \"__main__\":\n    main()"


Let me fix the syntax error in the script:

<function_calls>
<invoke name="edit_files">
<parameter name="title">Fix syntax error in verification script