#!/usr/bin/env python3
"""
Verification script to check if BUILD_FROM_SCRATCH.md contains all source files
from the original project.
"""

import os
import re
from pathlib import Path

def get_project_files():
    """Get all source code files from the project."""
    extensions = {'.py', '.ts', '.tsx', '.js', '.jsx', '.json', '.yaml', '.yml'}
    exclude_dirs = {'node_modules', '.git', '__pycache__', '.next', 'dist', 'build'}
    
    project_files = []
    
    for root, dirs, files in os.walk('.'):
        # Remove excluded directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        for file in files:
            if Path(file).suffix in extensions:
                file_path = os.path.join(root, file)
                project_files.append(file_path)
    
    return sorted(project_files)

def get_build_guide_files():
    """Extract file paths mentioned in BUILD_FROM_SCRATCH.md."""
    guide_files = []
    
    with open('BUILD_FROM_SCRATCH.md', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find all "cat > filename" patterns
    pattern = r'cat > ([^\s<]+)'
    matches = re.findall(pattern, content)
    
    for match in matches:
        # Convert to relative path format
        if match.startswith('./'):
            guide_files.append(match)
        else:
            guide_files.append('./' + match)
    
    return sorted(guide_files)

def count_code_lines_in_project():
    """Count actual lines of code in project files."""
    extensions = {'.py', '.ts', '.tsx', '.js', '.jsx'}
    exclude_dirs = {'node_modules', '.git', '__pycache__', '.next', 'dist', 'build'}
    
    total_lines = 0
    file_count = 0
    
    for root, dirs, files in os.walk('.'):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        for file in files:
            if Path(file).suffix in extensions:
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = len(f.readlines())
                        total_lines += lines
                        file_count += 1
                except:
                    continue
    
    return total_lines, file_count

def count_code_lines_in_guide():
    """Count lines of code embedded in BUILD_FROM_SCRATCH.md."""
    with open('BUILD_FROM_SCRATCH.md', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    code_lines = 0
    in_code_block = False
    
    for line in lines:
        line = line.strip()
        
        # Start of code block
        if line.startswith("cat >") and "<<" in line:
            in_code_block = True
            continue
        
        # End of code block
        if line == "EOF":
            in_code_block = False
            continue
        
        # Count code lines
        if in_code_block:
            code_lines += 1
    
    return code_lines

def main():
    print("🔍 Verifying BUILD_FROM_SCRATCH.md completeness...")
    print("=" * 60)
    
    # Get project files
    project_files = get_project_files()
    guide_files = get_build_guide_files()
    
    # Count lines of code
    project_lines, project_file_count = count_code_lines_in_project()
    guide_lines = count_code_lines_in_guide()
    
    print(f"📁 Original Project:")
    print(f"   Source files: {project_file_count}")
    print(f"   Total lines of code: {project_lines:,}")
    print()
    
    print(f"📖 BUILD_FROM_SCRATCH.md:")
    print(f"   Files included: {len(guide_files)}")
    print(f"   Lines of code: {guide_lines:,}")
    print()
    
    # Calculate coverage
    if project_file_count > 0:
        file_coverage = (len(guide_files) / project_file_count) * 100
    else:
        file_coverage = 0
    
    if project_lines > 0:
        line_coverage = (guide_lines / project_lines) * 100
    else:
        line_coverage = 0
    
    print(f"📊 Coverage Analysis:")
    print(f"   File coverage: {file_coverage:.1f}%")
    print(f"   Line coverage: {line_coverage:.1f}%")
    print()
    
    # Check for missing files
    project_file_names = {os.path.basename(f) for f in project_files}
    guide_file_names = {os.path.basename(f) for f in guide_files}
    
    missing_files = project_file_names - guide_file_names
    extra_files = guide_file_names - project_file_names
    
    if missing_files:
        print("❌ Missing files in BUILD_FROM_SCRATCH.md:")
        for file in sorted(missing_files)[:10]:  # Show first 10
            print(f"   - {file}")
        if len(missing_files) > 10:
            print(f"   ... and {len(missing_files) - 10} more")
        print()
    
    if extra_files:
        print("➕ Additional files in BUILD_FROM_SCRATCH.md:")
        for file in sorted(extra_files)[:10]:
            print(f"   - {file}")
        if len(extra_files) > 10:
            print(f"   ... and {len(extra_files) - 10} more")
        print()
    
    # Final assessment
    print("🎯 Final Assessment:")
    if file_coverage >= 95 and line_coverage >= 85:
        print("   ✅ BUILD_FROM_SCRATCH.md is COMPREHENSIVE")
        print("   ✅ Contains sufficient code coverage for full project reconstruction")
    elif file_coverage >= 80 and line_coverage >= 70:
        print("   ⚠️  BUILD_FROM_SCRATCH.md has GOOD coverage")
        print("   ⚠️  Most files included, minor gaps may exist")
    else:
        print("   ❌ BUILD_FROM_SCRATCH.md has INSUFFICIENT coverage")
        print("   ❌ Significant files or code may be missing")
    
    print(f"   Guide completeness: {min(file_coverage, line_coverage):.1f}%")

if __name__ == "__main__":
    main()