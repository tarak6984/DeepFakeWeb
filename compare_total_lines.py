#!/usr/bin/env python3
"""
Comprehensive comparison of total lines of code between the original project
and BUILD_FROM_SCRATCH.md guide
"""

import os
import re
from pathlib import Path

def count_project_lines():
    """Count all lines of code in the original project."""
    extensions = {'.py', '.ts', '.tsx', '.js', '.jsx', '.json', '.yaml', '.yml', '.md'}
    exclude_dirs = {'node_modules', '.git', '__pycache__', '.next', 'dist', 'build'}
    
    total_lines = 0
    total_files = 0
    file_breakdown = {}
    
    for root, dirs, files in os.walk('.'):
        # Remove excluded directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        for file in files:
            if Path(file).suffix in extensions:
                file_path = os.path.join(root, file)
                
                # Skip the BUILD_FROM_SCRATCH.md and verification scripts
                if file in ['BUILD_FROM_SCRATCH.md', 'compare_total_lines.py', 'verify_completeness.py', 'verify_core_fixed.py']:
                    continue
                
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = len(f.readlines())
                        total_lines += lines
                        total_files += 1
                        
                        ext = Path(file).suffix
                        if ext not in file_breakdown:
                            file_breakdown[ext] = {'files': 0, 'lines': 0}
                        file_breakdown[ext]['files'] += 1
                        file_breakdown[ext]['lines'] += lines
                        
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
                    continue
    
    return total_lines, total_files, file_breakdown

def count_guide_lines():
    """Count all lines of code embedded in BUILD_FROM_SCRATCH.md."""
    try:
        with open('BUILD_FROM_SCRATCH.md', 'r', encoding='utf-8') as f:
            content = f.read()
            total_guide_lines = len(content.split('\n'))
    except FileNotFoundError:
        print("❌ BUILD_FROM_SCRATCH.md not found!")
        return 0, 0, 0
    
    # Count code lines within code blocks
    code_lines = 0
    guide_files = 0
    in_code_block = False
    
    lines = content.split('\n')
    for line in lines:
        line = line.strip()
        
        # Start of code block
        if line.startswith("cat >") and "<<" in line:
            in_code_block = True
            guide_files += 1
            continue
        
        # End of code block
        if line == "EOF":
            in_code_block = False
            continue
        
        # Count code lines
        if in_code_block and line:  # Don't count empty lines
            code_lines += 1
    
    return total_guide_lines, code_lines, guide_files

def analyze_coverage():
    """Analyze what percentage of the project is covered."""
    # Count source code files only (excluding config files)
    source_extensions = {'.py', '.ts', '.tsx', '.js', '.jsx'}
    exclude_dirs = {'node_modules', '.git', '__pycache__', '.next', 'dist', 'build'}
    
    source_lines = 0
    source_files = 0
    
    for root, dirs, files in os.walk('.'):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        for file in files:
            if Path(file).suffix in source_extensions:
                file_path = os.path.join(root, file)
                
                # Skip verification scripts
                if file in ['compare_total_lines.py', 'verify_completeness.py', 'verify_core_fixed.py']:
                    continue
                
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = len(f.readlines())
                        source_lines += lines
                        source_files += 1
                except:
                    continue
    
    return source_lines, source_files

def main():
    print("🔍 COMPREHENSIVE PROJECT vs BUILD_FROM_SCRATCH.md COMPARISON")
    print("=" * 70)
    
    # Count original project lines
    print("📁 Analyzing Original Project...")
    project_lines, project_files, file_breakdown = count_project_lines()
    
    # Count guide lines
    print("📖 Analyzing BUILD_FROM_SCRATCH.md...")
    guide_total_lines, guide_code_lines, guide_files = count_guide_lines()
    
    # Count just source code
    source_lines, source_files = analyze_coverage()
    
    print(f"\n📊 COMPLETE COMPARISON RESULTS:")
    print("=" * 50)
    
    print(f"\n📁 ORIGINAL PROJECT (All Files):")
    print(f"   Total files: {project_files:,}")
    print(f"   Total lines: {project_lines:,}")
    print(f"   Source files only: {source_files:,}")  
    print(f"   Source lines only: {source_lines:,}")
    
    print(f"\n📖 BUILD_FROM_SCRATCH.md GUIDE:")
    print(f"   Guide total lines: {guide_total_lines:,}")
    print(f"   Code files included: {guide_files:,}")
    print(f"   Code lines embedded: {guide_code_lines:,}")
    
    print(f"\n🎯 COVERAGE ANALYSIS:")
    if project_files > 0:
        file_coverage = (guide_files / project_files) * 100
        print(f"   File coverage: {guide_files}/{project_files} = {file_coverage:.1f}%")
    
    if source_lines > 0:
        source_coverage = (guide_code_lines / source_lines) * 100
        print(f"   Source code coverage: {guide_code_lines:,}/{source_lines:,} = {source_coverage:.1f}%")
    
    if project_lines > 0:
        total_coverage = (guide_code_lines / project_lines) * 100
        print(f"   Total line coverage: {guide_code_lines:,}/{project_lines:,} = {total_coverage:.1f}%")
    
    print(f"\n📋 FILE TYPE BREAKDOWN (Original Project):")
    for ext, data in sorted(file_breakdown.items(), key=lambda x: x[1]['lines'], reverse=True):
        print(f"   {ext:<8}: {data['files']:>3} files, {data['lines']:>6,} lines")
    
    # Calculate efficiency
    if guide_total_lines > 0 and guide_code_lines > 0:
        efficiency = (guide_code_lines / guide_total_lines) * 100
        print(f"\n📈 GUIDE EFFICIENCY:")
        print(f"   Code density: {efficiency:.1f}% of guide is actual code")
        print(f"   Documentation: {100-efficiency:.1f}% is instructions/setup")
    
    # Final assessment
    print(f"\n🎯 FINAL ASSESSMENT:")
    
    if source_lines > 0:
        source_coverage = (guide_code_lines / source_lines) * 100
        if source_coverage >= 95:
            print("   ✅ EXCELLENT: BUILD_FROM_SCRATCH.md has comprehensive coverage")
            print("   ✅ Nearly all source code is included")
        elif source_coverage >= 80:
            print("   ✅ VERY GOOD: BUILD_FROM_SCRATCH.md has strong coverage")  
            print("   ✅ Most important source code is included")
        elif source_coverage >= 60:
            print("   ⚠️  GOOD: BUILD_FROM_SCRATCH.md has adequate coverage")
            print("   ⚠️  Core functionality is covered")
        else:
            print("   ❌ INSUFFICIENT: BUILD_FROM_SCRATCH.md needs more coverage")
    
    print(f"\n💡 SUMMARY:")
    print(f"   The BUILD_FROM_SCRATCH.md guide contains {guide_code_lines:,} lines")
    print(f"   of actual source code out of {source_lines:,} total source lines")
    print(f"   in the original project ({source_coverage:.1f}% coverage).")
    
    if source_coverage >= 80:
        print(f"   🎉 This is sufficient for complete project reconstruction!")
    else:
        missing_lines = source_lines - guide_code_lines
        print(f"   📝 Consider adding {missing_lines:,} more lines of code.")

if __name__ == "__main__":
    main()