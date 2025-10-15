#!/usr/bin/env python3
"""
Accurate comparison excluding build artifacts and generated files
"""

import os
import re
from pathlib import Path

def count_actual_source_lines():
    """Count only actual source code files, excluding build artifacts."""
    # Include only meaningful source code extensions
    source_extensions = {'.py', '.ts', '.tsx', '.js', '.jsx'}
    config_extensions = {'.json', '.yaml', '.yml', '.md', '.css', '.scss'}
    
    # Exclude build and generated directories
    exclude_dirs = {
        'node_modules', '.git', '__pycache__', '.next', 'dist', 'build', 
        'coverage', '.nyc_output', 'temp', 'tmp'
    }
    
    # Exclude generated files
    exclude_files = {
        'package-lock.json', 'yarn.lock', '.eslintcache',
        'BUILD_FROM_SCRATCH.md', 'compare_total_lines.py', 'verify_completeness.py', 
        'verify_core_fixed.py', 'accurate_comparison.py'
    }
    
    source_stats = {'files': 0, 'lines': 0}
    config_stats = {'files': 0, 'lines': 0}
    file_details = []
    
    for root, dirs, files in os.walk('.'):
        # Remove excluded directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        for file in files:
            if file in exclude_files:
                continue
                
            file_path = os.path.join(root, file)
            file_ext = Path(file).suffix
            
            if file_ext in source_extensions:
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = len([line for line in f.readlines() if line.strip()])  # Skip empty lines
                        source_stats['files'] += 1
                        source_stats['lines'] += lines
                        file_details.append({
                            'path': file_path,
                            'type': 'source',
                            'ext': file_ext,
                            'lines': lines
                        })
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
                    continue
                    
            elif file_ext in config_extensions and not file.startswith('.'):
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = len([line for line in f.readlines() if line.strip()])
                        config_stats['files'] += 1
                        config_stats['lines'] += lines
                        file_details.append({
                            'path': file_path,
                            'type': 'config',
                            'ext': file_ext,
                            'lines': lines
                        })
                except Exception as e:
                    continue
    
    return source_stats, config_stats, file_details

def count_guide_code_accurately():
    """Count code lines in BUILD_FROM_SCRATCH.md more accurately."""
    try:
        with open('BUILD_FROM_SCRATCH.md', 'r', encoding='utf-8') as f:
            content = f.read()
            guide_total_lines = len(content.split('\n'))
    except FileNotFoundError:
        print("❌ BUILD_FROM_SCRATCH.md not found!")
        return 0, 0, 0, {}
    
    code_stats = {
        'total_files': 0,
        'total_code_lines': 0,
        'source_files': 0,
        'source_lines': 0,
        'config_files': 0,
        'config_lines': 0
    }
    
    lines = content.split('\n')
    in_code_block = False
    current_file = None
    current_lines = 0
    
    for line in lines:
        line_stripped = line.strip()
        
        # Start of code block
        if line_stripped.startswith("cat >") and "<<" in line_stripped:
            # Extract filename
            match = re.search(r'cat > ([^\s<]+)', line_stripped)
            if match:
                current_file = match.group(1)
                current_lines = 0
                in_code_block = True
                code_stats['total_files'] += 1
            continue
        
        # End of code block
        if line_stripped == "EOF":
            if current_file and current_lines > 0:
                file_ext = Path(current_file).suffix
                if file_ext in {'.py', '.ts', '.tsx', '.js', '.jsx'}:
                    code_stats['source_files'] += 1
                    code_stats['source_lines'] += current_lines
                elif file_ext in {'.json', '.yaml', '.yml', '.md', '.css'}:
                    code_stats['config_files'] += 1
                    code_stats['config_lines'] += current_lines
                
                code_stats['total_code_lines'] += current_lines
                
            in_code_block = False
            current_file = None
            current_lines = 0
            continue
        
        # Count non-empty code lines
        if in_code_block and line_stripped:
            current_lines += 1
    
    return guide_total_lines, code_stats

def main():
    print("🔍 ACCURATE PROJECT vs BUILD_FROM_SCRATCH.md COMPARISON")
    print("=" * 65)
    print("(Excluding build artifacts, generated files, and empty lines)")
    
    # Analyze original project
    print("\n📁 Analyzing Original Project (Source Code Only)...")
    source_stats, config_stats, file_details = count_actual_source_lines()
    
    # Analyze guide
    print("📖 Analyzing BUILD_FROM_SCRATCH.md...")
    guide_total_lines, guide_stats = count_guide_code_accurately()
    
    print(f"\n📊 ACCURATE COMPARISON RESULTS:")
    print("=" * 50)
    
    print(f"\n📁 ORIGINAL PROJECT (Actual Source Code):")
    print(f"   Source files (.py, .ts, .tsx, .js, .jsx): {source_stats['files']:,}")
    print(f"   Source lines (non-empty): {source_stats['lines']:,}")
    print(f"   Config files (.json, .yaml, .md, etc.): {config_stats['files']:,}")
    print(f"   Config lines (non-empty): {config_stats['lines']:,}")
    print(f"   TOTAL FILES: {source_stats['files'] + config_stats['files']:,}")
    print(f"   TOTAL LINES: {source_stats['lines'] + config_stats['lines']:,}")
    
    print(f"\n📖 BUILD_FROM_SCRATCH.md GUIDE:")
    print(f"   Guide total lines: {guide_total_lines:,}")
    print(f"   Source files included: {guide_stats['source_files']:,}")
    print(f"   Source lines included: {guide_stats['source_lines']:,}")
    print(f"   Config files included: {guide_stats['config_files']:,}")
    print(f"   Config lines included: {guide_stats['config_lines']:,}")
    print(f"   TOTAL CODE FILES: {guide_stats['total_files']:,}")
    print(f"   TOTAL CODE LINES: {guide_stats['total_code_lines']:,}")
    
    print(f"\n🎯 ACCURATE COVERAGE ANALYSIS:")
    
    total_project_files = source_stats['files'] + config_stats['files']
    total_project_lines = source_stats['lines'] + config_stats['lines']
    
    if total_project_files > 0:
        file_coverage = (guide_stats['total_files'] / total_project_files) * 100
        print(f"   File coverage: {guide_stats['total_files']}/{total_project_files} = {file_coverage:.1f}%")
    
    if source_stats['lines'] > 0:
        source_coverage = (guide_stats['source_lines'] / source_stats['lines']) * 100
        print(f"   Source code coverage: {guide_stats['source_lines']:,}/{source_stats['lines']:,} = {source_coverage:.1f}%")
    
    if total_project_lines > 0:
        total_coverage = (guide_stats['total_code_lines'] / total_project_lines) * 100
        print(f"   Total line coverage: {guide_stats['total_code_lines']:,}/{total_project_lines:,} = {total_coverage:.1f}%")
    
    # Show top source files by size
    source_files = [f for f in file_details if f['type'] == 'source']
    source_files.sort(key=lambda x: x['lines'], reverse=True)
    
    print(f"\n📋 TOP SOURCE FILES (Original Project):")
    for i, file_info in enumerate(source_files[:10], 1):
        print(f"   {i:2}. {file_info['path']:<40} {file_info['lines']:>4} lines")
    
    # Calculate efficiency
    if guide_total_lines > 0:
        efficiency = (guide_stats['total_code_lines'] / guide_total_lines) * 100
        print(f"\n📈 GUIDE EFFICIENCY:")
        print(f"   Code density: {efficiency:.1f}% of guide is actual code")
        print(f"   Documentation: {100-efficiency:.1f}% is instructions/setup")
    
    # Final assessment
    print(f"\n🎯 FINAL ASSESSMENT:")
    
    if source_stats['lines'] > 0:
        source_coverage = (guide_stats['source_lines'] / source_stats['lines']) * 100
        
        if source_coverage >= 85:
            print("   🎉 EXCELLENT: BUILD_FROM_SCRATCH.md has comprehensive coverage!")
            print("   ✅ Nearly all essential source code is included")
            status = "EXCELLENT"
        elif source_coverage >= 70:
            print("   ✅ VERY GOOD: BUILD_FROM_SCRATCH.md has strong coverage")  
            print("   ✅ Most important source code is included")
            status = "VERY GOOD"
        elif source_coverage >= 50:
            print("   ⚠️  GOOD: BUILD_FROM_SCRATCH.md has adequate coverage")
            print("   ⚠️  Core functionality is well covered")
            status = "GOOD"
        else:
            print("   ❌ INSUFFICIENT: BUILD_FROM_SCRATCH.md needs more coverage")
            status = "INSUFFICIENT"
    
    print(f"\n💡 FINAL SUMMARY:")
    print(f"   📊 Source Code Coverage: {source_coverage:.1f}%")
    print(f"   📄 Total Files: {guide_stats['total_files']} included out of {total_project_files}")
    print(f"   📝 Actual Code Lines: {guide_stats['total_code_lines']:,} out of {total_project_lines:,}")
    print(f"   🏆 Assessment: {status}")
    
    if source_coverage >= 70:
        print(f"   🎉 The guide contains sufficient code for project reconstruction!")
    elif source_coverage >= 50:
        print(f"   ⚡ The guide covers core functionality well!")
    else:
        missing_lines = source_stats['lines'] - guide_stats['source_lines']
        print(f"   📝 Consider adding {missing_lines:,} more source code lines.")

if __name__ == "__main__":
    main()