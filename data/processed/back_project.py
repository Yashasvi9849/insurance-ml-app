import shutil
from pathlib import Path
from datetime import datetime
import os


def backup_project():
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    project_root = Path.cwd()
    
    backup_name = f"rag-dev_backup_{timestamp}"
    backup_parent = project_root.parent / "backups"
    backup_parent.mkdir(exist_ok=True)
    
    backup_path = backup_parent / backup_name
    
    print("\n" + "="*80)
    print("PROJECT BACKUP")
    print("="*80 + "\n")
    
    print(f"📦 Creating backup: {backup_name}\n")
    
    exclude_dirs = {'venv', '__pycache__', '.git', 'node_modules', '.DS_Store'}
    exclude_files = {'.pyc', '.pyo', '.pyd', '.so', '.dylib'}
    
    important_items = [
        'modules/',
        'data/processed/',
        'models/',
        'config.py',
        'organize_all_docs.py',
        'run_full_pipeline.py',
        'train_simple.py',
        'predict_new.py',
        'quick_fix_target.py',
        'test_extraction.py',
        'requirements.txt'
    ]
    
    backup_path.mkdir(parents=True, exist_ok=True)
    
    items_backed_up = 0
    
    for item_name in important_items:
        source = project_root / item_name
        
        if not source.exists():
            continue
        
        if source.is_file():
            dest = backup_path / item_name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
            items_backed_up += 1
            print(f"✅ {item_name}")
            
        elif source.is_dir():
            dest = backup_path / item_name
            
            def ignore_patterns(dir, files):
                return [f for f in files if f in exclude_dirs or 
                       any(f.endswith(ext) for ext in exclude_files)]
            
            shutil.copytree(source, dest, ignore=ignore_patterns, dirs_exist_ok=True)
            items_backed_up += 1
            print(f"✅ {item_name}")
    
    backup_size = sum(f.stat().st_size for f in backup_path.rglob('*') if f.is_file())
    backup_size_mb = backup_size / (1024 * 1024)
    
    readme_path = backup_path / "BACKUP_INFO.txt"
    with open(readme_path, 'w') as f:
        f.write(f"Project Backup\n")
        f.write(f"="*50 + "\n\n")
        f.write(f"Backup Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Items Backed Up: {items_backed_up}\n")
        f.write(f"Total Size: {backup_size_mb:.2f} MB\n\n")
        f.write(f"To Restore:\n")
        f.write(f"1. Copy contents to a new 'rag-dev' folder\n")
        f.write(f"2. Create virtual environment: python3 -m venv venv\n")
        f.write(f"3. Activate: source venv/bin/activate\n")
        f.write(f"4. Install: pip install -r requirements.txt\n")
        f.write(f"5. Place PDFs in data/raw_pdfs/\n")
        f.write(f"6. Run: python run_full_pipeline.py\n")
    
    print("\n" + "="*80)
    print("✅ BACKUP COMPLETE")
    print("="*80)
    print(f"📁 Location: {backup_path}")
    print(f"📊 Size: {backup_size_mb:.2f} MB")
    print(f"📄 Items: {items_backed_up}")
    print(f"\n💡 Backup saved to: {backup_path.absolute()}")
    print("="*80 + "\n")


if __name__ == "__main__":
    backup_project()