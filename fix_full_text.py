from pathlib import Path

# Read feature_engineer.py
with open('modules/feature_engineer.py', 'r') as f:
    lines = f.readlines()

# Find ClaimFeatures dataclass and add full_text after filename
new_lines = []
in_claim_features = False
full_text_added = False

for i, line in enumerate(lines):
    new_lines.append(line)
    
    # Detect start of ClaimFeatures
    if '@dataclass' in line and i+1 < len(lines) and 'class ClaimFeatures' in lines[i+1]:
        in_claim_features = True
    
    # Add full_text after filename
    if in_claim_features and not full_text_added and 'filename: str' in line:
        # Add full_text on next line
        indent = len(line) - len(line.lstrip())
        new_lines.append(' ' * indent + 'full_text: Optional[str] = None\n')
        full_text_added = True
        print(f"✅ Added full_text field after line {i+1}")

# Find extract_features method and ensure full_text is set
fixed_content = ''.join(new_lines)

# Update extract_features to include full_text
old_init = 'features = ClaimFeatures(filename=filename)'
new_init = 'features = ClaimFeatures(filename=filename, full_text=document_text)'

if old_init in fixed_content:
    fixed_content = fixed_content.replace(old_init, new_init)
    print("✅ Updated extract_features to preserve full_text")

# Save
with open('modules/feature_engineer.py', 'w') as f:
    f.write(fixed_content)

print("\n✅ Feature engineer fixed!")
print("Now re-run extraction with: python modules/feature_engineer.py")
