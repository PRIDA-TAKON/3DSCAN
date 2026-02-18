import os
from pathlib import Path

def build_kernel(kernel_dir, scripts, output_file):
    template_path = Path(kernel_dir) / "main.py"
    if not template_path.exists():
        print(f"❌ Template not found: {template_path}")
        return

    with open(template_path, "r", encoding="utf-8") as f:
        content = f.read()

    injection_logic = ""
    for script_name in scripts:
        script_path = Path("scripts") / script_name
        if not script_path.exists():
            print(f"⚠️ Script not found: {script_path}")
            continue
        
        with open(script_path, "r", encoding="utf-8") as f:
            script_content = f.read()
            # Escape triples and backslashes for python string
            escaped_content = script_content.replace("\\", "\\\\").replace("'''", "\\'\\'\\'").replace('"""', '\\"\\"\\"')
            injection_logic += f"\n# --- Injection of {script_name} ---\n"
            injection_logic += f"with open('{script_name}', 'w') as f:\n    f.write('''{escaped_content}''')\n"

    # We want to insert the injection logic near the top of main() or before main()
    if 'def install_dependencies():' in content:
        parts = content.split('def install_dependencies():')
        new_content = parts[0] + "def inject_scripts():\n" + "    print('💉 Injecting scripts...')\n" + injection_logic + "\n" + "def install_dependencies():" + parts[1]
        
        # Call inject_scripts() at the start of main()
        if "def main():" in new_content:
            parts_main = new_content.split("def main():")
            new_content = parts_main[0] + "def main():\n    inject_scripts()\n" + parts_main[1].split("\n", 1)[1]
    else:
        new_content = injection_logic + "\n" + content

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(new_content)
    print(f"✅ Merged kernel saved to: {output_file}")

if __name__ == "__main__":
    # Example: Build Kernel A
    build_kernel(
        "kernels/kernel_a_sfm", 
        ["step1_extract_frames.py", "step2_colmap_sfm.py"],
        "debug_kernel_a.py"
    )
