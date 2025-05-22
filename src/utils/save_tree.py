import os


def save_directory_tree(
    root_dir,
    output_file,
    include_files=False,
    allowed_extensions=None,
    exclude_dirs=None,
):
    if exclude_dirs is None:
        exclude_dirs = []
    if allowed_extensions is not None:
        allowed_extensions = [
            ext.lower() for ext in allowed_extensions
        ]  # 大文字小文字を統一

    with open(output_file, "w", encoding="utf-8") as f:
        for line in generate_tree(
            root_dir, exclude_dirs, include_files, allowed_extensions
        ):
            f.write(line + "\n")


def generate_tree(
    current_dir, exclude_dirs, include_files, allowed_extensions, prefix=""
):
    lines = []
    try:
        entries = sorted(os.listdir(current_dir))
    except PermissionError:
        return []

    entries = [e for e in entries if not e.startswith(".")]  # 隠しファイル除外
    entries = [e for e in entries if e not in exclude_dirs]

    visible_entries = []
    for e in entries:
        full_path = os.path.join(current_dir, e)
        if os.path.isdir(full_path):
            visible_entries.append(e)
        elif include_files and allowed_extensions:
            if os.path.splitext(e)[1].lower() in allowed_extensions:
                visible_entries.append(e)
        elif include_files and not allowed_extensions:
            visible_entries.append(e)

    for i, entry in enumerate(visible_entries):
        full_path = os.path.join(current_dir, entry)
        connector = "└── " if i == len(visible_entries) - 1 else "├── "
        lines.append(prefix + connector + entry)

        if os.path.isdir(full_path):
            new_prefix = prefix + ("    " if i == len(visible_entries) - 1 else "│   ")
            lines.extend(
                generate_tree(
                    full_path,
                    exclude_dirs,
                    include_files,
                    allowed_extensions,
                    new_prefix,
                )
            )

    return lines


# 実行例
save_directory_tree(
    root_dir="src",  # 起点ディレクトリ
    output_file="project_structure.txt",  # 出力ファイル
    include_files=True,
    allowed_extensions=[".py", ".md", ".ckpt", "pth"],
    exclude_dirs=[
        ".venv",
        "UniFormer",
        "outputs",
        "tb_logs",
        "__pycache__",
        "smaples",
        ".git",
    ],  # 除外したいフォルダ名
)
