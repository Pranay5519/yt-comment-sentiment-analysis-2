import os

def save_folder_structure(root_folder, output_file):

    skip_folders = {
        "__pycache__", ".git", ".venv", "venv",
        ".mypy_cache", ".history", ".pytest_cache"
    }

    skip_extensions = {".log", ".pyc", ".pyo"}

    with open(output_file, "w", encoding="utf-8") as f:

        root_name = os.path.basename(root_folder)
        f.write(f"{root_name}/\n")

        for item in os.listdir(root_folder):

            item_path = os.path.join(root_folder, item)

            # Handle folders
            if os.path.isdir(item_path):

                if item in skip_folders:
                    continue

                f.write(f"    {item}/\n")

                # Only list files inside this folder (no deeper folders)
                for sub_item in os.listdir(item_path):

                    sub_path = os.path.join(item_path, sub_item)

                    if os.path.isfile(sub_path):

                        ext = os.path.splitext(sub_item)[1]
                        if ext in skip_extensions:
                            continue

                        f.write(f"        {sub_item}\n")

            # Handle root files
            else:
                ext = os.path.splitext(item)[1]

                if ext in skip_extensions:
                    continue

                f.write(f"    {item}\n")

    print(f"✅ Folder structure saved to {output_file}")


# Example usage
save_folder_structure(
    r"D:\yt-comment-sentiment-analysis2\src",
    "folder_structure.txt"
)