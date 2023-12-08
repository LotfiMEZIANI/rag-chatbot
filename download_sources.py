import os
import shutil
import subprocess

from environements import sources_folder_path


def main():
    print("📢 script started !")

    doc_url = "https://nextjs.org/docs"
    include_directories = "/docs"

    if os.path.exists(sources_folder_path):
        print("🛠️ Cleaning sources folder...")

        try:
            shutil.rmtree(sources_folder_path)
            print("✅ Sources folder cleaned successfully")
        except Exception as e:
            print("❌ An error occurred: ", e)
            return

    command = f"wget --recursive --no-parent --convert-links --no-clobber --page-requisites --html-extension -e robots=off --include-directories={include_directories} --domains nextjs.org --no-check-certificate -P {sources_folder_path} {doc_url}"

    try:
        print(
            f"🛠️ Running command: {command}",
        )

        subprocess.run(command, shell=True, check=True)
        print("✅ Command executed successfully")

    except subprocess.CalledProcessError as e:
        print("❌ An error occurred: ", e)


if __name__ == "__main__":
    main()
