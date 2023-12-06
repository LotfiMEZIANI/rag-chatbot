import subprocess


def main():
    doc_url = "https://nextjs.org/docs"
    include_directories = "/docs"
    folder_name = "sources"

    command = f"wget --recursive --no-parent --convert-links --no-clobber --page-requisites --html-extension -e robots=off --include-directories={include_directories} --domains nextjs.org --no-check-certificate -P {folder_name} {doc_url}"

    try:
        subprocess.run(command, shell=True, check=True)
        print("Command executed successfully")

    except subprocess.CalledProcessError as e:
        print("An error occurred: ", e)


if __name__ == "__main__":
    main()
