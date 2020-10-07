"""This module contains a script to generates the html documentation of the
project using the Sphinx library. It uses the source files from the 'source'
directory and generates the output in the 'build' directory.
"""
import os
from os import path
import shutil

if __name__ == "__main__":
    # Save the current working directory
    cwd_backup = path.normpath(os.getcwd())
    # Set the current working directory to be the one containing this script
    cwd = path.dirname(path.normpath(__file__))
    os.chdir(cwd)

    # Remove the old source files
    print("Removing previous source files from '/source'.")
    keep_source_files = ['conf.py', 'index.rst', 'README_link.rst', '_templates']
    for file in os.listdir("source"):
        if file not in keep_source_files:
            rel_name = f"source/{file}"
            print(f"Removing file {rel_name}.")
            os.remove(rel_name)

    # Create the new source files calling the "sphinx-apidoc" console command
    print("\nCreating new source files in '/source'.")
    os.system("sphinx-apidoc .. -o source --templatedir source/_templates/apidoc")

    # Remove the old build files and replace them with the new ones using "make"
    print("\nCleaning previous build with 'make'.")
    os.system("make clean")
    print("\nCreating new build with 'make'.")
    os.system("make html")

    # Copy the custom 'css' into the new 'html' build
    print("\nCopying the custom 'css' into the 'html' build.")
    shutil.copyfile(src="source/_templates/html/custom.css",
                    dst="build/html/_static/custom.css")

    # Customize the 'html' code with some changes
    print("\nCustomizing the 'html' output.")
    tmp_file = "build/html/tmp"
    for file in os.listdir("build/html"):
        # Search for the 'html' files in the new build
        if path.splitext(file)[1] == ".html":
            rel_name = f"build/html/{file}"
            print(f"Customizing {rel_name}.")
            with open(rel_name, "r", encoding='utf-8') as f_in:
                with open(tmp_file, "w", encoding='utf-8') as f_out:
                    # Apply to the text the needed modifications
                    for line in f_in:
                        # Having the ',' inside the 'em' tag allows to have the
                        # parameters (for functions, methods, constructors...)
                        # listed one per line
                        f_out.write(line.replace("</em>,", ",</em>"))
            shutil.copyfile(tmp_file, rel_name)
            os.remove(tmp_file)

    # Go back to the previous working directory
    # NOTE: possibly useless
    os.chdir(cwd_backup)
