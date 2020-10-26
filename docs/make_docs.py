"""This module contains a script to generates the html documentation of the
project using the Sphinx library. It uses the basic source files from the
'source' directory ('conf.py', 'index.rst', 'README_link.rst', '_templates/')
and generates the output in the 'source' and 'build' directory.

The basic source files are:
 - *conf.py*: is the file containing the Sphinx configuration.
 - *index.rst*: is the file containing the documentation main page structure.
 - *README_link.rst*: is the file containing the link to the README.md file,
   that allows his conversion and representation in the 'rst' format.
 - *_templates/*: is a folder containing the templates used in the automatic
   generation of the other 'rst' files of all the modules and packages, and the
   custom 'css' for the 'html' documentation.
"""
import os
from os import listdir, path, remove, system
import shutil
import sphinx.ext.apidoc


def main():
    # Save the current working directory
    _cwd_backup = path.normpath(os.getcwd())
    # Set the current working directory to be the one containing this script
    cwd = path.dirname(path.normpath(__file__))
    os.chdir(cwd)

    print(f"\nRunning script '{path.normpath(__file__)}' from '{cwd}'.")

    project_dir = path.normpath("..")
    source_dir = path.normpath("source")
    build_dir = path.normpath("build")

    # Remove the old source files
    print(f"Removing previous source files from '{source_dir}'.")
    keep_source = \
        [f"{path.normpath(path.join(source_dir, name))}" for name in
         ['conf.py', 'index.rst', 'README_link.rst', '_templates']]
    for name in listdir(source_dir):
        abs_name = f"{path.normpath(path.join(source_dir, name))}"
        if abs_name not in keep_source:
            print(f"Removing file {abs_name}.")
            remove(abs_name)

    # Create the new source files calling the "sphinx-apidoc"
    print(f"\nCreating new source files in '{source_dir}'.")

    sphinx.ext.apidoc.main([
        f"{project_dir}",
        f"-o",  f"{source_dir}",
        f"--templatedir", f"{path.join(source_dir, '_templates', 'apidoc')}"
    ])

    # Remove the old build files and replace them with the new ones using "make"
    print("\nCleaning previous build with 'make'.")
    system("make clean")
    print("\nCreating new build with 'make'.")
    system("make html")

    # Copy the custom 'css' into the new 'html' build
    print("\nCopying the custom 'css' into the 'html' build.")
    shutil.copyfile(
        src=f"{path.join(source_dir, '_templates', 'html', 'custom.css')}",
        dst=f"{path.join(build_dir, 'html', '_static', 'custom.css')}"
    )

    # Customize the 'html' code with some changes
    print("\nCustomizing the 'html' output.")
    tmp_file = f"{path.join(build_dir, 'html', 'tmp')}"
    build_html_dir = f"{path.join(build_dir, 'html')}"
    for name in listdir(build_html_dir):
        abs_name = path.join(build_html_dir, name)
        # Search for the 'html' files in the new build
        if path.splitext(abs_name)[1] == ".html":
            print(f"Customizing {abs_name}.")
            with open(abs_name, "r", encoding='utf-8') as f_in:
                with open(tmp_file, "w", encoding='utf-8') as f_out:
                    # Apply to the text the needed modifications
                    for line in f_in:
                        # Having the ',' inside the 'em' tag allows to have the
                        # parameters (for functions, methods, constructors...)
                        # listed one per line
                        f_out.write(line.replace("</em>,", ",</em>"))
            shutil.copyfile(tmp_file, abs_name)
            remove(tmp_file)

    # Go back to the previous working directory
    # NOTE: possibly useless
    os.chdir(_cwd_backup)

    print(f"Ended script '{path.normpath(__file__)}'.\n")


if __name__ == "__main__":
    main()
