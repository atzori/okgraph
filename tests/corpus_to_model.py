import os
import argparse

from okgraph.file_converter import FileConverter


def convert(read_fname: str, save_fname: str):
    FileConverter.generate_gensim_model_from(read_fname=read_fname, save_fname=save_fname)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input",
        help="input file - corpus. A simple plain .txt or any compressed Gensim supported.",
        type=str,
        required=True)

    parser.add_argument(
        "-o",
        "--output",
        help="output file. If none, the input will be used appending .bin at the end.",
        type=str)

    args = parser.parse_args()
    input_file_path = os.path.expanduser(args.input)

    if not args.output:
        output_file_path = None
    else:
        output_file_path = os.path.expanduser(args.output)

    convert(input_file_path, output_file_path)
