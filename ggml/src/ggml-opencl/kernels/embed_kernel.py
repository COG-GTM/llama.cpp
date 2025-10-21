#

import sys
import os
import logging
logger = logging.getLogger("opencl-embed-kernel")


def main():
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) != 3:
        logger.info("Usage: python embed_kernel.py <input_file> <output_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    if not os.path.isfile(input_file):
        logger.error(f"Input file does not exist: {input_file}")
        sys.exit(1)
    
    ifile = open(input_file, "r")
    ofile = open(output_file, "w")

    for i in ifile:
        ofile.write('R"({})"\n'.format(i))

    ifile.close()
    ofile.close()


if __name__ == "__main__":
    main()
