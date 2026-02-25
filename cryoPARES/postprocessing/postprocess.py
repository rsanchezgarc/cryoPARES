"""
CryoPARES post-processing suite — single entry point with subcommand dispatch.

Usage
-----
    cryopares_postprocess bfactor --half1 h1.mrc --half2 h2.mrc --mask mask.mrc --output_dir out/
    cryopares_postprocess bfactor -h
    cryopares_postprocess -h
"""
import argparse

from argParseFromDoc import get_parser_from_function

from cryoPARES.postprocessing.methods.standard_bfactor import postprocess_bfactor

# Registry: name → callable.
# Each callable must have type-annotated parameters and :param: docstring entries
# so that get_parser_from_function() can auto-generate its CLI.
METHOD_REGISTRY = {
    "bfactor": postprocess_bfactor,
    # Future methods:
    # "localdeblur": postprocess_localdeblur,
}


def main():
    main_parser = argparse.ArgumentParser(
        prog="cryopares_postprocess",
        description="CryoPARES post-processing suite. "
                    "Choose a method subcommand (e.g. 'bfactor').")
    subparsers = main_parser.add_subparsers(
        dest="method", required=True,
        help="Post-processing method")

    for name, fn in METHOD_REGISTRY.items():
        first_doc_line = (fn.__doc__ or name).strip().splitlines()[0]
        sub = subparsers.add_parser(name, help=first_doc_line)
        get_parser_from_function(fn, parser=sub)

    args = main_parser.parse_args()
    fn = METHOD_REGISTRY[args.method]
    kwargs = {k: v for k, v in vars(args).items() if k != "method"}
    fn(**kwargs)


if __name__ == "__main__":
    main()
