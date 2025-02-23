import numpy

from dolo.misc.display import read_file_or_url       # Handle local/remote model files (snt3p5)
import yaml


def yaml_import(fname, check=True, check_only=False):

    txt = read_file_or_url(fname)                    # Load model spec from file/URL (snt3p5)

    try:
        data = yaml.compose(txt)                     # Parse YAML to AST for validation (snt3p5)
        # print(data)
        # return data
    except Exception as ex:
        print(
            "Error while parsing YAML file. Probable YAML syntax error in file : ",
            fname,
        )
        raise ex

    # if check:
    #     from dolo.linter import lint
    #     data = ry.load(txt, ry.RoundTripLoader)
    #     output = lint(data, source=fname)
    #     if len(output) > 0:
    #         print(output)

    # if check_only:
    #     return output

    data["filename"] = fname                         # Store source file info (snt3p5)

    from dolo.compiler.model import Model

    return Model(data, check=check)                  # Create model from YAML spec (snt3p5)
    # from dolo.compiler.model import SymbolicModel
