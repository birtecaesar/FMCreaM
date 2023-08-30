import pandas as pd

from FamilyCreator import (
    find_and_filter_all_keys,
    get_element_tree,
    get_optionals,
    parse_xml_to_dict,
)


FEATURE_MODEL_NAME = "TuggerTrain"
INPUT_FILEPATH = f"./{FEATURE_MODEL_NAME}/feature_model/{FEATURE_MODEL_NAME}2.xml"
OUTPUT_FILEPATH = f"./{FEATURE_MODEL_NAME}/reconfiguration_sheet/{FEATURE_MODEL_NAME}_reconfiguration_sheet_template.csv"
DEFAULT_COLUMNS = [
    "#",
    "ContextInformation",
    "RelationshipType",
    "RelatedContextInformation",
    "Property",
    "ContextDomain",
    "Unit",
    "DataType",
]


def main():
    """
    This script reads all optional features from a feature model using functions from FamilyCreator.
    It creates an empty CSV file with all necessary default values and optional features in the header.
    The CSV file can then be used to set up the reconfiguration trigger.
    """
    # extract feature model from xml
    root1 = get_element_tree(INPUT_FILEPATH)
    f = parse_xml_to_dict(next(iter(root1)))
    fm1_keys = find_and_filter_all_keys(f)

    # determine optional features
    optionals = {}
    get_optionals(fm1_keys, f[next(iter(f))], optionals)

    # prepare imp and imp not columns for optional features
    optional_columns = sorted(
        [f"{o.split('#')[-1]} {i}" for o in optionals for i in ["(imp)", "(imp not)"]]
    )

    # create dataframe and convert to csv
    all_columns = DEFAULT_COLUMNS + optional_columns
    df = pd.DataFrame(columns=all_columns)
    df.to_csv(OUTPUT_FILEPATH, sep=";", index=False)


if __name__ in "__main__":
    main()
