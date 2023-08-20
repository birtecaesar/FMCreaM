import json
from collections import defaultdict

import numpy as np
import pandas as pd

from FamilyCreator import get_element_tree, parse_xml_to_dict, parse_constraints_to_dict

FEATURE_MODEL_NAME = "TuggerTrain"
FILENAME = (
    f"./{FEATURE_MODEL_NAME}/reconfiguration_sheet/{FEATURE_MODEL_NAME}_reconfiguration_sheet.csv"
)
INPUT_FILEPATH = f"./{FEATURE_MODEL_NAME}/feature_model/{FEATURE_MODEL_NAME}2.xml"
DEFAULT_FEATURES = [
    "#",
    "ContextInformation",
    "RelationshipType",
    "RelatedContextInformation",
    "Property",
    "ContextDomain",
    "Unit",
    "DataType",
]


def main() -> None:
    # extract feature model from xml
    root1 = get_element_tree(INPUT_FILEPATH)
    f = parse_xml_to_dict(next(iter(root1)))

    # read feature model reconfiguration_sheet
    df = pd.read_csv(FILENAME, sep=";")

    # create reconfiguration trigger information
    contexts = create_contexts(df)
    capabilities_contexts = create_capabilities_contexts(df)
    context_constraints = create_context_constraints(df)
    optionals = create_optional_features(df)
    root_constraint, root_f = create_root_constraint(f)
    optional_combination_constraints = create_optional_combination_constraints(
        f, root_f
    )
    capabilities_constraint = create_capabilities_constraint(df, root_f)

    imp_and_eq_constraints = create_imp_and_eq_constraints()

    # create collected dict of all necessary reconfiguration trigger information
    context_sensitive_feature_model_dict = {
        "attributes": [],
        "contexts": contexts + capabilities_contexts,
        "configuration": {
            "selected_features": [],
            "attribute_values": [],
            "context_values": [],
        },
        "constraints": root_constraint
        + optional_combination_constraints
        + context_constraints
        + imp_and_eq_constraints
        + capabilities_constraint,
        "optional_features": optionals,
        "preferences": [],
    }

    # write to json
    with open(
        f"{FEATURE_MODEL_NAME}/reconfiguration_sheet/{FEATURE_MODEL_NAME}_reconfiguration_trigger.json",
        "w",
    ) as wtj:
        json.dump(context_sensitive_feature_model_dict, wtj, indent=4)


def create_imp_and_eq_constraints() -> list[str]:
    constraints = parse_constraints_to_dict(INPUT_FILEPATH)
    imp_and_eq_constraints = []
    for con_type, cons in constraints.items():
        if con_type == "eq":
            for k, vs in cons.items():
                for v in vs:
                    imp_and_eq_constraints.append(
                        f"(feature[{k.split('#')[-1]}] = 1) impl (feature[{v.split('#')[-1]}] = 0)"
                    )
                    imp_and_eq_constraints.append(
                        f"(feature[{v.split('#')[-1]}] = 1) impl (feature[{k.split('#')[-1]}] = 0)"
                    )
        else:  # imp
            for k, vs in cons.items():
                imp_ors = [f"(feature[{v.split('#')[-1]}] = 1)" for v in vs]
                if len(imp_ors) == 1:
                    imp_and_eq_constraints.append(
                        f"(feature[{k.split('#')[-1]}] = 1) impl {' or '.join(imp_ors)}"
                    )
                else:
                    imp_and_eq_constraints.append(
                        f"(feature[{k.split('#')[-1]}] = 1) impl ({' or '.join(imp_ors)})"
                    )
    return imp_and_eq_constraints


def create_capabilities_contexts(df):
    context_domain = []
    capabilities_contexts = []
    for idx_row, row in df.iterrows():
        if row["ContextInformation"] == "Capabilities":
            context_domain = [
                f"context[{x.strip()}]" for x in row["ContextDomain"].split(",")
            ]
    if context_domain:
        for c in context_domain:
            capabilities_contexts.append({"id": c.split("#")[-1], "min": 0, "max": 1})
    return capabilities_contexts


def create_capabilities_constraint(df: pd.DataFrame, root_f: str) -> list[str]:
    capabilities_constraint = []
    for idx_row, row in df.iterrows():
        if row["ContextInformation"] in ["Capabilities", "Capability"]:
            context_domain = [
                f"(context[{x.strip()}] = 0)" for x in row["ContextDomain"].split(",")
            ]
            if not context_domain:
                print("Found no ContextDomain for ContextInformation 'Capabilities'.")
                continue
            if len(context_domain) == 1:
                capabilities_constraint.append(
                    f"{context_domain[0]} impl not feature[{root_f}]"
                )
            else:
                capabilities_constraint.append(
                    f"({' and '.join(context_domain)}) impl (feature[{root_f}] = 0)"
                )
    if not capabilities_constraint:
        print("Warning: No capabilities defined!")
    return capabilities_constraint


def create_optional_combination_constraints(f: dict, root_f: str) -> list[str]:
    optional_combinations = defaultdict(list)
    determine_optional_combinations(root_f, f, optional_combinations)
    optional_combination_constraints = []
    for parent, children in optional_combinations.items():
        children = [f"(feature[{x}] = 1)" for x in children]
        if len(children) == 1:
            optional_combination_constraints.append(
                f"{children[0]} impl feature[{parent}]"
            )
        else:
            optional_combination_constraints.append(
                f"({' or '.join(children)}) impl feature[{parent}] = 1"
            )
    return optional_combination_constraints


def create_root_constraint(f: dict) -> tuple[list[str], str]:
    root_f = next(iter(f)).split("#")[-1]
    root_constraint = [f"feature[{root_f}]"]
    return root_constraint, root_f


def determine_optional_combinations(
    root_f: str, parent_dict: dict, optional_combinations: defaultdict
) -> None:
    for p_name, p_values in parent_dict.items():
        if p_name == root_f:
            # root is not really optional, but optional children should point to it nevertheless
            parent_is_optional = True
        else:
            # further down the hierarchy we determine the optionality of both parent and child
            # only if both are optional, we add them to the reconfiguration trigger, because mandatory parents don't
            # have to be provided as a constraint. they are part of the configuration anyway.
            if "attribues" in p_values:
                parent_is_optional = not p_values["attributes"].get("mandatory", False)
            else:
                parent_is_optional = not p_values.get("mandatory", False)
        for c_name, c_values in p_values.get("children", {}).items():
            child_is_optional = not c_values["attributes"].get("mandatory", False)
            if child_is_optional and parent_is_optional:
                optional_combinations[p_name.split("#")[-1]].append(
                    c_name.split("#")[-1]
                )
            if c_values.get("children", {}):
                # recursively add optional parent child combinations
                determine_optional_combinations(root_f, c_values, optional_combinations)


def create_optional_features(df):
    optionals = {
        x.replace(" (imp)", "").replace(" (imp not)", ""): []
        for x in df.columns
        if x not in DEFAULT_FEATURES
    }
    return optionals


def create_context_constraints(df: pd.DataFrame) -> list[str]:
    context_constraints = []
    ranges = {0: [], 1: []}
    for idx_row, row in df.iterrows():
        # get row properties
        props = get_row_properties(row)

        # ignore rows where optional features are not filled
        if all(not isinstance(x, str) and np.isnan(x) for x in props["opt"].values()):
            print(
                f"{idx_row=}: no information of optional features provided -> row will be ignored."
            )
            continue

        # add context_constraints
        property_ = props["dflt"]["Property"]
        relationship_type = props["dflt"]["RelationshipType"]
        context_information = props["dflt"]["ContextInformation"]
        context_constraint_name = (
            property_
            if isinstance(property_, str)
            else relationship_type
            if isinstance(relationship_type, str)
            else context_information
            if isinstance(context_information, str)
            else ""
        )

        for opt_name, opt_val in props["opt"].items():
            if not isinstance(opt_val, str) and np.isnan(opt_val):
                continue

            if "(imp not)" in opt_name:
                impl_bool = 0
                opt_name = opt_name.replace(" (imp not)", "")
            elif "(imp)" in opt_name:
                impl_bool = 1
                opt_name = opt_name.replace(" (imp)", "")
            else:
                print(
                    "WARNING: You have to specify (imp) or (imp not) in your table header for optional features!"
                    f"I don't now how to handle {opt_name=} and have to continue without it."
                )
                continue

            # range
            # => we have to collect all ranges and split them later => see split_ranges_by_constraint_and_optionals()
            if "-" in opt_val:
                min_val, max_val = opt_val.split("-")
                ranges[impl_bool].append(
                    {
                        "context_constraint_name": context_constraint_name,
                        "opt_name": opt_name,
                        "min": min_val,
                        "max": max_val,
                    }
                )
                continue

            # unequal
            if opt_val.startswith("!="):
                opt_val = int(opt_val[2:])
                context_constraints.append(
                    f"context[{context_constraint_name}] != {opt_val} impl (feature[{opt_name}] = {impl_bool})"
                )
                continue

            # single value
            try:
                opt_val = int(opt_val)
                is_num = True
            except ValueError:
                is_num = False
            if is_num:
                context_constraints.append(
                    f"context[{context_constraint_name}] = {opt_val} impl (feature[{opt_name}] = {impl_bool})"
                )
                continue

            # comparator
            if any(s in opt_val for s in ["<", "<=", ">", ">=", " or "]):
                if "or" in opt_val:
                    lower, upper = opt_val.split("or")
                    lower = lower.strip()
                    upper = upper.strip()
                    for b_type, b in zip(["lower", "upper"], [lower, upper]):
                        comp_sign = (
                            ">="
                            if ">=" in b
                            else "<="
                            if "<=" in b
                            else ">"
                            if ">" in b
                            else "<"
                            if "<" in b
                            else ""
                        )
                        if not comp_sign:
                            print(
                                f"I could not convert value of {opt_name=}, {opt_val=}. "
                                f"Please check for typos and correct specification."
                            )
                            continue
                        val = b[len(comp_sign):]
                        context_constraints.append(
                            f"context[{context_constraint_name}] {comp_sign} {val} impl (feature[{opt_name}] = "
                            f"{impl_bool})"
                        )
                    continue
                else:
                    comp_sign = (
                        ">="
                        if ">=" in opt_val
                        else "<="
                        if "<=" in opt_val
                        else ">"
                        if ">" in opt_val
                        else "<"
                        if "<" in opt_val
                        else ""
                    )
                    if not comp_sign:
                        print(
                            f"I could not convert value of {opt_name=}, {opt_val=}. "
                            f"Please check for typos and correct specification."
                        )
                        continue
                    val = opt_val[len(comp_sign):]
                    context_constraints.append(
                        f"context[{context_constraint_name}] {comp_sign} {val} impl (feature[{opt_name}] = "
                        f"{impl_bool})"
                    )
                    continue

            # bool
            str_cons = opt_val.split(",")
            str_cons = [x.strip() for x in str_cons]
            context_domain = props["dflt"]["ContextDomain"].split(",")
            context_domain = [x.strip() for x in context_domain]
            or_constraints = []
            for con in str_cons:
                con = con.replace(" ", "")
                if con not in context_domain:
                    print(
                        f"WARNING: this constraint ({con}) is not an element of ContextDomain ({context_domain}) and "
                        f"cannot be set as constraint. Please check the specification. I will ignore this constraint."
                    )
                    continue
                or_constraints.append(f"context[{con}] = 1")
            if not or_constraints:
                continue
            if len(or_constraints) == 1:
                context_constraints.append(
                    f"{or_constraints[0]} impl (feature[{opt_name}] = {impl_bool})"
                )
            else:
                context_constraints.append(
                    f"({' or '.join(or_constraints)}) impl (feature[{opt_name}] = {impl_bool})"
                )

    split_ranges_by_constraint_and_optionals(context_constraints, ranges)
    return context_constraints


def split_ranges_by_constraint_and_optionals(
    context_constraints: list[str], ranges: dict[int, list[dict]]
) -> None:
    """this function determines the full range of all optional features for each constraint and splits these ranges
    according to validity of each optional feature.

    example for hasHeight:
    ======================
    opt_a = (5,40)
    opt_b = (10, 50)
    =>
    "((context[hasHeight] >= 5) and (context[hasHeight] <= 9)) impl ((feature[opt_a] = 1))",
    "((context[hasHeight] >= 10) and (context[hasHeight] <= 40)) impl ((feature[opt_a] = 1) or (feature[opt_b] = 1))",
    "((context[hasHeight] >= 41) and (context[hasHeight] <= 50)) impl ((feature[opt_a] = 1))",
    """
    for impl_bool, ranges_by_bool in ranges.items():
        ranges_by_constraint = defaultdict(dict)
        for rbb in ranges_by_bool:
            ranges_by_constraint[rbb["context_constraint_name"]].update(
                {rbb["opt_name"]: {"min": rbb["min"], "max": rbb["max"]}}
            )
        for constraint, constraint_ranges in ranges_by_constraint.items():
            if len(constraint_ranges) == 1:
                opt_f = next(iter(constraint_ranges))
                range_vals = constraint_ranges[opt_f]
                context_constraints.append(
                    f"((context[{constraint}] >= {range_vals['min']}) and "
                    f"(context[{constraint}] <= {range_vals['max']})) impl"
                    f"(feature[{opt_f}] = {impl_bool})"
                )
            else:
                min_of_mins = min([int(v["min"]) for v in constraint_ranges.values()])
                max_of_maxs = max([int(v["max"]) for v in constraint_ranges.values()])
                range_dict = {n: set() for n in np.arange(min_of_mins, max_of_maxs + 1)}
                for opt_f, range_vals in constraint_ranges.items():
                    for nn in np.arange(
                        int(range_vals["min"]), int(range_vals["max"]) + 1
                    ):
                        range_dict[nn].add(opt_f)
                range_sections = {tuple(x) for x in range_dict.values()}
                for rs in range_sections:
                    rs = [x for x in rs if x]
                    nums = [
                        num
                        for num, opt_fs in range_dict.items()
                        if sorted(opt_fs) == sorted(rs)
                    ]
                    opt_fs_constraint = [
                        f"(feature[{opt_f}] = {impl_bool})" for opt_f in rs
                    ]
                    context_constraints.append(
                        f"((context[{constraint}] >= {min(nums)}) and "
                        f"(context[{constraint}] <= {max(nums)})) impl"
                        f"({' or '.join(opt_fs_constraint)})"
                    )


def create_contexts(df: pd.DataFrame) -> list[dict]:
    contexts = []
    for idx_row, row in df.iterrows():
        # get row properties
        props = get_row_properties(row)

        # ignore rows where optional features are not filled
        if all(not isinstance(x, str) and np.isnan(x) for x in props["opt"].values()):
            print(
                f"{idx_row=}: no information in CylindricalWorkpieceAdapter, RectangleWorkpieceAdapter, and "
                f"WorkpieceAdapterPlate provided."
            )
            continue

        # add context
        context_name = (
            f"context[{props['dflt']['Property']}]"
            if isinstance(props["dflt"]["Property"], str)
            else f"context[{props['dflt']['RelationshipType']}]"
            if isinstance(props["dflt"]["RelationshipType"], str)
            else f"context[{props['dflt']['ContextInformation']}]"
        )

        if "-" in props["dflt"]["ContextDomain"]:
            # range
            v_min, v_max = props["dflt"]["ContextDomain"].split("-")
            try:
                v_min = int(v_min)
                v_max = int(v_max)
            except Exception as e:
                print(
                    "Cannot convert values {v_min=} and/or {v_max=} to integer. "
                    "Please specify differently. "
                    "I'll continue without this feature."
                    f"({e!r})"
                )
                continue
            contexts.append({"id": context_name, "min": v_min, "max": v_max})
        elif "True" in props["dflt"]["ContextDomain"]:
            # bool
            contexts.append({"id": context_name, "min": 0, "max": 1})
        else:
            # enum
            enums = props["dflt"]["ContextDomain"].split(",")
            for enum in enums:
                context_name_enum = f"context[{enum.strip()}]"
                contexts.append({"id": context_name_enum, "min": 0, "max": 1})

    return contexts


def get_row_properties(row):
    props = {"dflt": {}, "opt": {}}
    for col in row.index:
        col = col.strip()
        if col in DEFAULT_FEATURES:
            props["dflt"].update({col: row[col]})
        else:
            props["opt"].update({col: row[col]})
    return props


if __name__ in "__main__":
    main()
