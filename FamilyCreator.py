import copy
import os
import sys
import xml.etree.ElementTree as eT
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from xml.etree.ElementTree import Element


INPUT_DIR = "./SoftGripper/input/"
OUTPUT_DIR = "./SoftGripper/output/"


def get_element_tree(file: str) -> "Element":
    print(f"getting element tree for {file=}")
    tree = eT.parse(file)
    root = tree.getroot()
    return root


def parse_xml_to_dict(element):
    """recursively loop through xml elements and extract information into a dictionary"""

    def _add_child_to_result(data, name, result_dict):
        """helper function to add children to the result dict"""
        if name in result_dict:
            if isinstance(result_dict[name], list):
                result_dict[name].append(data)
            else:
                result_dict[name] = [result_dict[name], data]
        else:
            result_dict.update(data)

    result = {}
    if "name" in element.attrib:
        # all features have a name attribute
        attributes = {"tag": element.tag}
        if element.attrib.get("mandatory", "not_set") == "true":
            attributes["mandatory"] = True
        if element.attrib.get("abstract", "not_set") == "true":
            attributes["abstract"] = True
        if len(element) == 0:
            return (
                {element.attrib["name"]: {"attributes": attributes, "children": result}}
                if attributes
                else {}
            )
        for child in element:
            child_data = parse_xml_to_dict(child)
            child_name = child.attrib.get("name")
            if child_name is None:
                continue
            _add_child_to_result(child_data, child_name, result)
        return (
            {element.attrib["name"]: {"attributes": attributes, "children": result}}
            if result
            else {}
        )
    elif element.tag == "struct":
        # struct is the wrapper around features
        # we are interested in the features, which are children of struct, hence we continue with the first entry of
        # struct (struct only has one entry! -> Which is the root feature. Feature models always have just one root
        # feature!)
        try:
            child = next(iter(element))
        except Exception as e:
            print(f"Feature model seems to be empty. Pleas check. {e!r}")
            sys.exit()
        child_data = parse_xml_to_dict(child)
        child_name = child.attrib.get("name")
        if child_name is None:
            return result if result else None
        _add_child_to_result(child_data, child_name, result)
        return result if result else None
    else:
        raise ValueError("I could not parse xml file and have to return empty-handed!")


def parse_constraints_to_dict(filename: str) -> dict[str, dict]:
    all_con_strs = {}
    con_str = []
    n = 1
    for event, elem in eT.iterparse(filename, events=["start", "end"]):
        if event == "start":
            if elem.tag in ["var", "not"]:
                con_str.append(elem.tag)
        elif event == "end":
            if elem.text:
                # ignore empty lines with strip
                text = elem.text.strip()
                if text:
                    con_str.append(text)
                    all_con_strs[n] = con_str
                    con_str = []
                    n += 1

    implies = {}
    excludes = {}
    for idx, con in all_con_strs.items():
        if idx % 2 == 0:
            # skip requiree (value), only add constraint by requirer (key)
            continue
        requiree = all_con_strs[idx + 1]
        k = con[-1]
        v = requiree[-1]
        if "not" in requiree:
            if k in excludes:
                excludes[k].append(v)
            else:
                excludes[k] = [v]
        else:
            if k in implies:
                implies[k].append(v)
            else:
                implies[k] = [v]

    return {"eq": excludes, "imp": implies}


def get_feature_element(d: dict, e: str, k: str = "") -> dict:
    """get entry (element) for given dict and key
    if no key is provided, use the first dict key
    if entry (element) is empty return an empty dict"""
    if not k:
        return d[next(iter(d))].get(e, {})
    else:
        return d.get(k, {}).get(e, {})


def compare_tag_attributes(a1: dict, a2: dict) -> str:
    if a1["tag"] == a2["tag"]:
        # case 1: both tags are identical
        tag = a1["tag"]
    else:
        if any("and" in a["tag"] for a in [a1, a2]):
            # case 2: any tag="and" => "and"
            tag = "and"
        elif any("or" in a["tag"] for a in [a1, a2]):
            # case 3: not any tag="and", but any tag="or" => "or"
            tag = "or"
        else:
            raise ValueError(
                f"Cannot handle this combination: {a1['tag']} vs {a2['tag']}"
            )
    return tag


def determine_system_and_attributes(
    all_systems, attrs1, attrs2, children1, children2, fm_dict1, fm_dict2
):
    system = next(iter(fm_dict1))
    if len(all_systems) == 1:
        print("...both feature models have the same root feature name")
        # both feature models have the same root feature name
        if sorted(children1) == sorted(children2):
            # case 1: both root features have exactly the same children (on the first child level)
            # => booth root features are identical
            print(
                "...entering case 1: both root features have exactly the same children (on the first child "
                "level)"
            )
            attrs = {
                "tag": compare_tag_attributes(attrs1, attrs2),
                "abstract": True,
            }
        else:
            # case 2: both root features have differences in their children (on the first child level)
            # => we check if first level group tag is an alternative, then we assume that some disjoint fms
            # were merged before
            print(
                "...entering case 2: both root features have differences in their children (on the first child "
                "level)"
            )
            if any(attr == "alt" for attr in (attrs1["tag"], attrs2["tag"])):
                attrs = {"tag": "alt", "abstract": True}
            # if no alternative tag is used, we apply the tag compare-rules
            else:
                attrs = {
                    "tag": compare_tag_attributes(attrs1, attrs2),
                    "abstract": True,
                }
    elif len(all_systems) == 2:
        print("...the feature models have different root feature names")
        # the feature models have different root feature names
        if sorted(children1) == sorted(children2):
            # case 1: both root features have exactly the same children (on the first child level)
            # => we assume that the root features are identical, but have different names
            print(
                "...entering case 1: both root features have exactly the same children (on the first child "
                "level)"
            )
            attrs = {
                "tag": compare_tag_attributes(attrs1, attrs2),
                "abstract": True,
            }
        else:
            # case 2: both root features have differences in their children (on the first child level)
            # => we check if first level group tag is an alternative, then we assume that some disjoint fms were
            # merged before
            print(
                "...entering case 2: both root features have differences in their children (on the first child "
                "level)"
            )
            system2 = next(iter(fm_dict2))
            if system in fm_dict2:
                attrs = attrs2
            elif system2 in fm_dict1:
                attrs = attrs1
            else:
                # => we have to check all children independently
                # ToDo: check the tag for children (and vs or...)
                attrs = {}
                print(
                    "WARNING: both root features have different names and different children on first child "
                    "level. "
                    "This case can not be handled yet."
                )
    else:
        raise ValueError(
            f"Cannot merge empty feature model and more than two should not exist, but got {all_systems=}"
        )
    return attrs, system


def get_all_keys(d):
    """extract all existing features and attribute keys"""
    for key, value in d.items():
        yield key
        if isinstance(value, dict):
            yield from get_all_keys(value)


def find_and_filter_all_keys(fm_dict):
    """filter all keys to receive features itself, not meta information of the features"""
    fm_keys = []
    for x in get_all_keys(fm_dict):
        not_relevant_list = [
            "and",
            "tag",
            "abstract",
            "mandatory",
            "attributes",
            "children",
        ]
        if x in not_relevant_list:
            continue
        if x not in fm_keys:
            fm_keys.append(x)
    return fm_keys


def compare_mandatory_attributes(a1: dict, a2: dict) -> Optional[bool]:
    mandatory1, mandatory2 = (d.get("mandatory", "") for d in [a1, a2])
    if mandatory1 != mandatory2 or not any(m for m in [mandatory1, mandatory2]):
        # case 1: mandatory attrs are either different or not set
        mandatory = None
    else:
        # case 2: both mandatory attrs are equal
        mandatory = True
    return mandatory


def get_avail_keys(d: dict, keys: set) -> set:
    for k, v in d.items():
        if k == "attributes":
            continue
        if k != "children":
            keys.add(k)
        if isinstance(v, dict):
            get_avail_keys(v, keys)
    return keys


def add_children_to_lowest_available_level(d: dict, f: str, c: dict, a_k: set) -> None:
    """
    Recursively search for lowest branch for feature (f) in root dict (d)
    - if the feature (f) is not yet known in the dict (d), i.e., not in avail_keys (a_k),
      add it directly to the root dict (d)
    - if the feature (f) is in available keys (a_k) but not in the highest branch, recursively search for the lowest
      branch and add the child dict (c) to d[...][f]["children"].
    """
    if f not in a_k:
        d[f] = c
    else:
        if f in d:
            d[f]["children"].update(c)
        else:
            for k, v in d.items():
                add_children_to_lowest_available_level(d[k]["children"], f, c, a_k)


def recursively_merge_children(
    c1: dict, c2: dict, results: dict, feature_name: str
) -> None:
    merged_child_dict = merge_children(c1, c2)
    add_children_to_lowest_available_level(
        results, feature_name, merged_child_dict, get_avail_keys(results, set())
    )
    for feature, values in merged_child_dict.items():
        children1, children2 = (
            get_feature_element(d, "children", feature) for d in [c1, c2]
        )
        if not children1 and not children2:
            # we reached the last level of this branch, no further recursion required
            continue
        else:
            recursively_merge_children(children1, children2, results, feature)


def merge_children(c1: dict, c2: dict) -> dict:
    """compare and merge children_dicts (c1 and c2) and update parent_dict (p)"""
    result_dict = {}
    all_occurrences = set(list(c1) + list(c2))
    duplicate_children = set(c1).intersection(set(c2))
    unique_children = all_occurrences.difference(duplicate_children)
    for child in duplicate_children:
        child1_attrs, child2_attrs = (
            get_feature_element(d, "attributes", child) for d in [c1, c2]
        )
        if child1_attrs == child2_attrs:
            # use attrs of first child
            attrs = child1_attrs
        else:
            # compare tags
            child_tag = compare_tag_attributes(child1_attrs, child2_attrs)
            child_abstract = True if child_tag != "feature" else None
            child_mandatory = compare_mandatory_attributes(child1_attrs, child2_attrs)
            # fill attrs
            attrs = {"tag": child_tag}
            if child_abstract:
                attrs.update({"abstract": child_abstract})
            if child_mandatory:
                attrs.update({"mandatory": child_mandatory})
        result_dict.update({child: {"attributes": attrs, "children": {}}})
    for child in unique_children:
        fm1_names, fm2_names = (find_and_filter_all_keys(c) for c in [c1, c2])
        really_unique = not (child in fm1_names and child in fm2_names)
        if really_unique:
            # case 1: child only occurs in one feature model disregarding the parent feature
            child1_attrs, child2_attrs = (
                get_feature_element(d, "attributes", child) for d in [c1, c2]
            )
            child_attrs = child1_attrs if child1_attrs else child2_attrs
            attrs = {k: v for k, v in child_attrs.items() if k != "mandatory"}
            result_dict.update({child: {"attributes": attrs, "children": {}}})
        else:
            print("WARNING: this case is not yet implemented and will be ignored!")
            # case 2: child occurs in both feature models, but has a different parent
            # ToDo: not yet clear how to handle this case, needs further discussion
            pass
    return result_dict


def create_constraints(
    f1: dict, f2: dict, f1con: dict, f2con: dict, breakpoint: bool
) -> dict:
    """get features that only exist in feature model 1 (f1) and features that only exist in feature model 2 (f2)
    if there are no such features no constraints will be added
    if there exists these features, we will create requires or exclude constraints (con) to maintain only the variants
    that exist at the current state of time
    """
    print("creating constraints...")

    # --- create diff_comps --------------------------------------------------------------------------------------------
    # system/module/component is not included in both fms
    fm1_keys = find_and_filter_all_keys(f1)
    fm2_keys = find_and_filter_all_keys(f2)
    diff_comp_in_fm1, diff_comp_in_fm2 = create_diff_comps(fm1_keys, fm2_keys)
    print(f"...done creating diff_comps: {diff_comp_in_fm1=}, {diff_comp_in_fm2=}")
    # ------------------------------------------------------------------------------------------------------------------

    # --- create excludes ----------------------------------------------------------------------------------------------
    # create constraints so that only the two variants compared can be selected + the variants that were already merged
    excludes = create_excludes_constraints(diff_comp_in_fm1, diff_comp_in_fm2)
    print(f"...done creating excludes from diff_comps => {excludes=}")
    for fcon in [f1con, f2con]:
        update_dict_with_ancient_knowledge(excludes, fcon, "eq")
    print(f"...done updating dict with ancient knowledge (eq) => {excludes=}")
    # ------------------------------------------------------------------------------------------------------------------

    # --- create mandatories -------------------------------------------------------------------------------------------
    # find all mandatory connections between features
    mandatories = {}
    for f in [f1, f2]:
        f_root = f[next(iter(f))]
        get_mandatories(diff_comp_in_fm1, diff_comp_in_fm2, f_root, mandatories)
    print(f"...done creating {mandatories=}")
    # ------------------------------------------------------------------------------------------------------------------

    if breakpoint:
        print("breakpoint")

    # --- create requires ----------------------------------------------------------------------------------------------
    # requires constraints between the features of fm 1, which do not occur in fm 2 -> same is applied to fm 2
    requires = {}
    mandatory_chain = {}
    for diff_comp in [diff_comp_in_fm1, diff_comp_in_fm2]:
        create_requires_constraints(diff_comp, mandatories, requires, mandatory_chain)
    print(f"...done creating requires from diff_comps => {requires=}")
    # _check_violation_of_excludes_cons_by_requires_cons(excludes, requires)
    for fcon in [f1con, f2con]:
        update_dict_with_ancient_knowledge(requires, fcon, "imp")
    print(f"...done updating dict with ancient knowledge (imp) => {requires=}")
    # ------------------------------------------------------------------------------------------------------------------

    if breakpoint:
        print("breakpoint")

    # --- filter requires ----------------------------------------------------------------------------------------------
    combined_diff_comp = diff_comp_in_fm1 + diff_comp_in_fm2
    requires = clean_up_requires_according_to_diff_comp(combined_diff_comp, requires)
    requires = _drop_duplicate_constraints(requires, "requires")
    # ------------------------------------------------------------------------------------------------------------------

    # --- filter excludes ----------------------------------------------------------------------------------------------
    excludes = clean_up_excludes_according_to_requires(excludes, requires)
    excludes = _drop_duplicate_constraints(excludes, "excludes")
    excludes = clean_up_excludes_according_to_mandatory_chain(excludes, mandatory_chain)
    excludes = clean_up_excludes_according_to_features_in_fm2(excludes, f2)
    # ------------------------------------------------------------------------------------------------------------------

    # --- create con ---------------------------------------------------------------------------------------------------
    con = {"eq": excludes}
    con.update({"imp": requires})
    print(f"...done creating constraints => {con=}")
    # ------------------------------------------------------------------------------------------------------------------

    if breakpoint:
        print("breakpoint")
    return con


def create_diff_comps(fm1_keys, fm2_keys):
    """determine unique features of fm1 and fm2"""
    diff_comp_in_fm1 = [x for x in fm1_keys if x not in fm2_keys]
    diff_comp_in_fm2 = [y for y in fm2_keys if y not in fm1_keys]
    return sorted(diff_comp_in_fm1), sorted(diff_comp_in_fm2)


def create_excludes_constraints(diff_comp_in_fm1, diff_comp_in_fm2):
    """assign all features of fm2 to each feature of fm1 and vice versa"""
    excludes = {diff: diff_comp_in_fm2.copy() for diff in diff_comp_in_fm1}
    excludes.update({diff: diff_comp_in_fm1.copy() for diff in diff_comp_in_fm2})
    return excludes


def clean_up_requires_according_to_diff_comp(full_diff_comp, requires):
    """if a key is in both fms, this means that it is not unique and therefore not in diff_comp. but if the feature
    that is required by this key is unique (hence in diff_comp) then it has to be removed from the requirements,
    because a feature that is in both features models cannot require a feature that is not present in both feature
    models => in that case, the constraint is not valid anymore."""
    updated_requires = {}
    for k, v in requires.items():
        if k not in full_diff_comp and any(vv in full_diff_comp for vv in v):
            to_add = sorted([vv for vv in v if vv not in full_diff_comp])
        else:
            to_add = v
        if to_add:
            if k in updated_requires:
                updated_requires[k] += to_add
            else:
                updated_requires[k] = to_add
    print(
        f"...done cleaning up requires according to diff_comp_in_fm => {updated_requires=}"
    )
    return updated_requires


def update_dict_with_ancient_knowledge(d, fcon, con_type):
    """add information of fm1 to requires or excludes depending on con_type"""
    if not fcon:
        return
    for k, v in fcon.get(con_type, {}).items():
        if k in d:
            d[k] += v
        else:
            d[k] = v


def _drop_duplicate_constraints(constraints: dict, con_type: str) -> dict:
    print(f"...dropping duplicates in {con_type}")
    clean_constraints = {}
    pairs = []
    for key, value in constraints.items():
        if isinstance(value, list):
            for single_value in value:
                pairs.append((key, single_value))
        else:
            pairs.append((key, value))
    if con_type == "excludes":
        unique_pairs = set(tuple(sorted(x)) for x in pairs)
    else:
        unique_pairs = set(pairs)
    for pair in sorted(unique_pairs):
        key = pair[0]
        value = pair[1]
        if key in clean_constraints:
            clean_constraints[key].append(value)
        else:
            clean_constraints.update({key: [value]})
    print(f"...done dropping duplicates in {con_type} => {clean_constraints=}")
    return clean_constraints


def clean_up_excludes_according_to_requires(excludes, requires):
    print(f"...cleaning up excludes according to requires...")
    updated_excludes = copy.deepcopy(excludes)
    for req_k, req_vs in requires.items():
        print(f"...working on {req_k=}")
        for req_v in req_vs:
            to_remove = []
            if all(req in updated_excludes for req in [req_k, req_v]):
                ex_vs_of_req_k = excludes[req_k]
                ex_vs_of_req_v = excludes[req_v]
                if ex_vs_of_req_k == ex_vs_of_req_v:
                    # if both lists are equal remove the whole entry of req_k from excludes
                    updated_excludes.pop(req_k, None)
                    to_remove = [req_k]
                    print(f"...removing keys {', '.join(to_remove)}")
                elif any(x in ex_vs_of_req_v for x in ex_vs_of_req_k):
                    # if any entry of the lists is equal, remove this entry from the values of excludes
                    duplicates = {x for x in ex_vs_of_req_k if x in ex_vs_of_req_v}
                    to_remove = [x for x in updated_excludes[req_k] if x in duplicates]
                    updated_excludes[req_k] = [
                        x for x in updated_excludes[req_k] if x not in duplicates
                    ]
                    print(f"...removing values {', '.join(to_remove)}")
                if to_remove:
                    # if we removed a key or a value previously, we also want to update the remaining counter-part
                    # constraints
                    print(f"...updating counter-parts of {', '.join(to_remove)}")
                    for ex_v_of_req_k in ex_vs_of_req_k:
                        if ex_v_of_req_k in updated_excludes:
                            updated_excludes[ex_v_of_req_k] = [
                                x for x in updated_excludes[ex_v_of_req_k] if x != req_k
                            ]
    # updated_excludes = {k: v for k, v in updated_excludes.items() if k not in requires}
    print(f"...done cleaning up excludes according to requires => {updated_excludes=}")
    return updated_excludes


def clean_up_excludes_according_to_mandatory_chain(excludes, mandatory_chain):
    """it is sufficient if one element of a chain of mandatory features that require each other excludes another
    feature. due to their chain connection this will mean that all elements in mandatory chain exclude this other
    feature."""
    excludes_in_mandatory_chain = sorted(mandatory_chain)
    if excludes_in_mandatory_chain:
        intersec = sorted(set(excludes_in_mandatory_chain).intersection(set(excludes)))
        excludes = {k: v for k, v in excludes.items() if k not in intersec}
        excludes = {
            k: [vv for vv in v if vv not in intersec[:1]] for k, v in excludes.items()
        }
    excludes = {k: v for k, v in excludes.items() if len(v)}
    print(f"...cleaned excludes according to mandatory chain => {excludes=}.")
    return excludes


def clean_up_excludes_according_to_features_in_fm2(excludes, f2):
    """in this function we check if the key-value combinations in excludes are both in the newly added feature model.
    if this is the case, they can no longer exclude each other and have to be removed from excludes."""
    features_in_fm2 = find_and_filter_all_keys(f2)
    updated_excludes = excludes.copy()
    for k, v in excludes.items():
        if k not in features_in_fm2:
            continue
        if any(vv in features_in_fm2 for vv in v):
            new_values = [vv for vv in v if vv not in features_in_fm2]
            if new_values:
                updated_excludes[k] = new_values
            else:
                updated_excludes.pop(k)
    print(f"...cleaned excludes according to features fms => {updated_excludes=}.")
    return updated_excludes


def create_requires_constraints(
    diff_comp_in_fm, mandatories, requires, mandatory_chain
):
    if len(diff_comp_in_fm) <= 1:
        print("...no update of requires necessary")
        return requires

    mandatory_diffs = sorted([x for x in diff_comp_in_fm if mandatories.get(x, False)])
    optional_diffs = sorted(
        [x for x in diff_comp_in_fm if not mandatories.get(x, False)]
    )

    # add mandatory requirement chain to requires
    if len(mandatory_diffs) > 1:
        for i_md, md in enumerate(mandatory_diffs, 1):
            if i_md == len(mandatory_diffs):
                # close circle by adding first element as requiree to last element (requirer)
                i_md = 0
            requirer = md
            requiree = mandatory_diffs[i_md]
            if requiree:
                to_add = {requirer: [requiree]}
                requires.update(to_add)
                mandatory_chain.update(to_add)

    # add optional requirements to requires
    for od in optional_diffs:
        if mandatory_diffs:
            requires.update({od: mandatory_diffs.copy()})

    print(f"...updated {requires=}")
    return requires


def get_mandatories(diff_comp_in_fm1, diff_comp_in_fm2, f, mandatories):
    if f.get("children", {}):
        for child_name, child_values in f["children"].items():
            if any(child_name in diff for diff in [diff_comp_in_fm1, diff_comp_in_fm2]):
                mandatories.update(
                    {child_name: child_values["attributes"].get("mandatory", False)}
                )
            get_mandatories(
                diff_comp_in_fm1, diff_comp_in_fm2, child_values, mandatories
            )
    return


def prettify_xml(element, indent="  "):
    queue = [(0, element)]  # (level, element)
    while queue:
        level, element = queue.pop(0)
        children = [(level + 1, child) for child in list(element)]
        if children:
            element.text = "\n" + indent * (level + 1)  # for child open
        if queue:
            element.tail = "\n" + indent * queue[0][0]  # for sibling open
        else:
            element.tail = "\n" + indent * (level - 1)  # for parent close
        queue[0:0] = children  # prepend so children come before siblings


def create_feature_model_xml(constraints_dict, merged_dict, filename):
    # define XML structure for the feature model file
    fm_xml = eT.Element("featureModel")
    # insert features with respective hierarchy
    struct = eT.SubElement(fm_xml, "struct")

    def _process_dict(dictionary, parent_element):
        for key, values in dictionary.items():
            tag = values["attributes"]["tag"]
            abstract = values["attributes"].get("abstract")
            mandatory = values["attributes"].get("mandatory")
            if abstract is None and mandatory is None:
                element = eT.SubElement(parent_element, tag, name=f"{key}")
            elif abstract is not None:
                element = eT.SubElement(
                    parent_element, tag, abstract="true", name=f"{key}"
                )
            elif mandatory is not None:
                element = eT.SubElement(
                    parent_element, tag, mandatory="true", name=f"{key}"
                )
            else:
                element = None
                print("WARNING: This case is not handled yet.")
            _process_dict(values["children"], element)

    _process_dict(merged_dict, struct)

    # insert requires and excludes constraints to the feature model
    constraints = eT.SubElement(fm_xml, "constraints")
    for feature_key in constraints_dict["eq"]:
        if len(constraints_dict["eq"][feature_key]) > 1:
            for f_value in constraints_dict["eq"][feature_key]:
                rule = eT.SubElement(constraints, "rule")
                eq = eT.SubElement(rule, "imp")
                f1 = eT.SubElement(eq, "var")
                f1.text = feature_key
                not_feature = eT.SubElement(eq, "not")
                f2 = eT.SubElement(not_feature, "var")
                f2.text = f_value
        else:
            rule = eT.SubElement(constraints, "rule")
            eq = eT.SubElement(rule, "imp")
            f1 = eT.SubElement(eq, "var")
            f1.text = feature_key
            not_feature = eT.SubElement(eq, "not")
            f2 = eT.SubElement(not_feature, "var")
            f2.text = constraints_dict["eq"][feature_key][0]
    for feature_key in constraints_dict["imp"]:
        # if len(constraints_dict["imp"][feature_key]) > 1:
        if isinstance(constraints_dict["imp"][feature_key], list):
            for f_value in constraints_dict["imp"][feature_key]:
                rule = eT.SubElement(constraints, "rule")
                imp = eT.SubElement(rule, "imp")
                f1 = eT.SubElement(imp, "var")
                f1.text = feature_key
                f2 = eT.SubElement(imp, "var")
                f2.text = f_value
        elif isinstance(constraints_dict["imp"][feature_key], dict):
            rule = eT.SubElement(constraints, "rule")
            imp = eT.SubElement(rule, "imp")
            f1 = eT.SubElement(imp, "var")
            f1.text = feature_key
            or_attribute = eT.SubElement(imp, "disj")
            for f_value in constraints_dict["imp"][feature_key]["disj"]:
                f2 = eT.SubElement(or_attribute, "var")
                f2.text = f_value
        else:
            raise TypeError(
                f"this type is not yet implemented, please check for errors in constraints dict: "
                f"{type(constraints_dict['imp'][feature_key])}"
            )
    prettify_xml(fm_xml)
    tree = eT.ElementTree(fm_xml)
    tree.write(filename, encoding="UTF-8", xml_declaration=True)
    print(f"done writing {filename=}.")
    return fm_xml


def main():
    input_dir = INPUT_DIR
    output_dir = OUTPUT_DIR
    for filepath in [input_dir, output_dir]:
        os.makedirs(filepath, exist_ok=True)

    input_files = os.listdir(input_dir)
    output_files = []
    for idx_f, file1 in enumerate(input_files):
        if idx_f == len(input_files) - 1:
            break
        file2 = input_files[idx_f + 1]
        if not idx_f:
            filename1 = f"{input_dir}{file1}"
            filename2 = f"{input_dir}{file2}"
        else:
            filename1 = output_files[-1]
            filename2 = f"{input_dir}{file2}"
        output_filename = f"{output_dir}SoftGripper{idx_f+1}.xml"

        # set breakpoint for debugging to a certain output_file
        breakpoint = False
        if output_filename.endswith("7.xml"):
            breakpoint = True

        print("############################################################")
        print(f"Merging {filename1=} and {filename2=} to {output_filename=}")
        print("############################################################")

        # Read xml data from file
        root1 = get_element_tree(filename1)
        root2 = get_element_tree(filename2)

        # iterate through each child element and convert xml to nested dictionary
        fm_dict1 = parse_xml_to_dict(next(iter(root1)))
        fm_dict2 = parse_xml_to_dict(next(iter(root2)))

        # iterate through all constraints and convert xml to nested dictionary
        fm_dict1_constraints = parse_constraints_to_dict(filename1)
        fm_dict2_constraints = parse_constraints_to_dict(filename2)

        # compare the two dictionaries
        # root feature layer
        print("comparing feature models...")
        fm_dicts = [fm_dict1, fm_dict2]
        all_systems = list(set(list(fm_dict1) + list(fm_dict2)))
        attrs1, attrs2 = (get_feature_element(d, "attributes") for d in fm_dicts)
        children1, children2 = (get_feature_element(d, "children") for d in fm_dicts)
        attrs, system = determine_system_and_attributes(
            all_systems, attrs1, attrs2, children1, children2, fm_dict1, fm_dict2
        )
        print(
            f"...done comparing feature models, found {attrs1=}, {attrs2=}, {len(children1)=}, {len(children2)=}"
        )

        # merge children to merged_dict
        print("merging children...")
        # initialize merged_dict with first level (root feature)
        merged_dict = {system: {"attributes": attrs, "children": {}}}
        # merge children and add them to merged_dict
        recursively_merge_children(children1, children2, merged_dict, system)
        print(f"...done merging children => {len(merged_dict)=}")

        # compare fms for distinct features and create requires and excludes constraints
        constraints_dict = create_constraints(
            fm_dict1,
            fm_dict2,
            fm_dict1_constraints,
            fm_dict2_constraints,
            breakpoint=breakpoint,
        )

        create_feature_model_xml(constraints_dict, merged_dict, output_filename)
        output_files.append(output_filename)

        print(f"done merging {filename1=} and {filename2=}\n")


if __name__ == "__main__":
    main()
