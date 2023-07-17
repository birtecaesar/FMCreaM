import copy
import os
import random
import sys
import xml.etree.ElementTree as ET
from itertools import combinations
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from xml.etree.ElementTree import Element


def flatten_list(list_of_lists):
    flat_list = []
    for sublist in list_of_lists:
        flat_list += sublist
    return flat_list


def get_attributes(elem, key):
    """get the attributes and children of a feature"""
    for elem_key, elem_val in elem.items():
        if elem_key == key:
            return elem_val
        elif isinstance(elem_val, dict):
            value = get_attributes(elem_val, key)
            if value is not None:
                return value


def parse_xml_to_dict(element):
    """recursively loop through xml elements and extract information into a dictionary"""

    def _add_child_to_result(data, name, result):
        """helper function to add children to the result dict"""
        if name in result:
            if isinstance(result[name], list):
                result[name].append(data)
            else:
                result[name] = [result[name], data]
        else:
            result.update(data)

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
        # struct (struct only has one entry! -> Which is the root feature. Feature models always have just one root feature!)
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


def parse_constraints_to_dict(element):
    excludes = {}
    implies = {}
    existing_constraints = {}
    # get both feature names of the excludes constraint and add them into the eq_dict
    for rules in element.findall(".//constraints/rule/eq/var"):
        feature1 = rules.text
        find = ".//constraints/rule/eq[var='" + feature1 + "']/not/var"
        for ex in element.findall(find):
            feature2 = ex.text
            excludes[feature1] = [feature2]
    existing_constraints.update({"eq": excludes})
    # get both feature names of the implies constraint and add them into the imp_dict
    imp_k = []
    imp_v = []
    for rule_idx, rules in enumerate(element.findall(".//constraints/rule/imp/var")):
        feature1 = rules.text
        if rule_idx % 2 == 0:
            imp_k.append(feature1)
        else:
            imp_v.append(feature1)
    for k, v in zip(imp_k, imp_v):
        if k in implies:
            implies[k].add(v)
        else:
            implies[k] = {v}
    implies = {k: list(v) for k, v in implies.items()}
    # implies = {k:[v] for k, v in zip(imp_k, imp_v)}
    # print(feature1)
    # find = ".//constraints/rule/imp[var='" + feature1 + "']/var"
    # # implies[feature1] = []
    # for ex in element.findall(find):
    #     feature2 = ex.text
    #     # implies[feature1].append(feature2)
    #     implies[feature1] = [feature2]
    #     print(f"   {feature2}")
    # for rule in element.iter("rule"):
    #     print(rule.items())
    #     if rule.find("imp/var"):
    #         f_imp = rule.find("imp/var").text
    #         for imp_f in rule.iter("var"):
    #             implies[f_imp] = imp_f.text
    #         disj_texts = []
    #         for disj in rule.iter("disj"):
    #             if disj:
    #                 for var in disj.iter("var"):
    #                     disj_texts.append(var.text)
    #                     disj = {"disj": disj_texts}
    #                     if implies[f_imp]:
    #                         implies.pop(f_imp)
    #                         implies[f_imp] = disj
    existing_constraints.update({"imp": implies})
    return existing_constraints


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


def compare_mandatory_attributes(a1: dict, a2: dict) -> Optional[bool]:
    mandatory1, mandatory2 = (d.get("mandatory", "") for d in [a1, a2])
    if mandatory1 != mandatory2 or not any(m for m in [mandatory1, mandatory2]):
        # case 1: mandatory attrs are either different or not set
        mandatory = None
    else:
        # case 2: both mandatory attrs are equal
        mandatory = True
    return mandatory


def compare_attributes(children1_attrs, children2_attrs):
    defaults = {"tag": "and", "mandatory": False, "abstract": False}
    avail_attrs = set(list(children1_attrs) + list(children2_attrs))
    attrs = {}
    for a_key1, a_val1 in children1_attrs.items():
        if a_key1 == "tag":
            a_val2 = children2_attrs.get(a_key1, "not_set")
            if a_val1 == a_val2:
                attrs.update({a_key1: a_val1})
            else:
                attrs.update({a_key1: "and"})


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
            # case 2: child occurs in both feature models, but has a different parent
            # ToDo: not yet clear how to handle this case, needs further discussion
            pass
    return result_dict


def create_constraints(
    f1: dict, f2: dict, f1con: dict, f2con: dict, disj_fm: bool
) -> dict:
    """get features that only exist in feature model 1 (f1) and features that only exist in feature model 2 (f2)
    if there are no such features no constraints will be added
    if there exists these features, we will create requires or exclude constraints (con) to maintain only the variants that exist at the current state of time
    """

    # system/module/component is not included in both fms
    fm1_keys = find_and_filter_all_keys(f1)
    fm2_keys = find_and_filter_all_keys(f2)
    diff_comp_in_fm1 = []
    diff_comp_in_fm2 = []
    for x in fm1_keys:
        if x not in fm2_keys:
            diff_comp_in_fm1.append(x)
    for y in fm2_keys:
        if y not in fm1_keys:
            diff_comp_in_fm2.append(y)

    # create constraints so that only the two variants compared can be selected + the variants that were already merged
    excludes = {}
    # excludes constraints are created for the features that do not occur in fm 1 -> same is applied to fm 2
    for x in range(len(diff_comp_in_fm1)):
        diff_comp_in_fm2_copy = diff_comp_in_fm2[:]
        if diff_comp_in_fm2_copy:
            excludes[diff_comp_in_fm1[x]] = diff_comp_in_fm2_copy
    for x in range(len(diff_comp_in_fm2)):
        diff_comp_in_fm1_copy = diff_comp_in_fm1[:]
        if diff_comp_in_fm1_copy:
            excludes[diff_comp_in_fm2[x]] = diff_comp_in_fm1[:]

    for fcon in [f1con, f2con]:
        update_dict_with_ancient_knowledge(excludes, fcon, "eq")

    # excludes constraints are extended by excludes constraints from each fm that still are valid
    # if f1con:
    #     _update_excludes_constraints(f1con, diff_comp_in_fm1, excludes)
    # if f2con:
    #     _update_excludes_constraints(f2con, diff_comp_in_fm2, excludes)

    # find all mandatory connections between features
    mandatories = {}
    for f in [f1, f2]:
        f_root = f[next(iter(f))]
        get_mandatories(diff_comp_in_fm1, diff_comp_in_fm2, f_root, mandatories)

    requires = {}
    # requires constraints between the features of fm 1, which do not occur in fm 2 -> same is applied to fm 2
    create_requires_constraints(diff_comp_in_fm1, mandatories, requires)
    create_requires_constraints(diff_comp_in_fm2, mandatories, requires)
    # Todo: weiter aufräumen, keine Kombinationen zwischen Eltern/Kindern bzw. Kindern unterschiedlicher Zweige (wenn tag==alt)
    # Es reicht eine Liste zu erstellen wo nur Leaf Feature drin sind. Dh. alle Elemente die den Tag 'feature' haben
    _check_violation_of_excludes_cons_by_requires_cons(excludes, requires)

    for fcon in [f1con, f2con]:
        update_dict_with_ancient_knowledge(requires, fcon, "imp")

    # existing_requires = f1con["imp"]
    # existing_requires.update(f2con["imp"])
    #
    # # if any diff_comp dict is empty, use the existing excludes and requires constraints
    # update_of_excludes_and_requires_necessary = True
    # if not all(diff for diff in [diff_comp_in_fm1, diff_comp_in_fm2]):
    #     diff_comp = diff_comp_in_fm1 + diff_comp_in_fm2
    #     existing_excludes = {k: v for k, v in existing_excludes.items() if k in diff_comp or any(vv in diff_comp for vv in v)}
    #     existing_requires = {k: v for k, v in existing_requires.items() if k in diff_comp and any(vv in diff_comp for vv in v)}
    #     update_of_excludes_and_requires_necessary = False

    con = {}
    if disj_fm:
        existing_excludes = f1con["eq"]
        existing_excludes.update(f2con["eq"])
        # if feature models are disjoint, constraints can not be created between alternative branches
        allowed_constraints = _find_allowed_constraints(f1)
        allowed_constraints.update(_find_allowed_constraints(f2))
        updated_requires = _update_constraints(allowed_constraints, requires)
        con.update({"imp": updated_requires})
        # clean excludes dict -> if fms are disjoint no excludes statements are added, only the exisiting ones for each fm are used, including the constraints for each alternative branch
        updated_excludes = _update_constraints(allowed_constraints, existing_excludes)
        con.update({"eq": updated_excludes})
    else:
        # filter requires
        updated_requires = {}
        clean_up_requires_according_to_diff_comp(
            diff_comp_in_fm1, requires, updated_requires
        )
        clean_up_requires_according_to_diff_comp(
            diff_comp_in_fm2, requires, updated_requires
        )

        # filter excludes
        excludes = clean_up_excludes_according_to_requires(excludes, requires)
        excludes = _drop_duplicate_excludes(excludes)

        con.update({"eq": excludes})
        con.update({"imp": updated_requires})
    return con


def clean_up_requires_according_to_diff_comp(
    diff_comp_in_fm, requires, updated_requires
):
    for k, v in requires.items():
        if k in diff_comp_in_fm and any(vv in diff_comp_in_fm for vv in v):
            # key and at least one value are only in one fm => we want to keep these constraints
            to_add = v  # [vv for vv in v if vv in diff_comp_in_fm]
        elif k not in diff_comp_in_fm and any(vv in diff_comp_in_fm for vv in v):
            # key is in both fms and therefore not in diff_comp, but at least one value is only in one fm
            # => we want to keep the key and remove the value that is only in one fm
            # => the constraint is not valid anymore, because a feature that is present in both fms cannot require a
            # feature that is not present in both fms
            to_add = [vv for vv in v if vv not in diff_comp_in_fm]
        # elif k in diff_comp_in_fm and not all(vv in diff_comp_in_fm for vv in v):
        #     # key is only in one fm and some values are in both fms
        #     # => we want to keep all values that are in both fms
        #     to_add = [vv for vv in v if vv not in diff_comp_in_fm]
        else:
            # we ignore everything else
            to_add = []
        if to_add:
            if k in updated_requires:
                updated_requires[k] += to_add
            else:
                updated_requires[k] = to_add


def update_dict_with_ancient_knowledge(d, fcon, con_type):
    for k, v in fcon.get(con_type, {}).items():
        if k in d:
            d[k] += v
        else:
            d[k] = v


def _check_violation_of_excludes_cons_by_requires_cons(excludes, requires):
    # checks if the new requires constraints do not violate any excludes constraints which are added from fm1 and fm2
    for f_imp, imp_fs in requires.items():
        for imp_f in imp_fs:
            if imp_f in excludes[f_imp]:
                requires[f_imp].remove(imp_f)
                requires[imp_f].remove(f_imp)
        # additional check to avoid contradicting constraints so that no configuration can be selected
        # -> If one feature requires several features, check if these do not exclude each other.
        if len(imp_fs) > 1:
            for combination in list(combinations(imp_fs, 2)):
                feat1 = combination[0]
                feat2 = combination[1]
                if feat2 in excludes[feat1]:
                    requires.update({f_imp: {"disj": imp_fs}})
                else:
                    if feat1 in excludes[feat2]:
                        requires.update({f_imp: {"disj": imp_fs}})
    # clean requires dict -> delete all features with empty constraints
    requires = {key: value for key, value in requires.items() if value}
    return requires


def _update_constraints(
    allowed_constraints: dict[str, list[str]], constraints: dict[str, list[str]]
) -> dict:
    updated_constraints = {}
    for f_constraints, constraint_fs in constraints.items():
        if "disj" in constraint_fs:
            # create disjoint constraints
            constraint_fs = constraint_fs["disj"]
            updated_constraint_fs = {
                "disj": [
                    rf
                    for rf in constraint_fs
                    if rf in allowed_constraints.get(f_constraints, [])
                ]
            }
        else:
            # create non-disjoint constraints (implies and excludes)
            updated_constraint_fs = [
                rf
                for rf in constraint_fs
                if rf in allowed_constraints.get(f_constraints, [])
            ]
        if updated_constraint_fs:
            updated_constraints[
                f_constraints
            ] = updated_constraint_fs  # Wieso wird hier das disj nicht hinzugefügt? Und auch die eq regel nicht obwohl alle bedingungen erfüllt werden?
    return updated_constraints


def _find_allowed_constraints(f: dict) -> dict:
    root = f[next(iter(f))]
    root_attrs = root.get("attributes", {})
    allowed_constraints = {}
    if root_attrs.get("tag", "unset") == "alt":
        for child_name, child_values in root.get("children", {}).items():
            allowed_keys = find_and_filter_all_keys(child_values)
            for allowed_key in allowed_keys:
                allowed_constraints[allowed_key] = [
                    ak for ak in allowed_keys if ak != allowed_key
                ]
    if not allowed_constraints:
        children = [c for c in root.get("children", {}).keys()]
        for child in children:
            allowed_constraints[child] = [c for c in children if c != child]
    return allowed_constraints


def _drop_duplicate_excludes(excludes: dict) -> dict:
    clean_excludes = {}
    pairs = []
    for key, value in excludes.items():
        if isinstance(value, list):
            for single_value in value:
                pairs.append((key, single_value))
        else:
            pairs.append((key, value))
    unique_pairs = set(tuple(sorted(x)) for x in pairs)
    for pair in unique_pairs:
        key = pair[0]
        value = pair[1]
        if key in clean_excludes.keys():
            clean_excludes[key].append(value)
        else:
            clean_excludes.update({key: [value]})
    return clean_excludes


def _update_excludes_constraints(fmcon: dict, only_in_fm: list, excludes: dict) -> None:
    # excludes constraints are extended by excludes constraints from each fm that still are valid
    for f_ex, ex_fs in fmcon["eq"].items():
        if f_ex in only_in_fm:
            for ex_f in ex_fs:
                if ex_f in only_in_fm:
                    excludes[f_ex].append(ex_f)


def pretty_print_dict(d):
    for k, v in d.items():
        print(k)
        if isinstance(v, list):
            for vv in v:
                print(f"|---- {vv}")
        elif isinstance(v, str):
            print(v)
        elif isinstance(v, dict):
            pretty_print_dict(v)
        else:
            print(f"Don't know what I should do with {v} -> {type(v)=}")


def clean_up_excludes_according_to_requires(excludes, requires):
    updated_excludes = copy.deepcopy(excludes)
    print(f"excludes prior to filtering:")
    pretty_print_dict(updated_excludes)
    for req_k, req_vs in requires.items():
        for req_v in req_vs:
            to_remove = False
            if all(req in excludes for req in [req_k, req_v]):
                ex_vs_of_req_k = excludes[req_k]
                ex_vs_of_req_v = excludes[req_v]
                if ex_vs_of_req_k == ex_vs_of_req_v:
                    # if both lists are equal remove the whole entry of req_k from excludes
                    print(f"remove {req_k=}")
                    updated_excludes.pop(req_k)
                    to_remove = True
                elif any(x in ex_vs_of_req_v for x in ex_vs_of_req_k):
                    # if any entry of the lists is equal, remove this entry from the values of excludes
                    duplicates = {x for x in ex_vs_of_req_k if x in ex_vs_of_req_v}
                    updated_excludes[req_k] = [
                        x for x in updated_excludes[req_k] if x not in duplicates
                    ]
                    to_remove = True
                if to_remove:
                    # if we removed a key or a value previously, we also want to update the remaining counter-part
                    # constraints
                    for ex_v_of_req_k in ex_vs_of_req_k:
                        if ex_v_of_req_k in updated_excludes:
                            updated_excludes[ex_v_of_req_k] = [
                                x for x in updated_excludes[ex_v_of_req_k] if x != req_k
                            ]

    print(f"excludes prior after filtering:")
    pretty_print_dict(updated_excludes)

    # all_required_keys = set(list(requires))
    # all_required_values = set(flatten_list(list(requires.values())))
    # for ex_k, ex_v in excludes.items():
    #     if not any(
    #         ex_k in all_requs for all_requs in [all_required_keys, all_required_values]
    #     ):
    #         # keep excludes that are neither in keys nor in values of requires
    #         updated_excludes[ex_k] = [x for x in ex_v if x in all_required_keys]
    #     elif all(
    #         ex_k in all_requs for all_requs in [all_required_keys, all_required_values]
    #     ):
    #         # remove excludes that are in both keys and values of requires
    #         continue
    #     elif ex_k in all_required_values:
    #         # keep excludes that are only in values of requires
    #         updated_excludes[ex_k] = [x for x in ex_v if x in all_required_keys]
    #     else:
    #         # remove excludes that are only in keys of requires
    #         continue
    # Todo: hier werfen wir gerade eine exclude zu viel raus => r50 muss r38 und r31 ausschließen
    return updated_excludes


def create_requires_constraints(diff_comp_in_fm, mandatories, requires):
    for x in diff_comp_in_fm:
        required_features = diff_comp_in_fm[:]
        required_features.remove(x)
        required_features = [
            rf
            for rf in required_features
            if not mandatories.get(x, False) and mandatories.get(rf, False)
        ]
        if required_features:
            requires[x] = required_features


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


def _update_requires(allowed_constraints, requires):
    updated_requires = {}
    for f_requires, required_fs in requires.items():
        updated_required_fs = [
            rf for rf in required_fs if rf in allowed_constraints.get(f_requires, [])
        ]
        if updated_required_fs:
            updated_requires[f_requires] = updated_required_fs
    return updated_requires


def get_feature_element(d: dict, e: str, k: str = "") -> dict:
    """get entry (element) for given dict and key
    if no key is provided, use the first dict key
    if entry (element) is empty return an empty dict"""
    if not k:
        return d[next(iter(d))].get(e, {})
    else:
        return d.get(k, {}).get(e, {})


def get_element_trees(file1: str, file2: str) -> tuple["Element", "Element"]:
    tree1 = ET.parse(file1)
    root1 = tree1.getroot()
    tree2 = ET.parse(file2)
    root2 = tree2.getroot()
    return root1, root2


def prettify(element, indent="  "):
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


def remove_mandatory(dictionary):
    iter_dict = dictionary.copy()
    for key, value in iter_dict.items():
        if isinstance(value, dict):
            remove_mandatory(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    remove_mandatory(item)
        if key == "mandatory" and value == True:
            del dictionary[key]


def generate_unique_name(f1_keys: list, f2_keys: list):
    random_numbers = [str(random.randint(0, 9)) for _ in range(2)]
    unique_name = "feature_" + "".join(random_numbers)
    if all(unique_name == name for name in (f1_keys, f2_keys)):
        generate_unique_name(f1_keys, f2_keys)
    return unique_name


def create_feature_model_xml(constraints_dict, merged_dict, filename):
    # define XML structure for the feature model file
    fm_xml = ET.Element("featureModel")
    # insert features with respective hierarchy
    struct = ET.SubElement(fm_xml, "struct")

    def process_dict(dictionary, parent_element):
        for key, values in dictionary.items():
            tag = values["attributes"]["tag"]
            abstract = values["attributes"].get("abstract")
            mandatory = values["attributes"].get("mandatory")
            if abstract is None and mandatory is None:
                element = ET.SubElement(parent_element, tag, name=f"{key}")
            elif abstract is not None:
                element = ET.SubElement(
                    parent_element, tag, abstract="true", name=f"{key}"
                )
            elif mandatory is not None:
                element = ET.SubElement(
                    parent_element, tag, mandatory="true", name=f"{key}"
                )
            process_dict(values["children"], element)

    process_dict(merged_dict, struct)

    # insert requires and excludes constraints to the feature model
    constraints = ET.SubElement(fm_xml, "constraints")
    for feature_key in constraints_dict["eq"]:
        if len(constraints_dict["eq"][feature_key]) > 1:
            for f_value in constraints_dict["eq"][feature_key]:
                rule = ET.SubElement(constraints, "rule")
                eq = ET.SubElement(rule, "eq")
                f1 = ET.SubElement(eq, "var")
                f1.text = feature_key
                not_feature = ET.SubElement(eq, "not")
                f2 = ET.SubElement(not_feature, "var")
                f2.text = f_value
        else:
            rule = ET.SubElement(constraints, "rule")
            eq = ET.SubElement(rule, "eq")
            f1 = ET.SubElement(eq, "var")
            f1.text = feature_key
            not_feature = ET.SubElement(eq, "not")
            f2 = ET.SubElement(not_feature, "var")
            f2.text = constraints_dict["eq"][feature_key][0]
    for feature_key in constraints_dict["imp"]:
        if len(constraints_dict["imp"][feature_key]) > 1:
            for f_value in constraints_dict["imp"][feature_key]:
                rule = ET.SubElement(constraints, "rule")
                imp = ET.SubElement(rule, "imp")
                f1 = ET.SubElement(imp, "var")
                f1.text = feature_key
                f2 = ET.SubElement(imp, "var")
                f2.text = f_value
        if isinstance(constraints_dict["imp"][feature_key], dict):
            rule = ET.SubElement(constraints, "rule")
            imp = ET.SubElement(rule, "imp")
            f1 = ET.SubElement(imp, "var")
            f1.text = feature_key
            or_attribute = ET.SubElement(imp, "disj")
            for f_value in constraints_dict["imp"][feature_key]["disj"]:
                f2 = ET.SubElement(or_attribute, "var")
                f2.text = f_value
        else:
            rule = ET.SubElement(constraints, "rule")
            imp = ET.SubElement(rule, "imp")
            f1 = ET.SubElement(imp, "var")
            f1.text = feature_key
            f2 = ET.SubElement(imp, "var")
            f2.text = constraints_dict["imp"][feature_key][0]
    prettify(fm_xml)
    tree = ET.ElementTree(fm_xml)
    # tree.write(
    #     "SoftGripper/output/SoftGripper4.xml", encoding="UTF-8", xml_declaration=True
    # )
    tree.write(filename, encoding="UTF-8", xml_declaration=True)
    return fm_xml


def main():
    input_dir = "./SoftGripper/input/"
    output_dir = "./SoftGripper/output/"
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

        # breakpoint
        if output_filename.endswith("3.xml"):
            print("break")

        print(f"Parsing {filename1=} and {filename2=}")

        # Read XML data from file
        # root1, root2 = get_element_trees("F3S60A30R31.xml", "F3S60A30R38.xml")
        # root1, root2 = get_element_trees(
        #     "./SoftGripper/output/SoftGripper3.xml", "./SoftGripper/input/F3S60A70D15R38.xml"
        # )
        root1, root2 = get_element_trees(filename1, filename2)

        # Iterate through each child element and convert XML to nested dictionary
        fm_dict1 = parse_xml_to_dict(next(iter(root1)))
        fm_dict2 = parse_xml_to_dict(next(iter(root2)))

        # Iterate through all constraints and convert XML to nested dictionary
        fm_dict1_constraints = parse_constraints_to_dict(root1)
        fm_dict2_constraints = parse_constraints_to_dict(root2)

        # Compare the two dictionaries
        # root feature layer
        fm_dicts = [fm_dict1, fm_dict2]
        all_systems = list(set(list(fm_dict1) + list(fm_dict2)))
        attrs1, attrs2 = (get_feature_element(d, "attributes") for d in fm_dicts)
        children1, children2 = (get_feature_element(d, "children") for d in fm_dicts)

        # SPECIAL CASE!: both fms are disjoint. No features are the same.
        fs_in_fm1 = set(find_and_filter_all_keys(children1))
        fs_in_fm2 = set(find_and_filter_all_keys(children2))
        if fs_in_fm1.isdisjoint(fs_in_fm2):
            system = next(iter(fm_dict1))
            if len(all_systems) == 1:
                if any(attr == "alt" for attr in (attrs1["tag"], attrs2["tag"])):
                    # both feature models have the same root feature name and at least one tag=='alt'
                    # -> one feature model already contains disjoint fms
                    attrs_dis = {"tag": "alt", "abstract": True}
                    fm1_names, fm2_names = (
                        find_and_filter_all_keys(c) for c in [fm_dict1, fm_dict2]
                    )
                    feature_name = generate_unique_name(fm1_names, fm2_names)
                    children = {
                        feature_name: {"attributes": attrs2, "children": children2}
                    }
                    children.update(children1)
                    merged_dict = {
                        system: {"attributes": attrs_dis, "children": children}
                    }
                    remove_mandatory(merged_dict)
                    constraints_dict = create_constraints(
                        fm_dict1,
                        fm_dict2,
                        fm_dict1_constraints,
                        fm_dict2_constraints,
                        disj_fm=True,
                    )
                    # create_feature_model_xml(constraints_dict, merged_dict)
                else:
                    # both feature models have the same root feature name and the tags are a combination of "and" and "or"
                    attrs_dis = {"tag": "alt", "abstract": True}
                    # attrs = {"tag": "and", "abstract": True}
                    fm1_names, fm2_names = (
                        find_and_filter_all_keys(c) for c in [fm_dict1, fm_dict2]
                    )
                    feature_name1 = generate_unique_name(fm1_names, fm2_names)
                    feature_name2 = generate_unique_name(fm1_names, fm2_names)
                    children = {
                        feature_name1: {"attributes": attrs1, "children": children1},
                        feature_name2: {"attributes": attrs2, "children": children2},
                    }
                    merged_dict = {
                        system: {"attributes": attrs_dis, "children": children}
                    }
                    remove_mandatory(merged_dict)
                    constraints_dict = create_constraints(
                        fm_dict1,
                        fm_dict2,
                        fm_dict1_constraints,
                        fm_dict2_constraints,
                        disj_fm=True,
                    )
                    # create_feature_model_xml(constraints_dict, merged_dict)
            else:
                # the feature models have different root feature names -> keep root features and add a new root above them
                attrs_dis = {"tag": "alt", "abstract": True}
                child1 = {system: {"attributes": attrs1, "children": children1}}
                child2 = {system: {"attributes": attrs2, "children": children2}}
                merged_dict = {
                    "root": {"attributes": attrs_dis, "children": {child1, child2}}
                }
                remove_mandatory(merged_dict)
                constraints_dict = create_constraints(
                    fm_dict1,
                    fm_dict2,
                    fm_dict1_constraints,
                    fm_dict2_constraints,
                    disj_fm=True,
                )
                # create_feature_model_xml(constraints_dict, merged_dict)
        # Normal Case: the fms share some features
        else:
            system = next(iter(fm_dict1))
            if len(all_systems) == 1:
                # both feature models have the same root feature name
                if sorted(children1) == sorted(children2):
                    # case 1: both root features have exactly the same children (on the first child level)
                    # => booth root features are identical
                    attrs = {
                        "tag": compare_tag_attributes(attrs1, attrs2),
                        "abstract": True,
                    }
                else:
                    # case 2: both root features have differences in their children (on the first child level)
                    # => we check if first level group tag is an alternative, then we assume that some disjoint fms were merged before
                    if any(attr == "alt" for attr in (attrs1["tag"], attrs2["tag"])):
                        attrs = {"tag": "alt", "abstract": True}
                    # if no alternative tag is used, we apply the tag compare-rules
                    else:
                        attrs = {
                            "tag": compare_tag_attributes(attrs1, attrs2),
                            "abstract": True,
                        }
            elif len(all_systems) == 2:
                # the feature models have different root feature names
                if sorted(children1) == sorted(children2):
                    # case 1: both root features have exactly the same children (on the first child level)
                    # => we assume that the root features are identical, but have different names
                    attrs = {
                        "tag": compare_tag_attributes(attrs1, attrs2),
                        "abstract": True,
                    }
                else:
                    # case 2: both root features have differences in their children (on the first child level)
                    # => we check if first level group tag is an alternative, then we assume that some disjoint fms were
                    # merged before
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
                            "Both root features have different names and different children on first child level. "
                            "This case can not be handled yet."
                        )
            else:
                raise ValueError(
                    f"Cannot merge empty feature model and more than two should not exist, but got {all_systems=}"
                )

            # initialize merged_dict with first level (root feature)
            merged_dict = {system: {"attributes": attrs, "children": {}}}

            # merge children and add them to merged_dict
            recursively_merge_children(children1, children2, merged_dict, system)

            # compare fms for distinct features and create requires and excludes constraints
            constraints_dict = create_constraints(
                fm_dict1,
                fm_dict2,
                fm_dict1_constraints,
                fm_dict2_constraints,
                disj_fm=False,
            )

        # if any disjoints are in constraints_dict, we have to adapt merged_dict, so that the disjoint featues become
        # alternative children of a new feature
        # (disj features can only occur in "imp", not in "eq"!)
        cons_to_remove = {}
        for feature, con_features in constraints_dict["imp"].items():
            if isinstance(con_features, dict):
                assert "disj" in con_features
                # we found disj features, now we have to adapt merged_dict
                merged_dict_root = merged_dict[next(iter(merged_dict))]
                if all(f in merged_dict_root["children"] for f in con_features["disj"]):
                    new_feature = {
                        f"feature_{random.randint(1, 100)}_disj": {
                            "attributes": {"tag": "alt", "abstract": True},
                            "children": {
                                f: merged_dict_root["children"][f]
                                for f in con_features["disj"]
                            },
                        }
                    }
                    merged_dict[next(iter(merged_dict))]["children"].update(new_feature)
                    for f in con_features["disj"]:
                        merged_dict[next(iter(merged_dict))]["children"].pop(f)
                    cons_to_remove.update({feature: con_features})
                for abstract_feature, leaf_features in merged_dict_root[
                    "children"
                ].items():
                    if all(
                        f in leaf_features["children"] for f in con_features["disj"]
                    ):
                        new_feature = {
                            f"feature_{random.randint(1,100)}_disj": {
                                "attributes": {"tag": "alt", "abstract": True},
                                "children": {
                                    f: leaf_features["children"][f]
                                    for f in con_features["disj"]
                                },
                            }
                        }
                        merged_dict[next(iter(merged_dict))]["children"][
                            abstract_feature
                        ]["children"].update(new_feature)
                        for f in con_features["disj"]:
                            merged_dict[next(iter(merged_dict))]["children"][
                                abstract_feature
                            ]["children"].pop(f)
                        cons_to_remove.update({feature: con_features})

        # after removing disj features from merged_dict, we also have to update constraints_dict, because disj features are
        # now new alternative features
        # get possible combinations
        combis = {}
        for con_type, constraints in constraints_dict.items():
            for k, v in constraints.items():
                if "disj" in v:
                    for c in list(combinations(v["disj"], 2)):
                        combis[c[0]] = [c[1]]
                        combis[c[1]] = [c[0]]
        # update constraints
        updated_constraints_dict = {}
        for con_type, constraints in constraints_dict.items():
            updated_constraints_dict[con_type] = {}
            for k, v in constraints.items():
                if k in cons_to_remove:
                    print(f"k in cons to remove, I won't use {k}")
                    pass
                elif combis.get(k, "any") == v:
                    print(f"combi of {k}:{v}. I won't use {k}")
                    pass
                else:
                    print(f"this looks good => {k}:{v}")
                    updated_constraints_dict[con_type].update({k: v})

        create_feature_model_xml(updated_constraints_dict, merged_dict, output_filename)
        output_files.append(output_filename)


if __name__ == "__main__":
    main()
