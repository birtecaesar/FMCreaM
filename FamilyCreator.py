import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from xml.etree.ElementTree import Element


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
        # struct (struct only has one entry!)
        child = next(iter(element))
        child_data = parse_xml_to_dict(child)
        child_name = child.attrib.get("name")
        if child_name is None:
            return result if result else None
        _add_child_to_result(child_data, child_name, result)
        return result if result else None
    else:
        raise ValueError("I could not parse xml file and have to return empty-handed!")


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
            # case 2: child occurs in both feature models, but has in different parent
            # ToDo: not yet clear how to handle this case, needs further discussion
            pass
    return result_dict


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


def main():
    # Read XML data from file
    root1, root2 = get_element_trees("sample.xml", "sampleC.xml")

    # Iterate through each child element and convert XML to nested dictionary
    fm_dict1 = parse_xml_to_dict(next(iter(root1)))
    fm_dict2 = parse_xml_to_dict(next(iter(root2)))

    # Compare the two dictionaries
    # root feature layer
    fm_dicts = [fm_dict1, fm_dict2]
    all_systems = list(set(list(fm_dict1) + list(fm_dict2)))
    attrs1, attrs2 = (get_feature_element(d, "attributes") for d in fm_dicts)
    children1, children2 = (get_feature_element(d, "children") for d in fm_dicts)
    system = next(iter(fm_dict1))
    if len(all_systems) == 1:
        # both feature models have the same root feature name
        attrs = {"tag": compare_tag_attributes(attrs1, attrs2), "abstract": True}
    elif len(all_systems) == 2:
        # the feature models have different root feature names
        if sorted(children1) == sorted(children2):
            # case 1: both root features have exactly the same children (on the first child level)
            # => we assume that the root features are identical, but have different names
            attrs = {"tag": compare_tag_attributes(attrs1, attrs2), "abstract": True}
        else:
            # case 2: both root features have differences in their children (on the first child level)
            # => we have to check all children independently
            # ToDo: check the tag for children (and vs or...)
            attrs = {}
    else:
        raise ValueError(
            f"Cannot merge empty feature model and more than two should not exist, but got {all_systems=}"
        )

    # initialize merged_dict with first level (root feature)
    merged_dict = {system: {"attributes": attrs, "children": {}}}

    # merge children and add them to merged_dict
    recursively_merge_children(children1, children2, merged_dict, system)

    # Compare each entry and create merged dict

    fm1_keys = find_and_filter_all_keys(fm_dict1)
    fm2_keys = find_and_filter_all_keys(fm_dict2)
    for x in fm1_keys:
        for y in fm2_keys:
            if y == x:  # root feature is equal
                children1 = get_attributes(fm_dict1, x)
                children2 = get_attributes(fm_dict2, x)
            else:  # root feature is not equal
                merged_dict.update({"root": {}})

    # System/Module/Component is not included in both FMs
    diff_comp = []
    for x in fm1_keys:
        if x not in fm2_keys:
            diff_comp.append(x)
        else:
            for y in fm2_keys:
                if y not in fm1_keys and y not in diff_comp:
                    diff_comp.append(y)

    # Set feature to optional
    for comp in diff_comp:
        attributes_fm1 = get_attributes(fm_dict1, comp)
        attributes_fm2 = get_attributes(fm_dict2, comp)
        if attributes_fm1 is not None:
            if attributes_fm1.get("mandatory") is not None:
                for att in attributes_fm1.values():
                    att["mandatory"] = False
            else:
                pass
        else:
            pass
        if attributes_fm2 is not None:
            if attributes_fm2.get("mandatory") is not None:
                for att in attributes_fm1.values():
                    att["mandatory"] = False
            else:
                pass
        else:
            pass


if __name__ == "__main__":
    main()
