import xml.etree.ElementTree as eT
from collections import defaultdict

from SPARQLWrapper import JSON, SPARQLWrapper, SPARQLWrapper2

ENDPOINT_URL = "http://localhost:7200/repositories/TuggerTrain"

MODULESQUERY = """
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
PREFIX vdi2206: <http://www.hsu-ifa.de/ontologies/VDI2206#>

SELECT ?module ?component
WHERE {
    ?component a vdi2206:Component .
    ?module a vdi2206:Module .
    ?module vdi2206:consistsOf ?component.
}"""

SYSTEMSQUERY = """
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
PREFIX vdi2206: <http://www.hsu-ifa.de/ontologies/VDI2206#>

SELECT ?mechatronicsystem ?system
WHERE {
    ?system a vdi2206:System .
    ?mechatronicsystem a vdi2206:System .
    ?mechatronicsystem vdi2206:consistsOf ?system.
}
"""

SYSMOQUERY = """
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
PREFIX vdi2206: <http://www.hsu-ifa.de/ontologies/VDI2206#>

SELECT ?mechatronicsystem ?module
WHERE {
    ?module a vdi2206:Module .
    ?mechatronicsystem a vdi2206:System .
    ?mechatronicsystem vdi2206:consistsOf ?module.
}
"""

SYSCOQUERY = """
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
PREFIX vdi2206: <http://www.hsu-ifa.de/ontologies/VDI2206#>

SELECT ?mechatronicsystem ?component 
WHERE {
    ?component a vdi2206:Component .
    ?mechatronicsystem a vdi2206:System .
    ?mechatronicsystem vdi2206:consistsOf ?component .
    }
"""
# PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
# PREFIX owl: <http://www.w3.org/2002/07/owl#>
# PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
# PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
# PREFIX vdi2206: <http://www.hsu-ifa.de/ontologies/VDI2206#>
#
# SELECT ?mechatronicsystem ?component ?module
# WHERE {
#   ?component a vdi2206:Component .
#   ?mechatronicsystem a vdi2206:System .
#   OPTIONAL {
#     ?module a vdi2206:Module .
#     ?mechatronicsystem vdi2206:consistsOf ?component .
#     FILTER NOT EXISTS { ?module vdi2206:consistsOf ?component }
#   }
#   FILTER(!BOUND(?module) || BOUND(?module) && EXISTS { ?module a vdi2206:Module })
# }
# """
SYSTEMQUERY = """
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
PREFIX vdi2206: <http://www.hsu-ifa.de/ontologies/VDI2206#>

SELECT ?system
WHERE {
    ?system a vdi2206:System .
}
"""


def get_sparql_results(endpoint_url, query):
    sparql = SPARQLWrapper2(endpoint_url)
    sparql.setQuery(query)
    # sparql.setReturnFormat(JSON)
    return sparql.queryAndConvert()


def add_to_lowest_layer(child, parents, query_dict):
    """add separate components to lowest available parent layer"""
    for k_1, layer_1 in query_dict.items():
        # check if lowest parent layer is a system
        if k_1 in parents:
            layer_1.update({child: {}})
            return
        for k_2, layer_2 in layer_1.items():
            # check if lowest parent layer is a module
            if k_2 in parents:
                layer_2.update({child: []})
                return
            for k_3, layer_3 in layer_2.items():
                # check if lowest parent layer is a component
                if k_3 in parents:
                    layer_3.append(child)
                    return
    # if no layer corresponds to any of the available parents, create a new level 1 layer
    query_dict.update({child: {}})
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


def main(filename: str) -> None:
    # get sparql query results
    component_results = get_sparql_results(ENDPOINT_URL, MODULESQUERY)
    systemofsystem_results = get_sparql_results(ENDPOINT_URL, SYSTEMSQUERY)
    module_results = get_sparql_results(ENDPOINT_URL, SYSMOQUERY)
    components_of_system_results = get_sparql_results(ENDPOINT_URL, SYSCOQUERY)
    system_results = get_sparql_results(ENDPOINT_URL, SYSTEMQUERY)

    # fill hierarchic dictionary of sparql query results
    components = create_components(component_results)
    modules = create_modules(components, module_results)
    systems = create_systems(modules, system_results, systemofsystem_results)
    separate_components = get_separate_components(
        components_of_system_results, component_results, systems
    )
    # use the highest available layer as final dict
    final_dict = (
        systems if systems else modules if modules else components if components else {}
    )

    # Define xml structure for the feature model file
    fm_xml = setup_xml_structure(final_dict, separate_components)
    prettify_xml(fm_xml)

    # write xml tree to file
    tree = eT.ElementTree(fm_xml)
    tree.write(filename, encoding="UTF-8", xml_declaration=True)


def setup_xml_structure(final_dict, separate_components):
    fm_xml = eT.Element("featureModel")
    struct = eT.SubElement(fm_xml, "struct")
    for sys_key, sys_values in final_dict.items():
        if sys_key in separate_components:
            root = eT.SubElement(struct, "feature", mandatory="true", name=f"{sys_key}")
        else:
            root = eT.SubElement(struct, "and", abstract="true", name=f"{sys_key}")
        if isinstance(sys_values, dict):
            # if no system is defined or there are no modules in the system, the components have to be created as leaf
            # feature directly unter the root feature
            for subs_key, subs_values in sys_values.items():
                if subs_key in separate_components:
                    system = eT.SubElement(
                        root, "feature", mandatory="true", name=f"{subs_key}"
                    )
                else:
                    system = eT.SubElement(
                        root, "and", abstract="true", name=f"{subs_key}"
                    )
                if isinstance(subs_values, dict):
                    # if there are no components not belonging to a module, the components are provided as list and not
                    # as a dict
                    for mod_key, mod_values in subs_values.items():
                        if mod_key in separate_components:
                            module = eT.SubElement(
                                system, "feature", mandatory="true", name=f"{mod_key}"
                            )
                        else:
                            module = eT.SubElement(
                                system, "and", abstract="true", name=f"{mod_key}"
                            )
                        for comp_value in mod_values:
                            eT.SubElement(
                                module,
                                "feature",
                                mandatory="true",
                                name=f"{comp_value}",
                            )
                else:
                    for comp_value in subs_values:
                        eT.SubElement(
                            system, "feature", mandatory="true", name=f"{comp_value}"
                        )
        else:
            for comp_value in sys_values:
                eT.SubElement(root, "feature", mandatory="true", name=f"{comp_value}")
    return fm_xml


def get_separate_components(components_of_system_results, component_results, systems):
    components_of_modules = set()
    if component_results:
        for c_result in component_results.bindings:
            components_of_modules.add(c_result["component"].value)
    separate_components = defaultdict(list)
    if components_of_system_results:
        for sc_result in components_of_system_results.bindings:
            key = sc_result["component"].value
            value = sc_result["mechatronicsystem"].value
            if key not in components_of_modules:
                separate_components[key].append(value)
        for component, parents in separate_components.items():
            add_to_lowest_layer(component, parents, systems)
    return separate_components


def create_systems(modules, system_results, systemofsystem_results):
    systems = defaultdict(dict)
    if systemofsystem_results:
        for s_result in systemofsystem_results.bindings:
            key = s_result["mechatronicsystem"].value
            value = s_result["system"].value
            systems[key].update({value: modules.get(value, {})})
        else:
            if system_results:
                for s in system_results.bindings:
                    value = s["system"].value
                    systems.update({value: modules.get(value, {})})
    return systems


def create_modules(components, module_results):
    modules = defaultdict(dict)
    if module_results:
        for m_result in module_results.bindings:
            key = m_result["mechatronicsystem"].value
            value = m_result["module"].value
            modules[key].update({value: components.get(value, [])})
    return modules


def create_components(component_results):
    components = defaultdict(list)
    if component_results:
        for c_result in component_results.bindings:
            key = c_result["module"].value
            value = c_result["component"].value
            components[key].append(value)
    return components


if __name__ == "__main__":
    filename = "TuggerTrain/input/GB1-C-Frame.xml"
    main(filename)
