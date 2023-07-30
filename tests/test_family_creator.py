import os

import FamilyCreator as mut


def assert_dict_equality(v_expected, v_test):
    if not v_expected.get("children", {}) or not v_expected.get("attributes", {}):
        return
    assert v_expected["attributes"] == v_test["attributes"]
    assert sorted(v_expected["children"]) == sorted(v_test["children"])
    assert_dict_equality(v_expected["children"], v_test["children"])


def test_family_creator_with_soft_gripper(tmp_path, monkeypatch):

    # set up temp dir for test output
    filepath_test_dir = tmp_path / "test_output/"
    filepath_test_dir.mkdir()

    # monkeypatch input and output filepaths
    monkeypatch.setattr(mut, "INPUT_DIR", "tests/data/SoftGripper/input/")
    monkeypatch.setattr(mut, "OUTPUT_DIR", f"{filepath_test_dir.as_posix()}/")

    # call FamilyCreator.main()
    mut.main()

    # assert all output files are as expected
    expected_files = sorted(os.listdir("tests/data/SoftGripper/expected_output/"))
    test_files = [
        f for f in sorted(os.listdir(filepath_test_dir)) if str(f).endswith("xml")
    ]
    assert expected_files == test_files

    for f_expected, f_test in zip(expected_files, test_files):
        # get trees
        fp_expected = f"tests/data/SoftGripper/expected_output/{f_expected}"
        fp_test = f"{filepath_test_dir.as_posix()}/{f_test}"
        root_expected = mut.get_element_tree(fp_expected)
        root_test = mut.get_element_tree(fp_test)

        # parse xml to dict
        fm_dict_expected = mut.parse_xml_to_dict(next(iter(root_expected)))
        fm_dict_test = mut.parse_xml_to_dict(next(iter(root_test)))

        # assert equality of feature model root
        assert sorted(fm_dict_expected) == sorted(fm_dict_test)

        # assert equality of feature model tree
        for k_expected, v_expected in fm_dict_test.items():
            v_test = fm_dict_test[k_expected]
            assert_dict_equality(v_expected, v_test)

        # parse constraints from xml to dict
        fm_dict_constraints_expected = mut.parse_constraints_to_dict(fp_expected)
        fm_dict_constraints_test = mut.parse_constraints_to_dict(fp_test)

        # assert equality of constraints
        for constraint in ["eq", "imp"]:
            assert sorted(fm_dict_constraints_expected.get(constraint, {})) == sorted(
                fm_dict_constraints_test.get(constraint, {})
            )
            for k_expected, v_expected in fm_dict_constraints_expected.get(
                constraint, {}
            ).items():
                v_test = fm_dict_constraints_test.get(constraint, {})[k_expected]
                assert sorted(v_expected) == sorted(v_test)
