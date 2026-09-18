import pytest

from scripts.formal.prepare_static_eval import validate_alfworld_binding


def binding():
    task = {"query": "Put a clean plate on the counter."}
    private = {
        "trajectory_relative_path": (
            "valid_seen/pick_clean_then_place_in_recep-Plate-None-CounterTop-19/trial"
        )
    }
    trajectory = {
        "task_type": "pick_clean_then_place_in_recep",
        "pddl_params": {
            "object_target": "Plate",
            "mrecep_target": "",
            "parent_target": "CounterTop",
            "toggle_target": "",
        },
        "turk_annotations": {
            "anns": [{"task_desc": "Put a clean plate on the counter."}]
        },
    }
    return task, private, trajectory


def test_alfworld_binding_accepts_matching_public_private_pair() -> None:
    validate_alfworld_binding(*binding())


@pytest.mark.parametrize("field", ["query", "object_target", "parent_target"])
def test_alfworld_binding_rejects_mismatched_public_private_pair(field: str) -> None:
    task, private, trajectory = binding()
    if field == "query":
        task[field] = "Wash a bowl."
    else:
        trajectory["pddl_params"][field] = "Mug" if field == "object_target" else "Drawer"
    with pytest.raises(ValueError):
        validate_alfworld_binding(task, private, trajectory)
