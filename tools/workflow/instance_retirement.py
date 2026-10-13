"""Keep retired hypotheses as evidence while excluding them from rendered IDs."""


def active_instances(instances, *, rejected_statuses=("REVIEWED_REJECTED",)):
    """Validate identity links, clear retired pixel IDs, and return active records.

    Records are updated in place only after all identity links validate. Raw mask
    paths, hashes, prompts, and review evidence are preserved for every record.
    This selects rendering candidates; it does not grant annotation approval.
    """
    objects = {row["candidate_id"]: row for row in instances}
    if len(objects) != len(instances):
        raise ValueError("duplicate candidate identity")
    rejected = set(rejected_statuses)
    for row in instances:
        parent = row.get("alias_of")
        if parent is None:
            continue
        if parent == row["candidate_id"] or parent not in objects:
            raise ValueError("unknown or self alias")
        target = objects[parent]
        if target.get("alias_of") is not None or target.get("status") in rejected:
            raise ValueError("alias parent is retired")
    active = []
    for row in instances:
        if row.get("alias_of") is not None or row.get("status") in rejected:
            row.pop("local_id", None)
        else:
            active.append(row)
    return active
