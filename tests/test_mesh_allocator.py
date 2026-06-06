from connect.mesh_allocator import _make_mesh, _split_counts, allocate_meshes


def test_split_counts():
    assert _split_counts(16) == (14, 2)
    assert _split_counts(8) == (7, 1)
    assert _split_counts(4) == (3, 1)
    assert _split_counts(2) == (1, 1)
    assert _split_counts(1) == (1, 0)


def test_all_target_policy(monkeypatch):
    monkeypatch.setenv("SSD_TPU_ROLE", "target")
    alloc = allocate_meshes()
    assert len(alloc.target_devices) >= 1
    assert len(alloc.draft_devices) == 0
    assert alloc.policy == "all_target"


def test_allocate_meshes_runs():
    alloc = allocate_meshes()
    assert len(alloc.target_devices) >= 1
    assert alloc.summary()
    if alloc.target_mesh is not None:
        assert alloc.target_mesh.devices.ndim == len(alloc.target_mesh.axis_names)
    if alloc.draft_mesh is not None:
        assert alloc.draft_mesh.devices.ndim == len(alloc.draft_mesh.axis_names)
