from connect.mesh_allocator import allocate_meshes, _split_counts


def test_split_counts():
    assert _split_counts(8) == (7, 1)
    assert _split_counts(4) == (3, 1)
    assert _split_counts(2) == (1, 1)
    assert _split_counts(1) == (1, 0)


def test_allocate_meshes_runs():
    alloc = allocate_meshes()
    assert len(alloc.target_devices) >= 1
    assert alloc.summary()
