"""COPC tiles must yield a writable plain LAZ with all other records kept."""
import laspy
import numpy as np
import pytest
from laspy.copc import CopcHierarchyVlr, CopcInfoVlr

from predict import _drop_copc_records


def _copc_like():
    header = laspy.LasHeader(point_format=6, version="1.4")
    header.vlrs.append(CopcInfoVlr())
    header.vlrs.append(laspy.VLR(user_id="test", record_id=7, record_data=b"kept"))
    las = laspy.LasData(header)
    las.x = np.array([0.0, 1.0])
    las.y = np.array([0.0, 1.0])
    las.z = np.array([0.0, 1.0])
    las.evlrs = [CopcHierarchyVlr()]
    return las


def test_copc_records_block_writing(tmp_path):
    with pytest.raises(NotImplementedError):
        _copc_like().write(tmp_path / "before.laz")


def test_drop_copc_records_keeps_other_records(tmp_path):
    las = _drop_copc_records(_copc_like())
    las.add_extra_dim(laspy.ExtraBytesParams(name="species_id", type=np.uint8))
    las.write(tmp_path / "after.laz")
    written = laspy.read(tmp_path / "after.laz")
    assert [r.user_id for r in written.vlrs if r.user_id == "copc"] == []
    assert [r.user_id for r in (written.evlrs or []) if r.user_id == "copc"] == []
    assert [r.record_data for r in written.vlrs if r.user_id == "test"] == [b"kept"]
    assert len(written.points) == 2
