import numpy as np
from scipy import ndimage


def keep_largest(mask: np.ndarray) -> np.ndarray:
    lab, n = ndimage.label(mask > 0)
    if n <= 1:
        return (mask > 0).astype(np.uint8)
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return (lab == sizes.argmax()).astype(np.uint8)


def remove_small(mask: np.ndarray, min_vox: int) -> np.ndarray:
    lab, n = ndimage.label(mask > 0)
    if n == 0:
        return (mask > 0).astype(np.uint8)
    sizes = np.bincount(lab.ravel())
    keep = {i for i in range(1, n + 1) if sizes[i] >= min_vox}
    return np.isin(lab, list(keep)).astype(np.uint8)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    m = ndimage.binary_fill_holes(mask > 0)
    for z in range(m.shape[2]):
        m[:, :, z] = ndimage.binary_fill_holes(m[:, :, z])
    return m.astype(np.uint8)


def morph_close(mask: np.ndarray, r: int = 1) -> np.ndarray:
    st = ndimage.generate_binary_structure(3, 1)
    return ndimage.binary_closing(mask > 0, structure=st, iterations=r).astype(np.uint8)


def postprocess_cavity(mask: np.ndarray, keep_ratio: float = 0.20, bridge_vox: int = 3) -> np.ndarray:
    m = mask > 0
    lab, n = ndimage.label(m)
    if n <= 1:
        return fill_holes(m.astype(np.uint8))
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    main = int(sizes.argmax())
    main_dil = ndimage.binary_dilation(lab == main, iterations=bridge_vox)
    keep = {main}
    for i in range(1, n + 1):
        if i != main and sizes[i] >= keep_ratio * sizes[main] and bool((main_dil & (lab == i)).any()):
            keep.add(i)
    return fill_holes(np.isin(lab, list(keep)).astype(np.uint8))


def _place(out, comp, val):
    out[(comp > 0) & (out == 0)] = val


def postprocess_cardiac(lab: np.ndarray, min_vox_pv: int = 80, close_r: int = 1) -> np.ndarray:
    out = np.zeros_like(lab, dtype=np.uint8)
    _place(out, fill_holes(keep_largest(lab == 1)), 1)
    _place(out, keep_largest(lab == 2), 2)
    _place(out, morph_close(remove_small(lab == 3, min_vox_pv), close_r), 3)
    return out


def _pv_adjacent_to_la(pv, la_dil):
    comp, n = ndimage.label(pv)
    keep = np.zeros_like(pv)
    for k in range(1, n + 1):
        c = comp == k
        if (c & la_dil).any():
            keep |= c
    return keep


def postprocess_cardiac_pvhyst(lab: np.ndarray, pv_prob: np.ndarray,
                               t_hi: float = 0.5, t_lo: float = 0.25) -> np.ndarray:
    out = np.zeros_like(lab, dtype=np.uint8)
    la = keep_largest(lab == 1)
    _place(out, fill_holes(la), 1)
    _place(out, keep_largest(lab == 2), 2)
    la_dil = ndimage.binary_dilation(la, iterations=3)
    seed = pv_prob >= t_hi
    cand = ((pv_prob >= t_lo) & (lab == 0)) | seed
    comp, n = ndimage.label(cand, structure=np.ones((3, 3, 3), np.uint8))
    if n:
        sl = np.unique(comp[seed])
        grown = np.isin(comp, sl[sl > 0])
    else:
        grown = np.zeros_like(cand)
    _place(out, _pv_adjacent_to_la((lab == 3) | grown, la_dil), 3)
    return out


def remove_scar_off_blob(scar: np.ndarray, cavity: np.ndarray, spacing, max_mm: float = 6.0) -> np.ndarray:
    dist = ndimage.distance_transform_edt(~(cavity > 0), sampling=spacing)
    lab, n = ndimage.label(scar > 0)
    if n == 0:
        return np.zeros_like(scar, dtype=np.uint8)
    ids = np.arange(1, n + 1)
    keep = ids[np.asarray(ndimage.minimum(dist, labels=lab, index=ids)) <= max_mm]
    return np.isin(lab, keep).astype(np.uint8)
