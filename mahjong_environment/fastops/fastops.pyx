# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False, initializedcheck=False
# distutils: language = c

cimport cython
from libc.stdint cimport int8_t
import numpy as np
cimport numpy as cnp

ctypedef cnp.int8_t INT8
ctypedef cnp.float32_t FLOAT32

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def build_observation_i8_f32(
    cnp.int8_t[:] hand,
    cnp.int8_t[:] drawn_tile,
    cnp.int8_t[:, :] rivers,
    cnp.int8_t[:, :] melds,
    cnp.int8_t[:] riichi,
    cnp.int8_t[:, :] dora,
    cnp.float32_t[:] scores,
    cnp.float32_t[:] game_info,
    cnp.int8_t[:] phase,
    cnp.int8_t[:] mask,
):
    cdef Py_ssize_t A = mask.shape[0]
    cdef Py_ssize_t i8_size = 34 + 34 + 136 + 136 + 4 + 170 + 3 + A
    cdef Py_ssize_t f32_size = 4 + 5

    cdef cnp.ndarray[INT8, ndim=1] out_i8 = np.empty((i8_size,), dtype=np.int8)
    cdef cnp.ndarray[FLOAT32, ndim=1] out_f32 = np.empty((f32_size,), dtype=np.float32)

    cdef Py_ssize_t off = 0
    cdef Py_ssize_t i, j

    for i in range(34): out_i8[off + i] = hand[i]
    off += 34
    for i in range(34): out_i8[off + i] = drawn_tile[i]
    off += 34
    for i in range(4):
        for j in range(34): out_i8[off + i*34 + j] = rivers[i, j]
    off += 136
    for i in range(4):
        for j in range(34): out_i8[off + i*34 + j] = melds[i, j]
    off += 136
    for i in range(4): out_i8[off + i] = riichi[i]
    off += 4
    for i in range(5):
        for j in range(34): out_i8[off + i*34 + j] = dora[i, j]
    off += 170
    for i in range(3): out_i8[off + i] = phase[i]
    off += 3
    for i in range(A): out_i8[off + i] = mask[i]

    cdef Py_ssize_t foff = 0
    for i in range(4): out_f32[foff + i] = scores[i]
    foff += 4
    for i in range(5): out_f32[foff + i] = game_info[i]

    return out_i8, out_f32

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def compute_fast_actions(
    cnp.int8_t[:] hand34,
    cnp.int8_t[:] drawn34,
    cnp.int8_t[:] riichi4,
    int phase,
    int player_id,
    int last_discard_idx,
    int last_discard_player,
) -> cnp.ndarray:
    cdef Py_ssize_t A = 112
    cdef cnp.ndarray[INT8, ndim=1] mask = np.zeros((A,), dtype=np.int8)
    cdef Py_ssize_t i
    cdef Py_ssize_t base

    if phase == 1:
        base = 0
        for i in range(34):
            if hand34[i] > 0 and base + i < A:
                mask[base + i] = 1
        return mask

    if phase == 2:
        mask[0] = 1
        if last_discard_idx >= 0:
            if 36 < A: mask[36] = 1
            if 37 < A: mask[37] = 1
            if 38 < A: mask[38] = 1
        return mask

    mask[0] = 1
    return mask


