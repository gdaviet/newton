# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Normal-cone classification of the contacts on triangulated surfaces of the LOX deformable bodies and colliders.

A contact direction is classified against the feature (vertex, edge, or face) that holds the
closest point on one primitive, with a small angular tolerance:

- *inside*: the direction lies in the normal cone of the feature; the contact keeps its
  closest-point direction.
- *outside*: the direction leaves the cone through a convex or flat part of the surface, so a
  neighboring feature holds the closest point; the contact is dropped.
- *concave*: the direction leaves the cone only through concave parts of the surface, where the
  closest point legitimately rests on the feature; the contact keeps the incident face normal
  closest to its direction.
- *penetrating*: the other primitive lies behind a one-sided feature; the contact keeps the
  outward normal of the incident face it exits through, which opposes moving further inside
  and admits moving out.

An edge is concave on the side whose faces both rise towards the direction. A vertex is
concave when every incident edge that rises towards the direction is a concave edge; a
saddle vertex with a rising convex edge is outside.

Each face has a side: zero for a two-sided face (cloth), and ``+1`` or ``-1`` for a one-sided
face (the boundary of a soft body or a collider), whose winding normal times the side points
outwards.
"""

import math

import numpy as np
import warp as wp

from .......utils.mesh import (
    MeshAdjacencyData,
    get_vertex_adjacent_edge_id_order,
    get_vertex_adjacent_face_id_order,
    get_vertex_num_adjacent_edges,
    get_vertex_num_adjacent_faces,
)

###
# Module interface
###

__all__ = [
    "CONTACT_CONE_CONCAVE",
    "CONTACT_CONE_INSIDE",
    "CONTACT_CONE_OUTSIDE",
    "CONTACT_CONE_PENETRATING",
    "SOFT_CONTACT_GEOMETRY_EPSILON",
    "classify_surface_contact",
    "find_surface_edge",
    "surface_face_sides",
    "surface_point_is_behind",
]

###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Constants
###


CONTACT_CONE_INSIDE = wp.constant(0)
"""The contact direction lies in the normal cone of its feature."""

CONTACT_CONE_OUTSIDE = wp.constant(1)
"""The contact direction leaves the normal cone of its feature through a convex or flat part of the surface."""

CONTACT_CONE_CONCAVE = wp.constant(2)
"""The contact direction leaves the normal cone of its feature only through concave parts of the surface."""

CONTACT_CONE_PENETRATING = wp.constant(3)
"""The other primitive lies behind a one-sided feature."""

SOFT_CONTACT_GEOMETRY_EPSILON = wp.constant(1.0e-12)
"""Length below which a direction or a normal is degenerate."""

_CONE_TOLERANCE = math.radians(5.0)
"""Angular tolerance of every normal cone [rad]."""

_CONE_SINE = wp.constant(math.sin(_CONE_TOLERANCE))
_CONE_COSINE = wp.constant(math.cos(_CONE_TOLERANCE))

_FEATURE_EPSILON = wp.constant(5.0e-3)
"""Barycentric weight below which a contact point lies on the opposite edge or vertex of its feature."""

_FEATURE_VERTEX = wp.constant(1)
_FEATURE_EDGE = wp.constant(2)
_FEATURE_FACE = wp.constant(3)


###
# Host functions
###


def surface_face_sides(tri_indices: np.ndarray, tet_indices: np.ndarray | None, particle_q: np.ndarray) -> np.ndarray:
    """Return the side of each surface triangle of the particles.

    A triangle that bounds a tetrahedron is one-sided, with the side that turns its winding normal
    away from the tetrahedron; the other triangles (cloth) are two-sided.

    Args:
        tri_indices: Surface triangles, shape ``[tri_count, 3]``.
        tet_indices: Tetrahedra, shape ``[tet_count, 4]``, or ``None``.
        particle_q: Rest positions of the particles, which orient the boundary triangles.

    Returns:
        The side of each triangle: ``0`` for two-sided, ``+1`` or ``-1`` for one-sided.
    """
    tri_indices = np.asarray(tri_indices, dtype=np.int64).reshape(-1, 3)
    sides = np.zeros(len(tri_indices), dtype=np.float32)
    if tet_indices is None or len(tet_indices) == 0 or len(tri_indices) == 0:
        return sides
    tet_indices = np.asarray(tet_indices, dtype=np.int64).reshape(-1, 4)
    behind = {}
    for tet in tet_indices:
        for corner in range(4):
            face = tuple(sorted(int(v) for k, v in enumerate(tet) if k != corner))
            behind[face] = int(tet[corner])
    q = np.asarray(particle_q, dtype=np.float64)
    for index, (a, b, c) in enumerate(tri_indices):
        opposite = behind.get(tuple(sorted((int(a), int(b), int(c)))))
        if opposite is None:
            continue
        normal = np.cross(q[b] - q[a], q[c] - q[a])
        sides[index] = -1.0 if normal @ (q[opposite] - q[a]) > 0.0 else 1.0
    return sides


###
# Functions
###


@wp.func
def _face_normal(face: int, position: wp.array[wp.vec3], tri_indices: wp.array2d[wp.int32]) -> wp.vec3:
    """Return the unit winding normal of a triangle, or zero for a degenerate triangle."""
    a = position[tri_indices[face, 0]]
    normal = wp.cross(position[tri_indices[face, 1]] - a, position[tri_indices[face, 2]] - a)
    length = wp.length(normal)
    if length <= SOFT_CONTACT_GEOMETRY_EPSILON:
        return wp.vec3(0.0)
    return normal / length


@wp.func
def _side_normal(
    face: int,
    direction: wp.vec3,
    position: wp.array[wp.vec3],
    tri_indices: wp.array2d[wp.int32],
    face_side: wp.array[wp.float32],
) -> wp.vec3:
    """Return the outward normal of a one-sided face, or the normal of a two-sided face towards the direction."""
    normal = _face_normal(face, position, tri_indices)
    side = face_side[face]
    if side != 0.0:
        return side * normal
    if wp.dot(normal, direction) < 0.0:
        return -normal
    return normal


@wp.func
def _opposite_vertex(face: int, vertex_0: int, vertex_1: int, tri_indices: wp.array2d[wp.int32]) -> int:
    """Return the vertex of a triangle opposite one of its edges."""
    for corner in range(3):
        vertex = tri_indices[face, corner]
        if vertex != vertex_0 and vertex != vertex_1:
            return vertex
    return -1


@wp.func
def find_surface_edge(
    vertex_0: int, vertex_1: int, edge_indices: wp.array2d[wp.int32], adjacency: MeshAdjacencyData
) -> int:
    """Return the surface edge joining two vertices, or ``-1`` if none exists."""
    for adjacent_edge in range(get_vertex_num_adjacent_edges(adjacency, vertex_0)):
        edge, local_slot = get_vertex_adjacent_edge_id_order(adjacency, vertex_0, adjacent_edge)
        other = int(-1)
        if local_slot == 2:
            other = edge_indices[edge, 3]
        elif local_slot == 3:
            other = edge_indices[edge, 2]
        if other == vertex_1:
            return edge
    return -1


@wp.func
def _find_surface_face(
    vertex_0: int, vertex_1: int, vertex_2: int, tri_indices: wp.array2d[wp.int32], adjacency: MeshAdjacencyData
) -> int:
    """Return the triangle with three given vertices, or ``-1`` if none exists."""
    for adjacent_face in range(get_vertex_num_adjacent_faces(adjacency, vertex_0)):
        face, _order = get_vertex_adjacent_face_id_order(adjacency, vertex_0, adjacent_face)
        found = int(0)
        for corner in range(3):
            vertex = tri_indices[face, corner]
            if vertex == vertex_1 or vertex == vertex_2:
                found += 1
        if found == 2:
            return face
    return -1


@wp.func
def _locate_feature(
    particles: wp.vec3i,
    weights: wp.vec3,
    tri_indices: wp.array2d[wp.int32],
    edge_indices: wp.array2d[wp.int32],
    adjacency: MeshAdjacencyData,
):
    """Return the kind and index (vertex, edge, or face) of the feature that holds a surface point, or kind zero."""
    support = wp.vec3i(-1)
    count = int(0)
    for slot in range(3):
        if particles[slot] >= 0 and weights[slot] > _FEATURE_EPSILON:
            support[count] = particles[slot]
            count += 1
    if count == 1:
        return _FEATURE_VERTEX, support[0]
    if count == 2:
        edge = find_surface_edge(support[0], support[1], edge_indices, adjacency)
        if edge >= 0:
            return _FEATURE_EDGE, edge
    if count == 3:
        face = _find_surface_face(support[0], support[1], support[2], tri_indices, adjacency)
        if face >= 0:
            return _FEATURE_FACE, face
    return 0, -1


@wp.func
def _feature_face(kind: int, index: int, slot: int, adjacency: MeshAdjacencyData) -> int:
    """Return one incident face of a feature, or ``-1`` past its last face."""
    if kind == _FEATURE_FACE:
        if slot == 0:
            return index
        return -1
    if kind == _FEATURE_EDGE:
        if slot < 2:
            return adjacency.edge_tri_indices[index, slot]
        return -1
    if slot < get_vertex_num_adjacent_faces(adjacency, index):
        face, _order = get_vertex_adjacent_face_id_order(adjacency, index, slot)
        return face
    return -1


@wp.func
def _feature_face_count(kind: int, index: int, adjacency: MeshAdjacencyData) -> int:
    """Return the number of face slots of a feature."""
    if kind == _FEATURE_FACE:
        return 1
    if kind == _FEATURE_EDGE:
        return 2
    return get_vertex_num_adjacent_faces(adjacency, index)


@wp.func
def _feature_exit(
    kind: int,
    index: int,
    direction: wp.vec3,
    position: wp.array[wp.vec3],
    tri_indices: wp.array2d[wp.int32],
    edge_indices: wp.array2d[wp.int32],
    adjacency: MeshAdjacencyData,
    face_side: wp.array[wp.float32],
):
    """Return whether a one-sided feature has the direction behind it, and the outward face normal to exit along.

    The direction is behind a face when it leaves the face plane inwards; behind a convex edge
    or a vertex when it is behind every incident face, and behind a concave edge when it is
    behind either face. The exit normal is the outward face normal most opposed to the direction.
    """
    one_sided = bool(True)
    found = bool(False)
    behind_all = bool(True)
    behind_any = bool(False)
    exit_normal = wp.vec3(0.0)
    best = float(-2.0)
    for slot in range(_feature_face_count(kind, index, adjacency)):
        face = _feature_face(kind, index, slot, adjacency)
        if face >= 0:
            found = True
            if face_side[face] == 0.0:
                one_sided = False
            normal = _side_normal(face, direction, position, tri_indices, face_side)
            alignment = wp.dot(normal, direction)
            if alignment < 0.0:
                behind_any = True
            else:
                behind_all = False
            if -alignment > best:
                best = -alignment
                exit_normal = normal
    behind = behind_all
    if kind == _FEATURE_EDGE and one_sided:
        if _edge_is_concave(index, direction, position, tri_indices, edge_indices, adjacency, face_side):
            behind = behind_any
    return found and one_sided and behind, exit_normal


@wp.func
def _edge_is_concave(
    edge: int,
    direction: wp.vec3,
    position: wp.array[wp.vec3],
    tri_indices: wp.array2d[wp.int32],
    edge_indices: wp.array2d[wp.int32],
    adjacency: MeshAdjacencyData,
    face_side: wp.array[wp.float32],
) -> bool:
    """Whether both faces of an interior edge rise towards the outside of the edge on the contact side."""
    face_0 = adjacency.edge_tri_indices[edge, 0]
    face_1 = adjacency.edge_tri_indices[edge, 1]
    if face_0 < 0 or face_1 < 0:
        return False
    vertex_0 = edge_indices[edge, 2]
    vertex_1 = edge_indices[edge, 3]
    axis = _side_normal(face_0, direction, position, tri_indices, face_side) + _side_normal(
        face_1, direction, position, tri_indices, face_side
    )
    axis_length = wp.length(axis)
    if axis_length <= SOFT_CONTACT_GEOMETRY_EPSILON:
        return False
    axis /= axis_length
    midpoint = 0.5 * (position[vertex_0] + position[vertex_1])
    for side in range(2):
        face = face_0
        if side == 1:
            face = face_1
        offset = position[_opposite_vertex(face, vertex_0, vertex_1, tri_indices)] - midpoint
        if wp.dot(axis, offset) <= _CONE_SINE * wp.length(offset):
            return False
    return True


@wp.func
def _classify_vertex(
    vertex: int,
    direction: wp.vec3,
    position: wp.array[wp.vec3],
    tri_indices: wp.array2d[wp.int32],
    edge_indices: wp.array2d[wp.int32],
    adjacency: MeshAdjacencyData,
    face_side: wp.array[wp.float32],
):
    """Classify a direction against the polar cone of the edges of a vertex."""
    rising = bool(False)
    for adjacent_edge in range(get_vertex_num_adjacent_edges(adjacency, vertex)):
        edge, local_slot = get_vertex_adjacent_edge_id_order(adjacency, vertex, adjacent_edge)
        other = int(-1)
        if local_slot == 2:
            other = edge_indices[edge, 3]
        elif local_slot == 3:
            other = edge_indices[edge, 2]
        if other >= 0:
            offset = position[other] - position[vertex]
            if wp.dot(direction, offset) > _CONE_SINE * wp.length(offset):
                # A rising convex or flat edge holds points closer than the vertex
                if not _edge_is_concave(edge, direction, position, tri_indices, edge_indices, adjacency, face_side):
                    return CONTACT_CONE_OUTSIDE, wp.vec3(0.0)
                rising = True
    if not rising:
        return CONTACT_CONE_INSIDE, wp.vec3(0.0)
    best = wp.vec3(0.0)
    best_alignment = float(-2.0)
    for adjacent_face in range(get_vertex_num_adjacent_faces(adjacency, vertex)):
        face, _order = get_vertex_adjacent_face_id_order(adjacency, vertex, adjacent_face)
        normal = _side_normal(face, direction, position, tri_indices, face_side)
        if wp.dot(normal, direction) > best_alignment:
            best_alignment = wp.dot(normal, direction)
            best = normal
    if best_alignment <= 0.0:
        return CONTACT_CONE_OUTSIDE, wp.vec3(0.0)
    return CONTACT_CONE_CONCAVE, best


@wp.func
def _classify_edge(
    edge: int,
    direction: wp.vec3,
    position: wp.array[wp.vec3],
    tri_indices: wp.array2d[wp.int32],
    edge_indices: wp.array2d[wp.int32],
    adjacency: MeshAdjacencyData,
    face_side: wp.array[wp.float32],
):
    """Classify a direction against the dihedral cone of an edge, or the half-space of a boundary edge."""
    face_0 = adjacency.edge_tri_indices[edge, 0]
    face_1 = adjacency.edge_tri_indices[edge, 1]
    vertex_0 = edge_indices[edge, 2]
    vertex_1 = edge_indices[edge, 3]
    if face_0 < 0 and face_1 < 0:
        return CONTACT_CONE_INSIDE, wp.vec3(0.0)
    if face_0 < 0 or face_1 < 0:
        # A boundary edge admits every direction that does not point into its face
        face = wp.max(face_0, face_1)
        tangent = wp.normalize(position[vertex_1] - position[vertex_0])
        inward = position[_opposite_vertex(face, vertex_0, vertex_1, tri_indices)] - position[vertex_0]
        inward -= wp.dot(inward, tangent) * tangent
        if wp.dot(direction, inward) > _CONE_SINE * wp.length(inward):
            return CONTACT_CONE_OUTSIDE, wp.vec3(0.0)
        return CONTACT_CONE_INSIDE, wp.vec3(0.0)
    normal_0 = _side_normal(face_0, direction, position, tri_indices, face_side)
    normal_1 = _side_normal(face_1, direction, position, tri_indices, face_side)
    if _edge_is_concave(edge, direction, position, tri_indices, edge_indices, adjacency, face_side):
        best = normal_0
        if wp.dot(normal_1, direction) > wp.dot(normal_0, direction):
            best = normal_1
        if wp.dot(best, direction) <= 0.0:
            return CONTACT_CONE_OUTSIDE, wp.vec3(0.0)
        return CONTACT_CONE_CONCAVE, best
    axis = normal_0 + normal_1
    axis_length = wp.length(axis)
    if axis_length <= SOFT_CONTACT_GEOMETRY_EPSILON:
        return CONTACT_CONE_INSIDE, wp.vec3(0.0)
    axis /= axis_length
    # Expand the half angle of the dihedral cone by the tolerance
    cosine = wp.clamp(wp.dot(axis, normal_0), -1.0, 1.0)
    sine = wp.sqrt(wp.max(0.0, 1.0 - cosine * cosine))
    if wp.dot(axis, direction) < cosine * _CONE_COSINE - sine * _CONE_SINE:
        return CONTACT_CONE_OUTSIDE, wp.vec3(0.0)
    return CONTACT_CONE_INSIDE, wp.vec3(0.0)


@wp.func
def _classify_face(
    face: int,
    direction: wp.vec3,
    position: wp.array[wp.vec3],
    tri_indices: wp.array2d[wp.int32],
    face_side: wp.array[wp.float32],
):
    """Classify a direction against the normal of a face."""
    alignment = wp.dot(_side_normal(face, direction, position, tri_indices, face_side), direction)
    if alignment < _CONE_COSINE:
        return CONTACT_CONE_OUTSIDE, wp.vec3(0.0)
    return CONTACT_CONE_INSIDE, wp.vec3(0.0)


@wp.func
def classify_surface_contact(
    particles: wp.vec3i,
    weights: wp.vec3,
    direction: wp.vec3,
    position: wp.array[wp.vec3],
    tri_indices: wp.array2d[wp.int32],
    edge_indices: wp.array2d[wp.int32],
    adjacency: MeshAdjacencyData,
    face_side: wp.array[wp.float32],
):
    """Classify a contact direction against the feature of a surface point.

    Args:
        particles: Surface vertices of the contact point, padded with ``-1``.
        weights: Interpolation weights of the vertices; a vertex below the feature epsilon drops
            the point onto the opposite edge or vertex.
        direction: Unit direction from the surface point towards the other primitive.
        position: Vertex positions of the surface.
        tri_indices: Triangles of the surface.
        edge_indices: Edges of the surface as ``(opposite_0, opposite_1, vertex_0, vertex_1)``.
        adjacency: Vertex, edge, and triangle adjacency of the surface.
        face_side: Side of each triangle; zero for a two-sided triangle.

    Returns:
        The :data:`CONTACT_CONE_INSIDE`, :data:`CONTACT_CONE_OUTSIDE`, :data:`CONTACT_CONE_CONCAVE`, or
        :data:`CONTACT_CONE_PENETRATING` verdict, and for a concave or penetrated feature the face
        normal along which the other primitive should move away from the surface.
    """
    kind, index = _locate_feature(particles, weights, tri_indices, edge_indices, adjacency)
    if kind == 0:
        return CONTACT_CONE_INSIDE, wp.vec3(0.0)
    behind, exit_normal = _feature_exit(
        kind, index, direction, position, tri_indices, edge_indices, adjacency, face_side
    )
    if behind:
        return CONTACT_CONE_PENETRATING, exit_normal
    if kind == _FEATURE_VERTEX:
        return _classify_vertex(index, direction, position, tri_indices, edge_indices, adjacency, face_side)
    if kind == _FEATURE_EDGE:
        return _classify_edge(index, direction, position, tri_indices, edge_indices, adjacency, face_side)
    return _classify_face(index, direction, position, tri_indices, face_side)


@wp.func
def surface_point_is_behind(
    particles: wp.vec3i,
    weights: wp.vec3,
    point: wp.vec3,
    other: wp.vec3,
    position: wp.array[wp.vec3],
    tri_indices: wp.array2d[wp.int32],
    edge_indices: wp.array2d[wp.int32],
    adjacency: MeshAdjacencyData,
    face_side: wp.array[wp.float32],
) -> bool:
    """Whether another point lies behind the one-sided feature of a surface point."""
    kind, index = _locate_feature(particles, weights, tri_indices, edge_indices, adjacency)
    if kind == 0:
        return False
    behind, _exit_normal = _feature_exit(
        kind, index, other - point, position, tri_indices, edge_indices, adjacency, face_side
    )
    return behind
