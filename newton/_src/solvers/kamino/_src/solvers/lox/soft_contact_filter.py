# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Normal-cone filters for LOX contacts on triangulated soft surfaces."""

import warp as wp

from ......utils.mesh import (
    MeshAdjacencyData,
    get_vertex_adjacent_edge_id_order,
    get_vertex_num_adjacent_edges,
)

# Angular expansion applied to every vertex, edge, and face normal cone [rad].
SOFT_CONTACT_NORMAL_CONE_TOLERANCE = wp.constant(0.08726646259971647)  # 5 degrees
SOFT_CONTACT_GEOMETRY_EPSILON = wp.constant(1.0e-12)
SOFT_CONTACT_PARAMETER_EPSILON = wp.constant(1.0e-4)
# The full-surface face optimizer converges only approximately on simplex boundaries.
SOFT_CONTACT_FACE_PARAMETER_EPSILON = wp.constant(5.0e-3)


@wp.func
def find_soft_edge(
    vertex_0: int,
    vertex_1: int,
    edge_indices: wp.array2d[wp.int32],
    adjacency: MeshAdjacencyData,
) -> int:
    """Return the mesh edge joining two vertices, or ``-1`` if none exists."""
    adjacent_edge_count = get_vertex_num_adjacent_edges(adjacency, vertex_0)
    for adjacent_edge in range(adjacent_edge_count):
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
def vertex_normal_cone_contains(
    vertex: int,
    direction: wp.vec3,
    position: wp.array[wp.vec3],
    edge_indices: wp.array2d[wp.int32],
    adjacency: MeshAdjacencyData,
) -> bool:
    """Test a feature-to-query direction against the polar cone of the incident mesh edges."""
    direction_length = wp.length(direction)
    if direction_length <= SOFT_CONTACT_GEOMETRY_EPSILON:
        return False
    angular_slack = wp.sin(SOFT_CONTACT_NORMAL_CONE_TOLERANCE)
    adjacent_edge_count = get_vertex_num_adjacent_edges(adjacency, vertex)
    for adjacent_edge in range(adjacent_edge_count):
        edge, local_slot = get_vertex_adjacent_edge_id_order(adjacency, vertex, adjacent_edge)
        other = int(-1)
        if local_slot == 2:
            other = edge_indices[edge, 3]
        elif local_slot == 3:
            other = edge_indices[edge, 2]
        if other >= 0:
            edge_direction = position[other] - position[vertex]
            if wp.dot(direction, edge_direction) > angular_slack * direction_length * wp.length(edge_direction):
                return False
    return True


@wp.func
def edge_normal_cone_contains(
    edge: int,
    direction: wp.vec3,
    cone_axis: wp.array[wp.vec3],
    cone_cosine: wp.array[float],
) -> bool:
    """Test a feature-to-query direction against an interior or boundary edge cone."""
    cosine = cone_cosine[edge]
    if cosine <= -1.0:
        axis = cone_axis[edge]
        axis_length = wp.length(axis)
        # Degenerate edges remain conservative because they have no reliable cone.
        if axis_length <= SOFT_CONTACT_GEOMETRY_EPSILON:
            return True
        direction_length = wp.length(direction)
        if direction_length <= SOFT_CONTACT_GEOMETRY_EPSILON:
            return False
        angular_slack = wp.sin(SOFT_CONTACT_NORMAL_CONE_TOLERANCE)
        return wp.dot(direction, axis) >= -angular_slack * direction_length * axis_length
    direction_length = wp.length(direction)
    if direction_length <= SOFT_CONTACT_GEOMETRY_EPSILON:
        return False
    tolerance_cosine = wp.cos(SOFT_CONTACT_NORMAL_CONE_TOLERANCE)
    tolerance_sine = wp.sin(SOFT_CONTACT_NORMAL_CONE_TOLERANCE)
    cone_sine = wp.sqrt(wp.max(0.0, 1.0 - cosine * cosine))
    expanded_cosine = cosine * tolerance_cosine - cone_sine * tolerance_sine
    return wp.abs(wp.dot(direction, cone_axis[edge])) >= expanded_cosine * direction_length


@wp.func
def edge_feature_normal_cone_contains(
    edge: int,
    parameter: float,
    direction: wp.vec3,
    position: wp.array[wp.vec3],
    edge_indices: wp.array2d[wp.int32],
    adjacency: MeshAdjacencyData,
    cone_axis: wp.array[wp.vec3],
    cone_cosine: wp.array[float],
) -> bool:
    """Test the vertex cone at an endpoint and the face-normal cone in the edge interior."""
    if parameter <= SOFT_CONTACT_PARAMETER_EPSILON:
        return vertex_normal_cone_contains(edge_indices[edge, 2], direction, position, edge_indices, adjacency)
    if parameter >= 1.0 - SOFT_CONTACT_PARAMETER_EPSILON:
        return vertex_normal_cone_contains(edge_indices[edge, 3], direction, position, edge_indices, adjacency)
    return edge_normal_cone_contains(edge, direction, cone_axis, cone_cosine)


@wp.func
def soft_edge_feature_normal_cone_contains(
    vertex_0: int,
    vertex_1: int,
    weight_0: float,
    weight_1: float,
    direction: wp.vec3,
    position: wp.array[wp.vec3],
    edge_indices: wp.array2d[wp.int32],
    adjacency: MeshAdjacencyData,
    cone_axis: wp.array[wp.vec3],
    cone_cosine: wp.array[float],
) -> bool:
    """Test an edge record, reducing endpoint contacts to their vertex cone."""
    edge = find_soft_edge(vertex_0, vertex_1, edge_indices, adjacency)
    weight_sum = weight_0 + weight_1
    if edge < 0 or weight_sum <= SOFT_CONTACT_GEOMETRY_EPSILON:
        return False
    parameter = float(0.0)
    if edge_indices[edge, 2] == vertex_0 and edge_indices[edge, 3] == vertex_1:
        parameter = weight_1 / weight_sum
    elif edge_indices[edge, 2] == vertex_1 and edge_indices[edge, 3] == vertex_0:
        parameter = weight_0 / weight_sum
    else:
        return False
    return edge_feature_normal_cone_contains(
        edge,
        parameter,
        direction,
        position,
        edge_indices,
        adjacency,
        cone_axis,
        cone_cosine,
    )


@wp.func
def face_interior_normal_cone_contains(
    vertex_0: int,
    vertex_1: int,
    vertex_2: int,
    barycentric: wp.vec3,
    direction: wp.vec3,
    position: wp.array[wp.vec3],
) -> bool:
    """Require a strict triangle-interior point and a two-sided face normal."""
    if (
        barycentric[0] <= SOFT_CONTACT_PARAMETER_EPSILON
        or barycentric[1] <= SOFT_CONTACT_PARAMETER_EPSILON
        or barycentric[2] <= SOFT_CONTACT_PARAMETER_EPSILON
    ):
        return False
    face_normal = wp.cross(position[vertex_1] - position[vertex_0], position[vertex_2] - position[vertex_0])
    face_normal_length = wp.length(face_normal)
    if face_normal_length <= SOFT_CONTACT_GEOMETRY_EPSILON:
        return False
    face_normal /= face_normal_length
    direction_length = wp.length(direction)
    if direction_length <= SOFT_CONTACT_GEOMETRY_EPSILON:
        return False
    return wp.abs(wp.dot(direction, face_normal)) >= (wp.cos(SOFT_CONTACT_NORMAL_CONE_TOLERANCE) * direction_length)


@wp.func
def triangle_feature_normal_cone_contains(
    vertex_0: int,
    vertex_1: int,
    vertex_2: int,
    barycentric: wp.vec3,
    direction: wp.vec3,
    position: wp.array[wp.vec3],
    edge_indices: wp.array2d[wp.int32],
    adjacency: MeshAdjacencyData,
    cone_axis: wp.array[wp.vec3],
    cone_cosine: wp.array[float],
) -> bool:
    """Test the normal cone of the vertex, edge, or face selected by barycentrics."""
    on_0 = barycentric[0] <= SOFT_CONTACT_FACE_PARAMETER_EPSILON
    on_1 = barycentric[1] <= SOFT_CONTACT_FACE_PARAMETER_EPSILON
    on_2 = barycentric[2] <= SOFT_CONTACT_FACE_PARAMETER_EPSILON
    if on_1 and on_2:
        return vertex_normal_cone_contains(vertex_0, direction, position, edge_indices, adjacency)
    if on_0 and on_2:
        return vertex_normal_cone_contains(vertex_1, direction, position, edge_indices, adjacency)
    if on_0 and on_1:
        return vertex_normal_cone_contains(vertex_2, direction, position, edge_indices, adjacency)
    if on_0:
        return soft_edge_feature_normal_cone_contains(
            vertex_1,
            vertex_2,
            barycentric[1],
            barycentric[2],
            direction,
            position,
            edge_indices,
            adjacency,
            cone_axis,
            cone_cosine,
        )
    if on_1:
        return soft_edge_feature_normal_cone_contains(
            vertex_0,
            vertex_2,
            barycentric[0],
            barycentric[2],
            direction,
            position,
            edge_indices,
            adjacency,
            cone_axis,
            cone_cosine,
        )
    if on_2:
        return soft_edge_feature_normal_cone_contains(
            vertex_0,
            vertex_1,
            barycentric[0],
            barycentric[1],
            direction,
            position,
            edge_indices,
            adjacency,
            cone_axis,
            cone_cosine,
        )
    return face_interior_normal_cone_contains(
        vertex_0,
        vertex_1,
        vertex_2,
        barycentric,
        direction,
        position,
    )


@wp.func
def soft_surface_contact_normal_cone_contains(
    particles: wp.vec3i,
    barycentric: wp.vec3,
    direction: wp.vec3,
    position: wp.array[wp.vec3],
    edge_indices: wp.array2d[wp.int32],
    adjacency: MeshAdjacencyData,
    cone_axis: wp.array[wp.vec3],
    cone_cosine: wp.array[float],
) -> bool:
    """Filter a soft feature using a direction from that feature toward the query point."""
    if particles[1] < 0:
        return vertex_normal_cone_contains(particles[0], direction, position, edge_indices, adjacency)
    if particles[2] < 0:
        return soft_edge_feature_normal_cone_contains(
            particles[0],
            particles[1],
            barycentric[0],
            barycentric[1],
            direction,
            position,
            edge_indices,
            adjacency,
            cone_axis,
            cone_cosine,
        )
    return triangle_feature_normal_cone_contains(
        particles[0],
        particles[1],
        particles[2],
        barycentric,
        direction,
        position,
        edge_indices,
        adjacency,
        cone_axis,
        cone_cosine,
    )


@wp.kernel
def compute_soft_edge_normal_cones(
    position: wp.array[wp.vec3],
    edge_indices: wp.array2d[wp.int32],
    cone_axis: wp.array[wp.vec3],
    cone_cosine: wp.array[float],
):
    """Precompute current incident-face cone axes and half-angle cosines."""
    edge = wp.tid()
    opposite_0 = edge_indices[edge, 0]
    opposite_1 = edge_indices[edge, 1]
    vertex_0 = edge_indices[edge, 2]
    vertex_1 = edge_indices[edge, 3]

    # A boundary edge has a 180-degree cone centered on the outward
    # in-face normal. A nonzero axis distinguishes it from an unavailable cone.
    if opposite_0 < 0 or opposite_1 < 0:
        opposite = opposite_0
        if opposite < 0:
            opposite = opposite_1
        edge_direction = position[vertex_1] - position[vertex_0]
        edge_length = wp.length(edge_direction)
        if opposite < 0 or edge_length <= SOFT_CONTACT_GEOMETRY_EPSILON:
            cone_axis[edge] = wp.vec3(0.0)
            cone_cosine[edge] = -1.0
            return
        edge_length_squared = edge_length * edge_length
        to_opposite = position[opposite] - position[vertex_0]
        inward = to_opposite - wp.dot(to_opposite, edge_direction) * edge_direction / edge_length_squared
        inward_length = wp.length(inward)
        if inward_length <= SOFT_CONTACT_GEOMETRY_EPSILON:
            cone_axis[edge] = wp.vec3(0.0)
            cone_cosine[edge] = -1.0
            return
        cone_axis[edge] = -inward / inward_length
        cone_cosine[edge] = -1.0
        return

    edge_0 = position[vertex_0] - position[opposite_0]
    edge_1 = position[vertex_1] - position[opposite_0]
    normal_0 = wp.cross(edge_0, edge_1)
    edge_2 = position[vertex_1] - position[opposite_1]
    edge_3 = position[vertex_0] - position[opposite_1]
    normal_1 = wp.cross(edge_2, edge_3)
    length_0 = wp.length(normal_0)
    length_1 = wp.length(normal_1)
    if length_0 <= SOFT_CONTACT_GEOMETRY_EPSILON or length_1 <= SOFT_CONTACT_GEOMETRY_EPSILON:
        cone_axis[edge] = wp.vec3(0.0)
        cone_cosine[edge] = -1.0
        return

    normal_0 /= length_0
    normal_1 /= length_1
    axis = normal_0 + normal_1
    axis_length = wp.length(axis)
    if axis_length <= SOFT_CONTACT_GEOMETRY_EPSILON:
        cone_axis[edge] = wp.vec3(0.0)
        cone_cosine[edge] = -1.0
        return

    axis /= axis_length
    cone_axis[edge] = axis
    cone_cosine[edge] = wp.min(wp.dot(axis, normal_0), wp.dot(axis, normal_1))
