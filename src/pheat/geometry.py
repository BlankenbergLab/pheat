"""Small vector geometry helpers used by the dependency-light core."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Optional, Sequence

from pheat.models import Coordinate


@dataclass(frozen=True)
class KabschTransform:
    """Rigid-body transform that superposes target coordinates onto reference."""

    reference_center: Coordinate
    target_center: Coordinate
    rotation: tuple[Coordinate, Coordinate, Coordinate]


def vec(values: Iterable[float]) -> Coordinate:
    items = tuple(float(value) for value in values)
    if len(items) != 3:
        raise ValueError("Expected a 3D vector.")
    return (items[0], items[1], items[2])


def add(a: Sequence[float], b: Sequence[float]) -> Coordinate:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def sub(a: Sequence[float], b: Sequence[float]) -> Coordinate:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def scale(a: Sequence[float], value: float) -> Coordinate:
    return (a[0] * value, a[1] * value, a[2] * value)


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a: Sequence[float], b: Sequence[float]) -> Coordinate:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def norm(a: Sequence[float]) -> float:
    return math.sqrt(dot(a, a))


def normalize(a: Sequence[float], *, fallback: Coordinate = (1.0, 0.0, 0.0)) -> Coordinate:
    length = norm(a)
    if length < 1e-12:
        return fallback
    return (a[0] / length, a[1] / length, a[2] / length)


def distance(a: Sequence[float], b: Sequence[float]) -> float:
    return norm(sub(a, b))


def angle_degrees(
    a: Sequence[float],
    b: Sequence[float],
    c: Sequence[float],
) -> float:
    """Return the A-B-C bond angle in degrees, with B as the center atom."""

    ba_raw = sub(a, b)
    bc_raw = sub(c, b)
    if norm(ba_raw) < 1e-12 or norm(bc_raw) < 1e-12:
        raise ValueError("Cannot compute angle for degenerate coordinates.")
    ba = normalize(ba_raw)
    bc = normalize(bc_raw)
    cosine = max(-1.0, min(1.0, dot(ba, bc)))
    return math.degrees(math.acos(cosine))


def dihedral_degrees(
    a: Sequence[float],
    b: Sequence[float],
    c: Sequence[float],
    d: Sequence[float],
) -> float:
    """Return the signed A-B-C-D dihedral angle in degrees."""

    b0 = sub(a, b)
    b1 = sub(c, b)
    b2 = sub(d, c)
    b1_unit = normalize(b1)

    v = sub(b0, scale(b1_unit, dot(b0, b1_unit)))
    w = sub(b2, scale(b1_unit, dot(b2, b1_unit)))
    if norm(v) < 1e-12 or norm(w) < 1e-12:
        raise ValueError("Cannot compute dihedral for degenerate coordinates.")

    x = dot(v, w)
    y = dot(cross(b1_unit, v), w)
    return _normalize_degrees(math.degrees(math.atan2(y, x)))


def centroid(coords: Sequence[Sequence[float]]) -> Coordinate:
    if not coords:
        raise ValueError("Cannot compute centroid of an empty coordinate set.")
    total = (0.0, 0.0, 0.0)
    for coord in coords:
        total = add(total, coord)
    return scale(total, 1.0 / len(coords))


def radius_of_gyration(
    coords: Sequence[Sequence[float]],
    *,
    weights: Optional[Sequence[float]] = None,
) -> float:
    """Return the root-mean-square distance of coordinates from their center.

    With ``weights`` omitted this is the usual unweighted geometric radius of
    gyration. With weights supplied, the center is the weighted center of mass
    and the squared distances are averaged by the same weights.
    """

    if not coords:
        raise ValueError("Cannot compute radius of gyration of an empty coordinate set.")
    normalized_coords = [vec(coord) for coord in coords]

    if weights is None:
        center = centroid(normalized_coords)
        squared_sum = sum(distance(coord, center) ** 2 for coord in normalized_coords)
        return math.sqrt(squared_sum / len(normalized_coords))

    if len(weights) != len(normalized_coords):
        raise ValueError("Radius of gyration weights must match the coordinate count.")
    normalized_weights = [float(weight) for weight in weights]
    if any(weight < 0.0 for weight in normalized_weights):
        raise ValueError("Radius of gyration weights must be non-negative.")
    total_weight = sum(normalized_weights)
    if total_weight <= 0.0:
        raise ValueError("Radius of gyration weights must have a positive total.")

    weighted_total = (0.0, 0.0, 0.0)
    for coord, weight in zip(normalized_coords, normalized_weights):
        weighted_total = add(weighted_total, scale(coord, weight))
    center = scale(weighted_total, 1.0 / total_weight)
    squared_sum = sum(
        weight * distance(coord, center) ** 2
        for coord, weight in zip(normalized_coords, normalized_weights)
    )
    return math.sqrt(squared_sum / total_weight)


def _normalize_degrees(angle: float) -> float:
    while angle <= -180.0:
        angle += 360.0
    while angle > 180.0:
        angle -= 360.0
    return angle


def rotate_around_axis(
    point: Sequence[float],
    axis_start: Sequence[float],
    axis_end: Sequence[float],
    angle_degrees: float,
) -> Coordinate:
    """Rotate a point around an axis using Rodrigues' formula."""

    theta = math.radians(angle_degrees)
    axis = normalize(sub(axis_end, axis_start))
    relative = sub(point, axis_start)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    rotated = add(
        add(scale(relative, cos_t), scale(cross(axis, relative), sin_t)),
        scale(axis, dot(axis, relative) * (1.0 - cos_t)),
    )
    return add(axis_start, rotated)


def place_atom(
    previous: Sequence[float],
    anchor: Sequence[float],
    parent: Sequence[float],
    *,
    length: float,
    angle_degrees: float,
    dihedral_degrees: float,
) -> Coordinate:
    """Place a new atom D from A-B-C using length C-D, angle B-C-D, dihedral A-B-C-D."""

    # This is a small Z-matrix/internal-coordinate construction used by
    # residue-geometry reconstruction. The three reference atoms define the local coordinate
    # frame; bond length, bond angle, and dihedral then place the new atom.
    bc = normalize(sub(parent, anchor))
    ba = normalize(sub(previous, anchor), fallback=(0.0, 1.0, 0.0))
    normal = normalize(cross(ba, bc), fallback=(0.0, 0.0, 1.0))
    binormal = normalize(cross(normal, bc), fallback=(0.0, 1.0, 0.0))

    theta = math.radians(angle_degrees)
    phi = math.radians(dihedral_degrees)
    direction = add(
        scale(bc, -math.cos(theta)),
        scale(add(scale(binormal, math.cos(phi)), scale(normal, math.sin(phi))), math.sin(theta)),
    )
    return add(parent, scale(normalize(direction), length))


def kabsch_align(
    reference: Sequence[Sequence[float]],
    target: Sequence[Sequence[float]],
) -> list[Coordinate]:
    """Return target coordinates optimally superposed onto reference."""

    if len(reference) == 0 and len(target) == 0:
        return []
    return apply_kabsch_transform(target, kabsch_transform(reference, target))


def kabsch_transform(
    reference: Sequence[Sequence[float]],
    target: Sequence[Sequence[float]],
) -> KabschTransform:
    """Return the least-squares rigid transform from target onto reference."""

    if len(reference) != len(target):
        raise ValueError("Alignment inputs must have the same length.")
    if len(reference) == 0:
        raise ValueError("Alignment inputs cannot be empty.")

    np = _require_numpy_for_kabsch()
    ref = _coordinate_array(np, reference)
    mob = _coordinate_array(np, target)
    if ref.shape != mob.shape:
        raise ValueError("Alignment inputs must have matching coordinate shapes.")

    # Kabsch first removes translation by centering both structures, then finds
    # the least-squares rotation that superposes target onto reference.
    ref_center = ref.mean(axis=0)
    mob_center = mob.mean(axis=0)
    ref_centered = ref - ref_center
    mob_centered = mob - mob_center
    covariance = mob_centered.T @ ref_centered
    left, _singular_values, right_t = np.linalg.svd(covariance)
    # Correct improper rotations so mirror images are not accepted as an optimal
    # rigid-body superposition.
    handedness = -1.0 if np.linalg.det(left @ right_t) < 0.0 else 1.0
    rotation = left @ np.diag([1.0, 1.0, handedness]) @ right_t
    rotation_rows = [vec(row) for row in rotation.tolist()]
    return KabschTransform(
        reference_center=vec(ref_center.tolist()),
        target_center=vec(mob_center.tolist()),
        rotation=(rotation_rows[0], rotation_rows[1], rotation_rows[2]),
    )


def apply_kabsch_transform(
    coords: Sequence[Sequence[float]],
    transform: KabschTransform,
) -> list[Coordinate]:
    """Apply a Kabsch rigid-body transform to any compatible coordinate set."""

    aligned = []
    rotation = transform.rotation
    for coord in coords:
        shifted = sub(coord, transform.target_center)
        aligned.append(
            (
                shifted[0] * rotation[0][0]
                + shifted[1] * rotation[1][0]
                + shifted[2] * rotation[2][0]
                + transform.reference_center[0],
                shifted[0] * rotation[0][1]
                + shifted[1] * rotation[1][1]
                + shifted[2] * rotation[2][1]
                + transform.reference_center[1],
                shifted[0] * rotation[0][2]
                + shifted[1] * rotation[1][2]
                + shifted[2] * rotation[2][2]
                + transform.reference_center[2],
            )
        )
    return aligned


def kabsch_rmsd(
    reference: Sequence[Sequence[float]],
    target: Sequence[Sequence[float]],
    *,
    aligned_target: Optional[Sequence[Sequence[float]]] = None,
) -> float:
    """Return RMSD after Kabsch superposition or against pre-aligned coordinates."""

    if len(reference) != len(target):
        raise ValueError("RMSD inputs must have the same length.")
    if len(reference) == 0:
        return 0.0

    np = _require_numpy_for_kabsch()
    ref = _coordinate_array(np, reference)
    if aligned_target is None:
        aligned = _coordinate_array(np, kabsch_align(reference, target))
    else:
        if len(reference) != len(aligned_target):
            raise ValueError("Aligned RMSD inputs must have the same length.")
        aligned = _coordinate_array(np, aligned_target)
    if ref.shape != aligned.shape:
        raise ValueError("RMSD inputs must have matching coordinate shapes.")

    diff = ref - aligned
    return float(np.sqrt((diff * diff).sum() / len(ref)))


def _require_numpy_for_kabsch():
    try:
        import numpy as np
    except Exception as exc:  # pragma: no cover - depends on optional scientific stack
        raise RuntimeError(
            "True Kabsch alignment requires NumPy. Install pheat[scientific] or use the "
            "local Miniforge environment from environment.yml."
        ) from exc
    return np


def _coordinate_array(np, coords: Sequence[Sequence[float]]):
    array = np.asarray(coords, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("Inputs must be sequences of 3D coordinates.")
    return array
