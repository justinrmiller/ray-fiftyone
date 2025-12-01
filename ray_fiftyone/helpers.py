from __future__ import annotations

import logging
from typing import (
    Any,
)


import fiftyone as fo

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Helper functions for serialization
# -----------------------------------------------------------------------------


def _serialize_label(label: Any) -> dict[str, Any] | None:
    """
    Serialize a FiftyOne label to a JSON-compatible dictionary.

    Args:
        label: A FiftyOne Label instance or None.

    Returns:
        A dictionary representation of the label, or None if label is None.
    """
    if label is None:
        return None

    # Use FiftyOne's built-in serialization
    if hasattr(label, "to_dict"):
        return label.to_dict()

    # Fallback for unknown types
    return str(label)


def _deserialize_label(
    label_dict: dict[str, Any] | None,
    label_type: str | None = None,
) -> Any:
    """
    Deserialize a dictionary back to a FiftyOne label.

    Args:
        label_dict: A dictionary representation of a label.
        label_type: Optional type hint for the label class.

    Returns:
        A FiftyOne Label instance, or None if label_dict is None.
    """
    if label_dict is None:
        return None

    # Import FiftyOne lazily to avoid import errors when not needed
    import fiftyone.core.labels as fol

    # Map of type names to FiftyOne label classes
    label_classes: dict[str, type] = {
        "Classification": fol.Classification,
        "Classifications": fol.Classifications,
        "Detection": fol.Detection,
        "Detections": fol.Detections,
        "Polyline": fol.Polyline,
        "Polylines": fol.Polylines,
        "Keypoint": fol.Keypoint,
        "Keypoints": fol.Keypoints,
        "Segmentation": fol.Segmentation,
        "Heatmap": fol.Heatmap,
        "TemporalDetection": fol.TemporalDetection,
        "TemporalDetections": fol.TemporalDetections,
        "GeoLocation": fol.GeoLocation,
        "GeoLocations": fol.GeoLocations,
        "Regression": fol.Regression,
    }

    # Determine the label type from the dict or hint
    type_name = label_dict.get("_cls") or label_type
    if type_name and type_name in label_classes:
        label_cls = label_classes[type_name]
        return label_cls.from_dict(label_dict)

    # Return raw dict if we can't determine the type
    return label_dict


def _sample_to_dict(
    sample: Any,
    fields: list[str] | None = None,
    include_id: bool = True,
    include_private: bool = False,
) -> dict[str, Any]:
    """
    Convert a FiftyOne Sample to a dictionary suitable for PyArrow.

    Args:
        sample: A FiftyOne Sample instance.
        fields: Optional list of fields to include. If None, includes all fields.
        include_id: Whether to include the sample ID.
        include_private: Whether to include private fields (starting with '_').

    Returns:
        A dictionary with serialized sample data.
    """
    # Get the sample as a dict using FiftyOne's serialization
    sample_dict = sample.to_dict(include_private=include_private)

    # Filter fields if specified
    if fields is not None:
        filtered_dict: dict[str, Any] = {}
        for field in fields:
            if field in sample_dict:
                filtered_dict[field] = sample_dict[field]
        sample_dict = filtered_dict

    # Include sample ID if requested
    if include_id and "id" not in sample_dict:
        sample_dict["id"] = str(sample.id)

    # Serialize complex label objects
    result: dict[str, Any] = {}
    for key, value in sample_dict.items():
        if hasattr(value, "to_dict"):
            result[key] = _serialize_label(value)
        else:
            result[key] = value

    return result


def _dict_to_sample(
    row: dict[str, Any],
    label_fields: dict[str, str] | None = None,
) -> Any:
    """
    Convert a dictionary row to a FiftyOne Sample.

    Args:
        row: A dictionary containing sample data.
        label_fields: Optional mapping of field names to label types.

    Returns:
        A FiftyOne Sample instance.
    """

    # Extract filepath (required)
    filepath = row.get("filepath")
    if filepath is None:
        raise ValueError("Row must contain 'filepath' field")

    # Create the sample
    sample = fo.Sample(filepath=filepath)

    # Add other fields
    for key, value in row.items():
        if key in ("filepath", "id", "_id"):
            continue

        # Check if this is a label field
        if label_fields and key in label_fields:
            value = _deserialize_label(value, label_fields[key])

        # Handle nested dicts that might be labels
        elif isinstance(value, dict) and "_cls" in value:
            value = _deserialize_label(value)

        sample[key] = value

    return sample
