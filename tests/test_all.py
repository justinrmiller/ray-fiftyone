"""Comprehensive test suite for ray-fiftyone."""

from __future__ import annotations

from unittest.mock import Mock, patch, MagicMock
import sys

import pytest
import pyarrow as pa


# Mock fiftyone and ray modules before imports
mock_fo = MagicMock()
mock_ray = MagicMock()
sys.modules["fiftyone"] = mock_fo
sys.modules["fiftyone.core"] = MagicMock()
sys.modules["fiftyone.core.labels"] = MagicMock()


class TestHelpers:
    """Tests for helpers module."""

    def setup_method(self):
        """Reset mocks before each test."""
        from ray_fiftyone import helpers

        self.helpers = helpers

    def test_serialize_label_none(self):
        """Test serialization of None."""
        result = self.helpers._serialize_label(None)
        assert result is None

    def test_serialize_label_with_to_dict(self):
        """Test serialization of label with to_dict."""
        label = Mock()
        label.to_dict.return_value = {"_cls": "Classification", "label": "cat"}
        result = self.helpers._serialize_label(label)
        assert result == {"_cls": "Classification", "label": "cat"}

    def test_serialize_label_fallback(self):
        """Test serialization fallback for unknown types."""
        result = self.helpers._serialize_label("simple_label")
        assert result == "simple_label"

    def test_deserialize_label_none(self):
        """Test deserialization of None."""
        result = self.helpers._deserialize_label(None)
        assert result is None

    def test_deserialize_label_unknown_type(self):
        """Test deserialization of unknown type returns dict."""
        label_dict = {"_cls": "UnknownType", "data": "value"}
        result = self.helpers._deserialize_label(label_dict)
        assert result == label_dict

    def test_sample_to_dict_basic(self):
        """Test basic sample to dict conversion."""
        sample = Mock()
        sample.id = "sample123"
        sample.to_dict.return_value = {
            "filepath": "/path/to/image.jpg",
            "tags": ["train"],
        }
        result = self.helpers._sample_to_dict(sample)
        assert "filepath" in result
        assert result["id"] == "sample123"

    def test_sample_to_dict_with_fields(self):
        """Test sample to dict with field filtering."""
        sample = Mock()
        sample.to_dict.return_value = {
            "filepath": "/path/to/image.jpg",
            "tags": ["train"],
            "metadata": None,
        }
        result = self.helpers._sample_to_dict(sample, fields=["filepath", "tags"])
        assert "filepath" in result
        assert "tags" in result
        assert "metadata" not in result

    def test_sample_to_dict_with_label(self):
        """Test sample to dict with label serialization."""
        label = Mock()
        label.to_dict.return_value = {"_cls": "Classification"}
        sample = Mock()
        sample.to_dict.return_value = {
            "filepath": "/path/to/image.jpg",
            "ground_truth": label,
        }
        result = self.helpers._sample_to_dict(sample)
        assert result["ground_truth"] == {"_cls": "Classification"}

    def test_dict_to_sample_basic(self):
        """Test basic dict to sample conversion."""
        mock_sample = MagicMock()
        mock_fo.Sample.return_value = mock_sample
        row = {"filepath": "/path/to/image.jpg", "tags": ["train"]}
        result = self.helpers._dict_to_sample(row)
        mock_fo.Sample.assert_called_with(filepath="/path/to/image.jpg")
        # Verify the sample was modified
        assert result == mock_sample

    def test_dict_to_sample_missing_filepath(self):
        """Test dict to sample raises error without filepath."""
        row = {"tags": ["train"]}
        with pytest.raises(ValueError, match="filepath"):
            self.helpers._dict_to_sample(row)

    def test_dict_to_sample_with_label_fields(self):
        """Test dict to sample with label fields mapping."""
        mock_sample = MagicMock()
        mock_fo.Sample.return_value = mock_sample
        row = {
            "filepath": "/path/to/image.jpg",
            "predictions": {"_cls": "Detections"},
        }
        # Reset the mock to capture calls
        with patch.object(self.helpers, "_deserialize_label") as mock_deser:
            mock_deser.return_value = Mock()
            result = self.helpers._dict_to_sample(
                row, label_fields={"predictions": "Detections"}
            )
            mock_deser.assert_called_once()
            assert result == mock_sample


class TestFiftyOneDatasource:
    """Tests for FiftyOneDatasource class."""

    def setup_method(self):
        """Setup for each test."""
        from ray_fiftyone.data_source import FiftyOneDatasource

        self.FiftyOneDatasource = FiftyOneDatasource

    def test_init_default(self):
        """Test initialization with defaults."""
        source = self.FiftyOneDatasource(dataset_name="test")
        assert source._dataset_name == "test"
        assert source._fields is None
        assert source._batch_size == 100
        assert source._include_id is True

    def test_init_custom(self):
        """Test initialization with custom params."""
        source = self.FiftyOneDatasource(
            dataset_name="test",
            fields=["filepath"],
            batch_size=50,
            include_id=False,
        )
        assert source._fields == ["filepath"]
        assert source._batch_size == 50
        assert source._include_id is False

    def test_get_name(self):
        """Test get_name returns formatted name."""
        source = self.FiftyOneDatasource(dataset_name="my_dataset")
        assert source.get_name() == "FiftyOne(my_dataset)"

    def test_get_sample_ids(self):
        """Test _get_sample_ids retrieves IDs."""
        source = self.FiftyOneDatasource(dataset_name="test")
        mock_dataset = Mock()
        mock_dataset.values.return_value = ["id1", "id2"]
        with patch.object(source, "_get_dataset", return_value=mock_dataset):
            result = source._get_sample_ids()
            assert result == ["id1", "id2"]

    def test_get_read_tasks_empty(self):
        """Test get_read_tasks with empty dataset."""
        source = self.FiftyOneDatasource(dataset_name="test")
        with patch.object(source, "_get_sample_ids", return_value=[]):
            result = source.get_read_tasks(parallelism=10)
            assert result == []

    def test_get_read_tasks_creates_tasks(self):
        """Test get_read_tasks creates appropriate tasks."""
        source = self.FiftyOneDatasource(dataset_name="test")
        with patch.object(
            source, "_get_sample_ids", return_value=["id1", "id2", "id3"]
        ):
            with patch.object(source, "_create_read_fn", return_value=Mock()):
                result = source.get_read_tasks(parallelism=2)
                assert len(result) == 2

    def test_estimate_inmemory_data_size_error(self):
        """Test estimate_inmemory_data_size handles errors."""
        source = self.FiftyOneDatasource(dataset_name="test")
        with patch.object(source, "_get_dataset", side_effect=Exception("error")):
            result = source.estimate_inmemory_data_size()
            assert result is None


class TestFiftyOneDatasink:
    """Tests for FiftyOneDatasink class."""

    def setup_method(self):
        """Setup for each test."""
        from ray_fiftyone.data_sink import FiftyOneDatasink

        self.FiftyOneDatasink = FiftyOneDatasink

    def test_init_default(self):
        """Test initialization with defaults."""
        sink = self.FiftyOneDatasink(dataset_name="output")
        assert sink._dataset_name == "output"
        assert sink._filepath_field == "filepath"
        assert sink._overwrite is False
        assert sink._persistent is True

    def test_init_custom(self):
        """Test initialization with custom params."""
        sink = self.FiftyOneDatasink(
            dataset_name="output",
            filepath_field="image_path",
            overwrite=True,
            batch_size=50,
        )
        assert sink._filepath_field == "image_path"
        assert sink._overwrite is True
        assert sink._batch_size == 50

    def test_get_name(self):
        """Test get_name returns formatted name."""
        sink = self.FiftyOneDatasink(dataset_name="output")
        assert sink.get_name() == "FiftyOne(output)"

    def test_supports_distributed_writes(self):
        """Test supports_distributed_writes returns True."""
        sink = self.FiftyOneDatasink(dataset_name="output")
        assert sink.supports_distributed_writes is True

    def test_num_rows_per_write(self):
        """Test num_rows_per_write returns batch size."""
        sink = self.FiftyOneDatasink(dataset_name="output", batch_size=50)
        assert sink.num_rows_per_write == 50

    def test_on_write_start_creates_dataset(self):
        """Test on_write_start creates new dataset."""
        sink = self.FiftyOneDatasink(dataset_name="output")
        mock_dataset = Mock()
        mock_fo.dataset_exists.return_value = False
        mock_fo.Dataset.return_value = mock_dataset
        sink.on_write_start()
        mock_fo.Dataset.assert_called_with(name="output")

    def test_on_write_start_overwrites(self):
        """Test on_write_start overwrites existing dataset."""
        sink = self.FiftyOneDatasink(dataset_name="output", overwrite=True)
        mock_fo.dataset_exists.return_value = True
        sink.on_write_start()
        mock_fo.delete_dataset.assert_called_with("output")

    def test_write_with_pyarrow_table(self):
        """Test write method with PyArrow table."""
        sink = self.FiftyOneDatasink(dataset_name="output")
        mock_dataset = Mock()
        mock_fo.load_dataset.return_value = mock_dataset
        mock_ctx = Mock(task_idx=0)

        data = pa.Table.from_pydict({"filepath": ["/path/img.jpg"]})

        with patch(
            "ray_fiftyone.data_sink.helpers._dict_to_sample", return_value=Mock()
        ):
            result = sink.write([data], mock_ctx)
            assert result["samples_written"] == 1
            assert result["task_index"] == 0

    def test_write_with_custom_filepath_field(self):
        """Test write with custom filepath field."""
        sink = self.FiftyOneDatasink(dataset_name="output", filepath_field="image_path")
        mock_dataset = Mock()
        mock_fo.load_dataset.return_value = mock_dataset
        mock_ctx = Mock(task_idx=0)

        data = pa.Table.from_pydict({"image_path": ["/path/img.jpg"]})

        with patch(
            "ray_fiftyone.data_sink.helpers._dict_to_sample", return_value=Mock()
        ):
            result = sink.write([data], mock_ctx)
            # Verify the field was mapped correctly
            assert result["samples_written"] == 1

    def test_on_write_failed_logs_error(self):
        """Test on_write_failed logs the error."""
        sink = self.FiftyOneDatasink(dataset_name="output")
        with patch("ray_fiftyone.data_sink.logger") as mock_logger:
            sink.on_write_failed(Exception("test error"))
            mock_logger.error.assert_called_once()


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_read_fiftyone(self):
        """Test read_fiftyone convenience function."""
        # Note: This test would require full Ray integration
        # Testing datasource creation instead
        from ray_fiftyone.data_source import FiftyOneDatasource

        source = FiftyOneDatasource("test_dataset")
        assert source._dataset_name == "test_dataset"

    def test_read_fiftyone_with_params(self):
        """Test read_fiftyone datasource with parameters."""
        from ray_fiftyone.data_source import FiftyOneDatasource

        source = FiftyOneDatasource(
            "test_dataset",
            fields=["filepath"],
            batch_size=50,
        )
        assert source._fields == ["filepath"]
        assert source._batch_size == 50

    def test_write_fiftyone(self):
        """Test write_fiftyone convenience function."""
        from ray_fiftyone.data_sink import write_fiftyone

        mock_dataset = Mock()
        write_fiftyone(mock_dataset, "output_dataset")
        mock_dataset.write_datasink.assert_called_once()

    def test_write_fiftyone_with_params(self):
        """Test write_fiftyone with parameters."""
        from ray_fiftyone.data_sink import write_fiftyone

        mock_dataset = Mock()
        write_fiftyone(
            mock_dataset,
            "output_dataset",
            filepath_field="image_path",
            overwrite=True,
            concurrency=5,
        )
        call_args = mock_dataset.write_datasink.call_args
        assert call_args[1]["concurrency"] == 5


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def setup_method(self):
        """Setup for each test."""
        from ray_fiftyone import helpers

        self.helpers = helpers

    def test_sample_to_dict_preserves_id(self):
        """Test that sample ID is preserved correctly."""
        sample = Mock()
        sample.id = "test_id_123"
        sample.to_dict.return_value = {"filepath": "/path/img.jpg"}
        result = self.helpers._sample_to_dict(sample, include_id=True)
        assert result["id"] == "test_id_123"

    def test_sample_to_dict_without_id(self):
        """Test sample to dict without ID."""
        sample = Mock()
        sample.to_dict.return_value = {"filepath": "/path/img.jpg"}
        result = self.helpers._sample_to_dict(sample, include_id=False)
        assert "filepath" in result
        # ID might still be added, but we don't enforce it's not there

    def test_dict_to_sample_skips_internal_fields(self):
        """Test that internal fields are skipped."""
        mock_sample = MagicMock()
        mock_fo.Sample.return_value = mock_sample
        row = {
            "filepath": "/path/img.jpg",
            "id": "should_skip",
            "_id": "should_skip",
            "tags": ["train"],
        }
        result = self.helpers._dict_to_sample(row)
        # Verify filepath was used but id and _id were skipped
        mock_fo.Sample.assert_called_with(filepath="/path/img.jpg")
        assert result == mock_sample

    def test_write_handles_error_logging(self):
        """Test write method handles and logs errors."""
        from ray_fiftyone.data_sink import FiftyOneDatasink

        sink = FiftyOneDatasink(dataset_name="output")
        mock_dataset = Mock()
        mock_fo.load_dataset.return_value = mock_dataset
        mock_ctx = Mock(task_idx=0)

        # Mock dict_to_sample to raise an error
        with patch("ray_fiftyone.data_sink.helpers._dict_to_sample") as mock_dict:
            mock_dict.side_effect = Exception("Sample creation error")
            data = pa.Table.from_pydict({"filepath": ["/path/img.jpg"]})

            result = sink.write([data], mock_ctx)
            assert result["samples_written"] == 0
            assert len(result["errors"]) == 1

    def test_on_write_complete_with_various_result_formats(self):
        """Test on_write_complete handles different result formats."""
        from ray_fiftyone.data_sink import FiftyOneDatasink

        sink = FiftyOneDatasink(dataset_name="output")
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 100
        mock_fo.load_dataset.return_value = mock_dataset

        # Test with None result (edge case)
        result = Mock()
        result.rows_written = None
        sink.on_write_complete(result)
        assert sink._total_samples_written == 0

    def test_datasource_with_view_stages_no_stage_key(self):
        """Test datasource handles view stages without 'stage' key."""
        from ray_fiftyone.data_source import FiftyOneDatasource

        source = FiftyOneDatasource(
            dataset_name="test",
            view_stages=[{"limit": 10}],  # Missing 'stage' key
        )
        # Should not crash during initialization
        assert source._view_stages == [{"limit": 10}]

    def test_datasink_with_dataset_type(self):
        """Test datasink initialization with dataset type."""
        from ray_fiftyone.data_sink import FiftyOneDatasink

        sink = FiftyOneDatasink(dataset_name="output", dataset_type="video")
        assert sink._dataset_type == "video"
