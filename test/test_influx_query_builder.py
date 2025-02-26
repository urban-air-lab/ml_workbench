import pytest

from database.influx_buckets import InfluxBuckets
from database.influx_query_builder import FluxQueryBuilder


def test_builder_init():
    query = FluxQueryBuilder()
    assert isinstance(query, FluxQueryBuilder)


def test_builder_without_bucket():
    with pytest.raises(ValueError, match="bucket must be set!"):
        query = FluxQueryBuilder().build()


def test_builder_bucket():
    with pytest.raises(ValueError, match="time range must be set!"):
        query = FluxQueryBuilder() \
            .set_bucket(InfluxBuckets.test_bucket.value) \
            .build()


def test_builder_bucket_range():
    with pytest.raises(ValueError, match="measurement must be set!"):
        query = FluxQueryBuilder() \
            .set_bucket(InfluxBuckets.test_bucket.value) \
            .set_range("2024-10-22T00:00:00Z", "2024-10-22T23:00:00Z") \
            .build()


def test_builder_bucket_range_measurement():
    query = FluxQueryBuilder() \
        .set_bucket(InfluxBuckets.test_bucket.value) \
        .set_range("2024-10-22T00:00:00Z", "2024-10-22T23:00:00Z") \
        .set_measurement("sont_c") \
        .build()
    assert query == '''from(bucket: "TestBucket")|> range(start: 2024-10-22T00:00:00Z, stop: 2024-10-22T23:00:00Z)|> filter(fn: (r) => r._measurement == "sont_c")'''


def test_builder_bucket_range_measurement_fields():
    query = FluxQueryBuilder() \
        .set_bucket(InfluxBuckets.test_bucket.value) \
        .set_range("2024-10-22T00:00:00Z", "2024-10-22T23:00:00Z") \
        .set_measurement("sont_c") \
        .set_fields(["CO", "NO"]) \
        .build()
    assert query == '''from(bucket: "TestBucket")|> range(start: 2024-10-22T00:00:00Z, stop: 2024-10-22T23:00:00Z)|> filter(fn: (r) => r._measurement == "sont_c")|> filter(fn: (r) => r._field == "CO" or r._field == "NO")'''