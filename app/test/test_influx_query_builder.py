import pytest
from app.database import InfluxBuckets
from app.database import InfluxQueryBuilder


def test_builder_init():
    query = InfluxQueryBuilder()
    assert isinstance(query, InfluxQueryBuilder)


def test_builder_without_bucket():
    with pytest.raises(ValueError, match="bucket must be set!"):
        query = InfluxQueryBuilder().build()


def test_builder_bucket():
    with pytest.raises(ValueError, match="time range must be set!"):
        query = InfluxQueryBuilder() \
            .set_bucket(InfluxBuckets.TEST_BUCKET.value) \
            .build()


def test_builder_bucket_range():
    with pytest.raises(ValueError, match="measurement must be set!"):
        query = InfluxQueryBuilder() \
            .set_bucket(InfluxBuckets.TEST_BUCKET.value) \
            .set_range("2024-10-22T00:00:00Z", "2024-10-22T23:00:00Z") \
            .build()


def test_builder_bucket_range_invalid_dates():
    with pytest.raises(ValueError, match="No valid date format - must be yyyy-mm-ddTHH:MM:SSZ"):
        query = InfluxQueryBuilder() \
            .set_bucket(InfluxBuckets.TEST_BUCKET.value) \
            .set_range("2024-10-22 00:00:00", "2024-10-22 23:00:00") \
            .build()


def test_builder_bucket_range_invalid_start_date():
    with pytest.raises(ValueError, match="No valid date format - must be yyyy-mm-ddTHH:MM:SSZ"):
        query = InfluxQueryBuilder() \
            .set_bucket(InfluxBuckets.TEST_BUCKET.value) \
            .set_range("2024-10-22 00:00:00", "2024-10-22T23:00:00Z") \
            .build()


def test_builder_bucket_range_end_date():
    with pytest.raises(ValueError, match="No valid date format - must be yyyy-mm-ddTHH:MM:SSZ"):
        query = InfluxQueryBuilder() \
            .set_bucket(InfluxBuckets.TEST_BUCKET.value) \
            .set_range("2024-10-22T00:00:00Z", "2024-10-22 23:00:00") \
            .build()


def test_builder_bucket_range_measurement():
    query = InfluxQueryBuilder() \
        .set_bucket(InfluxBuckets.TEST_BUCKET.value) \
        .set_range("2024-10-22T00:00:00Z", "2024-10-22T23:00:00Z") \
        .set_measurement("sont_c") \
        .build()
    assert query == '''from(bucket: "TestBucket")|> range(start: 2024-10-22T00:00:00Z, stop: 2024-10-22T23:00:00Z)|> filter(fn: (r) => r._measurement == "sont_c")|> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")'''


def test_builder_bucket_range_measurement_fields():
    query = InfluxQueryBuilder() \
        .set_bucket(InfluxBuckets.TEST_BUCKET.value) \
        .set_range("2024-10-22T00:00:00Z", "2024-10-22T23:00:00Z") \
        .set_measurement("sont_c") \
        .set_fields(["CO", "NO"]) \
        .build()
    assert query == '''from(bucket: "TestBucket")|> range(start: 2024-10-22T00:00:00Z, stop: 2024-10-22T23:00:00Z)|> filter(fn: (r) => r._measurement == "sont_c")|> filter(fn: (r) => r._field == "CO" or r._field == "NO")|> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")'''