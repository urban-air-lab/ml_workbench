from enum import Enum


class InfluxBuckets(Enum):
    """
    Contains strings of all used InfluxDB buckets
    """
    UAL_MINUTE_CALIBRATION_BUCKET = "ual-minute-calibration"
    UAL_MINUTE_MEASUREMENT_BUCKET = "ual-minute-measurement"
    LUBW_HOUR_BUCKET = "lubw-hour"
    LUBW_MINUTE_BUCKET = "lubw-minute"
    TEST_BUCKET = "test-data"
