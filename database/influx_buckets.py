from enum import Enum


class InfluxBuckets(Enum):
    """
    Contains strings of all used InfluxDB buckets
    """
    AQSN_minute_calibration_bucket = "AQSNMinuteCalibrationBucket"
    AQSN_minute_measurement_bucket = "AQSNMinuteMeasurementBucket"
    LUBW_hour_bucket = "LUBWHourBucket"
    LUBW_minute_bucket = "LUBWMinuteBucket"
    test_bucket = "TestBucket"
