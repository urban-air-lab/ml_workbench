from enum import Enum


class InfluxBuckets(Enum):
    """
    Contains strings of all used InfluxDB buckets
    """
    AQSN_MINUTE_CALIBRATION_BUCKET = "AQSNMinuteCalibrationBucket"
    AQSN_MINUTE_MEASUREMENT_BUCKET = "AQSNMinuteMeasurementBucket"
    LUBW_HOUR_BUCKET = "LUBWHourBucket"
    LUBW_MINUTE_BUCKET = "LUBWMinuteBucket"
    TEST_BUCKET = "TestBucket"
