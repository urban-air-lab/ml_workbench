from enum import Enum


class LUBWSensors(Enum):
    """
    All current in use sensors from lubw
    debw is english abbreviation for lubw
    """
    DEBW015 = "DEBW015"
    DEBW152 = "DEBW152"


class AQSNSensors(Enum):
    """
    All current in use custom sensors from AirUP! project
    """
    SONT_A = "sont_a"
    SONT_B = "sont_b"
    SONT_C = "sont_c"
