from enum import Enum


class LUBWSensors(Enum):
    """
    All current in use sensors from lubw
    debw is english abbreviation for lubw
    """
    DEBW015 = "DEBW015"
    DEBW152 = "DEBW152"


class UALSensors(Enum):
    """
    All current in use custom sensors from UrbanAirLab project
    """
    UAL_1 = "ual-1"
    UAL_2 = "ual-2"
    UAL_3 = "ual-3"
