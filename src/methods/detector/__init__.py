"""
Модуль детекции печатей на документах.
"""

from .stamp_detector import (
    StampDetector,
    detect_stamps_single,
    detect_stamps_folder,
    remove_stamps_from_image,
    remove_stamps_from_folder,
)

__all__ = [
    'StampDetector',
    'detect_stamps_single',
    'detect_stamps_folder',
    'remove_stamps_from_image',
    'remove_stamps_from_folder',
]

