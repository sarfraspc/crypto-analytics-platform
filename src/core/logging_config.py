"""Structured JSON logging configuration for the application."""

import logging.config
import json


class JSONFormatter(logging.Formatter):
    """Formatter that outputs log records as JSON objects."""

    def format(self, record):
        log_obj = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        return json.dumps(log_obj)


def setup_logging():
    """Configure application logging with JSON formatter and INFO level."""
    LOGGING_CONFIG = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'json': {
                '()': lambda: JSONFormatter(),
            },
        },
        'handlers': {
            'default': {
                'level': 'INFO',
                'formatter': 'json',
                'class': 'logging.StreamHandler',
            },
        },
        'loggers': {
            '': {
                'handlers': ['default'],
                'level': 'INFO',
                'propagate': True
            }
        }
    }
    logging.config.dictConfig(LOGGING_CONFIG)