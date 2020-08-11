import logging
import colorlog


def get_logger(name="__main__", debug=False):
    log_format = (
        "[%(levelname)s] "
        "%(asctime)s - "
        "%(name)s: "
        "%(funcName)s - "
        "%(message)s"
    )
    bold_seq = "\033[1m"
    colorlog_format = (
        f"{bold_seq} "
        "%(log_color)s "
        f"{log_format}"
    )
    colorlog.basicConfig(format=colorlog_format)

    logger = logging.getLogger(name)
    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    # Output full log
    fh = logging.FileHandler("app.log")
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter(log_format)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger
