import logging
import colorlog

def logger(loggername="Logger"):
    """
    A simple logger that doesn't duplicate the output in ipynb notebook

    OUTPUT:
    returns a logger instance
    """
    logger=logging.getLogger(loggername)
    if not len(logger.handlers):
        logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        formatter = colorlog.ColoredFormatter('%(log_color)s[%(asctime)s - %(name)s] - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger
