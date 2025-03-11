import sys
import pathlib
import logging
try:
    import coloredlogs
    coloredlogs.enable_ansi_support()
    __field_styles = coloredlogs.DEFAULT_FIELD_STYLES
    __field_styles["msecs"] = __field_styles["asctime"]
    __field_styles["levelname"] = dict(color='green')
    __level_styles = coloredlogs.DEFAULT_LEVEL_STYLES.update(
        spam=dict(color="blue", faint=True),
        debug=dict(color="blue"),
        verbose=dict(color="blue", bold=True),
        info=dict(),
        warning=dict(color=(255, 64, 0)),
        error=dict(color=(255, 0, 0)),
        fatal=dict(color=(255, 0, 0), bold=True, background=(255, 255, 0)),
        critical=dict(color=(0, 0, 0), bold=True, background=(255, 255, 0))
    )

    __formatter = coloredlogs.ColoredFormatter(f'%(asctime)s|%(name)s-%(levelname)s: %(message)s',
                                               datefmt='%Y-%m-%d %H:%M:%S.%f',
                                               field_styles=__field_styles, level_styles=__level_styles)
except ImportError:
    formatter_class = logging.Formatter
    __formatter = formatter_class('%(asctime)s.%(msecs).03d|%(name)s-%(levelname)s: %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')

# create logger
log = logging.getLogger(__name__)
log.level = logging.DEBUG

# Don't use colored logs for the file logs
__file_formatter = logging.Formatter('%(asctime)s.%(msecs).03d|%(name)s-%(levelname)5s %(threadName)s:%(filename)s:%(lineno)s:%(funcName)s| %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# create console handler
console_log_handler = logging.StreamHandler(sys.stdout)
console_log_handler.level = logging.DEBUG
console_log_handler.formatter = __formatter
log.addHandler(console_log_handler)  # add the handler to the logger

# create file handler which logs debug messages
try:
    __log_file_path = pathlib.Path(__file__).resolve().parent.parent / f'{log.name}.log'
    __fh = logging.FileHandler(__log_file_path, encoding='utf-8')
    __fh.level = -1
    __fh.formatter = __file_formatter
    # add the handler to the logger
    log.addHandler(__fh)
except IOError:
    console_log_handler.level = logging.DEBUG
    log.warning('Could not create log file. Redirecting messages to console output.')
