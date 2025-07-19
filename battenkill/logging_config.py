from textwrap import dedent
from pathlib import Path

# Create logs directory if it doesn't exist
log_dir = Path.cwd() / "reports" / "logs"
log_dir.mkdir(parents=True, exist_ok=True)

log_config_yaml = dedent(f"""\
version: 1
formatters:
    default:
        style: "{{"
        format: "{{message}}"
    timestamp:
        style: "{{"
        format: "{{asctime}} | {{levelname}} | {{message}}"
handlers:
    console:
        class: rich.logging.RichHandler
        formatter: default
    file:
        class: logging.FileHandler
        filename: {log_dir / "work.log"}
        formatter: timestamp
loggers:
    batten.expose:
        handlers:
        -   console
        -   file
    batten.write:
        handlers:
        -   file
root:
    level: DEBUG
""")
