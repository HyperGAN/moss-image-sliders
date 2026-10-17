"""Retired campaign dispatcher; GPU admission belongs to the coordinator."""


def main():
    raise SystemExit(
        "Legacy campaign dispatch is disabled. GPU admission belongs to the "
        "coordinator; use the explicit moss-sliders train-campaign command only "
        "after admission. This command starts no process."
    )


if __name__ == "__main__":
    main()
