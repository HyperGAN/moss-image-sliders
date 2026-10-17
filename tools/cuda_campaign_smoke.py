"""Retired autonomous CUDA smoke; CPU qualification uses the shared GAN."""


def main():
    raise SystemExit(
        "Autonomous CUDA smoke is disabled. Run moss-sliders smoke for the shared "
        "PG09 CPU check; native CUDA/BF16 qualification requires coordinator admission."
    )


if __name__ == "__main__":
    main()
