#!/usr/bin/env python3
"""Build one frozen expert-bank entry on a Cui published DGP."""

from __future__ import annotations

import build_frozen_expert_entry as bank_builder
import section4_cui_published_experiments as cui_published


def main() -> None:
    install_breadth = bank_builder.breadth._install_adapter

    def install_all(module) -> None:
        install_breadth(module)
        cui_published._install_adapter(module)

    bank_builder.breadth._install_adapter = install_all
    bank_builder.main()


if __name__ == "__main__":
    main()
