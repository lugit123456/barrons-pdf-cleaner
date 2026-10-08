from __future__ import annotations

import unittest

from metadata_parser import _parse_publication_type, parse_pdf_metadata


class MetadataParserTests(unittest.TestCase):
    def test_barrons_filename_maps_to_barrons(self) -> None:
        metadata = parse_pdf_metadata(
            "[lib.magstore.top]Barron's - October 5 2026.pdf"
        )
        self.assertEqual(metadata.publication_type, "BARRONS")
        self.assertEqual(metadata.publication_date, "2026-10-05")

    def test_the_economist_filename_maps_to_te(self) -> None:
        self.assertEqual(
            _parse_publication_type(
                "[lib.magstore.top]The Economist USA - July 25 2026.pdf"
            ),
            "TE",
        )


if __name__ == "__main__":
    unittest.main()
