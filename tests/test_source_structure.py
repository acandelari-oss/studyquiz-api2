import unittest

from document_extractors import ExtractedBlock, ExtractedDocument
from source_structure import (
    SOURCE_STRUCTURE_HIGH,
    SOURCE_STRUCTURE_MEDIUM,
    SOURCE_STRUCTURE_LOW,
    analyze_source_structure,
    format_source_structure_tree,
)


class SourceStructureDetectionTests(unittest.TestCase):
    def analyze_text(self, text):
        document = ExtractedDocument(
            filename="benchmark.pdf",
            file_format="pdf",
            file_size_bytes=len(text.encode("utf-8")),
            blocks=[
                ExtractedBlock(
                    text=text,
                    raw_text=text,
                    page=1,
                    block_index=1,
                )
            ],
            pages_detected=1,
        )
        return analyze_source_structure(document)

    def analyze_blocks(self, blocks):
        document = ExtractedDocument(
            filename="benchmark.pdf",
            file_format="pdf",
            file_size_bytes=sum(len(text.encode("utf-8")) for text, _page in blocks),
            blocks=[
                ExtractedBlock(
                    text=text,
                    raw_text=text,
                    page=page,
                    block_index=index,
                )
                for index, (text, page) in enumerate(blocks, start=1)
            ],
            pages_detected=max((page for _text, page in blocks), default=0),
        )
        return analyze_source_structure(document)

    def test_benchmark_numbered_operon_hierarchy_is_high(self):
        analysis = self.analyze_text(
            "\n".join([
                "5 Prokaryotic gene regulation",
                "Introductory paragraph.",
                "5.1 Why operons are useful",
                "Operons coordinate related genes.",
                "5.2 General operon logic",
                "Regulation depends on promoters and operators.",
                "5.3 Repressible versus inducible operons",
                "The two modes differ by default expression state.",
                "5.4 Trp operon",
                "The trp operon is repressible.",
                "5.5 Lac operon",
                "The lac operon is inducible.",
            ])
        )

        self.assertEqual(
            analysis.source_structure_confidence,
            SOURCE_STRUCTURE_HIGH,
        )
        self.assertEqual(analysis.numbered_heading_count, 6)
        self.assertEqual(analysis.hierarchy_depth, 2)
        self.assertEqual(analysis.strongest_parent, "5")
        self.assertEqual(analysis.strongest_parent_child_count, 5)

        tree = format_source_structure_tree(analysis)
        self.assertIn("5 Prokaryotic gene regulation", tree)
        self.assertIn("  5.4 Trp operon", tree)
        self.assertIn("page=1", tree)
        self.assertIn("order=4", tree)
        self.assertIn("method=numbered", tree)
        self.assertIn("level=2", tree)

    def test_split_level_one_heading_with_subsections_is_preserved(self):
        analysis = self.analyze_text(
            "\n".join([
                "5",
                "Prokaryotic gene regulation",
                "Introductory paragraph.",
                "5.1 Why operons are useful in bacteria",
                "Operons coordinate related genes.",
                "5.2 General operon logic",
                "Regulation depends on promoters and operators.",
                "5.3 Repressible versus inducible operons",
                "5.4 Trp operon: repressible operon",
                "5.5 Lac operon: inducible operon",
            ])
        )

        self.assertEqual(
            analysis.source_structure_confidence,
            SOURCE_STRUCTURE_HIGH,
        )
        detected_text = [heading.text for heading in analysis.headings]
        self.assertIn("5 Prokaryotic gene regulation", detected_text)
        self.assertIn("5.4 Trp operon: repressible operon", detected_text)
        self.assertNotIn("Prokaryotic gene regulation", detected_text)

        parent = analysis.headings[0]
        self.assertEqual(parent.numbering, "5")
        self.assertEqual(parent.hierarchy_level, 1)
        self.assertEqual(parent.detection_method, "numbered")

    def test_level_one_section_without_subsections_is_detected(self):
        analysis = self.analyze_text(
            "\n".join([
                "10",
                "Ribosome structure",
                "The ribosome contains small and large subunits.",
                "11 Translation initiation",
                "Initiation requires recognition of the start codon.",
                "12 Translation elongation",
                "Elongation proceeds through repeated tRNA selection.",
                "13",
                "Translation termination",
                "Release factors recognize stop codons.",
            ])
        )

        detected_text = [heading.text for heading in analysis.headings]
        self.assertIn("10 Ribosome structure", detected_text)
        self.assertIn("13 Translation termination", detected_text)
        self.assertTrue(
            all(
                heading.hierarchy_level == 1
                for heading in analysis.headings
                if heading.numbering in {"10", "11", "12", "13"}
            )
        )
        self.assertEqual(
            analysis.source_structure_confidence,
            SOURCE_STRUCTURE_HIGH,
        )
        self.assertGreaterEqual(analysis.accepted_node_count, 4)

    def test_academic_dotted_top_level_heading_with_children_is_high(self):
        analysis = self.analyze_text(
            "\n".join([
                "5. Prokaryotic gene regulation: operons",
                "This section introduces bacterial gene regulation.",
                "5.1 Why operons are useful in bacteria",
                "Operons coordinate gene expression.",
                "5.2 General operon logic",
                "Regulatory elements determine expression.",
                "5.3 Repressible versus inducible operons",
                "Repression and induction differ.",
            ])
        )

        self.assertEqual(analysis.source_structure_confidence, SOURCE_STRUCTURE_HIGH)
        detected_text = [heading.text for heading in analysis.headings]
        self.assertIn("5 Prokaryotic gene regulation: operons", detected_text)
        self.assertIn("5.1 Why operons are useful in bacteria", detected_text)
        self.assertEqual(analysis.strongest_parent, "5")

    def test_coherent_top_level_sequence_is_accepted_without_children(self):
        analysis = self.analyze_text(
            "\n".join([
                "10. Ribosome structure",
                "Ribosomes contain subunits.",
                "11. Translation initiation",
                "Initiation starts protein synthesis.",
                "12. Translation elongation",
                "Elongation grows the peptide.",
                "13. Translation termination",
                "Termination releases the polypeptide.",
                "14. Polyribosomes",
                "Multiple ribosomes translate together.",
            ])
        )

        self.assertEqual(analysis.source_structure_confidence, SOURCE_STRUCTURE_HIGH)
        self.assertEqual(analysis.numbered_heading_count, 5)
        self.assertEqual(analysis.hierarchy_depth, 1)

    def test_ordinary_numbered_list_does_not_become_high_structure(self):
        analysis = self.analyze_text(
            "\n".join([
                "These are the useful properties:",
                "1 small molecules can diffuse rapidly",
                "2 cells need energy for transport",
                "3 proteins can bind substrates",
                "4 enzymes accelerate reactions",
                "The numbered statements above are not a document outline.",
            ])
        )

        self.assertEqual(
            analysis.source_structure_confidence,
            SOURCE_STRUCTURE_LOW,
        )
        self.assertEqual(analysis.hierarchy_depth, 0)

    def test_split_numbered_list_does_not_become_headings(self):
        analysis = self.analyze_text(
            "\n".join([
                "Useful properties include:",
                "1",
                "small molecules can diffuse rapidly",
                "2",
                "cells need energy for transport",
                "3",
                "proteins can bind substrates",
            ])
        )

        self.assertEqual(
            analysis.source_structure_confidence,
            SOURCE_STRUCTURE_LOW,
        )
        self.assertEqual(analysis.numbered_heading_count, 0)

    def test_procedural_numbered_sentence_is_rejected(self):
        analysis = self.analyze_text(
            "\n".join([
                "13. Translation termination",
                "130. The ribosome reaches a stop codon.",
                "131. No tRNA corresponds to this codon.",
                "14. Polyribosomes",
                "Multiple ribosomes can translate one mRNA.",
                "15. Protein folding",
                "Folding follows translation.",
            ])
        )

        rejected = [
            node for node in analysis.nodes
            if node.status == "rejected"
        ]
        self.assertTrue(
            any("ribosome reaches" in node.title_original.lower() for node in rejected)
        )
        self.assertTrue(
            any("procedural" in node.reason.lower() for node in rejected)
        )

    def test_years_citations_and_formulas_are_rejected(self):
        analysis = self.analyze_text(
            "\n".join([
                "2024 Results from the experiment",
                "3 E = mc^2 + ATP",
                "7 doi references from the journal",
                "5.1 Valid Subsection",
            ])
        )

        self.assertEqual(
            analysis.source_structure_confidence,
            SOURCE_STRUCTURE_LOW,
        )
        detected_text = [heading.text for heading in analysis.headings]
        self.assertEqual(detected_text, ["5.1 Valid Subsection"])

    def test_textual_headings_can_be_reported_without_high_confidence(self):
        analysis = self.analyze_text(
            "\n".join([
                "",
                "OPERON REGULATION",
                "",
                "The following text explains regulatory logic.",
                "",
                "Lac Operon",
                "",
                "The lac operon responds to lactose availability.",
            ])
        )

        self.assertEqual(
            analysis.source_structure_confidence,
            SOURCE_STRUCTURE_LOW,
        )
        self.assertEqual(analysis.textual_heading_count, 2)
        self.assertIn(
            "operon regulation",
            [heading.normalized_title for heading in analysis.headings],
        )
        self.assertEqual(analysis.document.confidence, SOURCE_STRUCTURE_LOW)

    def test_one_one_one_hierarchy_is_constructed(self):
        analysis = self.analyze_text(
            "\n".join([
                "1 Gene expression",
                "1.1 Transcription",
                "1.1.1 Promoter recognition",
                "RNA polymerase binds promoter elements.",
                "1.2 Translation",
                "Ribosomes synthesize proteins.",
            ])
        )

        self.assertEqual(analysis.source_structure_confidence, SOURCE_STRUCTURE_HIGH)
        self.assertEqual(analysis.hierarchy_depth, 3)
        relationship_pairs = {
            (relationship.parent_node_id, relationship.child_node_id)
            for relationship in analysis.relationships
        }
        by_number = {
            node.numbering: node.id
            for node in analysis.nodes
            if node.status == "accepted" and node.numbering
        }
        self.assertIn((by_number["1"], by_number["1.1"]), relationship_pairs)
        self.assertIn((by_number["1.1"], by_number["1.1.1"]), relationship_pairs)

    def test_roman_numeral_sequence(self):
        analysis = self.analyze_text(
            "\n".join([
                "I Introduction to genetics",
                "Introductory text.",
                "II Mendelian inheritance",
                "Inheritance patterns.",
                "III Molecular inheritance",
                "DNA mechanisms.",
            ])
        )

        roman = [
            node for node in analysis.nodes
            if node.status == "accepted" and node.detection_method == "roman"
        ]
        self.assertEqual(len(roman), 3)
        self.assertEqual(analysis.document.confidence, SOURCE_STRUCTURE_MEDIUM)

    def test_part_chapter_numbered_section_pattern(self):
        analysis = self.analyze_text(
            "\n".join([
                "PARTE PRIMA Fondamenti",
                "CAPITOLO PRIMO Genetica molecolare",
                "1 DNA structure",
                "DNA has a double helix.",
                "2 RNA synthesis",
                "RNA is synthesized by transcription.",
                "3 Protein synthesis",
                "Proteins are synthesized by translation.",
            ])
        )

        self.assertEqual(analysis.source_structure_confidence, SOURCE_STRUCTURE_HIGH)
        self.assertTrue(
            any(
                relationship.relationship_type in {"explicit_structural_parent", "parent_child"}
                for relationship in analysis.relationships
            )
        )

    def test_repeated_textual_heading_candidates_are_medium_evidence(self):
        analysis = self.analyze_text(
            "\n".join([
                "",
                "SOMATIC MOSAICISM",
                "",
                "Cells carry distinct variants.",
                "",
                "GONADAL MOSAICISM",
                "",
                "Gametes carry variants.",
                "",
                "BLOOD CHIMERAS",
                "",
                "Blood lineages differ.",
            ])
        )

        accepted_textual = [
            node for node in analysis.nodes
            if node.status == "accepted" and node.detection_method == "textual"
        ]
        self.assertEqual(len(accepted_textual), 3)
        self.assertEqual(analysis.document.confidence, SOURCE_STRUCTURE_MEDIUM)

    def test_isolated_uppercase_false_positive_remains_uncertain(self):
        analysis = self.analyze_text(
            "\n".join([
                "The lecture discusses chromatin.",
                "",
                "H4 PHOSPHORYLATION S1",
                "",
                "This line is a local note rather than a stable document heading.",
            ])
        )

        self.assertEqual(analysis.source_structure_confidence, SOURCE_STRUCTURE_LOW)
        self.assertEqual(analysis.accepted_node_count, 0)
        self.assertEqual(analysis.uncertain_node_count, 1)

    def test_mixed_strong_and_weak_structural_regions(self):
        analysis = self.analyze_text(
            "\n".join([
                "5. Prokaryotic gene regulation",
                "5.1 Why operons are useful",
                "5.2 General operon logic",
                "",
                "FLOATING NOTE",
                "",
                "This is a weak local note.",
                "5.3 Repressible versus inducible operons",
            ])
        )

        self.assertEqual(analysis.source_structure_confidence, SOURCE_STRUCTURE_HIGH)
        self.assertGreaterEqual(analysis.accepted_node_count, 4)
        self.assertGreaterEqual(analysis.uncertain_node_count, 1)

    def test_low_confidence_unstructured_noisy_material(self):
        analysis = self.analyze_text(
            "\n".join([
                "arrows --> cell thing",
                "maybe ribosome here",
                "2024",
                "ATP + ADP = energy",
                "random note with no structure",
            ])
        )

        self.assertEqual(analysis.source_structure_confidence, SOURCE_STRUCTURE_LOW)
        self.assertEqual(analysis.accepted_node_count, 0)

    def test_textual_heading_inside_numbered_section_does_not_disrupt_hierarchy(self):
        analysis = self.analyze_text(
            "\n".join([
                "7 Epigenetics, DNA methylation and histone code",
                "7.1 DNA methylation",
                "Methylation affects gene expression.",
                "",
                "H4 phosphorylation S1",
                "",
                "This line is an unnumbered local heading.",
                "7.2 Histone acetylation",
                "Acetylation changes chromatin accessibility.",
                "7.3 Histone methylation",
                "Methylation effects depend on residue context.",
                "7.4 Chromatin remodeling",
                "Remodeling changes nucleosome positioning.",
            ])
        )

        numbered = [
            heading.text
            for heading in analysis.headings
            if heading.detection_method == "numbered"
        ]
        textual = [
            heading.text
            for heading in analysis.headings
            if heading.detection_method == "textual"
        ]

        self.assertEqual(numbered[0], "7 Epigenetics, DNA methylation and histone code")
        self.assertIn("7.4 Chromatin remodeling", numbered)
        self.assertIn("H4 phosphorylation S1", textual)
        self.assertEqual(analysis.strongest_parent, "7")
        self.assertEqual(analysis.strongest_parent_child_count, 4)

    def test_dense_local_enumeration_is_not_promoted_to_document_structure(self):
        analysis = self.analyze_text(
            "\n".join([
                "",
                "Key laboratory steps",
                "",
                "1 Prepare the sample",
                "2 Add the buffer",
                "3 Incubate the tube",
                "4 Measure the signal",
                "The list above is procedural support material.",
            ])
        )

        self.assertTrue(
            any(region.region_type == "local_enumeration" for region in analysis.regions)
        )
        self.assertEqual(analysis.source_structure_confidence, SOURCE_STRUCTURE_LOW)
        self.assertEqual(analysis.numbered_heading_count, 0)

    def test_procedural_numbered_sequence_is_region_aware(self):
        analysis = self.analyze_text(
            "\n".join([
                "1. The sample is placed on ice.",
                "2. The buffer is added slowly.",
                "3. The tube is centrifuged.",
                "4. The supernatant is collected.",
            ])
        )

        self.assertTrue(any(region.region_type == "procedural" for region in analysis.regions))
        self.assertTrue(
            any(sequence.sequence_type == "procedural_sequence" for sequence in analysis.sequences)
        )
        self.assertEqual(analysis.accepted_node_count, 0)

    def test_toc_like_outline_preserves_part_chapter_numbered_entries(self):
        analysis = self.analyze_text(
            "\n".join([
                "PARTE PRIMA Fondamenti",
                "CAPITOLO PRIMO Impresa e azienda",
                "1 L'azienda come sistema",
                "2 Soggetto economico",
                "3 Assetti istituzionali",
                "CAPITOLO SECONDO Bilancio",
                "4 Stato patrimoniale",
                "5 Conto economico",
                "6 Nota integrativa",
            ])
        )

        self.assertTrue(any(region.region_type == "toc_like" for region in analysis.regions))
        numbered_nodes = [
            node for node in analysis.nodes
            if node.status == "accepted" and node.numbering in {"1", "2", "3", "4", "5", "6"}
        ]
        self.assertTrue(numbered_nodes)
        self.assertTrue(all(node.hierarchy_level == 3 for node in numbered_nodes))
        self.assertEqual(analysis.source_structure_confidence, SOURCE_STRUCTURE_HIGH)

    def test_same_numbered_syntax_differs_by_body_context(self):
        analysis = self.analyze_blocks([
            (
                "\n".join([
                    "CHAPTER ONE Cellular organization",
                    "1 Membrane architecture",
                    "2 Organelle distribution",
                    "3 Cytoskeletal polarity",
                ]),
                1,
            ),
            (
                "\n".join([
                    "Useful review prompts:",
                    "1 Recall the definition",
                    "2 Compare the examples",
                    "3 Check the diagram",
                ]),
                2,
            ),
        ])

        self.assertTrue(any(region.confidence == SOURCE_STRUCTURE_HIGH for region in analysis.regions))
        self.assertTrue(any(region.region_type == "local_enumeration" for region in analysis.regions))
        accepted_titles = {
            node.title_original for node in analysis.nodes if node.status == "accepted"
        }
        self.assertIn("1 Membrane architecture", accepted_titles)
        self.assertNotIn("1 Recall the definition", accepted_titles)

    def test_random_numbered_prose_before_chapter_is_not_parent(self):
        analysis = self.analyze_text(
            "\n".join([
                "1 Random prefatory remark",
                "This is ordinary prose before the outline.",
                "",
                "CHAPTER ONE Gene expression",
                "1 Transcription control",
                "2 Translation control",
                "3 Protein turnover",
            ])
        )

        by_title = {
            node.title_original: node
            for node in analysis.nodes
            if node.status == "accepted"
        }
        chapter = by_title["CHAPTER ONE Gene expression"]
        child = by_title["1 Transcription control"]
        self.assertEqual(child.parent_id, chapter.id)
        self.assertNotIn("1 Random prefatory remark", by_title)

    def test_multiline_structural_heading_continuation_is_preserved(self):
        analysis = self.analyze_text(
            "\n".join([
                "CHAPTER ONE Foundations of",
                "Cellular Metabolism",
                "1 Glycolysis overview",
                "2 Oxidative phosphorylation",
                "3 ATP balance",
            ])
        )

        titles = [node.title_original for node in analysis.nodes if node.status == "accepted"]
        self.assertIn("CHAPTER ONE Foundations of Cellular Metabolism", titles)
        self.assertTrue(
            any(signal.signal_type == "multiline_heading_continuation_support" for signal in analysis.signals)
        )

    def test_mixed_document_keeps_high_and_low_regions_independent(self):
        analysis = self.analyze_blocks([
            (
                "\n".join([
                    "5 Prokaryotic gene regulation",
                    "Introductory paragraph.",
                    "5.1 Why operons are useful",
                    "Operons coordinate related genes.",
                    "5.2 General operon logic",
                    "Regulatory elements determine expression.",
                ]),
                1,
            ),
            (
                "\n".join([
                    "FLOATING NOTE",
                    "ATP + ADP = energy",
                    "Random sentence without a stable outline.",
                ]),
                2,
            ),
        ])

        self.assertTrue(any(region.confidence == SOURCE_STRUCTURE_HIGH for region in analysis.regions))
        self.assertTrue(any(region.confidence == SOURCE_STRUCTURE_LOW for region in analysis.regions))
        self.assertEqual(analysis.source_structure_confidence, SOURCE_STRUCTURE_HIGH)

    def test_parent_child_relationship_rejected_across_incompatible_regions(self):
        analysis = self.analyze_blocks([
            (
                "\n".join([
                    "1 Main structural heading",
                    "Body paragraph.",
                    "2 Second structural heading",
                    "Body paragraph.",
                    "3 Third structural heading",
                    "Body paragraph.",
                ]),
                1,
            ),
            (
                "\n".join([
                    "1.1 Local note that visually resembles a subsection",
                    "This line belongs to a different extracted block.",
                ]),
                2,
            ),
        ])

        local_subsection = next(
            node for node in analysis.nodes
            if node.numbering == "1.1"
        )
        self.assertIsNone(local_subsection.parent_id)

    def test_same_region_explicit_numbering_parent_is_preserved(self):
        analysis = self.analyze_text(
            "\n".join([
                "5 Prokaryotic gene regulation",
                "5.1 Why operons are useful in bacteria",
                "5.2 General operon logic",
            ])
        )

        by_number = {
            node.numbering: node
            for node in analysis.nodes
            if node.status == "accepted" and node.numbering
        }
        self.assertEqual(by_number["5.1"].parent_id, by_number["5"].id)
        self.assertEqual(by_number["5.2"].parent_id, by_number["5"].id)

    def test_cross_region_explicit_numbering_parent_is_preserved(self):
        analysis = self.analyze_text(
            "\n".join([
                "5 Prokaryotic gene regulation",
                "5.1 Why operons are useful in bacteria",
                "Operons coordinate related genes.",
                "This paragraph separates the next subsection.",
                "More source text keeps chronology clear.",
                "Additional source content appears here.",
                "A fifth explanatory line remains body text.",
                "A sixth explanatory line remains body text.",
                "A seventh explanatory line remains body text.",
                "An eighth explanatory line remains body text.",
                "A ninth explanatory line forces a new diagnostic region.",
                "5.2 General operon logic",
                "Regulatory elements determine expression.",
                "Another paragraph separates the final subsection.",
                "More body text.",
                "More body text.",
                "More body text.",
                "More body text.",
                "More body text.",
                "More body text.",
                "More body text.",
                "More body text.",
                "5.3 Repressible versus inducible operons",
            ])
        )

        by_number = {
            node.numbering: node
            for node in analysis.nodes
            if node.status == "accepted" and node.numbering
        }
        self.assertNotEqual(by_number["5"].region_id, by_number["5.2"].region_id)
        self.assertEqual(by_number["5.1"].parent_id, by_number["5"].id)
        self.assertEqual(by_number["5.2"].parent_id, by_number["5"].id)
        self.assertEqual(by_number["5.3"].parent_id, by_number["5"].id)
        self.assertTrue(
            all(
                relationship.relationship_type == "explicit_numbering_parent"
                for relationship in analysis.relationships
                if relationship.parent_node_id == by_number["5"].id
            )
        )

    def test_cross_block_page_explicit_numbering_parent_is_preserved(self):
        analysis = self.analyze_blocks([
            (
                "\n".join([
                    "7 Epigenetics, DNA methylation and histone code",
                    "7.1 DNA methylation",
                ]),
                1,
            ),
            (
                "\n".join([
                    "7.2 Histone acetylation",
                    "7.3 Histone methylation",
                ]),
                2,
            ),
        ])

        by_number = {
            node.numbering: node
            for node in analysis.nodes
            if node.status == "accepted" and node.numbering
        }
        self.assertNotEqual(by_number["7"].source_block, by_number["7.2"].source_block)
        self.assertEqual(by_number["7.2"].parent_id, by_number["7"].id)
        self.assertEqual(by_number["7.3"].parent_id, by_number["7"].id)

    def test_accepted_explicit_child_in_procedural_region_still_attaches(self):
        analysis = self.analyze_text(
            "\n".join([
                "5 Prokaryotic gene regulation",
                "5.1 Why operons are useful in bacteria",
                "5.2 General operon logic",
                "Body text separates the procedural-looking extracted region.",
                "More body text.",
                "More body text.",
                "More body text.",
                "More body text.",
                "More body text.",
                "More body text.",
                "More body text.",
                "More body text.",
                "101. The tube is centrifuged.",
                "102. The sample is transferred.",
                "103. The pellet is discarded.",
                "5.4 Trp operon: repressible operon",
            ])
        )

        by_number = {
            node.numbering: node
            for node in analysis.nodes
            if node.numbering
        }
        child = by_number["5.4"]
        self.assertEqual(child.status, "accepted")
        self.assertEqual(child.confidence, SOURCE_STRUCTURE_HIGH)
        self.assertEqual(child.parent_id, by_number["5"].id)
        child_region = next(region for region in analysis.regions if region.id == child.region_id)
        self.assertEqual(child_region.region_type, "procedural")

    def test_accepted_explicit_child_in_local_enumeration_region_still_attaches(self):
        analysis = self.analyze_text(
            "\n".join([
                "8 Protein translation and the genetic code",
                "8.1 Reading frames",
                "8.2 Codon degeneracy",
                "Body text separates the local enumeration region.",
                "More body text.",
                "More body text.",
                "More body text.",
                "More body text.",
                "More body text.",
                "More body text.",
                "More body text.",
                "More body text.",
                "8.3 Codon numbers",
                "1 Recall the codon table",
                "2 Compare the examples",
                "3 Check the diagram",
            ])
        )

        by_number = {
            node.numbering: node
            for node in analysis.nodes
            if node.numbering
        }
        child = by_number["8.3"]
        self.assertEqual(child.status, "accepted")
        self.assertEqual(child.parent_id, by_number["8"].id)
        child_region = next(region for region in analysis.regions if region.id == child.region_id)
        self.assertEqual(child_region.region_type, "local_enumeration")

    def test_same_region_procedural_classification_does_not_block_exact_numbering(self):
        analysis = self.analyze_text(
            "\n".join([
                "16 Post-translational modifications",
                "16.1 Why post-translational regulation matters",
                "16.2 Ubiquitination",
                "101. The sample is incubated.",
                "102. The supernatant is removed.",
            ])
        )

        by_number = {
            node.numbering: node
            for node in analysis.nodes
            if node.numbering
        }
        parent = by_number["16"]
        child = by_number["16.1"]
        self.assertEqual(parent.status, "accepted")
        self.assertEqual(child.status, "accepted")
        self.assertEqual(child.parent_id, parent.id)
        shared_region = next(region for region in analysis.regions if region.id == parent.region_id)
        self.assertEqual(shared_region.region_type, "procedural")

    def test_explicit_descendants_rehabilitate_weak_numbered_parent(self):
        analysis = self.analyze_text(
            "\n".join([
                "15 mRNA surveillance and nonsense-mediated decay in translation quality control stress responses and disease mechanisms",
                "15.1 Nonsense-mediated decay",
                "15.2 Ribosome stalling decay",
                "15.3 Nonstop decay",
            ])
        )

        by_number = {
            node.numbering: node
            for node in analysis.nodes
            if node.numbering
        }
        self.assertEqual(by_number["15"].status, "accepted")
        self.assertIn("Explicit numbered parent restored", by_number["15"].reason)
        self.assertTrue(
            any(
                signal.signal_type == "explicit_descendant_support"
                and signal.target_id == by_number["15"].id
                for signal in analysis.signals
            )
        )
        self.assertEqual(by_number["15.1"].parent_id, by_number["15"].id)
        self.assertEqual(by_number["15.2"].parent_id, by_number["15"].id)
        self.assertEqual(by_number["15.3"].parent_id, by_number["15"].id)

    def test_two_explicit_descendants_rehabilitate_weak_parent_as_medium(self):
        analysis = self.analyze_text(
            "\n".join([
                "15 mRNA surveillance and nonsense-mediated decay in translation quality control stress responses and disease mechanisms",
                "15.1 Exon-junction complexes",
                "15.2 Premature stop codons",
            ])
        )

        parent = next(node for node in analysis.nodes if node.numbering == "15")
        self.assertEqual(parent.status, "accepted")
        self.assertEqual(parent.confidence, SOURCE_STRUCTURE_MEDIUM)
        self.assertIn("Explicit numbered parent restored", parent.reason)

    def test_three_explicit_descendants_rehabilitate_scientific_acronym_parent_as_high(self):
        analysis = self.analyze_text(
            "\n".join([
                "15 mRNA surveillance and nonsense-mediated decay",
                "15.1 Exon-junction complexes",
                "15.2 How normal mRNA avoids NMD",
                "15.3 How premature stop codons trigger NMD",
            ])
        )

        by_number = {
            node.numbering: node
            for node in analysis.nodes
            if node.numbering
        }
        self.assertEqual(by_number["15"].status, "accepted")
        self.assertEqual(by_number["15"].confidence, SOURCE_STRUCTURE_HIGH)
        self.assertEqual(by_number["15.1"].parent_id, by_number["15"].id)
        self.assertEqual(by_number["15.2"].parent_id, by_number["15"].id)
        self.assertEqual(by_number["15.3"].parent_id, by_number["15"].id)

    def test_one_explicit_descendant_does_not_rehabilitate_parent(self):
        analysis = self.analyze_text(
            "\n".join([
                "12 Translation elongation is described in a single short local subsection",
                "12.1 Elongation cycle",
            ])
        )

        parent = next(node for node in analysis.nodes if node.numbering == "12")
        child = next(node for node in analysis.nodes if node.numbering == "12.1")
        self.assertNotEqual(parent.status, "accepted")
        self.assertIsNone(child.parent_id)

    def test_procedural_sequence_does_not_rehabilitate_numbered_parent(self):
        analysis = self.analyze_text(
            "\n".join([
                "20 Procedure",
                "20.1 Add reagent to the tube.",
                "20.2 Incubate for ten minutes.",
                "20.3 Remove the supernatant.",
            ])
        )

        accepted_numbers = {
            node.numbering
            for node in analysis.nodes
            if node.status == "accepted" and node.numbering
        }
        self.assertNotIn("20.1", accepted_numbers)
        self.assertFalse(
            any(
                signal.signal_type == "explicit_descendant_support"
                for signal in analysis.signals
            )
        )
        self.assertFalse(analysis.relationships)

    def test_obvious_numbered_prose_does_not_become_hierarchy(self):
        analysis = self.analyze_text(
            "\n".join([
                "6 The sample is then placed on ice.",
                "6.1 Add reagent to the tube.",
                "6.2 Incubate for ten minutes.",
            ])
        )

        self.assertEqual(analysis.accepted_node_count, 0)
        self.assertFalse(analysis.relationships)

    def test_strong_unsafe_parent_is_not_rehabilitated_by_descendants(self):
        analysis = self.analyze_text(
            "\n".join([
                "3 E = mc^2 + ATP",
                "3.1 Valid subsection",
                "3.2 Another valid subsection",
                "3.3 Third valid subsection",
            ])
        )

        parent = next(node for node in analysis.nodes if node.numbering == "3")
        self.assertEqual(parent.status, "rejected")
        self.assertFalse(
            any(
                signal.signal_type == "explicit_descendant_support"
                and signal.target_id == parent.id
                for signal in analysis.signals
            )
        )
        self.assertTrue(
            all(
                node.parent_id is None
                for node in analysis.nodes
                if node.numbering in {"3.1", "3.2", "3.3"}
            )
        )

    def test_italian_numbered_question_sequence_is_not_structural(self):
        analysis = self.analyze_text(
            "\n".join([
                "1 Cosa rappresenta la specifica operazione?",
                "Risposta descrittiva.",
                "2 Come si attua la specifica operazione?",
                "Risposta descrittiva.",
                "3 Quali sono le grandezze coinvolte?",
                "Risposta descrittiva.",
                "4 Come si registra il movimento?",
            ])
        )

        self.assertFalse(any(region.region_type == "structural_outline" for region in analysis.regions))
        self.assertFalse(any(sequence.sequence_type == "structural_sequence" for sequence in analysis.sequences))
        self.assertTrue(any(sequence.sequence_type == "local_enumeration" for sequence in analysis.sequences))
        self.assertNotEqual(analysis.global_structure_confidence, SOURCE_STRUCTURE_HIGH)
        self.assertFalse(analysis.relationships)

    def test_english_numbered_question_sequence_is_not_structural(self):
        analysis = self.analyze_text(
            "\n".join([
                "1 What is apoptosis?",
                "Answer text.",
                "2 How does apoptosis work?",
                "Answer text.",
                "3 Why is apoptosis important?",
            ])
        )

        self.assertFalse(any(sequence.sequence_type == "structural_sequence" for sequence in analysis.sequences))
        self.assertTrue(any(sequence.sequence_type == "local_enumeration" for sequence in analysis.sequences))
        self.assertNotEqual(analysis.global_structure_confidence, SOURCE_STRUCTURE_HIGH)

    def test_single_interrogative_does_not_destroy_academic_run(self):
        analysis = self.analyze_text(
            "\n".join([
                "1 What is apoptosis?",
                "Introductory content.",
                "2 Regulation of apoptosis",
                "Regulatory proteins control the pathway.",
                "3 Clinical relevance",
                "Clinical examples appear here.",
            ])
        )

        self.assertTrue(any(sequence.sequence_type == "structural_sequence" for sequence in analysis.sequences))
        self.assertTrue(
            all(
                node.status == "accepted" and node.confidence == SOURCE_STRUCTURE_HIGH
                for node in analysis.nodes
                if node.numbering in {"1", "2", "3"}
            )
        )

    def test_numbered_prose_sequence_is_not_semantic_local_structure(self):
        analysis = self.analyze_text(
            "\n".join([
                "7 Si tenga tuttavia presente l’eccezione rappresentata dal socio unico.",
                "Testo esplicativo.",
                "8 Negli enti pubblici esso viene denominato fondo di dotazione.",
                "Testo esplicativo.",
                "9 Può riscontrarsi anche il caso non infrequente nelle aziende.",
            ])
        )

        self.assertFalse(any(sequence.sequence_type == "semantic_local_sequence" for sequence in analysis.sequences))
        self.assertTrue(any(sequence.sequence_type == "local_enumeration" for sequence in analysis.sequences))
        self.assertFalse(any(relationship.relationship_type == "semantic_parent" for relationship in analysis.relationships))

    def test_section_scope_does_not_convert_uppercase_opener_plus_numbered_prose(self):
        analysis = self.analyze_text(
            "\n".join([
                "NECESSARIO PER SVOLGERE LA PROPRIA ATTIVITA'",
                "Testo introduttivo.",
                "7 Si tenga tuttavia presente l’eccezione rappresentata dal socio unico.",
                "Testo esplicativo.",
                "8 Negli enti pubblici esso viene denominato fondo di dotazione.",
                "Testo esplicativo.",
                "9 Può riscontrarsi anche il caso non infrequente nelle aziende.",
            ])
        )

        self.assertEqual(len(analysis.section_scopes), 0)
        self.assertFalse(any(sequence.sequence_type == "semantic_local_sequence" for sequence in analysis.sequences))
        self.assertFalse(any(relationship.relationship_type == "semantic_parent" for relationship in analysis.relationships))

    def test_true_ragioneria_chapter_structure_remains_structural(self):
        analysis = self.analyze_text(
            "\n".join([
                "PARTE TERZA",
                "CAPITOLO DODICESIMO",
                "1 Premessa",
                "2 L’Imposta sul Valore Aggiunto",
                "3 I resi, gli abbuoni, gli sconti",
            ])
        )

        self.assertEqual(analysis.global_structure_confidence, SOURCE_STRUCTURE_HIGH)
        self.assertTrue(any(sequence.sequence_type == "toc_sequence" for sequence in analysis.sequences))
        accepted = {
            node.title_original
            for node in analysis.nodes
            if node.status == "accepted"
        }
        self.assertIn("1 Premessa", accepted)
        self.assertIn("2 L’Imposta sul Valore Aggiunto", accepted)
        self.assertIn("3 I resi, gli abbuoni, gli sconti", accepted)

    def test_content_like_protection_preserves_mosaicism_semantic_local_scope(self):
        analysis = self.analyze_text(
            "\n".join([
                "Population and Mathematical Genetics",
                "1 Non-random mating",
                "2 Mutation",
                "3 Selection",
                "4 Small population size",
                "5 Gene flow",
            ])
        )

        self.assertEqual(analysis.local_structure_confidence, SOURCE_STRUCTURE_HIGH)
        self.assertTrue(any(sequence.sequence_type == "semantic_local_sequence" for sequence in analysis.sequences))
        self.assertTrue(any(relationship.relationship_type == "semantic_parent" for relationship in analysis.relationships))

    def test_content_like_protection_preserves_cellular_biology_explicit_hierarchy(self):
        analysis = self.analyze_text(
            "\n".join([
                "5 Prokaryotic gene regulation: operons",
                "5.1 Why operons are useful in bacteria",
                "5.2 General operon logic",
                "5.3 Repressible versus inducible operons",
                "5.4 Trp operon: repressible operon",
                "5.5 Lac operon: inducible operon",
            ])
        )

        by_number = {
            node.numbering: node
            for node in analysis.nodes
            if node.status == "accepted" and node.numbering
        }
        self.assertEqual(by_number["5.4"].parent_id, by_number["5"].id)
        self.assertEqual(by_number["5.5"].parent_id, by_number["5"].id)
        self.assertEqual(analysis.global_structure_confidence, SOURCE_STRUCTURE_HIGH)

    def test_global_skeleton_restores_true_top_level_headings(self):
        analysis = self.analyze_text(
            "\n".join([
                "8 Protein translation and the genetic code",
                "8.1 Reading frames",
                "8.2 Codon degeneracy",
                "8.3 Codon numbers",
                "8.4 Wobble",
                "9 tRNA charging",
                "9.1 Aminoacyl-tRNA synthetase reaction",
                "10 Ribosome structure",
                "11 Translation initiation",
                "11.1 Initiation factors",
                "11.2 Start codon recognition",
                "12 Translation elongation",
                "12.1 Elongation cycle",
                "13 Translation termination",
                "14 Polyribosomes",
                "15 mRNA surveillance and nonsense-mediated decay",
                "15.1 Exon-junction complexes",
                "15.2 How normal mRNA avoids NMD",
                "15.3 How premature stop codons trigger NMD",
                "16 Post-translational modifications",
                "16.1 Why post-translational regulation matters",
                "16.2 Ubiquitination",
                "16.3 SUMOylation",
            ])
        )

        by_number = {
            node.numbering: node
            for node in analysis.nodes
            if node.numbering
        }
        for number in {"9", "10", "12", "13"}:
            self.assertEqual(by_number[number].status, "accepted")
            self.assertTrue(
                any(
                    signal.signal_type == "global_numbering_skeleton_support"
                    and signal.target_id == by_number[number].id
                    for signal in analysis.signals
                )
            )
        self.assertEqual(by_number["9.1"].parent_id, by_number["9"].id)
        self.assertEqual(by_number["12.1"].parent_id, by_number["12"].id)
        self.assertEqual(by_number["16.3"].parent_id, by_number["16"].id)

    def test_one_descendant_without_global_skeleton_is_insufficient(self):
        analysis = self.analyze_text(
            "\n".join([
                "9 tRNA charging",
                "9.1 Aminoacyl-tRNA synthetase reaction",
            ])
        )

        parent = next(node for node in analysis.nodes if node.numbering == "9")
        child = next(node for node in analysis.nodes if node.numbering == "9.1")
        self.assertNotEqual(parent.status, "accepted")
        self.assertIsNone(child.parent_id)
        self.assertFalse(
            any(
                signal.signal_type == "global_numbering_skeleton_support"
                for signal in analysis.signals
            )
        )

    def test_local_enumeration_inside_dotted_subsection_is_excluded_from_global_skeleton(self):
        analysis = self.analyze_text(
            "\n".join([
                "2.3 How lncRNAs regulate gene expression",
                "Content before local list.",
                "1 Transcriptional interference with promoter activity",
                "Details.",
                "2 Chromatin remodeling",
                "Details.",
                "3 Histone modification",
                "Details.",
                "4 Alternative splicing modulation",
                "Details.",
                "8 Small RNA precursor",
                "Details.",
                "2.4 XIST and dosage compensation",
            ])
        )

        local_numbers = [
            node for node in analysis.nodes
            if node.numbering in {"1", "2", "3", "4", "8"}
        ]
        self.assertTrue(local_numbers)
        self.assertTrue(
            all(node.status != "accepted" for node in local_numbers)
        )
        self.assertTrue(
            any(
                signal.signal_type == "global_numbering_skeleton_exclusion"
                and signal.target_id in {node.id for node in local_numbers}
                for signal in analysis.signals
            )
        )

    def test_heading_like_local_enumeration_inside_dotted_subsection_cannot_be_global(self):
        analysis = self.analyze_text(
            "\n".join([
                "2 Molecular RNA regulation",
                "2.1 Coding and non-coding transcripts",
                "2.2 RNA processing",
                "2.3 How lncRNAs regulate gene expression",
                "This subsection introduces a local mechanism list.",
                "1 Transcriptional interference",
                "Mechanism detail.",
                "2 Chromatin remodeling",
                "Mechanism detail.",
                "3 Histone modification",
                "Mechanism detail.",
                "4 Alternative splicing modulation",
                "Mechanism detail.",
                "5 RNA-RNA interactions",
                "Mechanism detail.",
                "6 Protein binding",
                "Mechanism detail.",
                "7 Protein localization activity",
                "Mechanism detail.",
                "8 Small RNA precursor",
                "Mechanism detail.",
                "2.4 XIST as a key example of lncRNA function",
                "The next subsection resumes the explicit dotted hierarchy.",
            ])
        )

        by_number = {
            node.numbering: node
            for node in analysis.nodes
            if node.numbering
        }
        local_candidates = [
            node for node in analysis.nodes
            if node.numbering in {"1", "3", "4", "8"}
            and node.source_order > by_number["2.3"].source_order
            and node.source_order < by_number["2.4"].source_order
        ]
        self.assertTrue(local_candidates)
        self.assertTrue(
            all(node.structural_scope != "global" for node in local_candidates)
        )
        self.assertTrue(
            all(node.status != "accepted" for node in local_candidates)
        )
        self.assertEqual(by_number["2.3"].status, "accepted")
        self.assertEqual(by_number["2.4"].status, "accepted")

    def test_contained_duplicate_integer_does_not_shadow_true_global_parent(self):
        analysis = self.analyze_text(
            "\n".join([
                "1 RNA biology foundations",
                "Global section detail.",
                "2 Molecular RNA regulation",
                "Global section detail.",
                "2.1 Coding and non-coding transcripts",
                "2.2 RNA processing",
                "2.3 How lncRNAs regulate gene expression",
                "3 Histone modification",
                "Local enumeration detail.",
                "2.4 XIST as a key example of lncRNA function",
                "3 RNA-based diagnosis and therapy",
                "Global section detail.",
                "3.1 RNA interference therapeutics",
                "Explicit child detail.",
                "4 CRISPR-Cas9 and genome editing",
                "Global section detail.",
                "5 Protein translation and the genetic code",
                "Global section detail.",
            ])
        )

        local_three = next(
            node for node in analysis.nodes
            if node.numbering == "3"
            and "Histone" in node.title_original
        )
        global_three = next(
            node for node in analysis.nodes
            if node.numbering == "3"
            and "RNA-based" in node.title_original
        )
        child = next(node for node in analysis.nodes if node.numbering == "3.1")

        self.assertNotEqual(local_three.status, "accepted")
        self.assertEqual(global_three.status, "accepted")
        self.assertEqual(child.parent_id, global_three.id)

    def test_structural_sequence_support_cannot_override_dotted_subsection_containment(self):
        analysis = self.analyze_text(
            "\n".join([
                "2 Molecular RNA regulation",
                "2.1 Coding and non-coding transcripts",
                "2.2 RNA processing",
                "2.3 How lncRNAs regulate gene expression",
                "1 Transcriptional interference",
                "Long explanatory separation.",
                "2 Chromatin remodeling",
                "Long explanatory separation.",
                "3 Histone modification",
                "Long explanatory separation.",
                "4 Alternative splicing modulation",
                "Long explanatory separation.",
                "5 RNA-RNA interactions",
                "Long explanatory separation.",
                "6 Protein binding",
                "Long explanatory separation.",
                "7 Protein localization activity",
                "Long explanatory separation.",
                "8 Small RNA precursor",
                "Long explanatory separation.",
                "2.4 XIST as a key example of lncRNA function",
            ])
        )

        by_number = {
            node.numbering: node
            for node in analysis.nodes
            if node.numbering
        }
        contained = [
            node for node in analysis.nodes
            if node.numbering and node.numbering.isdigit()
            and by_number["2.3"].source_order < node.source_order < by_number["2.4"].source_order
        ]
        self.assertEqual({node.numbering for node in contained}, {"1", "2", "3", "4", "5", "6", "7", "8"})
        self.assertTrue(all(node.structural_scope != "global" for node in contained))
        self.assertTrue(all(node.status != "accepted" for node in contained))

    def test_section_scope_does_not_repromote_contained_dotted_subsection_integers(self):
        analysis = self.analyze_text(
            "\n".join([
                "2 Molecular RNA regulation",
                "2.1 Coding and non-coding transcripts",
                "2.2 RNA processing",
                "2.3 How lncRNAs regulate gene expression",
                "LOCAL MECHANISMS",
                "Introductory explanation.",
                "1 Transcriptional interference",
                "Mechanism detail.",
                "2 Chromatin remodeling",
                "Mechanism detail.",
                "3 Histone modification",
                "Mechanism detail.",
                "4 Alternative splicing modulation",
                "Mechanism detail.",
                "2.4 XIST as a key example of lncRNA function",
            ])
        )

        by_number = {
            node.numbering: node
            for node in analysis.nodes
            if node.numbering
        }
        contained = [
            node for node in analysis.nodes
            if node.numbering in {"1", "2", "3", "4"}
            and by_number["2.3"].source_order < node.source_order < by_number["2.4"].source_order
        ]
        self.assertTrue(contained)
        self.assertTrue(all(node.status != "accepted" for node in contained))
        self.assertTrue(all(node.structural_scope != "global" for node in contained))

    def test_local_integer_two_does_not_shadow_parent_of_two_four(self):
        analysis = self.analyze_text(
            "\n".join([
                "2.3 How lncRNAs regulate gene expression",
                "1 Transcriptional interference",
                "2 Chromatin remodeling",
                "3 Histone modification",
                "2.4 XIST and dosage compensation",
            ])
        )

        two_four = next(node for node in analysis.nodes if node.numbering == "2.4")
        local_two = next(node for node in analysis.nodes if node.numbering == "2")
        self.assertNotEqual(local_two.status, "accepted")
        self.assertNotEqual(two_four.parent_id, local_two.id)

    def test_noisy_numbered_fragments_do_not_gain_global_skeleton_support(self):
        analysis = self.analyze_text(
            "\n".join([
                "Bone tissue 2024",
                "1",
                "trabeculae random OCR fragment",
                "2",
                "Ca2+ + PO4 = mineral",
                "3 Fig. 12 irregular caption",
                "H4 PHOSPHORYLATION S1",
            ])
        )

        self.assertEqual(analysis.source_structure_confidence, SOURCE_STRUCTURE_LOW)
        self.assertFalse(
            any(
                signal.signal_type == "global_numbering_skeleton_support"
                for signal in analysis.signals
            )
        )

    def test_unsafe_numbering_is_not_rehabilitated(self):
        analysis = self.analyze_text(
            "\n".join([
                "31 Unsafe parent",
                "31.1 Unsafe child one",
                "31.2 Unsafe child two",
            ])
        )

        self.assertEqual(analysis.accepted_node_count, 0)
        self.assertFalse(analysis.relationships)

    def test_unknown_region_has_insufficient_evidence(self):
        analysis = self.analyze_text("8 Isolated plausible heading")

        self.assertTrue(any(region.region_type == "unknown" for region in analysis.regions))
        self.assertEqual(analysis.source_structure_confidence, SOURCE_STRUCTURE_LOW)
        self.assertEqual(analysis.accepted_node_count, 0)

    def test_explicit_numbered_hierarchy_is_not_semantic_parentage(self):
        analysis = self.analyze_text(
            "\n".join([
                "5. Prokaryotic gene regulation",
                "5.1 Why operons are useful",
                "5.2 General operon logic",
                "5.3 Repressible versus inducible operons",
            ])
        )

        self.assertEqual(analysis.global_structure_confidence, SOURCE_STRUCTURE_HIGH)
        self.assertFalse(
            any(relationship.relationship_type == "semantic_parent" for relationship in analysis.relationships)
        )
        self.assertTrue(
            any(relationship.relationship_type == "explicit_numbering_parent" for relationship in analysis.relationships)
        )

    def test_semantic_parent_with_local_numbered_sequence(self):
        analysis = self.analyze_text(
            "\n".join([
                "",
                "Population and Mathematical Genetics",
                "",
                "1 Non-random mating",
                "Population structure affects genotype frequencies.",
                "2 Mutation",
                "Mutation introduces new alleles.",
                "3 Selection",
                "Selection changes allele frequencies.",
                "4 Small population size",
                "Drift is stronger in small populations.",
                "5 Gene flow (migration).",
                "Migration exchanges alleles between populations.",
            ])
        )

        self.assertEqual(analysis.local_structure_confidence, SOURCE_STRUCTURE_HIGH)
        self.assertNotEqual(analysis.global_structure_confidence, SOURCE_STRUCTURE_HIGH)
        titles = {
            node.title_original: node
            for node in analysis.nodes
            if node.status == "accepted"
        }
        parent = titles["Population and Mathematical Genetics"]
        self.assertEqual(parent.structural_scope, "global")
        self.assertIn("5 Gene flow (migration).", titles)
        children = [
            node for node in titles.values()
            if node.parent_id == parent.id
        ]
        self.assertEqual(len(children), 5)
        self.assertTrue(all(child.structural_scope == "local" for child in children))
        self.assertTrue(
            all(
                relationship.relationship_type == "semantic_parent"
                for relationship in analysis.relationships
                if relationship.parent_node_id == parent.id
            )
        )

    def test_multiple_local_scopes_with_numbering_restart(self):
        analysis = self.analyze_text(
            "\n".join([
                "",
                "Mendelian Inheritance",
                "",
                "1 Law of Segregation",
                "2 Law of Dominance",
                "3 Law of Independent Assortment",
                "",
                "Chromosome Abnormalities",
                "",
                "1 Numerical abnormalities",
                "2 Structural abnormalities",
                "3 Different cell lines",
            ])
        )

        parents = {
            node.title_original: node
            for node in analysis.nodes
            if node.status == "accepted" and node.detection_method == "textual"
        }
        self.assertIn("Mendelian Inheritance", parents)
        self.assertIn("Chromosome Abnormalities", parents)
        for parent in parents.values():
            children = [
                node for node in analysis.nodes
                if node.status == "accepted" and node.parent_id == parent.id
            ]
            self.assertEqual(len(children), 3)
            self.assertTrue(all(child.structural_scope == "local" for child in children))
        self.assertEqual(analysis.local_structure_confidence, SOURCE_STRUCTURE_HIGH)
        self.assertNotEqual(analysis.global_structure_confidence, SOURCE_STRUCTURE_HIGH)

    def test_procedure_heading_does_not_create_semantic_local_taxonomy(self):
        analysis = self.analyze_text(
            "\n".join([
                "",
                "Procedure",
                "",
                "1 Add reagent to the tube.",
                "2 Incubate for 10 minutes.",
                "3 Centrifuge the sample.",
                "4 Remove the supernatant.",
            ])
        )

        self.assertFalse(
            any(relationship.relationship_type == "semantic_parent" for relationship in analysis.relationships)
        )
        self.assertEqual(analysis.accepted_node_count, 0)
        self.assertEqual(analysis.source_structure_confidence, SOURCE_STRUCTURE_LOW)

    def test_isolated_textual_heading_does_not_attach_unrelated_numbering_later(self):
        analysis = self.analyze_text(
            "\n".join([
                "",
                "Population and Mathematical Genetics",
                "",
                "This paragraph introduces the subject without a local list.",
                "Several explanatory lines intervene.",
                "The next numbers belong to an unrelated exercise.",
                "",
                "1 Recall the definition",
                "2 Compare the examples",
                "3 Check the diagram",
            ])
        )

        self.assertFalse(
            any(relationship.relationship_type == "semantic_parent" for relationship in analysis.relationships)
        )
        self.assertNotEqual(analysis.local_structure_confidence, SOURCE_STRUCTURE_HIGH)

    def test_synthetic_noisy_histology_like_document_remains_low(self):
        analysis = self.analyze_text(
            "\n".join([
                "Bone tissue 2024",
                "osteoblast arrows --> matrix",
                "Ca2+ + PO4 = mineral",
                "Fig. 12 irregular caption",
                "random OCR fragment trabeculae canaliculus",
                "45",
                "H4 phosphorylation S1",
            ])
        )

        self.assertEqual(analysis.source_structure_confidence, SOURCE_STRUCTURE_LOW)
        self.assertEqual(analysis.global_structure_confidence, SOURCE_STRUCTURE_LOW)
        self.assertEqual(analysis.local_structure_confidence, SOURCE_STRUCTURE_LOW)
        self.assertEqual(analysis.accepted_node_count, 0)

    def test_strong_local_structure_does_not_make_noisy_document_globally_high(self):
        analysis = self.analyze_blocks([
            (
                "\n".join([
                    "",
                    "Population and Mathematical Genetics",
                    "",
                    "1 Non-random mating",
                    "2 Mutation",
                    "3 Selection",
                    "4 Small population size",
                    "5 Gene flow",
                ]),
                1,
            ),
            (
                "\n".join([
                    "FLOATING NOTE",
                    "random OCR fragment",
                    "ATP + ADP = energy",
                    "another unstructured paragraph",
                ]),
                2,
            ),
        ])

        self.assertEqual(analysis.local_structure_confidence, SOURCE_STRUCTURE_HIGH)
        self.assertNotEqual(analysis.global_structure_confidence, SOURCE_STRUCTURE_HIGH)

    def test_section_scope_bridges_heading_body_text_and_later_sequence(self):
        analysis = self.analyze_text(
            "\n".join([
                "",
                "Population and Mathematical Genetics",
                "",
                "This section explains why population-level mechanisms matter.",
                "It connects mating patterns, mutation, selection and migration.",
                "The following local structure lists the main mechanisms.",
                "Each mechanism changes allele frequencies in a different way.",
                "The sequence below belongs to the population genetics section.",
                "Additional prose separates the heading from the list.",
                "More explanatory prose keeps the content continuous.",
                "One final sentence before the local sequence begins.",
                "1 Non-random mating",
                "2 Mutation",
                "3 Selection",
                "4 Small population size",
                "5 Gene flow",
            ])
        )

        self.assertEqual(len(analysis.section_scopes), 1)
        self.assertEqual(analysis.local_structure_confidence, SOURCE_STRUCTURE_HIGH)
        self.assertNotEqual(analysis.global_structure_confidence, SOURCE_STRUCTURE_HIGH)
        parent = next(
            node for node in analysis.nodes
            if node.title_original == "Population and Mathematical Genetics"
        )
        children = [
            node for node in analysis.nodes
            if node.parent_id == parent.id
        ]
        self.assertEqual(len(children), 5)
        self.assertTrue(all(child.structural_scope == "local" for child in children))
        self.assertTrue(
            all(
                relationship.relationship_type == "semantic_parent"
                for relationship in analysis.relationships
                if relationship.parent_node_id == parent.id
            )
        )
        self.assertNotEqual(parent.region_id, children[0].region_id)
        self.assertEqual(parent.section_scope_id, children[0].section_scope_id)

    def test_section_scope_can_continue_across_compatible_block_boundary(self):
        analysis = self.analyze_blocks([
            (
                "\n".join([
                    "",
                    "Population and Mathematical Genetics",
                    "",
                    "This section starts near the end of a page.",
                    "The local list continues in the next extracted block.",
                ]),
                1,
            ),
            (
                "\n".join([
                    "1 Non-random mating",
                    "2 Mutation",
                    "3 Selection",
                    "4 Small population size",
                    "5 Gene flow",
                ]),
                2,
            ),
        ])

        self.assertEqual(len(analysis.section_scopes), 1)
        self.assertEqual(analysis.local_structure_confidence, SOURCE_STRUCTURE_HIGH)
        parent = next(
            node for node in analysis.nodes
            if node.title_original == "Population and Mathematical Genetics"
        )
        self.assertEqual(
            len([node for node in analysis.nodes if node.parent_id == parent.id]),
            5,
        )

    def test_competing_heading_boundary_prevents_previous_scope_from_swallowing_sequence(self):
        analysis = self.analyze_text(
            "\n".join([
                "",
                "Mendelian Inheritance",
                "",
                "This section introduces laws of inheritance.",
                "",
                "Chromosome Abnormalities",
                "",
                "This section introduces chromosome-level problems.",
                "1 Numerical abnormalities",
                "2 Structural abnormalities",
                "3 Different cell lines",
            ])
        )

        self.assertEqual(len(analysis.section_scopes), 1)
        mendelian = next(
            node for node in analysis.nodes
            if node.title_original == "Mendelian Inheritance"
        )
        chromosome = next(
            node for node in analysis.nodes
            if node.title_original == "Chromosome Abnormalities"
        )
        self.assertIsNone(mendelian.section_scope_id)
        self.assertIsNotNone(chromosome.section_scope_id)
        self.assertEqual(
            len([node for node in analysis.nodes if node.parent_id == chromosome.id]),
            3,
        )

    def test_section_scope_does_not_convert_procedure_across_region_boundary(self):
        analysis = self.analyze_text(
            "\n".join([
                "",
                "Procedure",
                "",
                "The protocol is described below.",
                "Read every step before beginning.",
                "1 Add reagent to the tube.",
                "2 Incubate for 10 minutes.",
                "3 Centrifuge the sample.",
                "4 Remove the supernatant.",
            ])
        )

        self.assertEqual(len(analysis.section_scopes), 0)
        self.assertFalse(
            any(relationship.relationship_type == "semantic_parent" for relationship in analysis.relationships)
        )
        self.assertEqual(analysis.local_structure_confidence, SOURCE_STRUCTURE_LOW)


if __name__ == "__main__":
    unittest.main()
