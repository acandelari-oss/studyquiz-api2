"""Offline matrix and PDF boundary regression tests, no services."""
import copy
import unittest
from pdf_source_alignment import _upright_rendered_text, chapter_metadata
from test_pdf_source_alignment import run
from test_document_interpreter_pdf import _pdf_bytes, _text_command

class ComposedTransformTests(unittest.TestCase):
    def test_cancelling_vertical_reflections_are_upright(self):
        m={'text_matrix':[40.,0.,0.,-40.,0.,81.21206],
           'current_transformation_matrix':[.75,0.,0.,-.75,49.6063,770.]}
        before=copy.deepcopy(m)
        self.assertTrue(_upright_rendered_text(m))
        self.assertEqual(m,before)

    def test_genuine_rotation_skew_reflection_and_invalid_geometry_rejected(self):
        for tm in ([1,0,0,-1,0,0], [0,1,-1,0,0,0], [1,.1,0,1,0,0],
                   [0,0,0,1,0,0], [float('nan'),0,0,1,0,0], [1,0]):
            with self.subTest(tm=tm):
                self.assertFalse(_upright_rendered_text({'text_matrix':tm,
                    'current_transformation_matrix':[1,0,0,1,0,0]}))

    def test_real_pdf_equivalent_rendering_preserves_chapter_extents_and_chunks(self):
        ordinary,transformed=[],[]
        for title in ('1 Chapter Alpha','2 Chapter Beta'):
            lines=[(title,740,20),('Body one.',700,12),('Body two.',670,12),('Body three.',640,12)]
            ordinary.append([_text_command(text,x=72,y=y,size=size) for text,y,size in lines])
            transformed.append([f'q 1 0 0 -1 0 800 cm BT /F1 {size} Tf 1 0 0 -1 72 {800-y} Tm ({text}) Tj ET Q'
                                for text,y,size in lines])
        a,b=run(_pdf_bytes(ordinary)),run(_pdf_bytes(transformed))
        da,_=chapter_metadata(a);db,_=chapter_metadata(b)
        self.assertEqual(da['status'],'verified')
        self.assertEqual(db['status'],'verified')
        self.assertEqual(da['chapters'],db['chapters'])
        self.assertEqual([[c.text for c in x.alignment.chunks] for x in a.blocks],
                         [[c.text for c in x.alignment.chunks] for x in b.blocks])
        self.assertEqual(b.canonical_document.blocks[0].metadata['text_matrix'][3],-1)
