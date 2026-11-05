"""Offline chapter boundaries, persisted round trips and isolated ownership."""
import copy
import json
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from test_pdf_source_alignment import pdf, run
from test_document_interpreter_pdf import _pdf_bytes, _text_command
from pdf_source_alignment import chapter_metadata
from source_chunk_alignment import verified_chunk_chapters, metadata_revision
from taxonomy_organization import organize_verified_chapters
from taxonomy_coverage import consolidate_candidates
from taxonomy_assignment import plan_assignments


def fixture():
    pages = []
    for title in ("1 Chapter Alpha", "2 Chapter Beta"):
        pages.append([_text_command(title, x=72,y=740,size=20)] +
                     [_text_command(f"Body line {i}.",x=72,y=700-i*30,size=12) for i in range(3)])
    result = run(_pdf_bytes(pages))
    doc, provenance = chapter_metadata(result)
    chunks = [{"id": i+1, "document_id": "document-A", "chunk_text": b.alignment.chunks[0].text,
               "structural_provenance": provenance[i][0]} for i,b in enumerate(result.blocks)]
    return doc, chunks


class ChapterTests(unittest.TestCase):
    def test_verified_pdf_persisted_roundtrip_and_direct_categories(self):
        doc, chunks = fixture()
        self.assertEqual(doc["status"], "verified")
        owners = verified_chunk_chapters(json.loads(json.dumps(doc)), json.loads(json.dumps(chunks)), "document-A")
        self.assertEqual(len(set(owners.values())), 2)
        topics = [dict(assignment_id=str(i), evidence_ids=[i+1], title="Repeated subject") for i in range(2)]
        result = organize_verified_chapters({"categories":[{"name":"Old","topics":topics}]}, doc, owners)
        self.assertEqual([c["name"] for c in result["categories"]], ["1 Chapter Alpha","2 Chapter Beta"])
        self.assertIs(result["categories"][0]["topics"][0], topics[0])
        self.assertIs(result["categories"][1]["topics"][0], topics[1])

    def test_midpage_boundary_is_exact_but_cross_chapter_chunk_falls_back(self):
        result = run(pdf([('1 Chapter Alpha',72,740,20),('Body one.',72,710,12),
                          ('Body two.',72,680,12),('2 Chapter Beta',72,610,20),
                          ('Body three.',72,580,12),('Body four.',72,550,12)]))
        doc, provenance = chapter_metadata(result)
        self.assertEqual(doc["status"], "verified")
        start = doc["chapters"][1]["extent"]["start"]
        self.assertEqual((start["unit_index"],start["block_index"],start["start_offset"]),(1,4,0))
        chunk = result.blocks[0].alignment.chunks[0]
        ids = {i["chapter_id"] for i in provenance[0][0]["intervals"]} - {None}
        self.assertEqual(len(ids),2)  # both associations retained, never whole-page ownership
        self.assertIsNone(verified_chunk_chapters(doc,[{"id":1,"document_id":"document-A",
            "chunk_text":chunk.text,"structural_provenance":provenance[0][0]}],"document-A"))

    def test_recovery_rejects_stale_revision_text_and_other_documents(self):
        doc, chunks = fixture()
        for mutation in ("revision","text","document","gap"):
            changed=copy.deepcopy(chunks)
            if mutation=="revision": changed[0]["structural_provenance"]["revision"]="old"
            if mutation=="text": changed[0]["chunk_text"]+="changed"
            if mutation=="document": changed[0]["document_id"]="sibling"
            if mutation=="gap": changed[0]["structural_provenance"]["intervals"][0]["start"]=1
            self.assertIsNone(verified_chunk_chapters(doc,changed,"document-A"))
        changed=copy.deepcopy(doc);changed["chapters"][0]["title"]="Changed"
        self.assertIsNone(verified_chunk_chapters(changed,chunks,"document-A"))
        self.assertIsNone(verified_chunk_chapters(None,chunks,"document-A"))

    def test_unreliable_layout_and_minor_headings_fall_back(self):
        result=run(pdf([('2 Chapter Beta',330,100,20),('1 Chapter Alpha',72,740,20),
                        ('Body one.',72,710,12),('Body two.',72,680,12),('Body three.',72,650,12)]))
        doc,_=chapter_metadata(result)
        self.assertEqual(doc["status"],"unavailable")
        self.assertEqual(doc["chapters"],[])

    def test_existing_topic_with_cross_chapter_evidence_is_not_split(self):
        doc,chunks=fixture();owners=verified_chunk_chapters(doc,chunks,"document-A")
        topic={"assignment_id":"unchanged","evidence_ids":[1,2]}
        data={"categories":[{"name":"Old","topics":[topic]}]}
        before=copy.deepcopy(data)
        self.assertIsNone(organize_verified_chapters(data,doc,owners))
        self.assertEqual(data,before)

    def test_identical_objectives_in_different_chapters_never_reach_ai(self):
        cs=[dict(topic="Repeated subject",description="Same",category="Same label",importance=5,
                 embedding=[1.,0.],assignment_id=str(i),evidence_ids=[i+1],chapter_id=f"chapter{i}")
            for i in range(2)]
        result=consolidate_candidates(cs,lambda _:self.fail("cross-chapter AI review"),"")
        self.assertEqual(sum(len(c["topics"]) for c in result["categories"]),2)
        self.assertEqual({t["chapter_id"] for c in result["categories"] for t in c["topics"]},
                         {"chapter0","chapter1"})

    def test_assignment_primary_and_rescue_cannot_cross_chapters(self):
        cs=[dict(assignment_id=str(i),topic="Same",source_section="Same",embedding=[1.,0.],
                 chapter_id=f"chapter{i}") for i in range(2)]
        # Cross-chapter scores are highest; eligible own scores only meet rescue.
        rows=[(cid,"body","section",str(i),"Same","Same",.5 if cid==i+1 else .99)
              for cid in (1,2) for i in range(2)]
        db=SimpleNamespace(execute=lambda *a,**kw:SimpleNamespace(fetchall=lambda: (sorted({(r[0], r[1], r[2]) for r in rows}) if "cross join" not in str(a[0]) else [(r[0], r[3], r[6]) for r in rows])))
        with patch("taxonomy_assignment.source_snapshot",return_value={1:"a",2:"b"}):
            links,_=plan_assignments(db,cs,"p",lambda *a:a[-1],
                                    chapter_ownership={1:"chapter0",2:"chapter1"})
        self.assertEqual(links,{"0":{1},"1":{2}})


if __name__ == "__main__":
    unittest.main()

    def test_main_fragments_and_ocr_page_interval_preserve_content(self):
        from document_extractors import extract_pdf_document
        from pdf_source_alignment import align_digital_pdf
        from test_source_chunk_alignment import CLEAN, CHUNK
        pages = [[_text_command('MAIN',x=72,y=740,size=24),
                  _text_command(' CHAPTER',x=150,y=740,size=24),
                  _text_command('continued title',x=72,y=710,size=24),
                  _text_command('49 ref)',x=72,y=670,size=16),
                  *[_text_command('Body text',x=72,y=640-i*25,size=12) for i in range(4)]], [],
                 [_text_command('SECOND CHAPTER',x=72,y=740,size=24),
                  *[_text_command('Body text',x=72,y=700-i*25,size=12) for i in range(4)]]]
        data = _pdf_bytes(pages)
        supplied = extract_pdf_document(data,'fixture.pdf',inspect_acquisition=True)
        supplied.blocks[1].digital_text = supplied.blocks[1].text
        supplied.blocks[1].text = supplied.blocks[1].raw_text = 'OCR study content must survive'
        result = align_digital_pdf(data,'fixture.pdf','document-A',clean=CLEAN,chunk=CHUNK,
                                   extracted_document=supplied)
        doc, ps = chapter_metadata(result)
        self.assertEqual(doc['status'],'verified')
        self.assertEqual([c['title'] for c in doc['chapters']],
                         ['MAIN CHAPTER continued title','SECOND CHAPTER'])
        chunks = []
        for b, provs in zip(result.blocks,ps):
            for ch, prov in zip(b.alignment.chunks,provs):
                chunks.append(dict(id=len(chunks)+1,document_id='document-A',
                                   chunk_text=ch.text,structural_provenance=prov))
        owners = verified_chunk_chapters(doc,chunks,'document-A')
        self.assertEqual(len(owners),len(chunks))
        self.assertEqual(owners[2],doc['chapters'][0]['id'])
        self.assertEqual(chunks[1]['chunk_text'],CLEAN(supplied.blocks[1].text))
        self.assertTrue(all(not i['source_spans'] for i in ps[1][0]['intervals']))
        self.assertTrue(any(i['status']=='unresolved' for i in ps[1][0]['intervals']))
        changed=copy.deepcopy(chunks)
        changed[1]['structural_provenance']['intervals'][0]['source_page']=3
        self.assertIsNone(verified_chunk_chapters(doc,changed,'document-A'))

    def test_ocr_on_midpage_boundary_is_not_guessed(self):
        from document_extractors import extract_pdf_document
        from pdf_source_alignment import align_digital_pdf
        from test_source_chunk_alignment import CLEAN,CHUNK
        data=pdf([('1 Chapter Alpha',72,740,20),('Body',72,700,12),
                  ('2 Chapter Beta',72,610,20),('More body',72,570,12),('Last body',72,540,12)])
        supplied=extract_pdf_document(data,'fixture.pdf',inspect_acquisition=True)
        b=supplied.blocks[0];b.digital_text=b.text;b.text+=' OCR unknown position';b.raw_text=b.text
        result=align_digital_pdf(data,'fixture.pdf','document-A',clean=CLEAN,chunk=CHUNK,extracted_document=supplied)
        doc,ps=chapter_metadata(result)
        self.assertEqual(doc['status'],'verified')
        self.assertTrue(any(i['chapter_id'] is None and i['status']=='unresolved' for i in ps[0][0]['intervals']))
        self.assertIsNone(verified_chunk_chapters(doc,[dict(id=1,document_id='document-A',
            chunk_text=result.blocks[0].alignment.chunks[0].text,structural_provenance=ps[0][0])],'document-A'))
