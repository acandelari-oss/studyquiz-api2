"""Acquisition evidence for the existing extractors; no database or AI calls.

A failed unit has no recoverable text and is not confirmed blank, or its text
parser failed (even if OCR recovered a fallback). Partial units have usable text
but unresolved visual/embedded content. Partial != acquired. DOCX has no page
percentage. OCR recovers text only, never certifies visual meaning.
"""
import io
from collections import Counter
from zipfile import ZipFile

from document_extractors import DocumentExtractionError, ExtractedDocument, ExtractedBlock, extract_uploaded_document


def ocr_image(blob):
    from PIL import Image
    import pytesseract
    with Image.open(io.BytesIO(blob)) as image:
        return pytesseract.image_to_string(image)


def ocr_pdf(data, page):
    from pdf2image import convert_from_bytes
    import pytesseract
    images = convert_from_bytes(data, first_page=page, last_page=page, timeout=60)
    try:
        return "\n".join(pytesseract.image_to_string(image) for image in images)
    finally:
        for image in images:
            image.close()


def finish_report(filename, kind, units):
    counts = Counter(unit['status'] for unit in units)
    total = len(units) if kind != 'document' else None
    failed = counts['failed']
    rejected = (failed * 10 > total) if total is not None else bool(failed)
    return {
        'filename': filename, 'unit_kind': kind, 'total': total,
        'acquired': counts['acquired'], 'partial': counts['partial'],
        'failed': failed, 'blank': counts['blank'], 'units': units,
        'status': 'failed' if rejected else (
            'completed_with_warnings' if failed or counts['partial'] else 'completed'),
        'accepted': not rejected, 'limitations': [],
        'ocr_units': sum(bool(unit.get('ocr_attempted')) for unit in units),
    }


def _unit(number, text, visual=False, blank=False, errors=(), unsupported=False):
    reasons = list(errors)
    if visual:
        reasons.append('unresolved_visuals')
    if unsupported:
        reasons.append('unsupported_embedded_content')
    if 'parser_error' in reasons:
        status = 'failed'
    elif text.strip():
        status = 'partial' if reasons else 'acquired'
    elif blank and not reasons:
        status = 'blank'
    else:
        status = 'failed'
        if not reasons:
            reasons.append('no_recoverable_text')
    return {'number': number, 'status': status, 'reasons': list(dict.fromkeys(reasons)),
            'better_copy': status == 'failed' and 'parser_error' not in reasons and 'ocr_error' not in reasons}


def _supplement(existing, recovered):
    # Do not erase digital text. Remove exact repeated OCR lines only.
    lines = {line.strip() for line in existing.splitlines()}
    extra = [line for line in recovered.splitlines() if line.strip() and line.strip() not in lines]
    return '\n'.join([existing] + extra).strip()


def has_extraction_character_errors(value):
    """Detect explicit decoding damage, not foreign languages or mathematical glyphs.

    This cannot establish that visually plausible digits or formulas are correct.
    """
    return any(c == '\ufffd' or (ord(c) < 32 and c not in '\n\r\t\f')
               or 127 <= ord(c) <= 159 for c in value or '')


def acquire_document(data, filename):
    try:
        document = extract_uploaded_document(data, filename, inspect_acquisition=True)
    except DocumentExtractionError as exc:
        # No invented page denominator when the file cannot be opened.
        document = ExtractedDocument(filename, filename.rsplit('.', 1)[-1].lower(), len(data), [])
        report = finish_report(filename, 'document', [_unit(None, '', errors=['parser_error'])])
        report['message'] = str(exc)
        return document, report
    units = []
    if document.file_format == 'pdf':
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            reader.decrypt('')
        for block, page in zip(document.blocks, reader.pages):
            block.digital_text = block.text
            errors = [block.acquisition_error] if block.acquisition_error else []
            damaged_text = has_extraction_character_errors(block.digital_text)
            if damaged_text:
                errors.append('text_encoding_anomaly')
            visual = False
            blank = False
            try:
                content = page.get_contents()
                ops = {op for _, op in content.operations} if content is not None else set()
                # Painting, inline images, external forms/images and shading are
                # evidence of possible educational visuals, not blank space.
                visual = bool(ops & {b'Do', b'INLINE IMAGE', b'sh', b'S', b's', b'f', b'F', b'f*', b'B', b'B*', b'b', b'b*'})
                annotations = bool(page.get('/Annots'))
                if annotations:
                    errors.append('unsupported_embedded_content')
                blank = not block.text.strip() and not visual and not annotations and not (ops & {b'Tj', b'TJ', b"'", b'"'})
            except Exception:
                errors.append('inspection_error')
            # Reuse the existing page-local OCR once; do not guess missing digits,
            # erase original text or claim that OCR establishes exact provenance.
            ocr_attempted = damaged_text or visual or (not block.text.strip() and not blank)
            if ocr_attempted:
                try:
                    block.text = _supplement(block.text, ocr_pdf(data, block.page))
                except Exception:
                    errors.append('ocr_error')
            block.digital_text_start = (len(block.digital_text) - len(block.digital_text.lstrip())
                                        if block.text != block.digital_text else 0)
            block.raw_text = block.text
            units.append(_unit(block.page, block.text, visual, blank, errors))
            units[-1]["ocr_attempted"] = ocr_attempted
        kind = 'page'
    elif document.file_format == 'docx':
        from docx import Document
        source = Document(io.BytesIO(data))
        text = '\n'.join(block.text for block in document.blocks)
        errors = []
        images = []
        # All embedded image relationships, including headers/footers.
        for part in source.part.package.parts:
            if str(part.content_type).startswith('image/') and str(part.partname).startswith('/word/'):
                images.append(part.blob)
        for blob in images:
            try:
                recovered = ocr_image(blob).strip()
                if recovered:
                    document.blocks.append(ExtractedBlock(recovered, recovered, None, section='Image OCR', block_index=len(document.blocks)+1))
                    text += '\n' + recovered
            except Exception:
                errors.append('ocr_error')
        with ZipFile(io.BytesIO(data)) as archive:
            xml = '\n'.join(archive.read(name).decode('utf-8', errors='replace') for name in archive.namelist() if name.startswith(('word/document.xml', 'word/header', 'word/footer', 'word/footnotes.xml', 'word/endnotes.xml')))
            # Do not silently certify unsupported text boxes, charts, equations,
            # tracked deletions, footnotes, headers or embedded objects as acquired.
            unsupported = any(token in xml for token in ('<w:txbxContent', '<c:chart', '<m:oMath', '<w:object', '<w:del ', '<w:footnote ', '<w:endnote ', '<w:hdr', '<w:ftr', '<w:altChunk', '<dgm:', '<v:shape', '<w:sdt', '<w:smartTag', '<w:hyperlink'))
        units.append(_unit(None, text, bool(images), not text.strip() and not images and not unsupported, errors, unsupported))
        units[-1]['ocr_attempted'] = bool(images)
        kind = 'document'
    else:
        from pptx import Presentation
        from pptx.enum.shapes import MSO_SHAPE_TYPE
        source = Presentation(io.BytesIO(data))
        for index, slide in enumerate(source.slides, 1):
            blocks = [block for block in document.blocks if block.source_unit == index]
            # The extractor's fallback heading is metadata, not source text.
            if not (getattr(slide.shapes.title, 'text', '') or '').strip():
                for block in blocks:
                    if block.text.startswith(f'Slide {index}'):
                        block.text = block.text[len(f'Slide {index}'):].lstrip()
                        break
            errors = []
            visual = False
            unsupported = False
            def inspect(shapes, depth=0):
                nonlocal visual, unsupported
                for shape in shapes:
                    if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
                        if depth > 0:
                            unsupported = True  # legacy text extractor stops one level deep
                        inspect(shape.shapes, depth+1)
                    elif shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                        visual = True
                        try:
                            recovered = ocr_image(shape.image.blob).strip()
                            if recovered:
                                block = ExtractedBlock(recovered, recovered, None, section=f'Slide {index}', source_unit=index)
                                blocks.append(block)
                                document.blocks.append(block)
                        except Exception:
                            errors.append('ocr_error')
                    elif not getattr(shape, 'has_text_frame', False) and not getattr(shape, 'has_table', False):
                        visual = True
                        unsupported = True
                    elif shape.shape_type == MSO_SHAPE_TYPE.AUTO_SHAPE:
                        visual = True
            inspect(slide.shapes)
            # Background/master/layout graphics and speaker notes are not
            # guaranteed to be represented by slide.shapes.
            related_xml = ''.join(str(part._element.xml) for part in (slide, slide.slide_layout, slide.slide_layout.slide_master))
            unsupported = unsupported or any(token in related_xml for token in ('<a:blip', '<c:chart', '<dgm:', '<p:oleObj', '<m:oMath', '<a14:m'))
            if slide.has_notes_slide and (slide.notes_slide.notes_text_frame.text or '').strip():
                notes = slide.notes_slide.notes_text_frame.text.strip()
                block = ExtractedBlock(notes, notes, None, section=f'Slide {index} notes', source_unit=index)
                blocks.append(block)
                document.blocks.append(block)
            text = '\n'.join(block.text for block in blocks)
            for block in blocks:
                block.raw_text = block.text
            units.append(_unit(index, text, visual, not text.strip() and not visual and not unsupported, errors, unsupported))
        kind = 'slide'
    report = finish_report(filename, kind, units)
    if not any(block.text.strip() for block in document.blocks):
        report['accepted'] = False
        report['status'] = 'failed'
        report['limitations'].append('no_study_text')
    return document, report
