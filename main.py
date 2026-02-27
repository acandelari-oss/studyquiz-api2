@app.post("/projects/{project_id}/ingest")
def ingest(
    project_id: str,
    data: IngestRequest,
    api_key: str = Depends(verify_api_key)
):

    db = SessionLocal()

    try:

        for doc in data.documents:

            document_id = str(uuid.uuid4())

            # salva documento
            db.execute(sql_text("""
                insert into documents
                (id, project_id, filename)
                values
                (:id, :project_id, :filename)
            """), {
                "id": document_id,
                "project_id": project_id,
                "filename": doc.title
            })


            chunks = [doc.text]

            vectors = embed_texts(chunks)


            for chunk_text, vector in zip(chunks, vectors):

                db.execute(sql_text("""

                    insert into chunks
                    (
                        project_id,
                        doc_id,
                        doc_title,
                        chunk_text,
                        embedding
                    )

                    values
                    (
                        :project_id,
                        :doc_id,
                        :doc_title,
                        :chunk_text,
                        CAST(:embedding AS vector)
                    )

                """), {

                    "project_id": project_id,
                    "doc_id": document_id,
                    "doc_title": doc.title,
                    "chunk_text": chunk_text,
                    "embedding": vector

                })


        db.commit()

        return {"status": "ok"}

    except Exception as e:

        db.rollback()

        raise HTTPException(status_code=500, detail=str(e))

    finally:

        db.close()
