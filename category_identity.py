"""Category keys retain readable labels while qualifying ownership for selectors."""
def category_key(category, module_id=None, document_id=None, document_name=None):
    name = category or 'General'
    if document_id:
        label = f'{name} · {document_name}' if document_name else name
        return f'{label} ⟦document:{document_id}⟧'
    return name
