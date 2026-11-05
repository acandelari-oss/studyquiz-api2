"""Generate example diagnostic files using synthetic responses only (no API/DB)."""
import argparse
import os
from pathlib import Path
import sys
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ai_cost_test as cost


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('benchmark-output'))
    args = parser.parse_args()
    os.environ.update(DOUNO_AI_COST_TEST='1', DOUNO_AI_COST_TEST_OUTPUT_DIR=str(args.out.resolve()))
    def request(feature, sub, model='gpt-4o-mini', tokens=1000, **tags):
        response = NS(model=model, usage=NS(prompt_tokens=tokens, completion_tokens=200, total_tokens=tokens+200,
            prompt_tokens_details=NS(cached_tokens=0)))
        if model == 'text-embedding-3-small':
            response.usage = NS(prompt_tokens=tokens, total_tokens=tokens)
        return cost.call(lambda **kw:response,cost_feature=feature,cost_suboperation=sub,model=model,**tags)
    os.environ['DOUNO_AI_COST_TEST_ID']='MOCK-quiz-medium-balanced-15'
    @cost.operation('quiz')
    def quiz():
        cost.metrics(requested_count=15,difficulty='medium',style='balanced')
        request('quiz','retrieval','text-embedding-3-small',tokens=20)
        for _ in range(18):request('quiz','questions',cost_phase='initial')
        request('quiz','questions',cost_phase='refill',cost_attempt=1)
        request('quiz','explanation_repair',cost_phase='repair',cost_attempt=1)
        cost.metrics(generated_count=15,saved_count=15)
    quiz()
    os.environ['DOUNO_AI_COST_TEST_ID']='MOCK-ingestion-limited-200'
    @cost.operation('taxonomy')
    def taxonomy():
        cost.metrics(scope_kind='module',teaching_available=650,selected_rows=400,teaching_selected=400,selection_limit=400,topics_generated=40)
        request('taxonomy','consolidation')
    @cost.operation('ingestion')
    def ingest():
        cost.metrics(document_index=0,document_format='pdf',source_pages=200,extracted_pages=200,chunks_generated=700)
        request('ingestion','embedding','text-embedding-3-small',tokens=10000)
        return cost.background(taxonomy)
    ingest()()
    os.environ['DOUNO_AI_COST_TEST_ID']='MOCK-oral-with-unknown-audio'
    @cost.operation('oral_practice')
    def oral():
        request('oral_practice','question');request('oral_practice','evaluation')
        cost.call(lambda **kw:NS(content=b''),cost_feature='oral_practice',cost_suboperation='tts',model='gpt-4o-mini-tts')
    oral()
    print('Synthetic examples only. No provider or database requests made.')


if __name__ == '__main__': main()
