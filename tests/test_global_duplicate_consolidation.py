"""Deterministic coverage/decision tests. No live AI or embedding requests."""
import copy
import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

from taxonomy_coverage import (consolidate_candidates, nominate_relationships,
                               normalized_title, select_review_evidence)
from taxonomy_assignment import validate_evidence


def candidate(title, description=None, category="AUTO", vector=(1., 0.), i=0):
    return dict(topic=title, description=description or title, category=category, importance=5,
                embedding=vector, assignment_id=f"id-{i}", evidence_ids={i + 1})


def response(data):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(data)))])


class GlobalDuplicateTests(unittest.TestCase):
    def run_review(self, candidates, merge_ids=(), mutate=None, protected=False, duplicate_pairs=None):
        self.prompts, self.diag = [], {}
        def create(prompt):
            self.prompts.append(prompt)
            payload = json.loads(prompt.rsplit("REVIEW INPUT:\n", 1)[1])
            inputs = {c["input_id"]: c for c in payload["candidates"]}
            records = []
            for pair in payload["relationships"]:
                duplicate = (set(pair) <= set(merge_ids) if duplicate_pairs is None
                             else tuple(pair) in duplicate_pairs)
                records.append({"input_ids": pair, "decision": "duplicate" if duplicate else "distinct",
                    "reason": "Explicit mocked semantic decision", "category": inputs[pair[0]]["category"],
                    "evidence_checks": {i: [inputs[i]["validated_evidence_samples"][0]["chunk_id"]] for i in pair}})
            if mutate:
                mutate(records)
            return response({"decisions": records})
        self.logs = io.StringIO()
        with redirect_stdout(self.logs):
            result = consolidate_candidates(candidates, create, "English", protected_categories=protected,
                source_text={cid: c["description"] for c in candidates for cid in c["evidence_ids"]},
                diagnostics=self.diag, log_context={"project_id": "test"})
        validate_evidence(result, candidates, set().union(*(c["evidence_ids"] for c in candidates)))
        return [t for cat in result["categories"] for t in cat["topics"]], result

    def disjoint_pair_candidates(self, description=None):
        return [candidate(f"Subject {i // 2}", description=description,
                          vector=tuple(float(j == i // 2) for j in range(12)), i=i)
                for i in range(24)]

    def test_disjoint_pairs_pack_into_one_call_without_changing_results(self):
        from unittest.mock import patch
        cs = self.disjoint_pair_candidates()
        original = copy.deepcopy(cs)
        pairs = {(str(i), str(i + 1)) for i in range(0, 24, 2)}
        with patch('taxonomy_coverage.REVIEW_CANDIDATES_PER_REQUEST', 12):
            old_topics, old_result = self.run_review(cs, duplicate_pairs=pairs)
            old_calls = len(self.prompts)
        topics, result = self.run_review(cs, duplicate_pairs=pairs)
        self.assertEqual(old_calls, 2)
        self.assertEqual(len(self.prompts), 1)
        payload = json.loads(self.prompts[0].rsplit("REVIEW INPUT:\n", 1)[1])
        self.assertEqual(len(payload['candidates']), 24)
        self.assertEqual(len(payload['relationships']), 12)
        self.assertEqual({tuple(p) for p in payload['relationships']}, pairs)
        self.assertEqual(result, old_result)
        self.assertEqual(cs, original)
        self.assertEqual(len(topics), 12)
        self.assertLessEqual(len(self.prompts[0]), 48000)

    def test_larger_candidate_capacity_still_respects_character_budget(self):
        cs = self.disjoint_pair_candidates(description='Detailed objective ' * 200)
        topics, _ = self.run_review(cs)
        self.assertEqual(len(topics), 24)
        self.assertGreater(len(self.prompts), 1)
        reviewed = []
        for prompt in self.prompts:
            self.assertLessEqual(len(prompt), 48000)
            payload = json.loads(prompt.rsplit("REVIEW INPUT:\n", 1)[1])
            self.assertLessEqual(len(payload['candidates']), 24)
            self.assertLessEqual(len(payload['relationships']), 12)
            reviewed.extend(tuple(pair) for pair in payload['relationships'])
        self.assertEqual(len(reviewed), 12)
        self.assertEqual(len(set(reviewed)), 12)

    def test_twelve_equivalent_candidates_merge_across_overlapping_reviews(self):
        cs = [candidate("Bone-Lining Cells", "Same complete objective", i=i) for i in range(12)]
        topics, _ = self.run_review(cs, {str(i) for i in range(12)})
        self.assertEqual(len(topics), 1)
        self.assertEqual(topics[0]["candidate_ids"], list(map(str, range(12))))
        self.assertEqual(topics[0]["evidence_ids"], list(range(1, 13)))
        reviewed = [tuple(pair) for p in self.prompts for pair in json.loads(p.rsplit("REVIEW INPUT:\n",1)[1])["relationships"]]
        self.assertEqual(len(reviewed), 66)
        self.assertEqual(len(set(reviewed)), 66)
        self.assertGreater(len(self.prompts), 1)
        self.assertLessEqual(len(self.prompts), 12)
        self.assertTrue(all(len(p) <= 48000 for p in self.prompts))
        # Reuse original wording/vector: no repeated description concatenation.
        self.assertEqual(topics[0]["description"], "Same complete objective")
        self.assertEqual(topics[0]["embedding"], cs[0]["embedding"])

    def test_normalized_title_nomination_below_threshold(self):
        cs = [candidate("Bone-Lining Cells"), candidate("  bone lining   cells ", vector=(0.,1.), i=1)]
        topics, _ = self.run_review(cs, {"0","1"})
        self.assertEqual(len(topics), 1)
        self.assertEqual(self.diag["relationships"][0]["reason"], "normalized_title")
        self.assertEqual(normalized_title("BONE—LINING Cells"), "bone lining cells")
        self.assertEqual(normalized_title("Bone_lining cells"), "bone lining cells")

    def test_exact_title_nomination_below_threshold(self):
        cs = [candidate("Same title"), candidate("Same title", vector=(0.,1.), i=1)]
        topics, _ = self.run_review(cs, {"0","1"})
        self.assertEqual(len(topics), 1)
        self.assertEqual(self.diag["relationships"][0]["reason"], "exact_title")

    def test_dense_cluster_over_budget_is_explicitly_accounted(self):
        cs = [candidate("Same", i=i) for i in range(20)]
        topics, _ = self.run_review(cs)
        summary = self.diag["summary"]
        self.assertEqual(summary["nominated"], 190)
        self.assertEqual(summary["relationships_selected"], 160)
        self.assertEqual(summary["deferred"], {"exact_title": 30})
        self.assertEqual(sum(summary["outcomes"].values()), 160)
        self.assertEqual(len(topics), 20)
        self.assertLessEqual(summary["ai_calls"], 3 * len(cs))
        self.assertTrue(all(e["status"] != "pending" for e in self.diag["relationships"]))

    def test_global_different_titles_and_categories_across_multiple_batches(self):
        cs = [candidate(f"Independent {i}", vector=(0.,1.), i=i) for i in range(101)]
        cs[0] = candidate("Price responsiveness", "Demand response to price", "ECONOMICS")
        cs[100] = candidate("Price elasticity", "Demand response to price", "MARKETS", i=100)
        topics, _ = self.run_review(cs, {"0","100"})
        merged = next(t for t in topics if len(t["candidate_ids"]) == 2)
        self.assertEqual(merged["candidate_ids"], ["0","100"])
        self.assertEqual(merged["evidence_ids"], [1,101])
        self.assertEqual(len(topics), 100)

    def test_identical_titles_with_different_objectives_stay_separate(self):
        cs = [candidate("Determinant", "Define determinant"), candidate("Determinant", "Prove a determinant theorem", i=1)]
        topics, _ = self.run_review(cs)
        self.assertEqual(len(topics), 2)
        self.assertEqual(len(self.prompts), 1)

    def test_distinct_and_complementary_objectives_stay_separate(self):
        for decision in ("distinct", "complementary", "uncertain"):
            with self.subTest(decision=decision):
                topics, _ = self.run_review([candidate("Theory"), candidate("Application",i=1)],
                    mutate=lambda rs: rs[0].update(decision=decision))
                self.assertEqual(len(topics), 2)
                self.assertEqual(len(self.prompts), 1)

    def test_no_transitive_merging_of_unconfirmed_pair(self):
        cs = [candidate(str(i), i=i) for i in range(3)]
        topics, _ = self.run_review(cs, duplicate_pairs={("0","1"),("1","2")})
        self.assertEqual([t["candidate_ids"] for t in topics], [["0","1"],["2"]])
        self.assertEqual(self.diag["summary"]["confirmed_not_coalesced"], 1)

    def test_manual_boundaries_protect_title_and_semantic_matches(self):
        topics, _ = self.run_review([candidate("Same",category="A"),candidate("Same",category="B",i=1)],
                                    {"0","1"}, protected=True)
        self.assertEqual(len(topics),2)
        self.assertFalse(self.prompts)
        self.assertEqual(self.diag["summary"]["protected"],1)

    def test_category_decision_changes_only_merged_topics(self):
        cs = [candidate("Bone repair","Fracture repair","CARDIOVASCULAR"),
              candidate("Fracture healing","Fracture repair","BONE REPAIR",i=1),
              candidate("Valve mechanics",category="CARDIOVASCULAR",vector=(0.,1.),i=2)]
        topics, result = self.run_review(cs,{"0","1"},lambda rs: rs[0].update(category="BONE REPAIR"))
        self.assertEqual(len(topics),2)
        self.assertEqual([c["name"] for c in result["categories"]],["BONE REPAIR","CARDIOVASCULAR"])

    def test_unsupported_candidate_does_not_block_valid_pair(self):
        cs = [candidate("Alias",i=i) for i in range(3)]
        cs[2]["evidence_ids"] = set()
        topics, _ = self.run_review(cs,{"0","1","2"})
        self.assertEqual([t["candidate_ids"] for t in topics],[["0","1"],["2"]])
        self.assertEqual(self.diag["summary"]["outcomes"]["no_validated_excerpt"],2)

    def test_invalid_decision_does_not_discard_valid_pair_or_repeat_it(self):
        cs = [candidate(str(i),i=i) for i in range(4)]
        def corrupt(records):
            for r in records:
                if r["input_ids"] == ["2","3"]:
                    r["evidence_checks"] = {"2":[999],"3":[4]}
        topics, _ = self.run_review(cs, duplicate_pairs={("0","1"),("2","3")},mutate=corrupt)
        self.assertEqual([t["candidate_ids"] for t in topics],[["0","1"],["2"],["3"]])
        requests=[json.loads(p.rsplit("REVIEW INPUT:\n",1)[1])["relationships"] for p in self.prompts]
        self.assertEqual(sum(["0","1"] in req for req in requests),1)
        self.assertEqual(sum(["2","3"] in req for req in requests),3)

    def test_conflicting_responses_are_not_used(self):
        def conflict(records):
            records.append({**records[0],"decision":"distinct"})
        topics, _ = self.run_review([candidate("A"),candidate("A",i=1)],{"0","1"},conflict)
        self.assertEqual(len(topics),2)
        self.assertEqual(self.diag["summary"]["outcomes"],{"ambiguous_decisions":1})

    def test_evidence_object_requires_exact_pair_keys(self):
        for evidence in (None, [], {}, {"a": [1], "b": [2]}, {"0": [1]},
                         {"0": [1], "1": [2], "2": [3]}):
            with self.subTest(evidence=evidence):
                topics, _ = self.run_review([candidate("Same"), candidate("Same", i=1)], {"0", "1"},
                    lambda rs: rs[0].update(evidence_checks=evidence))
                self.assertEqual(len(topics), 2)
                self.assertEqual(self.diag["summary"]["outcomes"], {"missing_evidence_review": 1})
        topics, _ = self.run_review([candidate("Same"), candidate("Same", i=1)], {"0", "1"},
                                   lambda rs: rs[0].pop("evidence_checks"))
        self.assertEqual(len(topics), 2)
        self.assertEqual(self.diag["summary"]["outcomes"], {"missing_evidence_review": 1})

    def test_duplicate_requires_nonempty_integer_citations_from_each_input(self):
        for cited in ([], None, "1", ["1"], [True], [1.0], [2], [999]):
            with self.subTest(cited=cited):
                topics, _ = self.run_review([candidate("Same"), candidate("Same", i=1)], {"0", "1"},
                    lambda rs: rs[0].update(evidence_checks={"0": cited, "1": [2]}))
                self.assertEqual(len(topics), 2)
                self.assertEqual(self.diag["summary"]["outcomes"], {"unsupported_evidence": 1})

    def test_distinct_response_needs_no_duplicate_only_fields(self):
        def distinct_only(records):
            for record in records:
                record.pop("category")
                record.pop("evidence_checks")
        topics, _ = self.run_review([candidate("Same"), candidate("Same", i=1)], mutate=distinct_only)
        self.assertEqual(len(topics), 2)
        self.assertEqual(self.diag["summary"]["outcomes"], {"distinct": 1})
        self.assertEqual(len(self.prompts), 1)

    def test_prompt_json_example_satisfies_real_validator_and_preserves_unsampled_evidence(self):
        from taxonomy_coverage import REVIEW_INSTRUCTIONS
        example = json.loads(REVIEW_INSTRUCTIONS.split("not these example values:\n", 1)[1].split("\nDo not decide", 1)[0])
        cs = [candidate("Same objective", category="Example category", i=i) for i in range(2)]
        cs[0]["evidence_ids"] = {101, 102, 103}
        cs[1]["evidence_ids"] = {202, 203, 204}
        topics, result = self.run_review(cs, {"0", "1"}, lambda rs: rs.__setitem__(slice(None), example["decisions"]))
        self.assertEqual(len(topics), 1)
        self.assertEqual(topics[0]["evidence_ids"], [101, 102, 103, 202, 203, 204])
        self.assertEqual(self.diag["summary"]["outcomes"], {"duplicate": 1})
        prompt = self.prompts[0]
        for rule in ('EXACTLY the two input_id strings', 'nonempty array', 'integer chunk_id',
                     "THAT input's validated_evidence_samples", 'choose "uncertain", not',
                     'Non-duplicate decisions do not require category or evidence_checks'):
            self.assertIn(rule, prompt)
        payload = json.loads(prompt.rsplit("REVIEW INPUT:\n", 1)[1])
        self.assertEqual(sum(len(c["validated_evidence_samples"]) for c in payload["candidates"]), 4)
        # A validated association omitted from the sampled prompt is preserved in the
        # union, but must not be accepted as a reviewed citation.
        topics, _ = self.run_review(cs, {"0", "1"},
            lambda rs: rs[0].update(evidence_checks={"0": [103], "1": [202]}))
        self.assertEqual(len(topics), 2)
        self.assertEqual(self.diag["summary"]["outcomes"], {"unsupported_evidence": 1})

    def test_malformed_field_is_isolated_from_valid_pair(self):
        for field in ("decision", "category"):
            with self.subTest(field=field):
                def corrupt(records):
                    for r in records:
                        if r["input_ids"] == ["2","3"]:
                            r[field] = []
                topics, _ = self.run_review([candidate(str(i),i=i) for i in range(4)],
                    duplicate_pairs={("0","1"),("2","3")}, mutate=corrupt)
                self.assertEqual([t["candidate_ids"] for t in topics],[["0","1"],["2"],["3"]])

    def test_missing_and_unknown_pair_preserve_inputs(self):
        topics, _ = self.run_review([candidate("A"),candidate("A",i=1)],{"0","1"},
                                    lambda rs: rs[0].update(input_ids=["0","999"]))
        self.assertEqual(len(topics),2)
        self.assertEqual(len(self.prompts),3)

    def test_oversized_candidate_does_not_block_independent_pair(self):
        cs = [candidate("Huge","x"*49000),candidate("A",i=1),candidate("A",i=2)]
        topics, _ = self.run_review(cs,{"0","1","2"})
        self.assertEqual([t["candidate_ids"] for t in topics],[["0"],["1","2"]])
        self.assertEqual(self.diag["summary"]["outcomes"]["context_budget"],2)

    def test_evidence_windows_reach_relevant_text_beyond_first_800_chars(self):
        c=candidate("Bone lining cells","Mineral exchange by bone lining cells")
        c["evidence_ids"]={1,2,3}
        excerpts=select_review_evidence(c,{1:"Unrelated introduction "*100,
            2:"Other material "*100+"Bone lining cells regulate mineral exchange. "*10,
            3:"Bone lining cells support mineral exchange."})
        self.assertEqual({x["chunk_id"] for x in excerpts},{2,3})
        self.assertGreater(next(x["offset"] for x in excerpts if x["chunk_id"]==2),800)
        self.assertTrue(all(len(x["excerpt"])<=800 for x in excerpts))

    def test_tampered_evidence_and_objective_accounting_still_fail(self):
        cs=[candidate("A"),candidate("A",i=1)]
        _, data=self.run_review(cs,{"0","1"})
        for field,value in [("evidence_ids",[1]),("evidence_ids",[1,2,999]),("original_objectives",["A"])]:
            broken=copy.deepcopy(data);broken["categories"][0]["topics"][0][field]=value
            with self.assertRaises(ValueError):validate_evidence(broken,cs,{1,2})

    def test_diagnostics_do_not_log_passages_or_descriptions(self):
        self.run_review([candidate("A","PRIVATE_SOURCE_CONTENT"),candidate("A",i=1)],{"0","1"})
        self.assertNotIn("PRIVATE_SOURCE_CONTENT",self.logs.getvalue())
        self.assertIn('"event": "summary"',self.logs.getvalue())
        self.assertIn('"project_id": "test"',self.logs.getvalue())

    def test_realistic_surviving_duplicate_families(self):
        rows=json.loads((Path(__file__).parent/'fixtures/surviving_duplicate_examples.json').read_text())
        families=[lambda r: normalized_title(r['topic'])=='bone lining cells',
                  lambda r:r['topic']=='Osteocytes and Their Functions',
                  lambda r:r['topic']=='Osteoclasts and Bone Resorption']
        for family in families:
            selected=[r for r in rows if family(r)]
            with self.subTest(family=selected[0]['topic']):
                cs=[candidate(r['topic'],r['description'],r['category'],i=i) for i,r in enumerate(selected)]
                topics,_=self.run_review(cs,{str(i) for i in range(len(cs))})
                self.assertEqual(len(topics),1)
                self.assertEqual(topics[0]['evidence_ids'],list(range(1,len(cs)+1)))
                # Mocks assert equivalence; these are not live semantic evaluations.
                topics,_=self.run_review(cs)
                self.assertEqual(len(topics),len(cs))

    def test_old_export_complementary_osteoclast_objectives_are_not_forced(self):
        rows=json.loads((Path(__file__).parent/'fixtures/duplicate_taxonomy_examples.json').read_text())
        selected=[r for r in rows if r['topic'] in {'Osteoclast Formation and Activation',
                 'Osteoclasts: Characteristics and Activation','Osteoclasts and Bone Resorption','Osteoclasts: Function and Origin'}]
        cs=[candidate(r['topic'],r['description'],r['category'],i=i) for i,r in enumerate(selected)]
        topics,_=self.run_review(cs,mutate=lambda rs:[r.update(decision='complementary') for r in rs])
        self.assertEqual(len(topics),4)


class ConsolidationPackingPerformanceTests(unittest.TestCase):
    """Compare exact review/merge results with the previous scheduler offline."""
    run_review = GlobalDuplicateTests.run_review
    @staticmethod
    def old_pack(ledger, prompt_for):
        batch, ids = [], set()
        for entry in ledger:
            if entry['status'] != 'pending':
                continue
            union = ids | set(entry['pair'])
            if len(union) > 24 or len(prompt_for(batch + [entry])) > 48000:
                continue
            batch.append(entry)
            ids = union
            if len(batch) == 12:
                break
        return batch

    def test_dense_families_reduce_calls_and_repeated_context_with_identical_results(self):
        from unittest.mock import patch
        # Twelve repeated learning objectives in each of ten separate areas.
        cs = [candidate('Objective ' + str(i // 12),
                        'Complete evidence-supported learning objective ' + str(i // 12),
                        vector=tuple(float(j == i // 12) for j in range(10)), i=i)
              for i in range(120)]
        pairs = {(str(i), str(j)) for i in range(120) for j in range(i + 1, 120)
                 if i // 12 == j // 12}
        before = copy.deepcopy(cs)
        with patch('taxonomy_coverage.pack_review_batch', self.old_pack):
            _, old_result = self.run_review(cs, duplicate_pairs=pairs)
            old_prompts = self.prompts[:]
        _, result = self.run_review(cs, duplicate_pairs=pairs)
        self.assertEqual(result, old_result)
        self.assertEqual(cs, before)
        def reviewed(prompts):
            return [tuple(pair) for p in prompts
                    for pair in json.loads(p.rsplit('REVIEW INPUT:\n', 1)[1])['relationships']]
        self.assertEqual(set(reviewed(self.prompts)), pairs)
        self.assertEqual(len(reviewed(self.prompts)), len(pairs))
        self.assertLess(len(self.prompts), len(old_prompts))
        self.assertLess(sum(map(len, self.prompts)), sum(map(len, old_prompts)))
        self.assertEqual(self.diag['summary']['request_characters'], sum(map(len, self.prompts)))
        for prompt in self.prompts:
            data = json.loads(prompt.rsplit('REVIEW INPUT:\n', 1)[1])
            self.assertLessEqual(len(data['relationships']), 48)
            self.assertLessEqual(len(data['candidates']), 24)
            self.assertLessEqual(len(prompt), 48000)
        print('OFFLINE PACKING BENCHMARK', json.dumps({
            'candidates': len(cs), 'pairs': len(pairs),
            'old_calls': len(old_prompts), 'new_calls': len(self.prompts),
            'old_prompt_characters': sum(map(len, old_prompts)),
            'new_prompt_characters': sum(map(len, self.prompts))}))

    def test_large_batch_retries_only_invalid_pair(self):
        cs = [candidate('Same objective', i=i) for i in range(10)]
        initial = True
        def mutate(records):
            nonlocal initial
            if initial:
                self.assertEqual(len(records), 45)
                records[0].pop('evidence_checks')
                initial = False
        topics, _ = self.run_review(cs, {str(i) for i in range(10)}, mutate=mutate)
        self.assertEqual(len(topics), 1)
        self.assertEqual(len(self.prompts), 2)
        retry = json.loads(self.prompts[1].rsplit('REVIEW INPUT:\n', 1)[1])
        self.assertEqual(len(retry['relationships']), 1)
        self.assertEqual(self.diag['summary']['validation_failures'], 1)

    def test_usage_and_pending_progress_are_reported(self):
        cs = [candidate('Same objective', i=i) for i in range(2)]
        def create(prompt):
            result = response({'decisions': [{'input_ids': ['0', '1'],
                'decision': 'distinct', 'reason': 'Different learning objectives'}]})
            result.usage = SimpleNamespace(prompt_tokens=123, completion_tokens=45)
            return result
        diagnostics, logs = {}, io.StringIO()
        with redirect_stdout(logs):
            consolidate_candidates(cs, create, 'English', source_text={1: 'Evidence', 2: 'Evidence'},
                                   diagnostics=diagnostics)
        summary = diagnostics['summary']
        self.assertEqual((summary['input_tokens'], summary['output_tokens'], summary['calls_with_usage']), (123, 45, 1))
        self.assertGreaterEqual(summary['request_seconds'], 0)
        self.assertIn('"remaining_pairs": 0', logs.getvalue())
