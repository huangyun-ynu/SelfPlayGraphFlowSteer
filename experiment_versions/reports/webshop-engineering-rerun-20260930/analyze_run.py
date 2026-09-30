"""Audit the fresh affected-task evaluation, without additional model calls."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
RUN = Path((Path(__file__).resolve().parent/'run-path.txt').read_text().strip())
from audit_common import artifacts


def main():
    manifest = json.loads((RUN/'manifest.json').read_text())
    records = [json.loads(line) for line in (RUN/'results/records.jsonl').open()]
    assert len(records) == len({r['task_id'] for r in records}) == manifest["planned"]
    assert {r['task_id'] for r in records} == set(manifest['ids'])
    assert manifest['scheduling_policy'] == 'bounded_research_v1'
    assert manifest['purchase_budget_policy'] == 'completion_reserve_v2'
    assert manifest['route_concurrency'] == 50 and manifest['worker_thinking'] is False
    assert manifest['tool_budget'] == 16 and manifest['worker_token_limit'] == 350000
    assert all(hashlib.sha256((RUN/'evaluated_source'/p).read_bytes()).hexdigest()==h
               for p,h in manifest['source_sha256'].items())
    backend, rows = {}, []
    for record in records:
        trajectory = record['trajectory']
        metadata = trajectory['task']['metadata']
        env = metadata.get('webshop_environment_result') or {}
        receipt = trajectory.get('submission_receipt') or {}
        arts = artifacts(record)
        task_events = {}
        for event in [e for a in arts for e in a.get('backend_request_events', [])] + metadata.get('backend_request_events', []):
            if event.get('request_role')=='worker':
                backend[event['event_id']] = task_events[event['event_id']] = event
        tokens = sum(sum(e.get('completion_usage',{}).get(k,0) or 0 for k in ('token_in','token_out')) for e in task_events.values())
        assert tokens == record['token_cost']
        used = 0
        for art in arts:
            # The ledger charges execution exceptions conservatively; ordinary
            # preflight rejections do not consume an action.
            used += sum(t['observation'].get('status')=='ok' and t['action']['name'] in {'webshop_search','webshop_click'} for t in art['react_trace'])
        ledger = max(a.get('webshop_progress',{}).get('action_budget',{}).get('total_used',0) for a in arts)
        assert 0 <= used <= ledger <= 16
        purchased = bool(env.get('purchased'))
        if record['passed']:
            assert purchased and record['score']==1
        if record['submission_status']=='submitted':
            assert receipt['version']=='unified_submission_v1'
            assert receipt['payload']['purchased']==purchased
        deferrals = {}
        for art in arts:
            deferrals[art['agent_id']] = max(deferrals.get(art['agent_id'],0),
                art.get('webshop_progress',{}).get('completion_path_fuse_deferrals',0))
        assert all(value<=1 for value in deferrals.values())
        reserve_events = dict(enumerate(max((a.get('webshop_progress', {}).get('purchase_reservation_events', []) for a in arts), key=len)))
        reserves = [a.get('webshop_progress', {}).get('purchase_reservation') for a in arts]
        reserves = [r for r in reserves if r]
        assert reserves and all(r['policy']=='completion_reserve_v2' for r in reserves)
        previous_remaining = 16
        for a in arts:
            for t in a['react_trace']:
                remaining = t['remaining_budget']['total']
                assert remaining <= previous_remaining
                if t.get('observation', {}).get('status') == 'error':
                    if t['observation'].get('error', {}).get('code') in {'purchase_budget_reserved', 'completion_plan_required', 'invalid_completion_plan', 'purchase_plan_abandoned'}:
                        assert previous_remaining == remaining, t
                previous_remaining = remaining
        research_used = max(a.get('webshop_progress',{}).get('scheduling',{}).get('research_used',0) for a in arts)
        assert 0 <= research_used <= 8
        runtime_statuses = [a for a in arts if a.get('webshop_progress',{}).get('runtime_only')]
        for a in runtime_statuses:
            assert not a.get('react_trace') and not a.get('backend_request_events')
            assert not a.get('token_in') and not a.get('token_out')
            assert a['model'] == 'runtime-webshop-scheduler'
        canvas_events = [e['payload'] for e in trajectory['events'] if e['kind']=='canvas_step']
        zero_action_worker_executions = sum(not a.get('react_trace') and not a.get('webshop_progress',{}).get('runtime_only') for a in arts)
        memory_reads = [t for a in arts for t in a.get('webshop_progress', {}).get('memory_read_trace', [])]
        successful_reads = [t for t in memory_reads if not t['result'].get('error')]
        assert len(successful_reads) <= 4
        if manifest.get('memory_delivery') == 'automatic':
            assert not memory_reads
            assert all(t['action']['name'] in {'webshop_search','webshop_click'} for a in arts for t in a['react_trace'])
        assert all(t['environment_actions_charged'] == 0 for t in memory_reads)
        projections = [a['webshop_progress']['memory_projection'] for a in arts if 'memory_projection' in a.get('webshop_progress', {})]
        assert all(len(json.dumps(p, ensure_ascii=False)) <= 6000 for p in projections)
        memory_handles = [a['webshop_progress']['memory'] for a in arts if 'memory' in a.get('webshop_progress', {})]
        assert memory_handles
        rows.append(dict(task_id=record['task_id'],research_used=research_used,
            memory_reads=len(memory_reads), successful_memory_reads=len(successful_reads),
            memory_read_errors=[t['result']['error'] for t in memory_reads if 'error' in t['result']],
            max_endpoint_memory_chars=max([len(json.dumps(p,ensure_ascii=False)) for p in projections] or [0]),
            memory_read_exhausted=any(a.get('webshop_progress', {}).get('memory_reads_used',0)>=4 for a in arts),
            runtime_only_statuses=len(runtime_statuses),zero_action_worker_executions=zero_action_worker_executions,score=record['score'],passed=record['passed'],
            purchased=purchased,submission_status=record['submission_status'],outcome=record['outcome_status'],
            action_budget_used=ledger,successful_actions=used,token_cost=tokens,
            worker_requests=len(task_events),agents=sorted(deferrals),completion_deferrals=deferrals,
            stops=sorted({a.get('webshop_progress',{}).get('stop_reason','') for a in arts}),
            policy_failure=metadata.get('runtime_terminal_policy_failure'),
            reservation_events=list(reserve_events.values()),
            reservations_created=sum(e['event']=='reserved' for e in reserve_events.values()),
            reservation_rejections=dict(Counter(e.get('code') for e in reserve_events.values() if e['event']=='rejected')),
            final_reservation=reserves[-1],
            final_director_feedback=trajectory.get('director_run',{}).get('turns',[{}])[-1].get('feedback')))
    rows.sort(key=lambda r:r['task_id'])
    summary=dict(run=str(RUN),tasks=manifest["planned"],
        memory_reads=sum(r['memory_reads'] for r in rows),
        successful_memory_reads=sum(r['successful_memory_reads'] for r in rows),
        memory_read_exhausted_tasks=sum(r['memory_read_exhausted'] for r in rows),
        max_endpoint_memory_chars=max(r['max_endpoint_memory_chars'] for r in rows),full=sum(r['passed'] is True for r in rows),
        purchased=sum(r['purchased'] for r in rows),not_purchased=sum(not r['purchased'] for r in rows),
        submitted=sum(r['submission_status']=='submitted' for r in rows),
        unsubmitted=sum(r['submission_status']!='submitted' for r in rows),
        unknown=sum(r['score'] is None for r in rows),
        mean_reward_lower_bound=sum(r['score'] or 0 for r in rows)/len(rows),
        runtime_only_statuses=sum(r['runtime_only_statuses'] for r in rows),
        zero_action_worker_executions=sum(r['zero_action_worker_executions'] for r in rows),
        max_research_used=max(r['research_used'] for r in rows),
        worker_tokens=sum(r['token_cost'] for r in rows),worker_requests=len(backend),
        worker_request_statuses=dict(Counter(e.get('event') for e in backend.values())),
        execution_errors=len(list((RUN/'results/errors').glob('*.json'))),
        tasks_with_completion_deferral=[r['task_id'] for r in rows if any(r['completion_deferrals'].values())],
        tasks_with_reservation=[r['task_id'] for r in rows if r['reservations_created']],
        reservations_created=sum(r['reservations_created'] for r in rows),
        reservation_rejections=dict(sum((Counter(r['reservation_rejections']) for r in rows), Counter())),
        source_integrity=json.loads((RUN/'source_integrity.json').read_text()),
        cleanup=json.loads((RUN/'cleanup.json').read_text()))
    for name, value in [('summary.json',summary),('per-task.json',rows)]:
        text=json.dumps(value,ensure_ascii=False,indent=2)+'\n'
        (RUN/name).write_text(text)
        (Path(__file__).resolve().parent/name).write_text(text)
    store_manifest=json.loads((RUN/'memory-store-manifest.json').read_text())
    assert all(hashlib.sha256((RUN/p).read_bytes()).hexdigest()==h for p,h in store_manifest.items())
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
