"""Validate actual v4.3 smoke artifacts before a lane starts its full build."""
import argparse
import json
from pathlib import Path
import numpy as np
from action_head import protocol


def verify(root, version="v4.3"):
    P = protocol.protocol_module(version)
    provenance=json.loads((root/'provenance.json').read_text())
    P.assert_protocol_identity(provenance)
    rows=[json.loads(s) for s in (root/'manifest.jsonl').read_text().splitlines() if s.strip()]
    if not rows: raise ValueError('smoke generated no trainable samples')
    for row in rows:
        P.assert_protocol_identity(row)
        if row['slow_messages'][0]['content'] != [{'type':'text','text':P.SYSTEM_PROMPT}]:
            raise ValueError('incorrect system prompt')
        if version == 'v6':
            from sft.golden_context_beam import validate_context
            validate_context(row['slow_messages'], row['trajectory_map_path'])
            if row.get('decision_filter') != 'none' or not row.get('golden_anchor_id'):
                raise ValueError('v6 must be a no-direction-filter context beam')
            if not Path(row['trajectory_map_path']).is_file():
                raise FileNotFoundError(row['trajectory_map_path'])
        if row.get('conditioning_provenance'): raise ValueError('old teacher reconditioning is not recollection')
        if row.get('teacher_turns'): raise ValueError('turn must not affect golden supervision')
        for key in ('sample','reference_image','current_image'):
            if not Path(row[key]).is_file(): raise FileNotFoundError(row[key])
        with np.load(row['sample']) as data:
            for key in ('query_hidden','reference_visual_tokens','current_visual_tokens','target_actions'):
                if not np.isfinite(data[key]).all(): raise ValueError(f'nonfinite {key}')
            if data['target_actions'].shape != (16,3) or data['action_mask'].shape != (16,):
                raise ValueError('invalid action shape')
            if data['query_hidden'].shape != (4,2048): raise ValueError('invalid teacher query shape')
        if int(row.get('option_sequence_index') or 0)>0:
            if not any('direction_alignment' in part.get('text','')
                       for m in row['slow_messages'] if m['role']=='user'
                       for part in m['content']): raise ValueError('missing causal prior feedback')
    report={'ok':True,'samples':len(rows),'anchors':len({r['slow_anchor_id'] for r in rows}),
            'protocol':P.protocol_identity()}
    (root/'smoke_validated.json').write_text(json.dumps(report,indent=2)+'\n')
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--action-head-as-tool-version', choices=('v4.3', 'v6'), default='v4.3')
    args = parser.parse_args()
    print(json.dumps(verify(args.bundle, args.action_head_as_tool_version)))
