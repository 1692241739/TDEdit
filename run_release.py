#!/usr/bin/env python3
"""Run explicit paper settings; do not inherit legacy CLI defaults.

Help, mapping validation and --dry-run require only the Python standard library.
Actual generation requires the documented dependencies, weights and CUDA.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
PRIMARY={
    'text':dict(steps=12,strength=1.0,seed=0,guidance_t=2.1,cross_replace_steps=.8,expanded_subject_fill_px=6,result_branch='reference',annotation_view='source',local_blend_thresh_e=.6,local_blend_thresh_m=.65,drag_self_replace_steps=-1.0,ref_target_denoise_mix_max=.35,ref_target_denoise_mix_start=.6),
    'drag':dict(steps=17,strength=.75,seed=42,guidance_t=1.0,cross_replace_steps=.7,expanded_subject_fill_px=4,result_branch='target',annotation_view='modified',local_blend_thresh_e=.3,local_blend_thresh_m=.3),
    'joint':dict(steps=15,strength=.7,seed=42,guidance_t=1.5,cross_replace_steps=.7,expanded_subject_fill_px=6,result_branch='target',annotation_view='modified',local_blend_thresh_e=.3,local_blend_thresh_m=.3),
}
COMMON=dict(guidance_s=1.,self_replace_steps=.7,drag_cross_replace_steps=0.,drag_self_replace_steps=.7,
            influence_range=.5,pointcloud_domain='image',hole_fill_mode='sgf',attn_switch_mode='hard',
            ref_target_denoise_mix_max=.7,ref_target_denoise_mix_start=.9,start_step=1,start_layer=10)
PATH_OPTIONS={
    'model_path':'TDEDIT_MODEL_PATH', 'sam2_checkpoint':'TDEDIT_SAM2_CHECKPOINT',
    'sam2_config':'TDEDIT_SAM2_CONFIG', 'depth_v2_repo':'TDEDIT_DEPTH_V2_REPO',
    'depth_v2_checkpoint_dir':'TDEDIT_DEPTH_V2_CHECKPOINT_DIR',
}

def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mode',choices=PRIMARY,required=True)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--mapping',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True,help='New/empty directory; existing results are never silently overwritten.')
    p.add_argument('--device',type=int,nargs='+',default=[0],help='Visible CUDA indices, e.g. 0 or 0 1.')
    p.add_argument('--seed',type=int,help='Override primary seed; record separate runs for seeds 0,42,123.')
    p.add_argument('--limit',type=int,help='First N input records for a smoke test; omit for the complete supplied mapping.')
    p.add_argument('--dry-run',action='store_true',help='Validate mapping structure and print the exact command, without importing models or writing results.')
    for name,env in PATH_OPTIONS.items():
        p.add_argument('--'+name.replace('_','-'),type=Path,default=None,help='Model/dependency path; alternatively set '+env)
    args=p.parse_args(argv)
    if args.limit is not None and args.limit<=0:
        p.error('--limit must be positive')
    if any(i<0 for i in args.device) or len(set(args.device))!=len(args.device):
        p.error('--device must contain distinct nonnegative CUDA indices')
    return args

def selected_entry(raw,mode):
    if any(k in raw for k in ('source','modified','user_study')):
        base=dict(raw.get('source') or {})
        if mode!='text': base.update(raw.get('modified') or {})
        # The public runner requires explicit assets/prompts for the selected
        # author view; it never falls back to official masks silently.
        for key in ('image_path','mask_path','source_prompt','original_prompt','target_prompt','editing_prompt'):
            if not base.get(key) and raw.get(key): base[key]=raw[key]
        return base
    return dict(raw)

def load_cases(path,mode,limit=None,check_assets=False,data_root=None):
    data=json.loads(path.read_text())
    if not isinstance(data,dict) or not data:
        raise ValueError('Mapping must be a nonempty object keyed by case ID.')
    chosen=dict(list(data.items())[:limit]) if limit else data
    for key,raw in chosen.items():
        if not isinstance(key,str) or not re.fullmatch(r'[A-Za-z0-9_.-]+',key) or key in {'.','..'}:
            raise ValueError('Unsafe case ID in mapping (only letters, numbers, underscore, dot and dash).')
        if not isinstance(raw,dict): raise ValueError('Case '+key+' must be an object.')
        item=selected_entry(raw,mode)
        if not item.get('source_prompt',item.get('original_prompt','')):
            raise ValueError('Case '+key+' has no source prompt.')
        if mode=='joint' and not item.get('target_prompt',item.get('editing_prompt','')):
            raise ValueError('Joint case '+key+' has no target prompt.')
        if mode!='text':
            points=item.get('points')
            if not isinstance(points,list) or len(points)<2 or len(points)%2:
                raise ValueError('Case '+key+' needs alternating handle/target XY points.')
            if any(not isinstance(pt,(list,tuple)) or len(pt)!=2 or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in pt) for pt in points):
                raise ValueError('Case '+key+' has invalid XY coordinates.')
            if not item.get('mask_path'):
                raise ValueError('Case '+key+' needs the author mask hint; no official/empty-mask fallback is allowed.')
            if item.get('drag_type') not in {a+'-'+b for a in ('2D','3D') for b in ('Rigid','Non-Rigid','Hybrid')}:
                raise ValueError('Case '+key+' requires an explicit saved drag_type.')
        if check_assets:
            image=item.get('image_path',f'images/{key}.png')
            paths=[image]+([item['mask_path']] if mode!='text' else [])
            for value in paths:
                full=Path(value) if Path(value).is_absolute() else data_root/value
                if not full.is_file(): raise FileNotFoundError('Missing case asset: '+str(full))
    return chosen

def build_command(args,mapping):
    values={**COMMON,**PRIMARY[args.mode], 'mode':args.mode,'data_root':str(args.data_root.resolve()),
            'mapping_file':str(mapping),'output_dir':str(args.output.resolve())}
    if args.seed is not None: values['seed']=args.seed
    cmd=[sys.executable,str(ROOT/'run_batch.py')]
    for key,value in values.items(): cmd.extend(['--'+key,str(value)])
    cmd.extend(['--device',*[str(x) for x in args.device]])
    cmd.extend(['--no-joint_target_refine_mix','--no-drag_layout_latents','--no-drag_target_latents',
                '--use_expanded_subject_fill','--use_drag_guided_prefill','--no-use_saved_influence_range'])
    if args.mode=='text':
        cmd.extend(['--no-denoise','--no-ref_target_denoise_mix','--no-ref_kv_injection','--no-drag_target_q_layout_mix','--no-drag_clean_latents','--no-use_saved_drag_type'])
    else:
        cmd.extend(['--denoise','--ref_target_denoise_mix','--ref_kv_injection','--drag_target_q_layout_mix','--drag_clean_latents','--use_saved_drag_type','--skip_no_points'])
    return cmd,values

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main(argv=None):
    args=parse_args(argv)
    cases=load_cases(args.mapping,args.mode,args.limit,not args.dry_run,args.data_root.resolve())
    output=args.output.resolve()
    mapping=output/'selected_mapping.json'
    cmd,values=build_command(args,mapping)
    model_env={name:str(getattr(args,option).resolve()) for option,name in PATH_OPTIONS.items() if getattr(args,option) is not None}
    summary={'mode':args.mode,'cases':len(cases),'case_ids':list(cases),'parameters':values,'command':cmd,'model_environment':model_env,
             'note':'Paper-primary settings. A limited run is a smoke test, not reproduction of full reported results.'}
    if args.dry_run:
        print(json.dumps(summary,indent=2))
        print('\nCommand: '+shlex.join(cmd))
        return 0
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output directory is not empty; choose a new directory to preserve existing results.')
    env=os.environ.copy()
    env.update(model_env)
    output.mkdir(parents=True,exist_ok=True)
    mapping.write_text(json.dumps(cases,indent=2))
    summary.update({'input_mapping_sha256':sha(args.mapping),'runner_sha256':sha(ROOT/'run_batch.py'),
                    'core_sha256':sha(ROOT/'tdedit_core.py'),'started_unix':time.time(),'status':'running'})
    manifest=output/'release_run.json'
    manifest.write_text(json.dumps(summary,indent=2))
    # Legacy diagnostics are relative to cwd. Isolate them per run so they
    # neither dirty the source checkout nor overwrite another run's files.
    result=subprocess.run(cmd,cwd=output,env=env,check=False)
    missing=[key for key in cases if not (output/'results'/(key+'.png')).is_file()]
    summary.update({'returncode':result.returncode,'missing_outputs':missing,'finished_unix':time.time(),
                    'status':'passed' if result.returncode==0 and not missing else 'failed'})
    manifest.write_text(json.dumps(summary,indent=2))
    if result.returncode or missing:
        print('Generation incomplete; see release_run.json. Missing outputs: '+str(len(missing)),file=sys.stderr)
        return 1
    print(f'Completed {len(cases)} outputs. This verifies output coverage, not full benchmark equivalence.')
    return 0

if __name__=='__main__':
    try:
        sys.exit(main())
    except (ValueError,OSError,json.JSONDecodeError) as exc:
        print('Release runner: '+str(exc),file=sys.stderr)
        sys.exit(2)
