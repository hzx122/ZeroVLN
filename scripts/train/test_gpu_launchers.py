"""Regression tests for the actual GPU command path, without loading GPU models."""
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
import gpu_runtime
import run_rloo
from rloo_variants import PRESETS, RLOOSettings
from dataclasses import asdict
from hydra import compose, initialize_config_dir
from hydra.errors import ConfigCompositionException
from omegaconf import OmegaConf


def config(*overrides):
    with initialize_config_dir(version_base=None, config_dir=str(gpu_runtime.REPO/"configs/gpu")):
        return compose(config_name="rloo", overrides=list(overrides))


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    values = {'SLOW_MODEL': tmp_path/'brain', 'FAST_BACKBONE': tmp_path/'backbone',
              'INITIAL_CHECKPOINT': tmp_path/'checkpoint', 'TRAIN_CASES': tmp_path/'cases.jsonl',
              'VERL_SOURCE_ROOT': tmp_path/'verl', 'MEGATRON_ENV': tmp_path/'megatron',
              'SAGE_SOURCE_OBJECT_MAP': tmp_path/'objects.json'}
    for name,path in values.items():
        if path.suffix: path.write_text('{}')
        else: path.mkdir()
        monkeypatch.setenv(name,str(path))
    for path in ['checkpoint/action_head','checkpoint/dualvln_query','megatron/mcore','megatron/bridge/src']:
        (tmp_path/path).mkdir(parents=True,exist_ok=True)
    monkeypatch.setenv('RL_ARTIFACT_ROOT',str(tmp_path))
    monkeypatch.setenv('RAY_ADDRESS','127.0.0.1:6379')
    monkeypatch.setenv('BENCH_URLS',','.join(f'http://node{i//64}:{{}}'.format(8950+i%64) for i in range(512)))
    monkeypatch.setenv('GPU_CACHE_ROOT',str(tmp_path/'cache'))
    import v6_rl
    monkeypatch.setattr(v6_rl,'ARTIFACTS',tmp_path)
    return tmp_path


@pytest.mark.parametrize('variant',PRESETS)
def test_actual_trainer_command_and_rewards(inputs,variant):
    settings=PRESETS[variant]
    defaults=asdict(RLOOSettings())
    overrides=[f'rloo.{key}={value}' for key,value in asdict(settings).items() if value != defaults[key]]
    cfg=config('run.mode=prepare',f'run.output_dir={inputs}/run',*overrides)
    root,plan,command,env=run_rloo.build_launch(cfg)
    assert command[:4]==[sys.executable,'-B','-m','action_head_rl.trainer']
    assert '--text-backend' in command and command[command.index('--text-backend')+1]=='megatron'
    assert env['ACTION_HEAD_SEQUENCE_RLOO']==str(int(settings.credit_level=='sequence'))
    assert env['ACTION_HEAD_TRAIN_ACTION_HEAD']==str(int(settings.train_modules!='brain'))
    assert env['ACTION_HEAD_TRAIN_TEXT_POLICY']==str(int(settings.train_modules!='head'))
    assert float(env['ACTION_HEAD_RULE_INFORMATION_COEFFICIENT'])==settings.information_reward_coefficient
    assert float(env['ACTION_HEAD_RULE_DISTANCE_COEFFICIENT'])==settings.distance_reward_coefficient
    assert float(env['ACTION_HEAD_RULE_FIRST_ARRIVAL_REWARD'])==settings.first_arrival_reward
    assert float(env['ACTION_HEAD_RULE_TERMINAL_SUCCESS_REWARD'])==settings.terminal_success_reward
    assert not any('dlc' in part or 'submit' in part for part in command)


def test_run_reaches_gpu_command_without_platform(inputs,monkeypatch):
    calls=[]
    def execute(command,env,receipt):
        calls.append(command)
        return 17
    monkeypatch.setattr(run_rloo,'execute',execute)
    assert run_rloo.run(config(f'run.output_dir={inputs}/run'))==17
    assert calls[0][3]=='action_head_rl.trainer'
    assert (inputs/'run/launch/config.yaml').exists()


def test_prepare_never_executes(inputs,monkeypatch):
    monkeypatch.setattr(run_rloo,'execute',lambda *a:pytest.fail('prepare executed GPU work'))
    assert run_rloo.run(config('run.mode=prepare',f'run.output_dir={inputs}/prepared'))==0
    data=OmegaConf.load(inputs/'prepared/launch/config.yaml')
    assert data.rloo.credit_level == 'sequence'
    assert data.rloo == config().rloo
    assert data.recipe.terminal_success_reward == data.rloo.terminal_success_reward
    assert not (inputs/'prepared/launch/config.json').exists()


def test_supervisor_returns_real_exit_and_refuses_replay(tmp_path,monkeypatch):
    monkeypatch.setenv('GPU_CACHE_ROOT',str(tmp_path/'cache'))
    env=gpu_runtime.environment(tmp_path)
    receipt=tmp_path/'result.json'
    assert gpu_runtime.execute([sys.executable,'-c','raise SystemExit(7)'],env,receipt)==7
    assert json.loads(receipt.read_text())['returncode']==7
    with pytest.raises(FileExistsError):
        gpu_runtime.execute([sys.executable,'-c','raise SystemExit(0)'],env,receipt)


def test_expansion_rejects_missing_inputs(monkeypatch):
    monkeypatch.delenv('NOT_SET_GPU_TEST',raising=False)
    with pytest.raises(ValueError,match='NOT_SET_GPU_TEST'):
        gpu_runtime.expand('${NOT_SET_GPU_TEST}')


def test_invalid_old_submission_mode():
    with pytest.raises(ValueError,match='run.mode'):
        run_rloo.run(config('run.mode=submit'))


def test_openloop_prepares_and_dispatches_existing_training_entry(tmp_path,monkeypatch):
    import run_openloop_gpu as sft
    monkeypatch.setenv('RL_ARTIFACT_ROOT',str(tmp_path))
    pub=tmp_path/'inputs.json'
    pub.write_text(json.dumps({'sources':{name:{'counts':{'accepted_anchors':64},
        'manifest':{'path':str(tmp_path/(name+'.jsonl'))}} for name in ['sage','r2r','rxr']}}))
    config={'slow_model':str(tmp_path/'brain'),'fast_backbone':str(tmp_path/'backbone'),
        'verl_source_root':str(tmp_path/'verl'),'published_inputs':str(pub),
        'published_index':str(tmp_path/'indexes'),
        'initialization':{'base_head_checkpoint':str(tmp_path/'base_head'),
                          'base_query_checkpoint':str(tmp_path/'base_query')},
        'action_head_as_tool_version':'v8','source_sampling_ratio':{'sage':1,'r2r':1,'rxr':2},
        'fm_prefix_reuse':True,'local_replica_load':True,'checkpoint_stride_epochs':.2,
        'v8_label_profile':'exec4','learning_rate':5e-5}
    path=tmp_path/'sft.json';path.write_text(json.dumps(config))
    root=tmp_path/'sft'
    assert sft.main(['prepare','--config',str(path),'--output-dir',str(root)])==0
    launch=json.loads((root/'launch-plan.json').read_text())
    assert launch['environment']['NNODES']=='4'
    assert launch['environment']['NPROC_PER_NODE']=='16'
    assert launch['environment']['LEARNING_RATE']=='5e-05'
    calls=[]
    monkeypatch.setattr(sft,'execute',lambda command,env,receipt: calls.append((command,env)) or 0)
    assert sft.main(['initialize','--config',str(path),'--output-dir',str(root)])==0
    assert calls[-1][0][2].endswith('initialize_action_head_scale.py')
    (root/'initialization/d16').mkdir(parents=True)
    with pytest.raises(SystemExit):
        sft.main(['run','--config',str(path),'--output-dir',str(root)])
    gpu_runtime.save(root/'launch/initialize-node0.json',{'returncode':0})
    gpu_runtime.save(root/'initialization/d16/scale_report.json',{})
    for name in ('head','query'):
        (root/'initialization/d16'/name).mkdir()
    assert sft.main(['run','--config',str(path),'--output-dir',str(root),'--node-rank','2','--master-addr','127.0.0.1'])==0
    assert calls[-1][0][1].endswith('action_head_dualvln_sft/entry.sh')
    assert calls[-1][1]['NODE_RANK']=='2' and calls[-1][1]['MASTER_ADDR']=='127.0.0.1'


def test_services_use_same_reward_configuration(monkeypatch):
    import run_rl_services_gpu as services
    monkeypatch.setenv('TRAIN_CASES','/data/cases.jsonl')
    calls=[]
    monkeypatch.setattr(services.os,'execvpe',lambda file,command,env:calls.append((command,env)))
    services.run(config('rloo.credit_level=turn','rloo.terminal_success_reward=0'))
    command,env=calls[0]
    assert command[-1]=='run'
    assert float(env['ACTION_HEAD_RULE_TERMINAL_SUCCESS_REWARD'])==0
    assert env['ACTION_HEAD_SEQUENCE_RLOO']=='0'


def test_missing_binary_has_failure_receipt(tmp_path,monkeypatch):
    monkeypatch.setenv('GPU_CACHE_ROOT',str(tmp_path/'cache'))
    receipt=tmp_path/'bad.json'
    assert gpu_runtime.execute(['/missing/executable'],gpu_runtime.environment(tmp_path),receipt)==127
    assert json.loads(receipt.read_text())['phase']=='failed'


def test_hydra_typo_rejected_and_later_override_wins():
    with pytest.raises(ConfigCompositionException):
        config('rloo.information_reward_coeficient=0')
    cfg=config('rloo.information_reward_coefficient=0','rloo.information_reward_coefficient=0.2')
    assert run_rloo.settings_from_config(cfg).information_reward_coefficient == .2


def test_overrides_reach_actual_trainer_and_resolved_yaml(inputs,monkeypatch):
    cfg=config(f'run.output_dir={inputs}/custom','run.mode=prepare',
        'recipe.learning_rate=5e-7','recipe.opt_steps=12','run.batch_size=256',
        'rloo.terminal_success_reward=2','rloo.train_modules=head')
    _,doc,command,env=run_rloo.build_launch(cfg)
    assert float(command[command.index('--learning-rate')+1]) == 5e-7
    assert command[command.index('--opt-steps')+1] == '12'
    assert doc['recipe']['num_groups_per_step'] == 32
    assert doc['recipe']['case_source_groups'] == {'sage':16,'r2r':8,'rxr':8}
    assert float(command[command.index('--terminal-success-reward')+1]) == 2
    assert env['ACTION_HEAD_TRAIN_TEXT_POLICY'] == '0'
    monkeypatch.setattr(run_rloo,'execute',lambda *a:pytest.fail('prepare executed GPU work'))
    run_rloo.run(cfg)
    saved=OmegaConf.load(inputs/'custom/launch/config.yaml')
    assert saved.recipe.learning_rate == 5e-7
    assert saved.recipe.num_groups_per_step == 32
    assert saved.rloo.terminal_success_reward == 2
    assert '${' not in (inputs/'custom/launch/config.yaml').read_text()


def test_yaml_defaults_drive_both_trainer_and_services(inputs,monkeypatch):
    import run_rl_services_gpu as services
    cfg=config('rloo.credit_level=turn','rloo.terminal_success_reward=3',
        f'run.output_dir={inputs}/shared')
    _,_,_,train_env=run_rloo.build_launch(cfg)
    calls=[]
    monkeypatch.setattr(services.os,'execvpe',lambda file,command,env:calls.append(env))
    services.run(cfg)
    for key,value in run_rloo.settings_from_config(cfg).environment().items():
        assert calls[0][key] == train_env[key] == value
    with pytest.raises(ValueError,match='exactly'):
        run_rloo.settings_from_config(config('~rloo.first_arrival_reward'))


def test_shell_cli_uses_hydra_and_user_override_wins():
    script=gpu_runtime.REPO/'scripts/train/rloo/turn/run_rloo_turn_no_information.sh'
    output=subprocess.check_output([str(script),'rloo.information_reward_coefficient=0.3','--cfg','job'],text=True)
    cfg=OmegaConf.create(output)
    assert cfg.rloo.credit_level == 'turn'
    assert cfg.rloo.information_reward_coefficient == .3
    assert cfg.rloo.train_modules == 'joint'


def test_native_hydra_cli_prepare_and_custom_yaml_are_source_of_defaults(inputs):
    yaml=OmegaConf.load(gpu_runtime.REPO/'configs/gpu/rloo.yaml')
    yaml.rloo.information_reward_coefficient=.125
    yaml.rloo.train_modules='head'
    config_dir=inputs/'custom-config'
    config_dir.mkdir()
    OmegaConf.save(yaml,config_dir/'experiment.yaml')
    output=inputs/'cli-run'
    result=subprocess.run([str(gpu_runtime.REPO/'scripts/train/rloo/run_rloo.sh'),
        '--config-path',str(config_dir),'--config-name','experiment',
        'run.mode=prepare',f'run.output_dir={output}',
        'rloo.credit_level=turn'],cwd=inputs,text=True,capture_output=True)
    assert result.returncode == 0, result.stderr
    saved=OmegaConf.load(output/'launch/config.yaml')
    assert saved.rloo.credit_level == 'turn'
    assert saved.rloo.train_modules == 'head'
    assert saved.rloo.information_reward_coefficient == .125
    assert not (output/'launch/trainer.started.json').exists()
    assert not (inputs/'outputs').exists()
    assert not (inputs/'.hydra').exists()
    assert not list(inputs.glob('*.log'))
