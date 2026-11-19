"""RLOO experiment selection; sequence/full is our main method and default.

Only map hyperparameters to existing training switches. Reward calculation,
leave-one-out credit and frozen optimizers stay in their existing implementations.
Root launchers execute directly on user-owned GPU hosts. Old saved plans retain their explicit settings.
"""
from dataclasses import asdict, dataclass, replace
import math


@dataclass(frozen=True)
class RLOOSettings:
    credit_level: str = 'sequence'
    train_modules: str = 'joint'
    distance_reward_coefficient: float = 1.0
    information_reward_coefficient: float = .05
    first_arrival_reward: float = 1.0
    terminal_success_reward: float = 1.0

    def validate(self):
        if self.credit_level not in ('sequence', 'turn'):
            raise ValueError('credit_level must be sequence or turn')
        if self.train_modules not in ('joint', 'brain', 'head'):
            raise ValueError('train_modules must be joint, brain or head')
        for key, value in asdict(self).items():
            if key.endswith(('coefficient', 'reward')) and (
                    not math.isfinite(value) or value < 0):
                raise ValueError(key + ' must be finite and nonnegative')
        return self

    def environment(self):
        """The same reward/module switches for GPU trainers and environment services."""
        self.validate()
        return dict(ACTION_HEAD_SEQUENCE_RLOO=str(int(self.credit_level == 'sequence')),
            ACTION_HEAD_TRAIN_ACTION_HEAD=str(int(self.train_modules != 'brain')),
            ACTION_HEAD_TRAIN_TEXT_POLICY=str(int(self.train_modules != 'head')),
            ACTION_HEAD_RULE_FIRST_ARRIVAL_REWARD=str(self.first_arrival_reward),
            ACTION_HEAD_RULE_TERMINAL_SUCCESS_REWARD=str(self.terminal_success_reward),
            ACTION_HEAD_RULE_DISTANCE_COEFFICIENT=str(self.distance_reward_coefficient),
            ACTION_HEAD_RULE_INFORMATION_COEFFICIENT=str(self.information_reward_coefficient),
            SAGE_RULE_CREDIT='1', ACTION_HEAD_RULE_CREDIT='1',
            ACTION_HEAD_EXECUTABLE_POLICY='', ACTION_HEAD_FINAL_CONTRACT='')

    def check_plan(self, plan):
        """A fresh submission name must not silently adopt another variant."""
        expected = self.apply({'config': {}})
        for key in ('sequence_rloo', 'freeze_action_head', 'freeze_text_policy',
                    'rule_reward_coefficients', 'first_arrival_reward'):
            if plan.get(key) != expected[key]:
                raise ValueError('existing training variant differs at ' + key + '; use a fresh name')
        if plan['config']['terminal_success_reward'] != self.terminal_success_reward:
            raise ValueError('existing training terminal reward differs; use a fresh name')

    def apply(self, plan):
        """Explicitly overwrite all six knobs; never inherit another ablation."""
        self.validate()
        plan['config'].update(tool_credit_mode='rule_call_loo',
            rule_credit_version='v3_terminal_local', success_contract='native_success',
            terminal_success_reward=self.terminal_success_reward)
        plan.update(sequence_rloo=self.credit_level == 'sequence',
            freeze_action_head=self.train_modules == 'brain',
            freeze_text_policy=self.train_modules == 'head',
            rule_reward_coefficients=dict(distance=self.distance_reward_coefficient,
                information=self.information_reward_coefficient),
            first_arrival_reward=self.first_arrival_reward,
            rloo_hyperparameters=asdict(self))
        return plan


# Four reward removals, two module freezes and two previously used reward scales.
# The scale presets are available for both credit levels; availability is not
# evidence that every combination has already completed a GPU experiment.
ABLATIONS = {
    'full': ({}, '完整奖励，联合训练大脑与query/FM动作头'),
    'no-distance': ({'distance_reward_coefficient': 0.}, '仅去掉距离奖励'),
    'no-information': ({'information_reward_coefficient': 0.}, '仅去掉信息发现奖励'),
    'no-arrival': ({'first_arrival_reward': 0.}, '仅去掉首次到达奖励'),
    'no-terminal': ({'terminal_success_reward': 0.}, '仅去掉终局成功奖励，原生成功评分不变'),
    'brain-only': ({'train_modules': 'brain'}, '只训大脑，冻结query和FM动作头及其优化器/调度器'),
    'head-only': ({'train_modules': 'head'}, '只训query和FM动作头，冻结大脑优化器/调度器'),
    'information-01': ({'information_reward_coefficient': .1}, '信息发现奖励系数从0.05提高至0.1'),
    'terminal-2': ({'terminal_success_reward': 2.}, '终局成功奖励从1提高至2'),
}
PRESETS = {f'{level}-{name}': replace(RLOOSettings(), credit_level=level, **changes)
    for level in ('sequence', 'turn') for name, (changes, _) in ABLATIONS.items()}
DEFAULT_VARIANT = 'sequence-full'


def add_arguments(parser):
    parser.add_argument('--variant', choices=PRESETS, default=None,
        help='默认 sequence-full：我们的主方法，序列级RLOO＋完整奖励＋联合训练')
    parser.add_argument('--credit-level', choices=['sequence', 'turn'],
        help='sequence整轨迹回报；turn按轮次信用；两者均LOO、不除标准差')
    parser.add_argument('--train-modules', choices=['joint', 'brain', 'head'])
    parser.add_argument('--distance-reward-coefficient', type=float)
    parser.add_argument('--first-arrival-reward', type=float)
    # These legacy flags remain accepted by the existing submission API.
    parser.add_argument('--sequence-rloo', action='store_true', default=None,
        help='兼容旧命令；新命令优先用 --credit-level sequence')
    parser.add_argument('--rloo-ablation', choices=list(ABLATIONS),
        help='兼容旧序列级消融命令；与 --variant 互斥')
    parser.add_argument('--terminal-success-reward', type=float)
    parser.add_argument('--information-reward-coefficient', type=float)


def resolve(args):
    variant = getattr(args, 'variant', None)
    legacy = getattr(args, 'rloo_ablation', None)
    if variant and legacy:
        raise ValueError('choose --variant or --rloo-ablation, not both')
    variant = variant or ('sequence-' + legacy if legacy else DEFAULT_VARIANT)
    settings = PRESETS[variant]
    changes = {key: getattr(args, key) for key in asdict(settings)
        if getattr(args, key, None) is not None}
    if getattr(args, 'sequence_rloo', None):
        if changes.get('credit_level', settings.credit_level) != 'sequence':
            raise ValueError('--sequence-rloo conflicts with turn credit')
        changes['credit_level'] = 'sequence'
    return variant, replace(settings, **changes).validate()
