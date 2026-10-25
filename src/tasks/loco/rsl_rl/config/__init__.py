"""Training configurations for HIM and DreamWaQ variants."""

from .him_cfg import (
    HIMAMPOnPolicyRunnerCfg,
    HIMAMPActorCriticCfg,
    HIMAMPPPOAlgorithmCfg,
    HIMOnPolicyRunnerCfg,
    HIMActorCriticCfg,
    HIMPPOAlgorithmCfg,
)

__all__ = [
    "HIMActorCriticCfg", "HIMPPOAlgorithmCfg", "HIMOnPolicyRunnerCfg",
    "HIMAMPActorCriticCfg", "HIMAMPPPOAlgorithmCfg", "HIMAMPOnPolicyRunnerCfg",
]

from .dreamwaq_cfg import (
    DreamWaQActorCriticCfg,
    DreamWaQPPOAlgorithmCfg,
    DreamWaQOnPolicyRunnerCfg,
    DreamWaQAMPActorCriticCfg,
    DreamWaQAMPPPOAlgorithmCfg,
    DreamWaQAMPOnPolicyRunnerCfg,
)

__all__ += ['DreamWaQActorCriticCfg', 'DreamWaQPPOAlgorithmCfg', 'DreamWaQOnPolicyRunnerCfg', 'DreamWaQAMPActorCriticCfg', 'DreamWaQAMPPPOAlgorithmCfg', 'DreamWaQAMPOnPolicyRunnerCfg']

from .cts_cfg import (
    CTSActorCriticCfg,
    CTSAlgorithmCfg,
    CTSOnPolicyRunnerCfg,
    CTSAMPActorCriticCfg,
    CTSAMPAlgorithmCfg,
    CTSAMPOnPolicyRunnerCfg,
)

__all__ += ['CTSActorCriticCfg', 'CTSAlgorithmCfg', 'CTSOnPolicyRunnerCfg', 'CTSAMPActorCriticCfg', 'CTSAMPAlgorithmCfg', 'CTSAMPOnPolicyRunnerCfg']
