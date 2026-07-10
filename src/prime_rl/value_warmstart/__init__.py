"""Offline value-network warm-start: pretrain V and Q+ on Monte-Carlo returns
from the SFT policy so the action-level ARM run starts from warm value networks
instead of zero-init (thesis §8.2)."""
