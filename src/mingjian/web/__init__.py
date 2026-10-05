"""Small, dependency-free web workspace for the MingJian MVP."""

from .app import MingJianWebApp, create_server

__all__ = ["MingJianWebApp", "create_server"]
