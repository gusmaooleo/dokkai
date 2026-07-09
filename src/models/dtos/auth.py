"""
DTOs for the auth endpoints.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    token: str
    expires_at: datetime
    role: str
    default_admin_active: bool


class MeResponse(BaseModel):
    username: str
    role: str
    expires_at: datetime
    default_admin_active: bool


class StatusResponse(BaseModel):
    enabled: bool
    default_admin_active: bool


class UserDTO(BaseModel):
    id: int
    username: str
    role: str
    created_at: datetime


class CreateUserRequest(BaseModel):
    username: str
    password: str
    role: Literal["admin", "user", "viewer"]


class CreateUserResponse(BaseModel):
    id: int
    username: str
    role: str


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str
