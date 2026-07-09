"""
Auth API endpoints.

POST   /auth/login              — authenticate, get a bearer token (public)
POST   /auth/logout             — invalidate the current session
GET    /auth/me                 — current user info
GET    /auth/status             — auth-enabled flag + default admin warning (public)
GET    /auth/users              — admin: list users
POST   /auth/users              — admin: create a user
DELETE /auth/users/{id}         — admin: delete a user
PATCH  /auth/users/me/password  — any authenticated user: change own password
"""

from __future__ import annotations

import asyncpg
from fastapi import APIRouter, Depends, HTTPException, Request

from models.dtos.auth import (
    ChangePasswordRequest,
    CreateUserRequest,
    CreateUserResponse,
    LoginRequest,
    LoginResponse,
    MeResponse,
    StatusResponse,
    UserDTO,
)
from services.auth import (
    create_session,
    create_user,
    delete_session,
    delete_user,
    ensure_seeded,
    get_user_by_username,
    is_default_admin_active,
    list_users,
    purge_expired,
    require_auth,
    require_role,
    update_password,
    verify_password,
)
from services.db import DatabaseUnavailableError

router = APIRouter(prefix="/auth")

require_admin = require_role("admin")


@router.post("/login", response_model=LoginResponse)
async def login(data: LoginRequest):
    """
    Authenticate with username/password and receive a bearer token.

    Calls ``ensure_seeded`` first — self-healing a boot that happened while
    Postgres was down — and opportunistically purges expired sessions. A
    dead Postgres surfaces as 503; bad credentials are a single 401 message
    so a caller can't tell a wrong username from a wrong password.
    """
    try:
        await ensure_seeded()
        await purge_expired()
        user = await get_user_by_username(data.username)
        default_admin_active = await is_default_admin_active()
    except DatabaseUnavailableError as e:
        raise HTTPException(status_code=503, detail=str(e))

    if user is None or not verify_password(data.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="invalid username or password")

    token, expires_at = await create_session(user["id"])
    return LoginResponse(
        token=token,
        expires_at=expires_at,
        role=user["role"],
        default_admin_active=default_admin_active,
    )


@router.post("/logout")
async def logout(request: Request, user: dict = Depends(require_auth)):
    """Invalidate the session used for this request."""
    await delete_session(request.state.token)
    return {"message": "logged out"}


@router.get("/me", response_model=MeResponse)
async def me(user: dict = Depends(require_auth)):
    """Return the authenticated user's info."""
    try:
        default_admin_active = await is_default_admin_active()
    except DatabaseUnavailableError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return MeResponse(
        username=user["username"],
        role=user["role"],
        expires_at=user["expires_at"],
        default_admin_active=default_admin_active,
    )


@router.get("/status", response_model=StatusResponse)
async def status():
    """Public: whether auth is enabled and whether the default admin/admin
    credentials are still active."""
    try:
        default_admin_active = await is_default_admin_active()
    except DatabaseUnavailableError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return StatusResponse(enabled=True, default_admin_active=default_admin_active)


@router.get("/users", response_model=list[UserDTO])
async def get_users(admin: dict = Depends(require_admin)):
    """Admin: list every user. Never includes password hashes."""
    try:
        rows = await list_users()
    except DatabaseUnavailableError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return [
        UserDTO(id=row["id"], username=row["username"], role=row["role"], created_at=row["created_at"])
        for row in rows
    ]


@router.post("/users", response_model=CreateUserResponse, status_code=201)
async def post_user(data: CreateUserRequest, admin: dict = Depends(require_admin)):
    """Admin: create a user with a role."""
    try:
        user = await create_user(data.username, data.password, data.role)
    except asyncpg.UniqueViolationError:
        raise HTTPException(status_code=409, detail=f"username '{data.username}' already exists")
    except DatabaseUnavailableError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return CreateUserResponse(id=user["id"], username=user["username"], role=user["role"])


@router.delete("/users/{user_id}")
async def remove_user(user_id: int, admin: dict = Depends(require_admin)):
    """Admin: delete a user. Cannot delete yourself or the last admin."""
    if user_id == admin["id"]:
        raise HTTPException(status_code=400, detail="you cannot delete your own user")

    try:
        users = await list_users()
    except DatabaseUnavailableError as e:
        raise HTTPException(status_code=503, detail=str(e))

    target = next((u for u in users if u["id"] == user_id), None)
    if target is None:
        raise HTTPException(status_code=404, detail="user not found")

    admin_count = sum(1 for u in users if u["role"] == "admin")
    if target["role"] == "admin" and admin_count <= 1:
        raise HTTPException(status_code=400, detail="cannot delete the last admin user")

    try:
        await delete_user(user_id)
    except DatabaseUnavailableError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return {"message": "user deleted"}


@router.patch("/users/me/password")
async def change_own_password(
    data: ChangePasswordRequest, request: Request, user: dict = Depends(require_auth)
):
    """
    Change the authenticated user's own password.

    Verifies ``current_password`` first (401 on mismatch). Invalidates every
    OTHER session for this user — the session making this request stays
    valid.
    """
    try:
        full_user = await get_user_by_username(user["username"])
    except DatabaseUnavailableError as e:
        raise HTTPException(status_code=503, detail=str(e))

    if full_user is None or not verify_password(data.current_password, full_user["password_hash"]):
        raise HTTPException(status_code=401, detail="invalid current password")

    await update_password(user["id"], data.new_password, keep_token=request.state.token)
    return {"message": "password updated"}
